import numpy as np
import pandas as pd
import pytest
import zlib
from copy import deepcopy

from src.features.canonicalize import canonicalize_pdf_syntax
from src.features.engineered import EngineeredFeatureBuilder
from src.features.filters import (
    BoundedDecodeError,
    DecodeLimits,
    bounded_decode_stream,
    inspect_bounded_object_streams,
)
from src.features.pipeline import FeaturePipelineV2
from src.features.schema_v2 import BASE_FEATURE_COLUMNS, schema_dictionary
from src.features.selection import audit_file_size_shortcut
from src.features.structural import extract_structural_features_with_status
from src.features.vectorizer import extract_features_record, pdf_to_pipeline_vector
from src.data.splitter import SplitRequirements, build_frozen_splits_from_dataset
from src.data.validate import validate_feature_table
from src.experiment import create_experiment_identity, load_experiment_config
from tests.phase_helpers import canonical_frame


def test_name_escape_canonicalization_ignores_stream_payload():
    raw = (
        b"%PDF-1.7\n1 0 obj << /S /J#61vaScript /JS 2 0 R >> endobj\n"
        b"2 0 obj << /Length 23 >>\nstream\n/JavaScript /Launch\nendstream\nendobj\n"
        b"xref\ntrailer\nstartxref\n0\n%%EOF\n"
    )
    canonical, report = canonicalize_pdf_syntax(raw)
    assert b"/JavaScript" in canonical
    assert canonical.count(b"/JavaScript") == 1
    assert b"/Launch" not in canonical
    assert report.decoded_name_escapes == 1
    assert report.stream_bodies_skipped == 1


def test_canonicalization_keeps_comments_and_literal_strings_opaque():
    canonical, report = canonicalize_pdf_syntax(
        b"%PDF-1.7\n% /JavaScript\n1 0 obj << /Title (/JavaScript 100% safe) "
        b"/S /J#61vaScript >> endobj\n%%EOF"
    )
    assert canonical.count(b"/JavaScript") == 1
    assert report.comments_removed == 3


def test_structural_extractor_counts_obfuscated_name_outside_stream(tmp_path):
    path = tmp_path / "safe-fixture.pdf"
    path.write_bytes(
        b"%PDF-1.7\n1 0 obj << /S /J#61vaScript >> endobj\n"
        b"xref\ntrailer\nstartxref\n0\n%%EOF\n"
    )
    result = extract_structural_features_with_status(path)
    assert result.features["javascript_count"] == 1
    assert result.features["obfuscation_count"] == 1
    assert not result.status.parse_failure


def test_every_engineered_feature_has_formula_unit_range_rationale_and_lineage():
    specs = EngineeredFeatureBuilder().feature_specs()
    assert specs
    for spec in specs:
        assert spec.formula and spec.unit and spec.rationale and spec.lineage
        assert spec.minimum is not None
    dictionary = schema_dictionary()
    assert dictionary["feature_count"] == 37
    assert all("train_available" in feature for feature in dictionary["features"])


def test_every_actual_model_input_has_complete_catalog_entry():
    frame = canonical_frame(100)
    pipeline = FeaturePipelineV2(model_family="neural").fit(
        frame, partition_name="train"
    )
    specs = pipeline.output_feature_specs()
    assert [spec["name"] for spec in specs] == list(pipeline.output_feature_names_)
    assert len(specs) == len({spec["name"] for spec in specs})
    for spec in specs:
        assert spec["formula"] and spec["unit"] and spec["rationale"]
        assert spec["lineage"]
        assert "minimum" in spec and "maximum" in spec
        assert spec["range_basis"]


def test_pipeline_serialization_fails_without_verified_upstream_provenance(tmp_path):
    pipeline = FeaturePipelineV2(model_family="tree").fit(
        canonical_frame(100), partition_name="train"
    )
    with pytest.raises(RuntimeError, match="provenance"):
        pipeline.save(tmp_path / "unverified.pkl")


def test_file_size_shortcut_audit_uses_train_direction_and_held_out_validation(
    tmp_path,
):
    train = canonical_frame(100)
    validation = canonical_frame(100).copy()
    for frame in (train, validation):
        frame["Class"] = np.tile([0, 1], 50)
        frame["pdf_size"] = np.where(frame["Class"].eq(1), 10_000.0, 100.0)
    train_path = tmp_path / "train"
    validation_path = tmp_path / "validation"
    train_path.mkdir()
    validation_path.mkdir()
    train.to_parquet(train_path / "part.parquet", index=False)
    validation.to_parquet(validation_path / "part.parquet", index=False)
    audit = audit_file_size_shortcut(train_path, validation_path)
    assert audit.train_direction == "increasing"
    assert audit.validation_auc_train_direction == 1.0
    assert audit.high_risk_shortcut
    assert set(audit.excluded_identity_fields) == {"creator", "producer"}


def test_golden_batch_record_feature_parity_on_100_sanitized_rows():
    frame = canonical_frame(100)
    pipeline = FeaturePipelineV2(model_family="tree")
    batch = pipeline.fit_transform(frame, partition_name="train")
    assert len(batch) == 100
    for position, (_, row) in enumerate(frame.iterrows()):
        record = {name: float(row[name]) for name in BASE_FEATURE_COLUMNS}
        single = pipeline.transform_record(record)
        np.testing.assert_allclose(
            single.to_numpy(), batch.iloc[[position]].to_numpy(), rtol=0, atol=0
        )


def _safe_pdf_bytes(marker: int) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
        b"/Resources << /Font << >> >> /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    content = bytearray(b"%PDF-1.7\n" + (b"% safe-padding\n" * marker))
    offsets = [0]
    for number, value in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode() + value + b"\nendobj\n")
    xref_offset = len(content)
    content.extend(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return bytes(content)


def test_golden_live_batch_parity_on_100_safe_pdf_fixtures(tmp_path):
    paths = []
    records = []
    for index in range(100):
        path = tmp_path / f"safe-{index:03d}.pdf"
        path.write_bytes(_safe_pdf_bytes(index % 5))
        record, diagnostics = extract_features_record(path)
        assert not diagnostics["abstain_recommended"]
        paths.append(path)
        records.append(record)
    frame = pd.DataFrame(records)
    pipeline = FeaturePipelineV2(model_family="tree").fit(
        frame, partition_name="train"
    )
    batch = pipeline.transform(frame).to_numpy(dtype=np.float32)
    for index, path in enumerate(paths):
        live, _, _ = pdf_to_pipeline_vector(path, pipeline=pipeline)
        np.testing.assert_array_equal(live, batch[index])


def test_pipeline_fit_save_load_requires_verified_phase1_and_sealed_phase2(
    tmp_path, monkeypatch
):
    import src.data.splitter as splitter_module
    import src.data.validate as validate_module

    config = deepcopy(load_experiment_config())
    config["split_version"] = "integration-v2"
    config["acceptance_gates"].update(
        minimum_input_rows=120,
        minimum_clean_rows=120,
        minimum_train_rows=1,
        minimum_validation_rows=1,
        minimum_test_rows=1,
    )
    monkeypatch.setattr(validate_module, "load_experiment_config", lambda: config)
    monkeypatch.setattr(splitter_module, "load_experiment_config", lambda: config)
    source = tmp_path / "approved.parquet"
    canonical_frame(120).to_parquet(source, index=False)
    validate_feature_table(
        source,
        source_id="integration-approved",
        output_dir=tmp_path / "clean",
        report_dir=tmp_path / "reports",
        batch_size=23,
        enforce_gates=True,
    )
    requirements = SplitRequirements(1, 1, 1, 0.995, 0.2, 0.2, "integration-v2")
    split_root = tmp_path / "split"
    split_manifest = build_frozen_splits_from_dataset(
        tmp_path / "clean",
        requirements=requirements,
        output_root=split_root,
        batch_size=19,
    )
    pipeline = FeaturePipelineV2(model_family="neural")
    pipeline.fit_dataset(
        split_root / "train" / "features",
        partition_name="train",
        sealed_split_root=split_root,
        dataset_quality_path=tmp_path / "reports" / "dataset_quality.json",
        transformation_manifest_path=tmp_path / "reports" / "transformation_manifest.json",
        batch_size=13,
        quantile_sample_rows=40,
    )
    assert pipeline.metadata()["fit_partition"] == "train"
    assert pipeline.preprocessor.fit_rows_ == split_manifest.row_counts["train"]
    assert pipeline.preprocessor.quantile_estimation_rows_ == 40
    identity = create_experiment_identity(config, code_commit="integration-test")
    artifact = pipeline.save(
        tmp_path / "feature_pipeline_v2.pkl", identity=identity
    )
    loaded = FeaturePipelineV2.load(
        artifact,
        identity=identity,
        split_manifest_path=split_root / "split_manifest.json",
        dataset_quality_path=tmp_path / "reports" / "dataset_quality.json",
        transformation_manifest_path=tmp_path / "reports" / "transformation_manifest.json",
    )
    assert loaded.metadata()["split_manifest_hash"]


def test_bounded_filter_decoding_and_object_stream_header_validation():
    decoded = b"10 0 " + b"<< /S /JavaScript >>"
    compressed = zlib.compress(decoded)
    assert bounded_decode_stream(compressed, ["/FlateDecode"]) == decoded
    raw = (
        b"1 0 obj << /Type /ObjStm /N 1 /First 5 /Filter /FlateDecode >>\n"
        b"stream\n" + compressed + b"\nendstream\nendobj"
    )
    inspection = inspect_bounded_object_streams(raw)
    assert inspection.object_streams_seen == 1
    assert inspection.object_streams_validated == 1
    with pytest.raises(BoundedDecodeError, match="expansion-ratio"):
        bounded_decode_stream(
            zlib.compress(b"A" * 10_000),
            ["/FlateDecode"],
            limits=DecodeLimits(maximum_expansion_ratio=2.0),
        )


def test_phase_three_orchestrator_does_not_open_sealed_test(monkeypatch, tmp_path):
    from src import run_all

    observed_partitions = []

    class FakeAudit:
        def to_dict(self):
            return {"status": "test"}

    class FakeBuilder:
        def feature_specs(self):
            return []

        def lineage(self):
            return {}

    class FakePipeline:
        def __init__(self, model_family):
            self.model_family = model_family

        def fit_dataset(self, path, **_kwargs):
            observed_partitions.append(path.parent.name)
            return self

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.model_family.encode("utf-8"))
            return path

        def metadata(self):
            return {"model_family": self.model_family}

        def output_feature_specs(self):
            return []

    def fake_materialize(source, _destination, **_kwargs):
        observed_partitions.append(source.parent.name)
        return {"partition": source.parent.name}

    monkeypatch.setattr(run_all, "EngineeredFeatureBuilder", FakeBuilder)
    monkeypatch.setattr(run_all, "FeaturePipelineV2", FakePipeline)
    monkeypatch.setattr(run_all, "audit_file_size_shortcut", lambda *_a, **_k: FakeAudit())
    monkeypatch.setattr(run_all, "materialize_engineered_layer", fake_materialize)
    monkeypatch.setattr(run_all, "schema_dictionary", lambda: {})
    monkeypatch.setattr(run_all, "atomic_write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(run_all, "DATA_REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(run_all, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(run_all, "MODELS_DIR", tmp_path / "models")

    run_all._phase_three(
        tmp_path / "split",
        split_version="test-v1",
        compatibility_model_family="tree",
        batch_size=10,
    )

    assert set(observed_partitions) == {"train", "validation"}
    assert "test" not in observed_partitions
