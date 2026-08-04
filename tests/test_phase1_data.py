import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.deduplicate import StreamingDeduplicator
from src.data.schema import FeatureTableSchema
from src.data.validate import (
    DatasetGateError,
    row_validation_reasons,
    validate_feature_table,
    validate_registered_source,
)
from tests.phase_helpers import canonical_frame


def test_strict_schema_rejects_fractional_count_without_coercion():
    frame = canonical_frame(2)
    frame.loc[0, "obj_count"] = 1.25
    reasons = row_validation_reasons(frame, FeatureTableSchema())
    assert "non_integral_count:obj_count" in reasons.iloc[0]
    assert reasons.iloc[1] == ""


def test_two_pass_dedup_removes_both_sides_of_label_conflict(tmp_path):
    frame = canonical_frame(4)
    duplicate = frame.iloc[[0]].copy()
    duplicate["Class"] = 1
    combined = pd.concat([frame, duplicate], ignore_index=True)
    with StreamingDeduplicator(tmp_path / "dedup.sqlite") as deduper:
        deduper.observe_batch(combined)
        reasons = deduper.selection_reasons(combined)
        stats = deduper.stats()
    assert (reasons == "conflicting_label_for_sample_id").sum() == 2
    assert stats.conflicting_sample_ids == 1
    assert stats.conflicting_rows == 2
    assert stats.conflicting_feature_hashes == 1


def test_validation_writes_clean_quarantine_and_failed_gate_report(tmp_path):
    frame = canonical_frame(8)
    frame.loc[1, "Class"] = 9
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    source = tmp_path / "approved.parquet"
    frame.to_parquet(source, index=False)
    report = validate_feature_table(
        source,
        source_id="test-approved",
        output_dir=tmp_path / "clean",
        report_dir=tmp_path / "reports",
        batch_size=3,
        enforce_gates=False,
    )
    assert report.row_flow["received"] == 9
    assert report.row_flow["deduplicated"] == 7
    assert report.status == "failed_acceptance_gates"
    assert (tmp_path / "reports" / "dataset_quality.json").is_file()
    assert (tmp_path / "reports" / "transformation_manifest.json").is_file()
    assert list((tmp_path / "clean").glob("*.parquet"))
    assert list((tmp_path / "clean_quarantine").glob("*.parquet"))
    arrow_schema = pq.ParquetFile(
        next((tmp_path / "clean").glob("*.parquet"))
    ).schema_arrow
    assert arrow_schema.field("obj_count").type == pa.int32()
    assert arrow_schema.field("header_valid").type == pa.int8()
    assert arrow_schema.field("pdf_size").type == pa.float32()


def test_payload_column_is_rejected_without_persisting_payload(tmp_path):
    frame = canonical_frame(3)
    frame["raw_bytes"] = [b"MZ-danger"] * 3
    source = tmp_path / "unsafe.parquet"
    frame.to_parquet(source, index=False)
    report = validate_feature_table(
        source,
        source_id="unsafe-test",
        output_dir=tmp_path / "clean",
        report_dir=tmp_path / "reports",
        enforce_gates=False,
    )
    assert report.row_flow["fully_valid"] == 0
    quarantine = pd.read_parquet(next((tmp_path / "clean_quarantine").glob("*.parquet")))
    assert "raw_bytes" not in quarantine.columns
    assert "MZ-danger" not in quarantine.to_string()


def test_disabled_or_unapproved_source_never_enters_validation():
    with pytest.raises(DatasetGateError, match="not eligible"):
        validate_registered_source("cic-evasive-pdfmal2022")
