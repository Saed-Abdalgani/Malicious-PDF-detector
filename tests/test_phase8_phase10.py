from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.components.analyzer import PDFAnalyzer
from scripts.sync_results_docs import sync_results_docs
from src.artifacts import metadata_path
from src.config import EXPERIMENT_CONFIG_PATH, PROJECT_ROOT, RESULTS_DIR
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.llm.prompts import EVIDENCE_ONLY_SYSTEM_PROMPT, build_evidence_only_prompt
from src.models.bundle import CalibratedMember, Phase4ModelBundle
from src.models.calibration import ProbabilityCalibrator
from src.models.deployment import DeploymentBundle, observed_indicators
from src.models.phase8 import Phase8GateError, Phase8Runner, _natural_reference_indices
from src.run_all import _stage_main
from tests.phase_helpers import canonical_frame


@pytest.fixture
def deployment_bundle(tmp_path: Path) -> DeploymentBundle:
    frame = canonical_frame(140, malicious_indices=range(0, 140, 10))
    pipeline = FeaturePipelineV2(model_family="tree").fit(frame, partition_name="train")
    matrix = pipeline.transform(frame).to_numpy(dtype=np.float32)
    labels = frame["Class"].to_numpy(dtype=np.int8)
    estimator = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42, solver="liblinear"
    )
    estimator.fit(matrix, labels)
    calibrator = ProbabilityCalibrator("identity").fit(
        np.where(np.tile([0, 1], 70) == 1, 0.8, 0.2),
        np.tile([0, 1], 70),
        partition_name="validation_calibration_fit",
    )
    digest = "a" * 64
    model = Phase4ModelBundle(
        model_name="test_logistic",
        model_family="tree",
        variant="unit_test",
        feature_names=tuple(pipeline.output_feature_names_),
        members=[
            CalibratedMember(
                seed=42,
                estimator=estimator,
                calibrator=calibrator,
                training_configuration={"model_name": "test_logistic"},
            )
        ],
        thresholds={
            "fpr_lte_0_001": {
                "threshold": 0.5,
                "selected_on": "validation_threshold_selection",
            }
        },
        selected_policy="fpr_lte_0_001",
        provenance={
            "dataset_quality_sha256": digest,
            "split_manifest_sha256": digest,
            "feature_pipeline_sha256": digest,
        },
        training_evidence={},
        calibration_evidence={},
    )
    paths = {}
    for name in ("model", "pipeline", "quality", "split", "transformation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    return DeploymentBundle.create(
        model=model,
        feature_pipeline=pipeline,
        explanation_background=matrix[:120],
        source_model_path=paths["model"],
        source_pipeline_path=paths["pipeline"],
        dataset_quality_path=paths["quality"],
        split_manifest_path=paths["split"],
        transformation_manifest_path=paths["transformation"],
        maximum_pdf_bytes=2048,
        uncertainty_probability_margin=0.0,
        ood_feature_fraction_threshold=1.0,
    )


def test_deployment_bundle_three_way_contract_and_evidence_separation(deployment_bundle):
    row = canonical_frame(1).iloc[0]
    record = {name: float(row[name]) for name in deployment_bundle.feature_pipeline.required_input_columns}
    record["js_count"] = 2.0
    decision = deployment_bundle.predict_record(
        record, {"abstain_recommended": False}, include_explanation=True
    )
    assert decision.outcome in {"benign", "malicious"}
    assert decision.malicious_probability is not None
    assert decision.threshold == 0.5
    assert any(item["feature"] == "js_count" for item in decision.raw_indicators)
    assert all(item["evidence_type"].endswith("observation") for item in decision.raw_indicators)
    assert all(item["method"].startswith("local_") for item in decision.model_attributions)

    abstained = deployment_bundle.predict_record(
        record,
        {"abstain_recommended": True, "parse_failure": 1.0},
    )
    assert abstained.outcome == "uncertain/abstain"
    assert abstained.malicious_probability is None
    assert "parse_failure" in abstained.abstention_reasons


def test_corrupted_and_oversized_uploads_fail_closed_without_persistence(deployment_bundle, tmp_path):
    analyzer = PDFAnalyzer(bundle=deployment_bundle)

    class Upload:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def getvalue(self) -> bytes:
            return self.data

    corrupt = analyzer.analyze(Upload(b"not a PDF"))
    oversized = analyzer.analyze(Upload(b"x" * 2049))
    assert corrupt.outcome == oversized.outcome == "uncertain/abstain"
    assert corrupt.model_attributions == oversized.model_attributions == ()
    assert not list(tmp_path.glob("*.pdf"))


def test_deployment_bundle_roundtrip_and_schema_tamper_rejection(deployment_bundle, tmp_path):
    identity = create_experiment_identity(load_experiment_config())
    path = deployment_bundle.save(tmp_path / "bundle.joblib", identity=identity)
    loaded = DeploymentBundle.load(path, identity=identity)
    assert loaded.schema_sha256 == deployment_bundle.schema_sha256
    assert metadata_path(path).is_file()

    tampered = joblib.load(path)
    tampered.schema_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="schema digest"):
        tampered.validate()


def test_evidence_only_llm_contract_rejects_extra_fields():
    evidence = {
        "outcome": "uncertain/abstain",
        "malicious_probability": None,
        "locked_threshold": 0.42,
        "threshold_policy": "fpr_lte_0_001",
        "abstention_reasons": ["parse_failure"],
        "raw_actionable_indicators": [],
        "model_attributions": [],
    }
    prompt = build_evidence_only_prompt(evidence)
    assert "parse_failure" in prompt
    for phrase in ("Never invent", "Do not change", "static evidence"):
        assert phrase.lower() in EVIDENCE_ONLY_SYSTEM_PROMPT.lower()
    with pytest.raises(ValueError, match="Unsupported"):
        build_evidence_only_prompt({**evidence, "pdf_bytes": "forbidden"})


def test_phase8_and_stage_commands_fail_closed_on_current_phase0_state():
    with pytest.raises(Phase8GateError, match="completed.*Phase 7"):
        Phase8Runner().run()
    with pytest.raises(RuntimeError, match="requires upstream status"):
        _stage_main(["train", "--config", str(EXPERIMENT_CONFIG_PATH)])


def test_phase8_reference_is_real_deterministic_and_natural_prevalence():
    labels = np.r_[np.zeros(995, dtype=np.int8), np.ones(5, dtype=np.int8)]
    first = _natural_reference_indices(labels, maximum_rows=200, random_seed=42)
    second = _natural_reference_indices(labels, maximum_rows=200, random_seed=42)
    assert np.array_equal(first, second)
    assert len(first) == 200
    assert labels[first].sum() == 1


def test_documentation_sync_binds_all_generated_outputs():
    manifest_path = sync_results_docs()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manual_metrics_labeled_unverified"] is True
    assert manifest["final_metrics_present"] is False
    assert manifest["experiment_summary_sha256"] == sha256_file(
        RESULTS_DIR / "experiment_summary.json"
    )
    for entry in manifest["outputs"].values():
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file()
        assert sha256_file(path) == entry["sha256"]
    text = (PROJECT_ROOT / "docs/generated/results_summary.md").read_text(encoding="utf-8")
    assert "author-reported and unverified" in text
    assert "MLP (PyTorch) | 92.00% | 86.36%" in text
    assert "no verified final success metric above 90%" in text.lower()


def test_observed_indicators_never_invent_absent_activity():
    indicators = observed_indicators({}, {})
    assert indicators == ()
