from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from src.data.access import SealedTestLedger, SplitAccessError
from src.data.splitter import SplitRequirements, build_frozen_splits_from_dataset
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.models.analysis import (
    drift_table,
    metrics_by_subgroup,
    sanitized_hard_errors,
)
from src.models.bundle import CalibratedMember, Phase4ModelBundle
from src.models.calibration import ProbabilityCalibrator, fit_and_select_calibrator
from src.models.matrix import load_model_matrix, materialize_model_matrix
from src.models.metrics import (
    metric_definitions,
    metric_report,
    select_validation_thresholds,
)
from src.models.phase4 import (
    assign_validation_roles,
    default_candidate_specs,
    select_development_indices,
    validate_candidate_contract,
)
from src.models.phase5 import main as phase5_main
from src.models.tabular_transformer import AsymmetricFocalLoss, FTTransformer
from src.models.uncertainty import (
    group_confusion_bootstrap,
    paired_group_bootstrap_difference,
)
from tests.phase_helpers import canonical_frame


def test_complete_metric_formulas_at_realistic_prevalence():
    labels = np.r_[np.zeros(995, dtype=np.int8), np.ones(5, dtype=np.int8)]
    probabilities = np.r_[
        np.full(994, 0.001),
        np.array([0.9]),
        np.array([0.99, 0.8, 0.7, 0.1, 0.05]),
    ]
    report = metric_report(labels, probabilities, threshold=0.5)
    assert report["true_positive"] == 3
    assert report["false_positive"] == 1
    assert report["false_negative"] == 2
    assert report["true_negative"] == 994
    assert report["precision"] == pytest.approx(3 / 4)
    assert report["recall"] == pytest.approx(3 / 5)
    assert report["f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))
    assert report["f2"] == pytest.approx(5 * 3 / (5 * 3 + 4 * 2 + 1))
    assert report["f0_5"] == pytest.approx(1.25 * 3 / (1.25 * 3 + 0.25 * 2 + 1))
    assert report["pr_no_skill_baseline"] == pytest.approx(0.005)
    assert report["false_positives_per_million_benign"] == pytest.approx(
        1_000_000 / 995
    )
    assert report["roc_auc"] is not None
    assert report["pr_auc_average_precision"] is not None


def test_metric_definitions_explain_required_operational_meaning():
    definitions = metric_definitions()
    for name in (
        "precision",
        "recall",
        "f0_5",
        "f1",
        "f2",
        "roc_auc",
        "partial_roc_auc",
        "pr_auc_average_precision",
    ):
        assert definitions[name]["formula"]
        assert definitions[name]["indicates"]
    assert "prevalence" in definitions["precision"]["indicates"].lower()
    assert "ranking" in definitions["roc_auc"]["indicates"].lower()


def test_thresholds_are_validation_only_and_meet_low_fpr_constraints():
    labels = np.r_[np.zeros(20_000, dtype=np.int8), np.ones(100, dtype=np.int8)]
    probabilities = np.r_[
        np.linspace(0.0, 0.2, 20_000),
        np.linspace(0.1, 0.95, 100),
    ]
    with pytest.raises(RuntimeError, match="validation"):
        select_validation_thresholds(labels, probabilities, partition_name="test")
    decisions = select_validation_thresholds(
        labels,
        probabilities,
        partition_name="validation_threshold_selection",
    )
    assert set(decisions) == {
        "fixed_0_5",
        "max_f1",
        "max_f2",
        "fpr_lte_0_001",
        "fpr_lte_0_0001",
    }
    assert (
        decisions["fpr_lte_0_001"].validation_metrics["false_positive_rate"]
        <= 0.001
    )
    assert (
        decisions["fpr_lte_0_0001"].validation_metrics["false_positive_rate"]
        <= 0.0001
    )


def test_low_fpr_threshold_can_lock_zero_alert_point_without_test_tuning():
    labels = np.r_[np.zeros(1_000, dtype=np.int8), np.ones(5, dtype=np.int8)]
    probabilities = np.r_[np.array([0.99]), np.full(999, 0.01), np.full(5, 0.5)]
    decisions = select_validation_thresholds(
        labels,
        probabilities,
        partition_name="validation_threshold_selection",
    )
    strict = decisions["fpr_lte_0_0001"]
    assert strict.validation_metrics["false_positive_rate"] == 0.0
    assert strict.validation_metrics["false_positive"] == 0


def test_calibration_fit_and_selection_are_disjoint_validation_roles():
    fit_labels = np.tile([0, 1], 100)
    select_labels = np.tile([0, 1], 100)
    fit_probability = np.where(fit_labels == 1, 0.65, 0.35)
    select_probability = np.where(select_labels == 1, 0.65, 0.35)
    with pytest.raises(RuntimeError, match="partition"):
        ProbabilityCalibrator("sigmoid").fit(
            fit_probability, fit_labels, partition_name="train"
        )
    calibrator, evidence = fit_and_select_calibrator(
        fit_probability,
        fit_labels,
        select_probability,
        select_labels,
        fit_partition_name="validation_calibration_fit",
        selection_partition_name="validation_calibration_selection",
    )
    assert evidence.fit_on == "validation_calibration_fit"
    assert evidence.selected_on == "validation_calibration_selection"
    predicted = calibrator.predict(select_probability)
    assert np.all((predicted > 0) & (predicted < 1))


def test_group_bootstrap_and_paired_identity_difference():
    labels = np.tile(np.array([0, 0, 0, 1], dtype=np.int8), 30)
    probability = np.where(labels == 1, 0.9, 0.01)
    groups = np.repeat([f"group-{index:03d}" for index in range(30)], 4)
    intervals = group_confusion_bootstrap(
        labels,
        probability,
        groups,
        threshold=0.5,
        replicates=100,
    )
    assert intervals["f2"].point == 1.0
    assert intervals["f2"].lower_95 == 1.0
    difference = paired_group_bootstrap_difference(
        labels,
        probability,
        probability,
        groups,
        candidate_threshold=0.5,
        reference_threshold=0.5,
        replicates=100,
    )
    assert difference.point == 0.0
    assert difference.lower_95 == difference.upper_95 == 0.0


def test_ft_transformer_has_feature_tokens_attention_and_trainable_focal_loss():
    model = FTTransformer(
        12, token_dimension=32, blocks=2, heads=4, residual_dropout=0.0
    )
    features = torch.randn(8, 12)
    tokens = model.tokenize(features)
    logits = model(features)
    assert tokens.shape == (8, 13, 32)
    assert logits.shape == (8, 1)
    loss = AsymmetricFocalLoss(positive_weight=20.0)(
        logits, torch.tensor([[0], [0], [0], [0], [0], [0], [0], [1]])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.feature_identity.grad is not None


def test_required_model_set_and_weighting_ablation_are_predeclared():
    specs = default_candidate_specs()
    required = {
        "always_benign",
        "logistic_regression",
        "extra_trees",
        "lightgbm",
        "xgboost_hist",
        "fully_connected_mlp",
        "ft_transformer",
    }
    assert {spec.model_name for spec in specs} == required
    for model_name in required - {"always_benign"}:
        variants = {spec.variant for spec in specs if spec.model_name == model_name}
        assert variants == {"unweighted", "cost_sensitive"}

    with pytest.raises(RuntimeError, match="inventory"):
        validate_candidate_contract(
            [spec for spec in specs if spec.model_name != "lightgbm"], required
        )
    with pytest.raises(RuntimeError, match="ablation"):
        validate_candidate_contract(
            [
                spec
                for spec in specs
                if not (
                    spec.model_name == "extra_trees"
                    and spec.variant == "cost_sensitive"
                )
            ],
            required,
        )


def test_validation_roles_are_group_disjoint_and_strictly_temporal():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for group_index in range(30):
        for label in (0, 1):
            rows.append(
                {
                    "group_id": f"group-{group_index:03d}",
                    "first_seen_at": start + timedelta(days=group_index),
                    "Class": label,
                }
            )
    metadata = pd.DataFrame(rows)
    roles, manifest = assign_validation_roles(
        metadata, split_manifest_sha256="a" * 64
    )
    assert manifest.group_overlap_zero
    assert manifest.strict_temporal_order
    assert set(roles) == {
        "validation_calibration_fit",
        "validation_calibration_selection",
        "validation_threshold_selection",
    }


def test_development_subset_is_deterministic_and_keeps_complete_groups():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for group_index in range(40):
        for row_index in range(4):
            rows.append(
                {
                    "group_id": f"group-{group_index:03d}",
                    "source_id": f"source-{group_index % 3}",
                    "first_seen_at": start + timedelta(days=group_index),
                    "Class": int(row_index == 0 and group_index % 2 == 0),
                }
            )
    metadata = pd.DataFrame(rows)
    first, audit = select_development_indices(
        metadata, maximum_rows=80, random_seed=42
    )
    second, _ = select_development_indices(
        metadata, maximum_rows=80, random_seed=42
    )
    np.testing.assert_array_equal(first, second)
    selected_groups = set(metadata.iloc[first]["group_id"])
    for group in selected_groups:
        assert set(metadata.index[metadata["group_id"] == group]).issubset(set(first))
    assert len(first) <= 80
    assert audit["strategy"].startswith("complete_group")


def test_subgroup_drift_and_sanitized_error_outputs_are_actionable_and_safe():
    rows = 200
    metadata = pd.DataFrame(
        {
            "sample_id": [f"secret-sample-{index}" for index in range(rows)],
            "source_id": np.where(np.arange(rows) % 2, "source-b", "source-a"),
            "first_seen_at": pd.date_range(
                "2024-01-01", periods=rows, freq="h", tz="UTC"
            ),
            "pdf_size": np.arange(rows) + 100,
            "is_encrypted": np.arange(rows) % 2,
            "js_count": np.arange(rows) % 3,
            "parse_failure": np.arange(rows) % 7 == 0,
            "obfuscation_count": np.arange(rows) % 8,
            "missing_feature_count": np.arange(rows) % 3,
        }
    )
    labels = np.tile([0, 0, 0, 1], rows // 4).astype(np.int8)
    probabilities = np.where(labels == 1, 0.8, 0.1).astype(float)
    probabilities[0] = 0.9
    probabilities[3] = 0.01
    subgroup, unavailable = metrics_by_subgroup(
        metadata, labels, probabilities, threshold=0.5, minimum_rows=10
    )
    assert {"source", "time_window", "parser_status"}.issubset(
        set(subgroup["subgroup_type"])
    )
    assert {entry["subgroup"] for entry in unavailable} == {
        "malware_family",
        "campaign",
    }
    errors = sanitized_hard_errors(
        metadata,
        labels,
        probabilities,
        threshold=0.5,
        experiment_id="experiment-test",
    )
    assert set(errors["error_type"]) == {"false_positive", "false_negative"}
    assert "sample_id" not in errors
    assert not any("secret-sample" in value for value in errors.astype(str).to_numpy().ravel())
    drift = drift_table(
        np.arange(400, dtype=float).reshape(200, 2),
        np.arange(400, 800, dtype=float).reshape(200, 2),
        ("feature_a", "feature_b"),
    )
    assert list(drift["feature"]) == ["feature_a", "feature_b"]
    assert (drift["population_stability_index"] >= 0).all()


def test_phase5_cli_requires_explicit_irreversible_confirmation():
    with pytest.raises(SystemExit) as error:
        phase5_main([])
    assert error.value.code == 2


def _identity_calibrator(probability, labels):
    return ProbabilityCalibrator("identity").fit(
        probability,
        labels,
        partition_name="validation_calibration_fit",
    )


def test_phase4_bundle_roundtrip_and_upstream_tampering(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, 4))
    labels = (features[:, 0] > 0).astype(np.int8)
    estimator = LogisticRegression().fit(features, labels)
    raw_probability = estimator.predict_proba(features)[:, 1]
    members = [
        CalibratedMember(
            seed=seed,
            estimator=estimator,
            calibrator=_identity_calibrator(raw_probability, labels),
            training_configuration={"seed": seed},
        )
        for seed in (42, 1337, 2026)
    ]
    quality = tmp_path / "quality.json"
    split = tmp_path / "split.json"
    pipeline = tmp_path / "pipeline.pkl"
    for path, value in ((quality, b"quality"), (split, b"split"), (pipeline, b"pipe")):
        path.write_bytes(value)
    decisions = select_validation_thresholds(
        labels,
        raw_probability,
        partition_name="validation_threshold_selection",
    )
    bundle = Phase4ModelBundle(
        model_name="logistic_regression",
        model_family="linear",
        variant="cost_sensitive",
        feature_names=("a", "b", "c", "d"),
        members=members,
        thresholds={name: value.to_dict() for name, value in decisions.items()},
        selected_policy="fpr_lte_0_001",
        provenance={
            "dataset_quality_sha256": sha256_file(quality),
            "split_manifest_sha256": sha256_file(split),
            "feature_pipeline_sha256": sha256_file(pipeline),
        },
        training_evidence={"fit_partition": "train_full"},
        calibration_evidence={"fit_partition": "validation_calibration_fit"},
    )
    config = deepcopy(load_experiment_config())
    identity = create_experiment_identity(config, code_commit="phase45-test")
    finalist = bundle.save_finalist(tmp_path / "finalist.pkl", identity=identity)
    loaded = Phase4ModelBundle.load_finalist(
        finalist,
        identity=identity,
        dataset_quality_path=quality,
        split_manifest_path=split,
        feature_pipeline_path=pipeline,
    )
    np.testing.assert_allclose(loaded.predict_proba(features), bundle.predict_proba(features))
    champion = bundle.save_champion(tmp_path / "champion.pkl", identity=identity)
    loaded_champion = Phase4ModelBundle.load_champion(
        champion,
        identity=identity,
        dataset_quality_path=quality,
        split_manifest_path=split,
        feature_pipeline_path=pipeline,
    )
    assert loaded_champion.threshold == bundle.threshold
    quality.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="mismatch"):
        Phase4ModelBundle.load_finalist(
            finalist,
            identity=identity,
            dataset_quality_path=quality,
            split_manifest_path=split,
            feature_pipeline_path=pipeline,
        )


def test_model_matrix_is_immutable_typed_and_checksum_verified(tmp_path):
    frame = canonical_frame(120)
    source = tmp_path / "source"
    source.mkdir()
    frame.to_parquet(source / "part.parquet", index=False)
    pipeline = FeaturePipelineV2(model_family="tree").fit(
        frame, partition_name="train"
    )
    pipeline_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "matrix"
    materialize_model_matrix(
        source,
        output,
        pipeline=pipeline,
        pipeline_path=pipeline_path,
        split_manifest_path=split_manifest,
        partition_name="train",
        batch_size=23,
    )
    features, labels, metadata, manifest = load_model_matrix(
        output,
        expected_partition="train",
        feature_pipeline_path=pipeline_path,
        split_manifest_path=split_manifest,
    )
    assert features.dtype == np.float32
    assert labels.dtype == np.int8
    assert features.shape == (120, manifest.columns)
    assert len(metadata) == 120
    with pytest.raises(FileExistsError, match="immutable"):
        materialize_model_matrix(
            source,
            output,
            pipeline=pipeline,
            pipeline_path=pipeline_path,
            split_manifest_path=split_manifest,
            partition_name="train",
        )
    with (output / "X.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(RuntimeError, match="mismatch"):
        load_model_matrix(
            output,
            expected_partition="train",
            feature_pipeline_path=pipeline_path,
            split_manifest_path=split_manifest,
        )


def test_sealed_test_ledger_is_exclusive_and_one_shot(tmp_path):
    frame = canonical_frame(120)
    source = tmp_path / "clean"
    source.mkdir()
    frame.to_parquet(source / "part.parquet", index=False)
    split_root = tmp_path / "split"
    requirements = SplitRequirements(1, 1, 1, 0.5, 0.2, 0.2, "ledger-v1")
    build_frozen_splits_from_dataset(
        source,
        requirements=requirements,
        output_root=split_root,
        batch_size=31,
    )
    champion = tmp_path / "champion.pkl"
    champion.write_bytes(b"champion")
    output = tmp_path / "metric.csv"
    output.write_text("metric,value\nf2,0.5\n", encoding="utf-8")
    config = deepcopy(load_experiment_config())
    config["split_version"] = "ledger-v1"
    identity = create_experiment_identity(config, code_commit="ledger-test")
    path = tmp_path / "SEALED_TEST_EVALUATION.json"
    ledger = SealedTestLedger(path)
    token = ledger.claim(
        identity=identity,
        split_root=split_root,
        champion_bundle_path=champion,
    )
    assert ledger.authorized_test_features(split_root, token=token).name == "features"
    with pytest.raises(SplitAccessError, match="already"):
        SealedTestLedger(path).claim(
            identity=identity,
            split_root=split_root,
            champion_bundle_path=champion,
        )
    ledger.complete(token=token, output_artifacts={"metric": str(output)})
    assert "completed_test_closed" in path.read_text(encoding="utf-8")
    with pytest.raises(SplitAccessError, match="already"):
        SealedTestLedger(path).claim(
            identity=identity,
            split_root=split_root,
            champion_bundle_path=champion,
        )
