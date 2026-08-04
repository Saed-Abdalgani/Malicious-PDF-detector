"""One-shot sealed-test evaluation with complete Phase 5 scientific evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (
    DATA_REPORTS_DIR,
    FIGURES_DIR,
    GROUP_ID_COLUMN,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SAMPLE_ID_COLUMN,
    SPLITS_DIR,
)
from src.data.access import SealedTestLedger, verify_training_split_access
from src.data.validate import verify_validated_dataset
from src.experiment import (
    ExperimentIdentity,
    create_experiment_identity,
    load_experiment_config,
    sha256_file,
)
from src.features.pipeline import FeaturePipelineV2
from src.models.analysis import (
    calibration_table,
    drift_table,
    metrics_by_subgroup,
    sanitized_hard_errors,
)
from src.models.bundle import Phase4ModelBundle
from src.models.matrix import load_model_matrix, materialize_model_matrix
from src.models.metrics import metric_definitions, metric_report
from src.models.uncertainty import (
    group_confusion_bootstrap,
    group_probability_bootstrap,
    paired_group_bootstrap_difference,
)
from src.utils.atomic import atomic_write_json, atomic_write_via


PHASE5_VERSION = "1.0.0"


class Phase5GateError(RuntimeError):
    """Raised before or during the irrevocable sealed-test evaluation."""


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_write_via(path, lambda temporary: frame.to_csv(temporary, index=False))


def _atomic_figure(path: Path, draw: Any) -> Path:
    def writer(temporary: Path) -> None:
        figure = draw()
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    return atomic_write_via(path, writer)


def select_phase6_handoff_indices(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    *,
    threshold: float,
    maximum_rows: int,
    random_seed: int,
) -> np.ndarray:
    """Select deterministic representative test cases without selection feedback."""
    if maximum_rows < 8 or not (len(labels) == len(probabilities) == len(metadata)):
        raise ValueError("Phase 6 handoff inputs or maximum_rows are invalid.")
    predicted = probabilities >= threshold
    outcomes = np.select(
        (
            (labels == 1) & predicted,
            (labels == 0) & ~predicted,
            (labels == 0) & predicted,
            (labels == 1) & ~predicted,
        ),
        ("true_positive", "true_negative", "false_positive", "false_negative"),
        default="invalid",
    )
    identifiers = (
        metadata[SAMPLE_ID_COLUMN].astype(str).to_numpy()
        if SAMPLE_ID_COLUMN in metadata
        else np.arange(len(labels)).astype(str)
    )
    hashes = np.array(
        [
            hashlib.sha256(f"{random_seed}:{value}".encode("utf-8")).hexdigest()
            for value in identifiers
        ]
    )
    selected: list[int] = []
    quota = max(1, maximum_rows // 8)
    for outcome in (
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ):
        positions = np.flatnonzero(outcomes == outcome)
        positions = positions[np.argsort(hashes[positions], kind="mergesort")]
        selected.extend(positions[:quota].tolist())
    confidence_distance = np.abs(probabilities - threshold)
    priority_sets = (
        np.argsort(confidence_distance, kind="mergesort"),
        np.argsort(-confidence_distance, kind="mergesort"),
        np.argsort(hashes, kind="mergesort"),
    )
    seen = set(selected)
    for positions in priority_sets:
        for position in positions:
            value = int(position)
            if value not in seen:
                selected.append(value)
                seen.add(value)
            if len(selected) >= min(maximum_rows, len(labels)):
                return np.array(sorted(selected), dtype=np.int64)
    return np.array(sorted(selected), dtype=np.int64)


def _plot_roc(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    low_fpr: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, probability in sorted(probabilities.items()):
        fpr, tpr, _ = roc_curve(labels, probability)
        axis.plot(fpr, tpr, linewidth=1.8, label=name)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="random ranking")
    if low_fpr:
        axis.set_xlim(0.0, 0.001)
        axis.set_ylim(0.0, 1.0)
        axis.set_title("ROC — operational low-FPR region")
    else:
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Full ROC curves")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("Recall / true-positive rate")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return figure


def _plot_precision_recall(
    labels: np.ndarray, probabilities: dict[str, np.ndarray]
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, probability in sorted(probabilities.items()):
        precision, recall, _ = precision_recall_curve(labels, probability)
        axis.plot(recall, precision, linewidth=1.8, label=name)
    prevalence = float(np.mean(labels))
    axis.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"no-skill baseline = prevalence ({prevalence:.6f})",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision–Recall curves at natural prevalence")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return figure


def _plot_calibration(calibration: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 6))
    for model_name, frame in calibration.groupby("bundle_key"):
        valid = frame[frame["rows"] > 0]
        axis.plot(
            valid["mean_probability"],
            valid["observed_malicious_rate"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=model_name,
        )
    axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean calibrated probability")
    axis.set_ylabel("Observed malicious rate")
    axis.set_title("Reliability curves")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return figure


def _plot_confusion(report: dict[str, Any], title: str) -> plt.Figure:
    matrix = np.array(
        [
            [report["true_negative"], report["false_positive"]],
            [report["false_negative"], report["true_positive"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_xticks((0, 1), labels=("Predicted benign", "Predicted malicious"))
    axis.set_yticks((0, 1), labels=("Actual benign", "Actual malicious"))
    axis.set_title(title)
    figure.colorbar(image, ax=axis)
    return figure


class Phase5Runner:
    """Irrevocably open the sealed test once and publish all required evidence."""

    def __init__(
        self,
        *,
        split_root: Path | None = None,
        phase4_manifest_path: Path | None = None,
        batch_size: int = 100_000,
    ) -> None:
        self.config = load_experiment_config()
        self.identity: ExperimentIdentity = create_experiment_identity(self.config)
        self.split_root = Path(
            split_root or SPLITS_DIR / str(self.config["split_version"])
        )
        self.phase4_manifest_path = Path(
            phase4_manifest_path or RESULTS_DIR / "phase4_champion.json"
        )
        self.batch_size = batch_size
        self.phase5_config = self.config.get("phase5", {})

    def _verify_phase4(
        self,
    ) -> tuple[dict[str, Any], Phase4ModelBundle, dict[str, Phase4ModelBundle]]:
        verify_validated_dataset()
        verify_training_split_access(self.split_root)
        if not self.phase4_manifest_path.is_file():
            raise Phase5GateError("Phase 4 champion manifest is missing.")
        summary_path = RESULTS_DIR / "experiment_summary.json"
        if not summary_path.is_file():
            raise Phase5GateError("Experiment summary is missing before Phase 5.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        phase4_summary = summary.get("phase_4", {})
        if (
            Path(phase4_summary.get("champion_manifest", "")).resolve()
            != self.phase4_manifest_path.resolve()
            or phase4_summary.get("champion_manifest_sha256")
            != sha256_file(self.phase4_manifest_path)
        ):
            raise Phase5GateError("Experiment summary does not bind the Phase 4 manifest.")
        manifest = json.loads(self.phase4_manifest_path.read_text(encoding="utf-8"))
        expected = self.identity.to_dict()
        for field in (
            "experiment_id",
            "dataset_version",
            "feature_schema_version",
            "split_version",
            "code_commit",
            "random_seed",
            "config_sha256",
        ):
            if manifest.get("experiment", {}).get(field) != expected[field]:
                raise Phase5GateError(f"Phase 4 experiment mismatch: {field}.")
        if manifest.get("sealed_test_opened") is not False:
            raise Phase5GateError("Phase 4 manifest does not prove test remained sealed.")
        model_matrices = manifest.get("model_matrices", {})
        if set(model_matrices) != {"tree", "neural"}:
            raise Phase5GateError("Phase 4 requires both tree and neural matrices.")
        for family, partitions in model_matrices.items():
            if family not in {"tree", "neural"} or set(partitions) != {
                "train",
                "validation",
            }:
                raise Phase5GateError("Phase 4 model-matrix inventory is incomplete.")
            for evidence in partitions.values():
                matrix_manifest_path = Path(evidence["path"])
                if (
                    not matrix_manifest_path.is_file()
                    or sha256_file(matrix_manifest_path) != evidence["sha256"]
                ):
                    raise Phase5GateError("Phase 4 model-matrix manifest was tampered.")
        champion_path = Path(manifest["champion_bundle"])
        if (
            not champion_path.is_file()
            or sha256_file(champion_path) != manifest.get("champion_bundle_sha256")
        ):
            raise Phase5GateError("Champion bundle inventory mismatch.")
        champion_pipeline_path = Path(manifest["champion_feature_pipeline"])
        if (
            not champion_pipeline_path.is_file()
            or sha256_file(champion_pipeline_path)
            != manifest["champion_feature_pipeline_sha256"]
        ):
            raise Phase5GateError("Champion feature pipeline inventory mismatch.")
        champion = Phase4ModelBundle.load_champion(
            champion_path,
            identity=self.identity,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            split_manifest_path=self.split_root / "split_manifest.json",
            feature_pipeline_path=champion_pipeline_path,
        )
        finalists: dict[str, Phase4ModelBundle] = {}
        for key, evidence in manifest.get("finalists", {}).items():
            path = Path(evidence["path"])
            if not path.is_file() or sha256_file(path) != evidence["sha256"]:
                raise Phase5GateError(f"Finalist inventory mismatch: {key}")
            finalist_pipeline_path = Path(evidence["feature_pipeline_path"])
            if (
                not finalist_pipeline_path.is_file()
                or sha256_file(finalist_pipeline_path)
                != evidence["feature_pipeline_sha256"]
            ):
                raise Phase5GateError(f"Finalist pipeline inventory mismatch: {key}")
            bundle = Phase4ModelBundle.load_finalist(
                path,
                identity=self.identity,
                dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
                split_manifest_path=self.split_root / "split_manifest.json",
                feature_pipeline_path=finalist_pipeline_path,
                require_three_seeds=False,
            )
            bundle.validate(require_three_seeds=bundle.model_family != "dummy")
            if key != bundle.variant_key:
                raise Phase5GateError(f"Finalist key/bundle mismatch: {key}")
            finalists[key] = bundle
        required_models = set(self.config["phase4"]["required_models"])
        if {bundle.model_name for bundle in finalists.values()} != required_models:
            raise Phase5GateError("Finalist model inventory is incomplete.")
        for model_name in required_models:
            variants = {
                bundle.variant
                for bundle in finalists.values()
                if bundle.model_name == model_name
            }
            expected_variants = (
                {"unweighted"}
                if model_name == "always_benign"
                else {"unweighted", "cost_sensitive"}
            )
            if variants != expected_variants:
                raise Phase5GateError(
                    f"Finalist weighting ablation is incomplete for {model_name}."
                )
        selected_key = manifest["selected_bundle_key"]
        if selected_key not in finalists:
            raise Phase5GateError("Champion is absent from finalist inventory.")
        selected = finalists[selected_key]
        if (
            selected.model_name != champion.model_name
            or selected.variant != champion.variant
            or selected.selected_policy != champion.selected_policy
            or selected.feature_names != champion.feature_names
            or [member.seed for member in selected.members]
            != [member.seed for member in champion.members]
        ):
            raise Phase5GateError("Champion copy and selected finalist disagree.")
        return manifest, champion, finalists

    def _required_outputs_absent(self) -> None:
        paths = [
            RESULTS_DIR / "metrics_by_model.csv",
            RESULTS_DIR / "metrics_by_threshold.csv",
            RESULTS_DIR / "metrics_by_subgroup.csv",
            RESULTS_DIR / "bootstrap_differences.csv",
            RESULTS_DIR / "calibration.csv",
            RESULTS_DIR / "temporal_feature_drift.csv",
            RESULTS_DIR / "sanitized_hard_errors.csv",
            RESULTS_DIR / "unavailable_subgroups.json",
            RESULTS_DIR / "metric_definitions.json",
            RESULTS_DIR / "phase5_manifest.json",
            RESULTS_DIR / "phase6_handoff",
            FIGURES_DIR / "roc_full.png",
            FIGURES_DIR / "roc_low_fpr.png",
            FIGURES_DIR / "precision_recall.png",
            FIGURES_DIR / "calibration.png",
            FIGURES_DIR / "confusion_matrices",
        ]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise Phase5GateError(
                "Phase 5 outputs already exist; sealed test cannot be rerun or overwritten: "
                f"{existing}"
            )

    def run(self) -> Path:
        phase4_manifest, champion, finalists = self._verify_phase4()
        self._required_outputs_absent()
        champion_path = Path(phase4_manifest["champion_bundle"])
        ledger = SealedTestLedger(RESULTS_DIR / "SEALED_TEST_EVALUATION.json")
        token = ledger.claim(
            identity=self.identity,
            split_root=self.split_root,
            champion_bundle_path=champion_path,
        )
        # From this point onward, any failure intentionally leaves a claimed
        # ledger. The test may not be reopened under this experiment version.
        test_path = ledger.authorized_test_features(self.split_root, token=token)
        matrix_root = (
            PROCESSED_DATA_DIR
            / "model_matrices_v2"
            / str(self.config["split_version"])
        )
        pipeline_by_family: dict[str, tuple[FeaturePipelineV2, Path]] = {}
        for key, bundle in finalists.items():
            matrix_family = "neural" if bundle.model_family == "neural" else "tree"
            evidence = phase4_manifest["finalists"][key]
            path = Path(evidence["feature_pipeline_path"])
            if matrix_family in pipeline_by_family:
                if pipeline_by_family[matrix_family][1].resolve() != path.resolve():
                    raise Phase5GateError(
                        f"Finalists disagree on the {matrix_family} pipeline."
                    )
                continue
            pipeline_by_family[matrix_family] = (
                FeaturePipelineV2.load(
                    path,
                    split_manifest_path=self.split_root / "split_manifest.json",
                    dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
                    transformation_manifest_path=DATA_REPORTS_DIR
                    / "transformation_manifest.json",
                ),
                path,
            )
        X_test_by_family: dict[str, np.ndarray] = {}
        test_matrix_manifest_by_family: dict[str, Any] = {}
        y_test = None
        test_metadata = None
        for family, (pipeline, pipeline_path) in pipeline_by_family.items():
            test_matrix_dir = matrix_root / family / "test"
            materialize_model_matrix(
                test_path,
                test_matrix_dir,
                pipeline=pipeline,
                pipeline_path=pipeline_path,
                split_manifest_path=self.split_root / "split_manifest.json",
                partition_name="test",
                batch_size=self.batch_size,
                allow_test=True,
            )
            X_family, y_family, metadata_family, matrix_manifest = load_model_matrix(
                test_matrix_dir,
                expected_partition="test",
                feature_pipeline_path=pipeline_path,
                split_manifest_path=self.split_root / "split_manifest.json",
            )
            if y_test is None:
                y_test = y_family
                test_metadata = metadata_family
            elif not np.array_equal(y_test, y_family):
                raise Phase5GateError("Tree/neural test matrices have different labels.")
            X_test_by_family[family] = X_family
            test_matrix_manifest_by_family[family] = matrix_manifest
        assert y_test is not None and test_metadata is not None
        champion_family = "neural" if champion.model_family == "neural" else "tree"
        champion_pipeline_path = pipeline_by_family[champion_family][1]
        X_train, _, _, train_matrix_manifest = load_model_matrix(
            matrix_root / champion_family / "train",
            expected_partition="train",
            feature_pipeline_path=champion_pipeline_path,
            split_manifest_path=self.split_root / "split_manifest.json",
            load_metadata=False,
        )
        groups = test_metadata[GROUP_ID_COLUMN].astype(str).to_numpy()
        probability_by_model = {
            key: bundle.predict_proba(
                X_test_by_family[
                    "neural" if bundle.model_family == "neural" else "tree"
                ]
            )
            for key, bundle in finalists.items()
        }
        confusion_replicates = int(
            self.phase5_config.get("bootstrap_replicates", 1_000)
        )
        probability_replicates = int(
            self.phase5_config.get("probability_bootstrap_replicates", 200)
        )
        random_seed = int(self.config["random_seed"])
        model_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        subgroup_frames: list[pd.DataFrame] = []
        unavailable_subgroups: dict[str, list[dict[str, str]]] = {}
        calibration_frames: list[pd.DataFrame] = []
        selected_reports: dict[str, dict[str, Any]] = {}
        confidence_intervals: dict[str, dict[str, Any]] = {}
        for index, (key, bundle) in enumerate(finalists.items()):
            probabilities = probability_by_model[key]
            for policy, decision in bundle.thresholds.items():
                report = metric_report(
                    y_test, probabilities, threshold=float(decision["threshold"])
                )
                threshold_rows.append(
                    {
                        "bundle_key": key,
                        "model_name": bundle.model_name,
                        "variant": bundle.variant,
                        "partition": "sealed_test_one_shot",
                        "threshold_policy": policy,
                        **report,
                    }
                )
            selected_report = metric_report(
                y_test, probabilities, threshold=bundle.threshold
            )
            selected_reports[key] = selected_report
            confusion_ci = group_confusion_bootstrap(
                y_test,
                probabilities,
                groups,
                threshold=bundle.threshold,
                replicates=confusion_replicates,
                random_seed=random_seed + index,
            )
            probability_ci = group_probability_bootstrap(
                y_test,
                probabilities,
                groups,
                replicates=probability_replicates,
                random_seed=random_seed + index,
            )
            intervals = {
                **{name: value.to_dict() for name, value in confusion_ci.items()},
                **{name: value.to_dict() for name, value in probability_ci.items()},
            }
            confidence_intervals[key] = intervals
            row: dict[str, Any] = {
                "bundle_key": key,
                "model_name": bundle.model_name,
                "model_family": bundle.model_family,
                "variant": bundle.variant,
                "partition": "sealed_test_one_shot",
                "threshold_policy": bundle.selected_policy,
                **selected_report,
            }
            for metric_name, interval in intervals.items():
                row[f"{metric_name}_ci_lower_95"] = interval["lower_95"]
                row[f"{metric_name}_ci_upper_95"] = interval["upper_95"]
            model_rows.append(row)
            subgroup, unavailable = metrics_by_subgroup(
                test_metadata,
                y_test,
                probabilities,
                threshold=bundle.threshold,
                minimum_rows=int(self.phase5_config.get("minimum_subgroup_rows", 100)),
            )
            subgroup.insert(0, "bundle_key", key)
            subgroup_frames.append(subgroup)
            unavailable_subgroups[key] = unavailable
            calibration = calibration_table(y_test, probabilities)
            calibration.insert(0, "bundle_key", key)
            calibration_frames.append(calibration)

        selected_key = phase4_manifest["selected_bundle_key"]
        champion_probability = probability_by_model[selected_key]
        differences = []
        for index, (key, bundle) in enumerate(finalists.items()):
            if key == selected_key:
                continue
            interval = paired_group_bootstrap_difference(
                y_test,
                probability_by_model[key],
                champion_probability,
                groups,
                candidate_threshold=bundle.threshold,
                reference_threshold=champion.threshold,
                metric="f2",
                replicates=confusion_replicates,
                random_seed=random_seed + index,
            )
            differences.append(
                {
                    "candidate_bundle_key": key,
                    "reference_bundle_key": selected_key,
                    "metric": "f2",
                    "difference_direction": "candidate_minus_champion",
                    **interval.to_dict(),
                }
            )

        metrics_by_model_path = _atomic_csv(
            RESULTS_DIR / "metrics_by_model.csv", pd.DataFrame(model_rows)
        )
        metrics_by_threshold_path = _atomic_csv(
            RESULTS_DIR / "metrics_by_threshold.csv", pd.DataFrame(threshold_rows)
        )
        subgroup_frame = pd.concat(subgroup_frames, ignore_index=True)
        metrics_by_subgroup_path = _atomic_csv(
            RESULTS_DIR / "metrics_by_subgroup.csv", subgroup_frame
        )
        bootstrap_path = _atomic_csv(
            RESULTS_DIR / "bootstrap_differences.csv", pd.DataFrame(differences)
        )
        calibration_frame = pd.concat(calibration_frames, ignore_index=True)
        calibration_path = _atomic_csv(
            RESULTS_DIR / "calibration.csv", calibration_frame
        )
        drift_path = _atomic_csv(
            RESULTS_DIR / "temporal_feature_drift.csv",
            drift_table(
                X_train,
                X_test_by_family[champion_family],
                train_matrix_manifest.feature_names,
                maximum_rows=int(self.phase5_config.get("drift_maximum_rows", 250_000)),
            ),
        )
        errors_path = _atomic_csv(
            RESULTS_DIR / "sanitized_hard_errors.csv",
            sanitized_hard_errors(
                test_metadata,
                y_test,
                champion_probability,
                threshold=champion.threshold,
                experiment_id=self.identity.experiment_id,
            ),
        )
        unavailable_path = RESULTS_DIR / "unavailable_subgroups.json"
        atomic_write_json(unavailable_path, {"by_model": unavailable_subgroups})
        definitions_path = RESULTS_DIR / "metric_definitions.json"
        atomic_write_json(
            definitions_path,
            {
                "positive_class": "malicious_pdf",
                "natural_prevalence": float(np.mean(y_test)),
                "definitions": metric_definitions(),
            },
        )

        phase6_handoff_dir = RESULTS_DIR / "phase6_handoff"
        phase6_handoff_dir.mkdir(parents=True, exist_ok=False)
        handoff_indices = select_phase6_handoff_indices(
            y_test,
            champion_probability,
            test_metadata,
            threshold=champion.threshold,
            maximum_rows=int(
                self.config.get("phase6", {}).get(
                    "test_handoff_maximum_rows", 5_000
                )
            ),
            random_seed=random_seed,
        )
        handoff_matrix_path = phase6_handoff_dir / "test_cases.npz"

        def write_handoff_matrix(temporary: Path) -> None:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    X=np.asarray(
                        X_test_by_family[champion_family][handoff_indices],
                        dtype=np.float32,
                    ),
                    y=np.asarray(y_test[handoff_indices], dtype=np.int8),
                    probability=np.asarray(
                        champion_probability[handoff_indices], dtype=np.float64
                    ),
                )

        atomic_write_via(handoff_matrix_path, write_handoff_matrix)
        handoff_metadata = test_metadata.iloc[handoff_indices].reset_index(drop=True).copy()
        if SAMPLE_ID_COLUMN in handoff_metadata:
            handoff_metadata["sample_id_sha256"] = [
                hashlib.sha256(
                    f"{self.identity.experiment_id}:{value}".encode("utf-8")
                ).hexdigest()
                for value in handoff_metadata.pop(SAMPLE_ID_COLUMN).astype(str)
            ]
        if GROUP_ID_COLUMN in handoff_metadata:
            handoff_metadata["group_id_sha256"] = [
                hashlib.sha256(
                    f"{self.identity.experiment_id}:group:{value}".encode("utf-8")
                ).hexdigest()
                for value in handoff_metadata.pop(GROUP_ID_COLUMN).astype(str)
            ]
        handoff_metadata["outcome"] = np.select(
            (
                (y_test[handoff_indices] == 1)
                & (champion_probability[handoff_indices] >= champion.threshold),
                (y_test[handoff_indices] == 0)
                & (champion_probability[handoff_indices] < champion.threshold),
                (y_test[handoff_indices] == 0)
                & (champion_probability[handoff_indices] >= champion.threshold),
                (y_test[handoff_indices] == 1)
                & (champion_probability[handoff_indices] < champion.threshold),
            ),
            ("true_positive", "true_negative", "false_positive", "false_negative"),
        )
        handoff_metadata_path = phase6_handoff_dir / "test_cases_metadata.parquet"
        atomic_write_via(
            handoff_metadata_path,
            lambda temporary: handoff_metadata.to_parquet(
                temporary, index=False, compression="zstd"
            ),
        )
        handoff_manifest_path = phase6_handoff_dir / "manifest.json"
        atomic_write_json(
            handoff_manifest_path,
            {
                "version": "1.0.0",
                "experiment": self.identity.to_dict(),
                "source_partition": "sealed_test_one_shot_bounded_handoff",
                "selection_used_for_model_or_threshold": False,
                "rows": int(len(handoff_indices)),
                "feature_names": list(
                    test_matrix_manifest_by_family[champion_family].feature_names
                ),
                "champion_bundle_key": selected_key,
                "champion_threshold": champion.threshold,
                "matrix": {
                    "path": str(handoff_matrix_path.resolve()),
                    "sha256": sha256_file(handoff_matrix_path),
                },
                "metadata": {
                    "path": str(handoff_metadata_path.resolve()),
                    "sha256": sha256_file(handoff_metadata_path),
                },
                "raw_pdf_content_present": False,
                "raw_sample_identifiers_present": False,
                "raw_group_identifiers_present": False,
            },
        )

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        roc_full_path = _atomic_figure(
            FIGURES_DIR / "roc_full.png",
            lambda: _plot_roc(y_test, probability_by_model, low_fpr=False),
        )
        roc_low_path = _atomic_figure(
            FIGURES_DIR / "roc_low_fpr.png",
            lambda: _plot_roc(y_test, probability_by_model, low_fpr=True),
        )
        pr_path = _atomic_figure(
            FIGURES_DIR / "precision_recall.png",
            lambda: _plot_precision_recall(y_test, probability_by_model),
        )
        calibration_figure_path = _atomic_figure(
            FIGURES_DIR / "calibration.png",
            lambda: _plot_calibration(calibration_frame),
        )
        confusion_directory = FIGURES_DIR / "confusion_matrices"
        confusion_directory.mkdir()
        confusion_paths: dict[str, Path] = {}
        for key, report in selected_reports.items():
            path = confusion_directory / f"{key}.png"
            confusion_paths[key] = _atomic_figure(
                path, lambda report=report, key=key: _plot_confusion(report, key)
            )

        champion_report = selected_reports[selected_key]
        champion_intervals = confidence_intervals[selected_key]
        final_metrics = {
            "model": champion.model_name,
            "variant": champion.variant,
            "partition": "sealed_test_one_shot",
            "threshold_policy": champion.selected_policy,
            "threshold": champion.threshold,
            "metrics": {
                name: {
                    "value": champion_report[name],
                    "lower_95": champion_intervals.get(name, {}).get("lower_95"),
                    "upper_95": champion_intervals.get(name, {}).get("upper_95"),
                }
                for name in (
                    "precision",
                    "recall",
                    "f0_5",
                    "f1",
                    "f2",
                    "roc_auc",
                    "partial_roc_auc_standardized_fpr_0_001",
                    "partial_roc_auc_standardized_fpr_0_0001",
                    "pr_auc_average_precision",
                    "specificity",
                    "false_positive_rate",
                    "matthews_correlation_coefficient",
                    "brier_score",
                )
            },
            "confusion_counts": {
                name: champion_report[name]
                for name in (
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                )
            },
            "false_positives_per_million_benign": champion_report[
                "false_positives_per_million_benign"
            ],
        }
        output_paths = {
            "metrics_by_model": metrics_by_model_path,
            "metrics_by_threshold": metrics_by_threshold_path,
            "metrics_by_subgroup": metrics_by_subgroup_path,
            "bootstrap_differences": bootstrap_path,
            "calibration": calibration_path,
            "temporal_feature_drift": drift_path,
            "sanitized_hard_errors": errors_path,
            "unavailable_subgroups": unavailable_path,
            "metric_definitions": definitions_path,
            "phase6_handoff_matrix": handoff_matrix_path,
            "phase6_handoff_metadata": handoff_metadata_path,
            "phase6_handoff_manifest": handoff_manifest_path,
            "roc_full": roc_full_path,
            "roc_low_fpr": roc_low_path,
            "precision_recall": pr_path,
            "calibration_figure": calibration_figure_path,
            **{f"confusion_{key}": path for key, path in confusion_paths.items()},
        }
        phase5_manifest_path = RESULTS_DIR / "phase5_manifest.json"
        phase5_manifest = {
            "phase5_version": PHASE5_VERSION,
            "experiment": self.identity.to_dict(),
            "phase4_champion_manifest_sha256": sha256_file(self.phase4_manifest_path),
            "test_matrix_manifests": {
                family: {
                    "path": str((matrix_root / family / "test" / "matrix_manifest.json").resolve()),
                    "sha256": sha256_file(
                        matrix_root / family / "test" / "matrix_manifest.json"
                    ),
                }
                for family in test_matrix_manifest_by_family
            },
            "test_rows": int(len(y_test)),
            "test_malicious_prevalence": float(np.mean(y_test)),
            "evaluation_count": 1,
            "thresholds_selected_on": "validation_threshold_selection",
            "test_used_for_selection": False,
            "final_metrics": final_metrics,
            "outputs": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in output_paths.items()
            },
        }
        atomic_write_json(phase5_manifest_path, phase5_manifest)
        output_paths["phase5_manifest"] = phase5_manifest_path
        ledger_path = ledger.complete(
            token=token,
            output_artifacts={name: str(path) for name, path in output_paths.items()},
        )
        summary_path = RESULTS_DIR / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "status": "phase_5_complete_sealed_test_closed",
                "data_gate_passed": True,
                "phase_5": {
                    "manifest": str(phase5_manifest_path.resolve()),
                    "manifest_sha256": sha256_file(phase5_manifest_path),
                    "sealed_test_ledger": str(ledger_path.resolve()),
                    "sealed_test_ledger_sha256": sha256_file(ledger_path),
                    "evaluation_count": 1,
                },
                "final_metrics": final_metrics,
            }
        )
        atomic_write_json(summary_path, summary)
        return phase5_manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--confirm-sealed-test-evaluation",
        action="store_true",
        help="Explicitly authorize the irreversible single sealed-test evaluation.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_sealed_test_evaluation:
        parser.error(
            "This command irreversibly claims the sealed test. Re-run with "
            "--confirm-sealed-test-evaluation after reviewing Phase 4 evidence."
        )
    print(Phase5Runner(batch_size=args.batch_size).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
