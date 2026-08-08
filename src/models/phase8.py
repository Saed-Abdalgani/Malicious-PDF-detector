"""Phase 8: build and validate the single application deployment bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.artifacts import metadata_path
from src.config import DATA_REPORTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, SPLITS_DIR
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.features.vectorizer import extract_features_record
from src.inference import analyze_pdf_bytes
from src.models.bundle import Phase4ModelBundle
from src.models.deployment import DeploymentBundle
from src.models.matrix import load_model_matrix
from src.security.adversarial import build_inert_pdf
from src.utils.atomic import atomic_write_json


PHASE8_VERSION = "1.0.0"


class Phase8GateError(RuntimeError):
    """Raised when the application bundle cannot be proven deployment-safe."""


def _natural_reference_indices(
    labels: np.ndarray, *, maximum_rows: int, random_seed: int
) -> np.ndarray:
    """Select bounded real train rows while preserving natural class prevalence."""
    values = np.asarray(labels, dtype=np.int8)
    if maximum_rows < 100 or not np.isin(values, (0, 1)).all():
        raise Phase8GateError("Phase 8 train reference request is invalid.")
    if np.unique(values).size != 2:
        raise Phase8GateError("Phase 8 train reference requires both classes.")
    rows = min(maximum_rows, len(values))
    malicious_target = max(1, min(rows - 1, int(round(rows * float(values.mean())))))
    targets = {1: malicious_target, 0: rows - malicious_target}
    generator = np.random.default_rng(random_seed)
    selected: list[np.ndarray] = []
    for label, target in targets.items():
        candidates = np.flatnonzero(values == label)
        if len(candidates) < target:
            raise Phase8GateError("Phase 8 train reference class coverage is insufficient.")
        selected.append(generator.choice(candidates, size=target, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def _json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase8GateError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8GateError(f"{description} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise Phase8GateError(f"{description} must be a JSON object.")
    return value


def _checked(path: Path, digest: str, description: str) -> Path:
    target = Path(path)
    if not target.is_file() or sha256_file(target) != digest:
        raise Phase8GateError(f"{description} checksum mismatch: {target}")
    return target


class Phase8Runner:
    def __init__(self, *, split_root: Path | None = None) -> None:
        self.config = load_experiment_config()
        self.phase8 = self.config["phase8"]
        self.identity = create_experiment_identity(self.config)
        self.split_root = Path(split_root or SPLITS_DIR / str(self.config["split_version"]))
        self.output_root = MODELS_DIR / "deployment"

    def _verify_upstream(self) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
        if self.output_root.exists() or (RESULTS_DIR / "phase8_manifest.json").exists():
            raise Phase8GateError("Phase 8 artifacts are immutable and already exist.")
        summary = _json(RESULTS_DIR / "experiment_summary.json", "Experiment summary")
        if summary.get("status") != "phase_7_complete_safe_adversarial_evaluation":
            raise Phase8GateError("Phase 8 requires completed, checksummed Phase 7 evidence.")
        phase7_entry = summary.get("phase_7", {})
        phase7_path = _checked(
            Path(str(phase7_entry.get("manifest", ""))),
            str(phase7_entry.get("manifest_sha256", "")),
            "Phase 7 manifest",
        )
        phase7 = _json(phase7_path, "Phase 7 manifest")
        if not phase7.get("gate_passed") or phase7.get("live_malware_used"):
            raise Phase8GateError("Phase 7 safety/evidence gate did not pass.")
        phase4_path = RESULTS_DIR / "phase4_champion.json"
        phase4 = _json(phase4_path, "Phase 4 champion manifest")
        pipeline_path = _checked(
            Path(phase4["champion_feature_pipeline"]),
            str(phase4["champion_feature_pipeline_sha256"]),
            "Champion feature pipeline",
        )
        model_path = _checked(
            Path(phase4["champion_bundle"]),
            str(phase4["champion_bundle_sha256"]),
            "Champion model bundle",
        )
        return summary, phase4, pipeline_path, model_path

    def run(self) -> Path:
        summary, phase4, pipeline_path, model_path = self._verify_upstream()
        pipeline = FeaturePipelineV2.load(
            pipeline_path,
            identity=self.identity,
            split_manifest_path=self.split_root / "split_manifest.json",
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
        )
        model = Phase4ModelBundle.load_champion(
            model_path,
            identity=self.identity,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            split_manifest_path=self.split_root / "split_manifest.json",
            feature_pipeline_path=pipeline_path,
        )
        family = "neural" if model.model_family == "neural" else "tree"
        matrix_root = PROCESSED_DATA_DIR / "model_matrices_v2" / str(self.config["split_version"])
        train_matrix_dir = matrix_root / family / "train"
        train_features, train_labels, _train_metadata, train_manifest = load_model_matrix(
            train_matrix_dir,
            expected_partition="train",
            feature_pipeline_path=pipeline_path,
            split_manifest_path=self.split_root / "split_manifest.json",
            load_metadata=False,
        )
        if tuple(train_manifest.feature_names) != tuple(model.feature_names):
            raise Phase8GateError("Train reference/model feature names do not align.")
        indices = _natural_reference_indices(
            train_labels,
            maximum_rows=int(self.phase8["explanation_background_rows"]),
            random_seed=int(self.config["random_seed"]),
        )
        background = np.asarray(train_features[indices], dtype=np.float32)
        bundle = DeploymentBundle.create(
            model=model,
            feature_pipeline=pipeline,
            explanation_background=background,
            source_model_path=model_path,
            source_pipeline_path=pipeline_path,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            split_manifest_path=self.split_root / "split_manifest.json",
            transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
            maximum_pdf_bytes=int(self.phase8["maximum_pdf_bytes"]),
            uncertainty_probability_margin=float(self.phase8["uncertainty_probability_margin"]),
            ood_robust_z_threshold=float(self.phase8["ood_robust_z_threshold"]),
            ood_feature_fraction_threshold=float(self.phase8["ood_feature_fraction_threshold"]),
            provenance={
                "phase7_manifest_sha256": summary["phase_7"]["manifest_sha256"],
                "train_matrix_manifest_sha256": sha256_file(train_matrix_dir / "matrix_manifest.json"),
                "train_reference_rows": len(background),
                "train_reference_indices_sha256": __import__("hashlib").sha256(indices.tobytes()).hexdigest(),
            },
        )

        temporary = Path(tempfile.mkdtemp(prefix=".phase8-", dir=MODELS_DIR))
        try:
            bundle_path = bundle.save(temporary / "deployment_bundle_v1.joblib", identity=self.identity)
            reloaded = DeploymentBundle.load(bundle_path, identity=self.identity)
            golden_rows: list[dict[str, Any]] = []
            count = int(self.phase8["golden_fixture_count"])
            for serial in range(count):
                data = build_inert_pdf(security_marker=bool(serial % 2), serial=serial)
                app_result = analyze_pdf_bytes(data, reloaded, include_explanation=False)
                temp_pdf: str | None = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
                        handle.write(data)
                        temp_pdf = handle.name
                    record, diagnostics = extract_features_record(temp_pdf)
                    direct = reloaded.predict_record(record, diagnostics, include_explanation=False)
                finally:
                    if temp_pdf and os.path.exists(temp_pdf):
                        os.remove(temp_pdf)
                if app_result.decision.to_dict() != direct.to_dict():
                    raise Phase8GateError(f"Golden app/direct prediction mismatch at fixture {serial}.")
                golden_rows.append(
                    {
                        "fixture_sha256": app_result.file_sha256,
                        "security_marker": bool(serial % 2),
                        "outcome": direct.outcome,
                        "malicious_probability": direct.malicious_probability,
                        "threshold": direct.threshold,
                    }
                )

            resource_started = time.perf_counter()
            corrupted = analyze_pdf_bytes(b"not-a-pdf", reloaded, include_explanation=False)
            oversized = analyze_pdf_bytes(
                b"x" * (reloaded.maximum_pdf_bytes + 1),
                reloaded,
                include_explanation=False,
            )
            fail_closed_seconds = time.perf_counter() - resource_started
            if corrupted.decision.outcome != "uncertain/abstain" or oversized.decision.outcome != "uncertain/abstain":
                raise Phase8GateError("Corrupt/oversized application fixtures did not fail closed.")
            if fail_closed_seconds > float(self.phase8["maximum_fail_closed_seconds"]):
                raise Phase8GateError("Corrupt/oversized fail-closed resource gate exceeded.")

            atomic_write_json(
                temporary / "golden_predictions.json",
                {
                    "fixture_count": count,
                    "app_bundle_exact_match": True,
                    "uploaded_pdf_bytes_persisted": False,
                    "rows": golden_rows,
                },
            )
            atomic_write_json(
                temporary / "resource_gate.json",
                {
                    "corrupted_outcome": corrupted.decision.outcome,
                    "oversized_outcome": oversized.decision.outcome,
                    "elapsed_seconds": fail_closed_seconds,
                    "maximum_seconds": float(self.phase8["maximum_fail_closed_seconds"]),
                    "passed": True,
                },
            )
            manifest = {
                "phase8_version": PHASE8_VERSION,
                "experiment": self.identity.to_dict(),
                "phase7_manifest_sha256": summary["phase_7"]["manifest_sha256"],
                "bundle": {
                    "path": "deployment_bundle_v1.joblib",
                    "sha256": sha256_file(bundle_path),
                    "metadata_path": metadata_path(bundle_path).name,
                    "metadata_sha256": sha256_file(metadata_path(bundle_path)),
                },
                "golden_predictions_sha256": sha256_file(temporary / "golden_predictions.json"),
                "resource_gate_sha256": sha256_file(temporary / "resource_gate.json"),
                "three_way_outcomes": ["benign", "malicious", "uncertain/abstain"],
                "raw_indicators_separate_from_model_attributions": True,
                "uploaded_pdf_bytes_persisted": False,
                "gate_passed": True,
            }
            atomic_write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, self.output_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        manifest_path = self.output_root / "manifest.json"
        result_manifest = RESULTS_DIR / "phase8_manifest.json"
        atomic_write_json(result_manifest, _json(manifest_path, "Phase 8 manifest"))
        summary.update(
            {
                "status": "phase_8_complete_deployment_bundle_verified",
                "phase_8": {
                    "manifest": str(result_manifest.resolve()),
                    "manifest_sha256": sha256_file(result_manifest),
                    "bundle": str((self.output_root / "deployment_bundle_v1.joblib").resolve()),
                    "bundle_sha256": sha256_file(self.output_root / "deployment_bundle_v1.joblib"),
                },
            }
        )
        atomic_write_json(RESULTS_DIR / "experiment_summary.json", summary)
        return result_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(Phase8Runner().run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
