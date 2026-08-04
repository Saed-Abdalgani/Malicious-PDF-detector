"""Unified calibrated Phase 4 model bundle with locked validation thresholds."""

from __future__ import annotations

import os
import string
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from src.artifacts import (
    ArtifactCompatibilityError,
    verify_artifact_compatibility,
    verify_deployable_model_artifact,
    write_artifact_metadata,
    write_model_artifact_metadata,
)
from src.experiment import ExperimentIdentity, sha256_file
from src.models.calibration import ProbabilityCalibrator


BUNDLE_VERSION = "1.0.0"


def positive_class_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from supported sklearn/torch estimators."""
    if isinstance(estimator, torch.nn.Module):
        estimator.eval()
        device = next(estimator.parameters()).device
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), 8_192):
                tensor = torch.as_tensor(
                    np.asarray(features[start : start + 8_192]),
                    dtype=torch.float32,
                    device=device,
                )
                logits = estimator(tensor).reshape(-1)
                outputs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64, copy=False)
    if hasattr(estimator, "predict_proba"):
        value = np.asarray(estimator.predict_proba(features), dtype=np.float64)
        if value.ndim == 2:
            classes = np.asarray(getattr(estimator, "classes_", (0, 1)))
            matches = np.flatnonzero(classes == 1)
            if len(matches) != 1:
                raise RuntimeError("Estimator has no unique malicious class probability.")
            value = value[:, matches[0]]
        return value.reshape(-1)
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(features), dtype=np.float64)
        decision = np.clip(decision.reshape(-1), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-decision))
    raise TypeError(f"Unsupported estimator probability API: {type(estimator)!r}")


@dataclass
class CalibratedMember:
    seed: int
    estimator: Any
    calibrator: ProbabilityCalibrator
    training_configuration: dict[str, Any]
    resource_usage: dict[str, Any] = field(default_factory=dict)

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(
            positive_class_probability(self.estimator, features)
        )


@dataclass
class Phase4ModelBundle:
    """One model variant, its seed ensemble, calibration, and thresholds."""

    model_name: str
    model_family: str
    variant: str
    feature_names: tuple[str, ...]
    members: list[CalibratedMember]
    thresholds: dict[str, dict[str, Any]]
    selected_policy: str
    provenance: dict[str, str]
    training_evidence: dict[str, Any]
    calibration_evidence: dict[str, Any]
    bundle_version: str = BUNDLE_VERSION

    def validate(self, *, require_three_seeds: bool = True) -> None:
        if self.bundle_version != BUNDLE_VERSION:
            raise RuntimeError("Unsupported Phase 4 bundle version.")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise RuntimeError("Bundle feature names must be non-empty and unique.")
        if not self.members:
            raise RuntimeError("Bundle has no trained members.")
        seeds = [member.seed for member in self.members]
        if len(set(seeds)) != len(seeds):
            raise RuntimeError("Bundle seed members must be unique.")
        if require_three_seeds and self.model_family != "dummy" and len(seeds) < 3:
            raise RuntimeError("Every non-dummy finalist requires at least three seeds.")
        if self.selected_policy not in self.thresholds:
            raise RuntimeError("Bundle selected threshold policy is absent.")
        for name, decision in self.thresholds.items():
            threshold = decision.get("threshold")
            if not isinstance(threshold, (int, float)) or not 0.0 < threshold < 1.0:
                raise RuntimeError(f"Bundle threshold {name!r} is invalid.")
            if decision.get("selected_on") != "validation_threshold_selection":
                raise RuntimeError(f"Bundle threshold {name!r} was not selected on validation.")
        for field_name in (
            "dataset_quality_sha256",
            "split_manifest_sha256",
            "feature_pipeline_sha256",
        ):
            digest = self.provenance.get(field_name, "")
            if len(digest) != 64 or any(
                character not in string.hexdigits for character in digest
            ):
                raise RuntimeError(f"Bundle provenance lacks {field_name}.")

    @property
    def threshold(self) -> float:
        return float(self.thresholds[self.selected_policy]["threshold"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.validate(require_three_seeds=False)
        matrix = np.asarray(features)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected model matrix with {len(self.feature_names)} columns."
            )
        probabilities = np.vstack(
            [member.predict_probability(matrix) for member in self.members]
        )
        return probabilities.mean(axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int8)

    def _atomic_dump(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(self, temporary, compress=3)
            with temporary.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return output

    def save_finalist(
        self,
        path: Path,
        *,
        identity: ExperimentIdentity,
        require_three_seeds: bool = True,
    ) -> Path:
        self.validate(require_three_seeds=require_three_seeds)
        output = self._atomic_dump(path)
        write_artifact_metadata(
            output,
            "phase4_finalist_bundle",
            identity,
            extra={
                "bundle_version": self.bundle_version,
                "model_name": self.model_name,
                "model_family": self.model_family,
                "variant": self.variant,
                "seeds": [member.seed for member in self.members],
                "selected_policy": self.selected_policy,
                "threshold": self.threshold,
                "threshold_selected_on": "validation",
                **self.provenance,
            },
        )
        return output

    def save_champion(self, path: Path, *, identity: ExperimentIdentity) -> Path:
        self.validate(require_three_seeds=True)
        if self.model_family == "dummy":
            raise RuntimeError("The always-benign baseline cannot be deployed as champion.")
        output = self._atomic_dump(path)
        write_model_artifact_metadata(
            output,
            identity,
            model_type=self.model_name,
            threshold=self.threshold,
            dataset_quality_sha256=self.provenance["dataset_quality_sha256"],
            split_manifest_sha256=self.provenance["split_manifest_sha256"],
            feature_pipeline_sha256=self.provenance["feature_pipeline_sha256"],
            extra={
                "bundle_version": self.bundle_version,
                "model_family": self.model_family,
                "variant": self.variant,
                "selected_policy": self.selected_policy,
                "calibration": self.calibration_evidence,
                "seeds": [member.seed for member in self.members],
                "phase4_champion": True,
            },
        )
        return output

    @classmethod
    def load_finalist(
        cls,
        path: Path,
        *,
        identity: ExperimentIdentity,
        dataset_quality_path: Path,
        split_manifest_path: Path,
        feature_pipeline_path: Path,
        require_three_seeds: bool = True,
    ) -> "Phase4ModelBundle":
        metadata = verify_artifact_compatibility(path, identity)
        if metadata.artifact_kind != "phase4_finalist_bundle":
            raise ArtifactCompatibilityError("Artifact is not a Phase 4 finalist bundle.")
        upstream = {
            "dataset_quality_sha256": Path(dataset_quality_path),
            "split_manifest_sha256": Path(split_manifest_path),
            "feature_pipeline_sha256": Path(feature_pipeline_path),
        }
        for field_name, upstream_path in upstream.items():
            if not upstream_path.is_file() or metadata.extra.get(field_name) != sha256_file(
                upstream_path
            ):
                raise ArtifactCompatibilityError(
                    f"Finalist bundle upstream mismatch: {field_name}."
                )
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Artifact at {path} is not Phase4ModelBundle.")
        value.validate(require_three_seeds=require_three_seeds)
        return value

    @classmethod
    def load_champion(
        cls,
        path: Path,
        *,
        identity: ExperimentIdentity,
        dataset_quality_path: Path,
        split_manifest_path: Path,
        feature_pipeline_path: Path,
    ) -> "Phase4ModelBundle":
        metadata = verify_deployable_model_artifact(
            path,
            identity,
            dataset_quality_path=dataset_quality_path,
            split_manifest_path=split_manifest_path,
            feature_pipeline_path=feature_pipeline_path,
        )
        if not metadata.extra.get("phase4_champion"):
            raise ArtifactCompatibilityError("Deployable artifact is not the Phase 4 champion.")
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Artifact at {path} is not Phase4ModelBundle.")
        value.validate(require_three_seeds=True)
        if abs(value.threshold - float(metadata.extra["threshold"])) > 1e-15:
            raise ArtifactCompatibilityError("Champion threshold differs from sidecar.")
        return value
