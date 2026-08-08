"""Self-contained, fail-closed deployment bundle for Phase 8.

The application is allowed to load exactly one artifact.  That artifact binds
the feature pipeline, calibrated model ensemble, validation-selected threshold,
schema, provenance, and a real train-only explanation/OOD reference.
"""

from __future__ import annotations

import hashlib
import os
import string
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from src.artifacts import ArtifactCompatibilityError, verify_artifact_compatibility, write_artifact_metadata
from src.experiment import ExperimentIdentity, canonical_json, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.features.schema_v2 import schema_dictionary
from src.models.bundle import Phase4ModelBundle


DEPLOYMENT_BUNDLE_VERSION = "1.0.0"
DEPLOYMENT_ARTIFACT_KIND = "phase8_deployment_bundle"


def current_schema_digest(pipeline: FeaturePipelineV2) -> str:
    """Hash the complete input and model-output schema contract."""
    payload = {
        "base_schema": schema_dictionary(),
        "pipeline_metadata": pipeline.metadata(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(character in string.hexdigits for character in value)


@dataclass(frozen=True)
class DeploymentDecision:
    """A three-way deployment decision with evidence kept in separate layers."""

    outcome: str
    malicious_probability: float | None
    threshold: float
    threshold_policy: str
    abstention_reasons: tuple[str, ...] = ()
    raw_indicators: tuple[dict[str, Any], ...] = ()
    model_attributions: tuple[dict[str, Any], ...] = ()
    ood_feature_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["abstention_reasons"] = list(self.abstention_reasons)
        value["raw_indicators"] = list(self.raw_indicators)
        value["model_attributions"] = list(self.model_attributions)
        return value


def observed_indicators(record: Mapping[str, float], diagnostics: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return only directly observed, actionable indicators; make no causal claim."""
    candidates = {
        "javascript_actions": ("js_count", "Embedded JavaScript action tokens were observed."),
        "automatic_open_action": ("openaction_count", "An automatic document-open action was observed."),
        "launch_action": ("launch_count", "An external launch action was observed."),
        "embedded_files": ("embedded_file_count", "Embedded-file objects were observed."),
        "form_submission": ("submitform_count", "Form-submission actions were observed."),
        "obfuscation_tokens": ("obfuscation_count", "Encoded or obfuscated token patterns were observed."),
        "object_streams": ("objstm_count", "PDF object streams were observed."),
        "encryption": ("is_encrypted", "The PDF is encrypted."),
        "parser_disagreement": ("parser_disagreement", "Independent structural counts materially disagree."),
    }
    rows: list[dict[str, Any]] = []
    for indicator, (feature, description) in candidates.items():
        value = float(record.get(feature, 0.0))
        if value > 0:
            rows.append(
                {
                    "indicator": indicator,
                    "feature": feature,
                    "observed_value": value,
                    "description": description,
                    "evidence_type": "raw_static_observation",
                }
            )
    for name in (
        "parse_failure",
        "recovery_mode",
        "invalid_header",
        "invalid_eof",
        "truncated_structure",
        "extraction_limit_reached",
        "extraction_timeout",
        "file_too_large",
    ):
        if float(diagnostics.get(name, 0.0) or 0.0) > 0:
            rows.append(
                {
                    "indicator": name,
                    "feature": name,
                    "observed_value": 1.0,
                    "description": "The bounded parser reported this extraction condition.",
                    "evidence_type": "parser_health_observation",
                }
            )
    return tuple(rows)


@dataclass
class DeploymentBundle:
    """The only model artifact accepted by the Phase 8 application."""

    model: Phase4ModelBundle
    feature_pipeline: FeaturePipelineV2
    explanation_background: np.ndarray
    reference_median: np.ndarray
    reference_mad: np.ndarray
    schema_sha256: str
    source_model_sha256: str
    source_pipeline_sha256: str
    dataset_quality_sha256: str
    split_manifest_sha256: str
    transformation_manifest_sha256: str
    maximum_pdf_bytes: int = 50 * 1024 * 1024
    uncertainty_probability_margin: float = 0.02
    ood_robust_z_threshold: float = 8.0
    ood_feature_fraction_threshold: float = 0.10
    bundle_version: str = DEPLOYMENT_BUNDLE_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.bundle_version != DEPLOYMENT_BUNDLE_VERSION:
            raise RuntimeError("Unsupported deployment bundle version.")
        self.model.validate(require_three_seeds=False)
        self.feature_pipeline.metadata()
        names = tuple(self.feature_pipeline.output_feature_names_)
        if names != tuple(self.model.feature_names):
            raise RuntimeError("Deployment model/pipeline feature schema mismatch.")
        if self.schema_sha256 != current_schema_digest(self.feature_pipeline):
            raise RuntimeError("Deployment bundle schema digest mismatch.")
        for name in (
            "source_model_sha256",
            "source_pipeline_sha256",
            "dataset_quality_sha256",
            "split_manifest_sha256",
            "transformation_manifest_sha256",
        ):
            if not _valid_digest(str(getattr(self, name))):
                raise RuntimeError(f"Deployment provenance lacks {name}.")
        background = np.asarray(self.explanation_background, dtype=np.float32)
        median = np.asarray(self.reference_median, dtype=np.float32).reshape(-1)
        mad = np.asarray(self.reference_mad, dtype=np.float32).reshape(-1)
        if background.ndim != 2 or background.shape[0] < 100 or background.shape[1] != len(names):
            raise RuntimeError("Deployment explanation background must contain at least 100 aligned train rows.")
        if median.shape != (len(names),) or mad.shape != (len(names),):
            raise RuntimeError("Deployment OOD reference does not align with the feature schema.")
        if not np.isfinite(background).all() or not np.isfinite(median).all() or not np.isfinite(mad).all():
            raise RuntimeError("Deployment reference contains non-finite values.")
        if not 0 < int(self.maximum_pdf_bytes) <= 100 * 1024 * 1024:
            raise RuntimeError("Deployment PDF byte limit is invalid.")
        if not 0.0 <= float(self.uncertainty_probability_margin) < 0.5:
            raise RuntimeError("Deployment uncertainty margin is invalid.")
        if not 0.0 < float(self.ood_feature_fraction_threshold) <= 1.0:
            raise RuntimeError("Deployment OOD fraction threshold is invalid.")

    @classmethod
    def create(
        cls,
        *,
        model: Phase4ModelBundle,
        feature_pipeline: FeaturePipelineV2,
        explanation_background: np.ndarray,
        source_model_path: Path,
        source_pipeline_path: Path,
        dataset_quality_path: Path,
        split_manifest_path: Path,
        transformation_manifest_path: Path,
        maximum_pdf_bytes: int = 50 * 1024 * 1024,
        uncertainty_probability_margin: float = 0.02,
        ood_robust_z_threshold: float = 8.0,
        ood_feature_fraction_threshold: float = 0.10,
        provenance: Mapping[str, Any] | None = None,
    ) -> "DeploymentBundle":
        reference = np.asarray(explanation_background, dtype=np.float32)
        median = np.median(reference, axis=0).astype(np.float32)
        mad = np.median(np.abs(reference - median), axis=0).astype(np.float32)
        value = cls(
            model=model,
            feature_pipeline=feature_pipeline,
            explanation_background=reference,
            reference_median=median,
            reference_mad=mad,
            schema_sha256=current_schema_digest(feature_pipeline),
            source_model_sha256=sha256_file(source_model_path),
            source_pipeline_sha256=sha256_file(source_pipeline_path),
            dataset_quality_sha256=sha256_file(dataset_quality_path),
            split_manifest_sha256=sha256_file(split_manifest_path),
            transformation_manifest_sha256=sha256_file(transformation_manifest_path),
            maximum_pdf_bytes=maximum_pdf_bytes,
            uncertainty_probability_margin=uncertainty_probability_margin,
            ood_robust_z_threshold=ood_robust_z_threshold,
            ood_feature_fraction_threshold=ood_feature_fraction_threshold,
            provenance=dict(provenance or {}),
        )
        value.validate()
        return value

    def _local_attributions(self, row: np.ndarray, *, top_k: int = 8) -> tuple[dict[str, Any], ...]:
        """Use real-background median replacement as a local perturbation check."""
        baseline_probability = float(self.model.predict_proba(row.reshape(1, -1))[0])
        perturbed = np.repeat(row.reshape(1, -1), len(row), axis=0)
        perturbed[np.arange(len(row)), np.arange(len(row))] = self.reference_median
        probabilities = self.model.predict_proba(perturbed)
        contributions = baseline_probability - probabilities
        order = np.argsort(np.abs(contributions))[::-1][: min(top_k, len(row))]
        names = tuple(self.model.feature_names)
        return tuple(
            {
                "feature": names[index],
                "contribution": float(contributions[index]),
                "direction": "increases_malicious_score" if contributions[index] > 0 else "decreases_malicious_score",
                "method": "local_train_median_replacement_probability_delta",
                "reference": "real_train_only_background",
            }
            for index in order
            if abs(float(contributions[index])) > 1e-12
        )

    def predict_record(
        self,
        record: Mapping[str, float],
        diagnostics: Mapping[str, Any],
        *,
        include_explanation: bool = True,
    ) -> DeploymentDecision:
        self.validate()
        indicators = observed_indicators(record, diagnostics)
        if bool(diagnostics.get("abstain_recommended", False)):
            reasons = tuple(
                sorted(
                    name
                    for name in (
                        "parse_failure", "extraction_timeout", "file_too_large",
                        "extraction_limit_reached", "invalid_header", "invalid_eof",
                        "truncated_structure",
                    )
                    if float(diagnostics.get(name, 0.0) or 0.0) > 0
                )
            ) or ("parser_health_gate",)
            return DeploymentDecision(
                outcome="uncertain/abstain",
                malicious_probability=None,
                threshold=self.model.threshold,
                threshold_policy=self.model.selected_policy,
                abstention_reasons=reasons,
                raw_indicators=indicators,
            )

        row = self.feature_pipeline.transform_record(record).to_numpy(dtype=np.float32)[0]
        scale = np.maximum(self.reference_mad * 1.4826, 1e-6)
        robust_z = np.abs((row - self.reference_median) / scale)
        ood_fraction = float(np.mean(robust_z > self.ood_robust_z_threshold))
        probability = float(self.model.predict_proba(row.reshape(1, -1))[0])
        reasons: list[str] = []
        if ood_fraction > self.ood_feature_fraction_threshold:
            reasons.append("out_of_distribution_feature_profile")
        if abs(probability - self.model.threshold) <= self.uncertainty_probability_margin:
            reasons.append("probability_within_abstention_margin")
        outcome = (
            "uncertain/abstain"
            if reasons
            else ("malicious" if probability >= self.model.threshold else "benign")
        )
        attributions = self._local_attributions(row) if include_explanation else ()
        return DeploymentDecision(
            outcome=outcome,
            malicious_probability=probability,
            threshold=self.model.threshold,
            threshold_policy=self.model.selected_policy,
            abstention_reasons=tuple(reasons),
            raw_indicators=indicators,
            model_attributions=attributions,
            ood_feature_fraction=ood_fraction,
        )

    def save(self, path: Path, *, identity: ExperimentIdentity) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
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
        write_artifact_metadata(
            output,
            DEPLOYMENT_ARTIFACT_KIND,
            identity,
            extra={
                "bundle_version": self.bundle_version,
                "schema_sha256": self.schema_sha256,
                "model_name": self.model.model_name,
                "threshold": self.model.threshold,
                "threshold_policy": self.model.selected_policy,
                "source_model_sha256": self.source_model_sha256,
                "source_pipeline_sha256": self.source_pipeline_sha256,
                "dataset_quality_sha256": self.dataset_quality_sha256,
                "split_manifest_sha256": self.split_manifest_sha256,
                "transformation_manifest_sha256": self.transformation_manifest_sha256,
            },
        )
        return output

    @classmethod
    def load(cls, path: Path, *, identity: ExperimentIdentity) -> "DeploymentBundle":
        metadata = verify_artifact_compatibility(path, identity)
        if metadata.artifact_kind != DEPLOYMENT_ARTIFACT_KIND:
            raise ArtifactCompatibilityError("Artifact is not a Phase 8 deployment bundle.")
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Artifact at {path} is not DeploymentBundle.")
        value.validate()
        checks = {
            "bundle_version": value.bundle_version,
            "schema_sha256": value.schema_sha256,
            "threshold_policy": value.model.selected_policy,
            "source_model_sha256": value.source_model_sha256,
            "source_pipeline_sha256": value.source_pipeline_sha256,
        }
        mismatches = {name: (metadata.extra.get(name), expected) for name, expected in checks.items() if metadata.extra.get(name) != expected}
        if mismatches:
            raise ArtifactCompatibilityError(f"Deployment sidecar/bundle mismatch: {mismatches}")
        if abs(float(metadata.extra.get("threshold", -1)) - value.model.threshold) > 1e-15:
            raise ArtifactCompatibilityError("Deployment sidecar threshold mismatch.")
        return value

