"""Bounded local PDF inference for the Streamlit application and golden tests."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from src.features.vectorizer import FEATURE_COLUMNS, extract_features_record
from src.models.deployment import DeploymentBundle, DeploymentDecision
from src.security.file_validation import validate_pdf_envelope


@dataclass(frozen=True)
class PDFInferenceResult:
    decision: DeploymentDecision
    features: dict[str, float]
    elapsed_ms: float
    file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "features": self.features,
            "elapsed_ms": self.elapsed_ms,
            "file_sha256": self.file_sha256,
        }


def _envelope_abstention(bundle: DeploymentBundle, reason: str) -> DeploymentDecision:
    return DeploymentDecision(
        outcome="uncertain/abstain",
        malicious_probability=None,
        threshold=bundle.model.threshold,
        threshold_policy=bundle.model.selected_policy,
        abstention_reasons=(reason,),
        raw_indicators=(
            {
                "indicator": reason,
                "feature": "file_envelope",
                "observed_value": 1.0,
                "description": "The upload failed a bounded PDF envelope check.",
                "evidence_type": "file_envelope_observation",
            },
        ),
    )


def analyze_pdf_bytes(
    data: bytes,
    bundle: DeploymentBundle,
    *,
    include_explanation: bool = True,
) -> PDFInferenceResult:
    """Analyze bytes locally; uploaded content is never retained or sent elsewhere."""
    started = time.perf_counter()
    payload = bytes(data)
    digest = hashlib.sha256(payload).hexdigest()
    envelope = validate_pdf_envelope(payload, maximum_bytes=bundle.maximum_pdf_bytes)
    if not envelope.valid:
        return PDFInferenceResult(
            decision=_envelope_abstention(bundle, envelope.reason),
            features={},
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            file_sha256=digest,
        )

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(payload)
            temporary_path = handle.name
        record, diagnostics = extract_features_record(temporary_path)
        decision = bundle.predict_record(record, diagnostics, include_explanation=include_explanation)
        raw = {name: float(record[name]) for name in FEATURE_COLUMNS}
        return PDFInferenceResult(
            decision=decision,
            features=raw,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            file_sha256=digest,
        )
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

