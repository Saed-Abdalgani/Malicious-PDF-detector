"""Application bridge to the single checksummed Phase 8 deployment bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MODELS_DIR
from src.experiment import create_experiment_identity, load_experiment_config
from src.inference import analyze_pdf_bytes
from src.models.deployment import DeploymentBundle


DEPLOYMENT_BUNDLE_PATH = MODELS_DIR / "deployment" / "deployment_bundle_v1.joblib"


@dataclass(frozen=True)
class AnalysisResult:
    outcome: str
    malicious_probability: float | None
    threshold: float
    threshold_policy: str
    abstention_reasons: tuple[str, ...]
    raw_indicators: tuple[dict[str, Any], ...]
    model_attributions: tuple[dict[str, Any], ...]
    features: dict[str, float]
    time_ms: float
    file_hash: str

    @property
    def prediction(self) -> str:
        return {
            "benign": "Benign",
            "malicious": "Malicious",
            "uncertain/abstain": "Uncertain / Abstain",
        }[self.outcome]


class PDFAnalyzer:
    def __init__(
        self,
        *,
        bundle_path: Path = DEPLOYMENT_BUNDLE_PATH,
        bundle: DeploymentBundle | None = None,
    ) -> None:
        """Load one self-contained artifact; separate model/scaler fallback is forbidden."""
        if bundle is None:
            identity = create_experiment_identity(load_experiment_config())
            bundle = DeploymentBundle.load(Path(bundle_path), identity=identity)
        bundle.validate()
        self.bundle = bundle

    def analyze(self, uploaded_file: Any) -> AnalysisResult:
        result = analyze_pdf_bytes(
            uploaded_file.getvalue(), self.bundle, include_explanation=True
        )
        decision = result.decision
        return AnalysisResult(
            outcome=decision.outcome,
            malicious_probability=decision.malicious_probability,
            threshold=decision.threshold,
            threshold_policy=decision.threshold_policy,
            abstention_reasons=decision.abstention_reasons,
            raw_indicators=decision.raw_indicators,
            model_attributions=decision.model_attributions,
            features=result.features,
            time_ms=result.elapsed_ms,
            file_hash=result.file_sha256,
        )

    @staticmethod
    def get_feature_breakdown(features: dict[str, float]) -> pd.DataFrame:
        frame = pd.DataFrame(features.items(), columns=["Feature", "Observed value"])
        frame["Observed value"] = frame["Observed value"].map(lambda value: f"{value:g}")
        return frame
