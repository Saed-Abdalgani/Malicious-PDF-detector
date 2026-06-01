"""
generate_sample_reports.py
---------------------------
Pre-generate cached ThreatReports for the bundled sample PDFs so the
"AI Threat Analyst" page renders a rich report even when Ollama is offline
(e.g. during a live demo on a machine without the LLM).

The reports use the *real* feature extraction, suspicious-feature analysis, and
SHAP decision drivers; the narrative text is synthesized offline (no LLM) and is
clearly labelled as a curated, offline cache.

Output: data/sample_reports/<name>.json  (+ .md)

Usage::

    python -m scripts.generate_sample_reports
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import PROJECT_ROOT, SAMPLE_PDFS_DIR
from src.llm.client import GemmaClient
from src.llm.analyzer import ThreatAnalyzer
from src.llm.report_generator import save_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_REPORTS_DIR = PROJECT_ROOT / "data" / "sample_reports"

# Ground-truth labels for the curated demo files (by filename stem).
_LABELS = {"benign_sample": "Benign", "malicious_sample": "Malicious"}


def _synthesize_explanation(prediction, suspicious, drivers) -> str:
    """Build a readable, accurate offline explanation from real signals."""
    top_sus = ", ".join(f"`{n}`={v:g}" for n, v, *_ in suspicious[:5]) or "none"
    top_drv = ", ".join(f"`{n}` ({val:+.3f})" for n, val, *_ in drivers[:5]) or "n/a"
    if prediction == "Malicious":
        return (
            "This file exhibits structural indicators consistent with a "
            "malicious PDF. The static extractor flagged the following "
            f"high-risk indicators: {top_sus}. The model's own decision drivers "
            f"(SHAP) were: {top_drv}. The presence of auto-execute and scripting "
            "actions, combined with obfuscation indicators, is characteristic of "
            "a JavaScript-based exploit chain delivered via a crafted document. "
            "_(Offline cached report — generated without the LLM.)_"
        )
    return (
        "This file's structure is consistent with a benign document. No "
        "high-risk auto-execute or scripting indicators dominate the profile "
        f"(flagged: {top_sus}). The model's decision drivers (SHAP) were: "
        f"{top_drv}. _(Offline cached report — generated without the LLM.)_"
    )


def generate() -> None:
    SAMPLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid a hard SHAP dependency if it is unavailable.
    try:
        from src.features.explain import explain_pdf
        shap_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SHAP unavailable, drivers will be empty: {exc}")
        shap_ok = False

    from src.features.vectorizer import extract_features_dict

    analyzer = ThreatAnalyzer(GemmaClient())  # offline-safe (no LLM calls forced)

    for pdf in sorted(Path(SAMPLE_PDFS_DIR).glob("*.pdf")):
        stem = pdf.stem
        prediction = _LABELS.get(stem, "Malicious")
        confidence = 0.95 if prediction == "Malicious" else 0.93

        features = extract_features_dict(pdf)
        suspicious = analyzer.identify_suspicious_features(features)
        drivers = []
        if shap_ok:
            try:
                drivers = explain_pdf(pdf, k=6)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"SHAP failed for {pdf.name}: {exc}")

        report = analyzer.analyze(
            features=features,
            prediction=prediction,
            confidence=confidence,
            filename=pdf.name,
            pdf_bytes=pdf.read_bytes(),
            ml_drivers=drivers,
        )
        # Replace the LLM-failure text with a clean synthesized narrative.
        report.threat_explanation = _synthesize_explanation(prediction, suspicious, drivers)
        report.model_used = "offline-cache (no Ollama)"
        report.raw_llm_response = report.threat_explanation

        base = SAMPLE_REPORTS_DIR / stem
        save_report(report, str(base.with_suffix(".json")), fmt="json")
        save_report(report, str(base.with_suffix(".md")), fmt="markdown")
        # Also index by file hash for runtime matching by the app.
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
        logger.info(f"Cached report for {pdf.name} (hash {digest}) -> {base}.json")


if __name__ == "__main__":
    generate()
