"""
analyzer.py
-----------
Threat analysis pipeline that combines ML classification results with
LLM-powered threat intelligence via the Gemma 4 E4B model.

The ``ThreatAnalyzer`` class orchestrates the full analysis workflow:

    1. Receives ML classification result + raw feature vector
    2. Identifies suspicious features (>2σ deviation from benign baseline)
    3. Constructs context-rich prompts using cybersecurity templates
    4. Generates threat analysis via the local LLM (Ollama)
    5. Parses the structured output into a ``ThreatReport`` dataclass

This module is the bridge between the ML detection pipeline and the
LLM explainability layer. The LLM does NOT replace ML classification —
it *explains* and *enriches* the ML verdict with actionable intelligence.

Usage:
    from src.llm.analyzer import ThreatAnalyzer
    from src.llm.client import GemmaClient

    client = GemmaClient()
    client.auto_initialize()

    analyzer = ThreatAnalyzer(client)
    report = analyzer.analyze(
        features={"js_count": 5, "openaction_count": 1, ...},
        prediction="Malicious",
        confidence=0.973,
    )
    print(report.to_markdown())
"""

import re
import time
from typing import Dict, Generator, List, Optional, Tuple


from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.llm.client import GemmaClient
from src.llm.prompts import (
    FEATURE_DESCRIPTIONS,
    FOLLOW_UP_TEMPLATE,
    JAVASCRIPT_ANALYSIS_TEMPLATE,
    QUICK_SUMMARY_TEMPLATE,
    SYSTEM_PROMPT,
    THREAT_ANALYSIS_TEMPLATE,
    format_feature_summary,
    format_suspicious_features,
)
from src.llm.report_generator import ThreatReport, generate_file_hash
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Benign baseline path
BASELINE_PATH = MODELS_DIR / "benign_baseline.pkl"

# Deviation threshold for flagging suspicious features
_DEVIATION_THRESHOLD = 2.0

# Features strongly associated with malicious behavior
_HIGH_RISK_FEATURES = {
    "js_count", "javascript_count", "openaction_count",
    "launch_count", "submitform_count", "xfa_count",
    "obfuscation_count", "richmedia_count", "jbig2decode_count",
    "aa_count", "embedded_file_count",
}


class ThreatAnalyzer:
    """Orchestrates LLM-powered threat analysis for PDF classification results.

    Combines ML verdicts with feature-level analysis and LLM intelligence
    to produce comprehensive ``ThreatReport`` objects.

    The analyzer uses benign baseline statistics (mean/std per feature from
    the training set) to identify features that deviate significantly from
    normal behavior, providing the LLM with targeted context for its analysis.

    Attributes:
        client (GemmaClient): Ollama client for LLM inference.
        baseline (dict): Benign baseline statistics ``{feature: {mean, std}}``.
        _analysis_cache (dict): Cache of previous analyses (by feature hash).

    Example:
        >>> analyzer = ThreatAnalyzer(client)
        >>> report = analyzer.analyze(features, "Malicious", 0.97)
        >>> print(report.risk_severity)
        'Critical'
    """

    def __init__(self, client: GemmaClient):
        """Initialize the ThreatAnalyzer.

        Args:
            client: An initialized ``GemmaClient`` instance.
        """
        self.client = client
        self.baseline: Dict[str, Dict[str, float]] = {}
        self._analysis_cache: Dict[str, ThreatReport] = {}

        # Attempt to load benign baseline
        self._load_baseline()

        logger.info("ThreatAnalyzer initialized")

    # ------------------------------------------------------------------
    # Baseline loading
    # ------------------------------------------------------------------

    def _load_baseline(self) -> None:
        """Load benign baseline statistics from disk.

        The baseline contains per-feature mean and standard deviation
        computed from benign samples in the training set. Used to
        identify features that deviate significantly from normal.
        """
        try:
            import joblib
            if BASELINE_PATH.exists():
                self.baseline = joblib.load(BASELINE_PATH)
                logger.info(
                    f"Benign baseline loaded — {len(self.baseline)} features "
                    f"from {BASELINE_PATH}"
                )
            else:
                logger.warning(
                    f"Benign baseline not found at {BASELINE_PATH}. "
                    f"Suspicious feature detection will use fallback heuristics. "
                    f"Run vectorizer.compute_benign_baseline() to generate."
                )
                self._build_fallback_baseline()
        except Exception as exc:
            logger.warning(f"Failed to load baseline: {exc}. Using fallback.")
            self._build_fallback_baseline()

    def _build_fallback_baseline(self) -> None:
        """Build a conservative fallback baseline for common features.

        Used when the proper baseline file is not available. Values are
        based on typical benign PDF characteristics from the CIC dataset
        documentation.
        """
        self.baseline = {}
        # Most keyword counts are 0 or near-0 in benign PDFs
        for feature in FEATURE_COLUMNS:
            if feature in _HIGH_RISK_FEATURES:
                # High-risk features: very low mean, any non-zero is notable
                self.baseline[feature] = {"mean": 0.0, "std": 0.1}
            elif "count" in feature:
                # General count features: moderate baseline
                self.baseline[feature] = {"mean": 5.0, "std": 10.0}
            elif feature == "pdf_size":
                self.baseline[feature] = {"mean": 50000.0, "std": 100000.0}
            elif feature == "avg_stream_size":
                self.baseline[feature] = {"mean": 500.0, "std": 1000.0}
            elif feature in ("is_encrypted", "has_text", "header_valid"):
                self.baseline[feature] = {"mean": 0.5, "std": 0.5}
            else:
                self.baseline[feature] = {"mean": 1.0, "std": 2.0}

        logger.info(
            f"Fallback baseline built — {len(self.baseline)} features "
            f"(conservative estimates)"
        )

    # ------------------------------------------------------------------
    # Suspicious feature identification
    # ------------------------------------------------------------------

    def identify_suspicious_features(
        self,
        features: Dict[str, float],
    ) -> List[Tuple[str, float, float, str]]:
        """Compare each feature against the benign baseline.

        Identifies features that deviate more than 2σ from the benign
        mean, indicating potential malicious indicators.

        Args:
            features: Raw feature dictionary ``{name: value}``.

        Returns:
            list: Sorted list of ``(feature_name, value, deviation, description)``
                  tuples, ordered by deviation magnitude (highest first).
        """
        suspicious = []

        for feature_name in FEATURE_COLUMNS:
            value = float(features.get(feature_name, 0.0))
            stats = self.baseline.get(feature_name, {"mean": 0.0, "std": 1.0})

            mean = stats["mean"]
            std = stats["std"]

            # Avoid division by zero
            if std < 1e-8:
                std = 0.1

            deviation = (value - mean) / std

            # Flag if deviation exceeds threshold
            if abs(deviation) > _DEVIATION_THRESHOLD:
                description = FEATURE_DESCRIPTIONS.get(
                    feature_name,
                    "Feature value deviates significantly from benign baseline"
                )
                suspicious.append((feature_name, value, deviation, description))

            # Also flag high-risk features that have any non-zero value
            elif feature_name in _HIGH_RISK_FEATURES and value > 0:
                description = FEATURE_DESCRIPTIONS.get(
                    feature_name,
                    "High-risk feature with non-zero value"
                )
                suspicious.append((feature_name, value, deviation, description))

        # Sort by absolute deviation (most suspicious first)
        suspicious.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(
            f"Identified {len(suspicious)} suspicious features "
            f"(threshold: {_DEVIATION_THRESHOLD} sigma)"
        )
        return suspicious

    # ------------------------------------------------------------------
    # Full threat analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        features: Dict[str, float],
        prediction: str,
        confidence: float,
        filename: str = "unknown.pdf",
        pdf_bytes: Optional[bytes] = None,
        processing_time_ms: float = 0.0,
    ) -> ThreatReport:
        """Run full LLM-powered threat analysis on a classified PDF.

        Constructs a detailed prompt with ML results and suspicious
        feature highlights, sends it to the LLM, and parses the response
        into a structured ``ThreatReport``.

        Args:
            features: Raw (unscaled) feature dictionary.
            prediction: ML classification result ("Malicious" or "Benign").
            confidence: Classification confidence (0.0 to 1.0).
            filename: Original PDF filename (sanitized).
            pdf_bytes: Raw PDF bytes for hash generation (optional).
            processing_time_ms: ML pipeline processing time.

        Returns:
            ThreatReport: Complete structured threat report.
        """
        start_time = time.perf_counter()

        # Generate file hash
        file_hash = ""
        if pdf_bytes:
            file_hash = generate_file_hash(pdf_bytes)

        # Identify suspicious features
        suspicious = self.identify_suspicious_features(features)

        # Select template based on prediction
        if prediction.lower() in ("malicious", "suspicious"):
            template = THREAT_ANALYSIS_TEMPLATE
        else:
            template = QUICK_SUMMARY_TEMPLATE

        # Build prompt
        prompt = template.format(
            prediction=prediction,
            confidence=confidence * 100,
            processing_time=processing_time_ms,
            feature_summary=format_feature_summary(features),
            suspicious_features=format_suspicious_features(suspicious),
        )

        # Call LLM
        try:
            logger.info(
                f"Sending analysis request to LLM — "
                f"prediction={prediction}, confidence={confidence:.3f}, "
                f"suspicious_features={len(suspicious)}"
            )
            raw_response = self.client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
            )
        except (ConnectionError, RuntimeError) as exc:
            logger.error(f"LLM analysis failed: {exc}")
            raw_response = (
                f"⚠️ LLM analysis unavailable: {exc}\n\n"
                f"The ML model classified this PDF as **{prediction}** "
                f"with {confidence * 100:.1f}% confidence."
            )

        # Parse LLM response into structured fields
        risk_severity = self._extract_severity(raw_response, prediction, suspicious)
        threat_explanation = self._extract_section(
            raw_response, "Threat Assessment", raw_response
        )
        attack_vector = self._extract_section(
            raw_response, "Attack Vector", ""
        )
        remediation = self._extract_section(
            raw_response, "Remediation", self._default_remediation(prediction)
        )

        total_time = (time.perf_counter() - start_time) * 1000

        # Build report
        report = ThreatReport(
            file_hash=file_hash,
            filename=filename,
            ml_prediction=prediction,
            ml_confidence=confidence,
            risk_severity=risk_severity,
            threat_explanation=threat_explanation,
            attack_vector=attack_vector,
            suspicious_features=suspicious,
            remediation=remediation,
            processing_time_ms=total_time,
            model_used=self.client.model or "unknown",
            raw_llm_response=raw_response,
            feature_summary=features,
        )

        logger.info(
            f"Analysis complete — severity={risk_severity}, "
            f"time={total_time:.0f}ms"
        )

        return report

    # ------------------------------------------------------------------
    # Streaming analysis
    # ------------------------------------------------------------------

    def analyze_stream(
        self,
        features: Dict[str, float],
        prediction: str,
        confidence: float,
    ) -> Generator[str, None, None]:
        """Stream LLM threat analysis token by token.

        Used by the Streamlit UI for real-time response display.

        Args:
            features: Raw feature dictionary.
            prediction: ML prediction.
            confidence: ML confidence.

        Yields:
            str: Individual response tokens.
        """
        suspicious = self.identify_suspicious_features(features)

        if prediction.lower() in ("malicious", "suspicious"):
            template = THREAT_ANALYSIS_TEMPLATE
        else:
            template = QUICK_SUMMARY_TEMPLATE

        prompt = template.format(
            prediction=prediction,
            confidence=confidence * 100,
            processing_time=0,
            feature_summary=format_feature_summary(features),
            suspicious_features=format_suspicious_features(suspicious),
        )

        yield from self.client.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
        )

    # ------------------------------------------------------------------
    # JavaScript analysis
    # ------------------------------------------------------------------

    def analyze_javascript(
        self,
        js_code: str,
        trigger_mechanism: str = "/OpenAction → /JS",
        obfuscation_level: str = "Unknown",
    ) -> str:
        """Analyze extracted JavaScript code for malicious behavior.

        Args:
            js_code: JavaScript code extracted from the PDF.
            trigger_mechanism: How the JS is triggered (e.g., /OpenAction).
            obfuscation_level: Estimated obfuscation level.

        Returns:
            str: LLM-generated JavaScript analysis.
        """
        prompt = JAVASCRIPT_ANALYSIS_TEMPLATE.format(
            js_code=js_code,
            trigger_mechanism=trigger_mechanism,
            obfuscation_level=obfuscation_level,
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.2,
            )
            return response
        except (ConnectionError, RuntimeError) as exc:
            logger.error(f"JavaScript analysis failed: {exc}")
            return f"⚠️ Analysis unavailable: {exc}"

    # ------------------------------------------------------------------
    # Quick summary (for benign files)
    # ------------------------------------------------------------------

    def quick_summary(
        self,
        features: Dict[str, float],
        prediction: str,
        confidence: float,
    ) -> str:
        """Generate a short safety summary for benign PDFs.

        Uses the ``QUICK_SUMMARY_TEMPLATE`` for a concise response.

        Args:
            features: Raw feature dictionary.
            prediction: ML prediction (should be "Benign").
            confidence: ML confidence.

        Returns:
            str: Short safety confirmation text.
        """
        prompt = QUICK_SUMMARY_TEMPLATE.format(
            prediction=prediction,
            confidence=confidence * 100,
            feature_summary=format_feature_summary(features),
        )

        try:
            return self.client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
            )
        except (ConnectionError, RuntimeError) as exc:
            logger.error(f"Quick summary failed: {exc}")
            return (
                f"✅ This PDF has been classified as **{prediction}** "
                f"with {confidence * 100:.1f}% confidence. "
                f"No significant suspicious indicators were detected."
            )

    # ------------------------------------------------------------------
    # Follow-up Q&A
    # ------------------------------------------------------------------

    def follow_up(
        self,
        user_question: str,
        previous_report: ThreatReport,
    ) -> str:
        """Answer a follow-up question about a previous analysis.

        Args:
            user_question: The user's follow-up question.
            previous_report: The ThreatReport from the initial analysis.

        Returns:
            str: LLM response to the follow-up question.
        """
        # Build key findings summary
        key_findings = ", ".join(
            f"{name}={value:.0f}" for name, value, _, _ in
            previous_report.suspicious_features[:5]
        ) if previous_report.suspicious_features else "None identified"

        prompt = FOLLOW_UP_TEMPLATE.format(
            previous_analysis=previous_report.threat_explanation[:2000],
            prediction=previous_report.ml_prediction,
            confidence=previous_report.ml_confidence,
            key_findings=key_findings,
            user_question=user_question,
        )

        try:
            return self.client.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.4,
            )
        except (ConnectionError, RuntimeError) as exc:
            logger.error(f"Follow-up failed: {exc}")
            return f"⚠️ Unable to answer: {exc}"

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_severity(
        response: str,
        prediction: str,
        suspicious: List[Tuple],
    ) -> str:
        """Extract or infer severity rating from LLM response.

        First attempts to parse severity from the LLM text. If not found,
        infers based on the number and type of suspicious features.

        Args:
            response: Raw LLM response text.
            prediction: ML prediction.
            suspicious: List of suspicious feature tuples.

        Returns:
            str: Severity level ("Critical"/"High"/"Medium"/"Low").
        """
        # Try to extract from response
        response_lower = response.lower()
        for level in ["critical", "high", "medium", "low"]:
            patterns = [
                rf"severity[:\s]*\**\s*{level}",
                rf"risk[:\s]*\**\s*{level}",
                rf"rating[:\s]*\**\s*{level}",
                rf"🔴\s*\**\s*{level}" if level == "critical" else None,
                rf"🟠\s*\**\s*{level}" if level == "high" else None,
                rf"🟡\s*\**\s*{level}" if level == "medium" else None,
                rf"🟢\s*\**\s*{level}" if level == "low" else None,
            ]
            for pattern in patterns:
                if pattern and re.search(pattern, response_lower):
                    return level.capitalize()

        # Fallback: infer from features
        if prediction.lower() == "benign":
            return "Low"

        high_risk_count = sum(
            1 for name, _, _, _ in suspicious
            if name in _HIGH_RISK_FEATURES
        )
        total_suspicious = len(suspicious)

        if high_risk_count >= 3 or total_suspicious >= 8:
            return "Critical"
        elif high_risk_count >= 2 or total_suspicious >= 5:
            return "High"
        elif high_risk_count >= 1 or total_suspicious >= 3:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def _extract_section(response: str, heading: str, default: str) -> str:
        """Extract a specific section from the LLM response.

        Looks for markdown-style headings (## or **heading**) and returns
        the content until the next heading.

        Args:
            response: Raw LLM response text.
            heading: Section heading to search for.
            default: Default text if section is not found.

        Returns:
            str: Extracted section content or default.
        """
        patterns = [
            rf"#+\s*\d*\.?\s*{re.escape(heading)}[^\n]*\n(.*?)(?=\n#+|\Z)",
            rf"\*\*\d*\.?\s*{re.escape(heading)}[^\n]*\*\*[^\n]*\n(.*?)(?=\*\*\d|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    return content

        return default

    @staticmethod
    def _default_remediation(prediction: str) -> str:
        """Generate default remediation text when LLM extraction fails."""
        if prediction.lower() == "malicious":
            return (
                "1. **Quarantine** — Isolate this file immediately\n"
                "2. **Do not open** — Do not open in any PDF reader\n"
                "3. **Investigate source** — Check the email/download origin\n"
                "4. **Alert security team** — Escalate for manual review\n"
                "5. **Block indicators** — Add file hash to blocklist"
            )
        return "No immediate action required. File appears benign."


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ThreatAnalyzer — Feature Analysis Demo")
    print("=" * 60)

    # Demo: identify suspicious features without LLM
    from src.llm.client import GemmaClient

    client = GemmaClient()

    analyzer = ThreatAnalyzer(client)

    # Simulate malicious features
    sample_features = {col: 0.0 for col in FEATURE_COLUMNS}
    sample_features.update({
        "js_count": 5.0,
        "javascript_count": 3.0,
        "openaction_count": 1.0,
        "action_count": 2.0,
        "obfuscation_count": 12.0,
        "filter_count": 8.0,
        "obj_count": 45.0,
        "stream_count": 15.0,
        "uri_count": 4.0,
        "pdf_size": 85000.0,
        "page_count": 1.0,
    })

    suspicious = analyzer.identify_suspicious_features(sample_features)
    print(f"\nSuspicious features found: {len(suspicious)}")
    for name, value, dev, desc in suspicious[:5]:
        print(f"  {'🔴' if abs(dev) > 5 else '🟠' if abs(dev) > 3 else '🟡'} "
              f"{name} = {value:.0f} ({dev:+.1f}σ) — {desc[:60]}")

    # Severity inference
    severity = ThreatAnalyzer._extract_severity(
        "", "Malicious", suspicious
    )
    print(f"\nInferred severity: {severity}")

    print("=" * 60)
