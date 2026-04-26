"""
report_generator.py
-------------------
Structured security report builder for LLM-generated threat analyses.

Provides the ``ThreatReport`` dataclass and ``generate_file_hash()`` utility
for creating, formatting, and exporting comprehensive PDF threat analysis
reports in multiple formats (Markdown, JSON, dict).

The ``ThreatReport`` is the primary output artifact of the LLM analysis
pipeline. It captures both ML classification results and LLM-generated
threat intelligence in a structured format suitable for:

    - Streamlit dashboard display (``to_dict()``)
    - Downloadable security reports (``to_markdown()``)
    - API/integration export (``to_json()``)

Usage:
    from src.llm.report_generator import ThreatReport, generate_file_hash

    report = ThreatReport(
        file_hash=generate_file_hash(pdf_bytes),
        ml_prediction="Malicious",
        ml_confidence=0.973,
        risk_severity="Critical",
        threat_explanation="This PDF contains embedded JavaScript...",
        attack_vector="JavaScript exploit via /OpenAction → /JS chain",
        suspicious_features=[("js_count", 5.0, 8.2, "...")],
        remediation="Quarantine immediately. Do not open in any PDF reader.",
        processing_time_ms=42.5,
    )
    print(report.to_markdown())
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File hash utility
# ---------------------------------------------------------------------------

def generate_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file contents.

    Used to uniquely identify analyzed PDF files in reports and for
    cross-referencing analyses.

    Args:
        data: Raw file bytes.

    Returns:
        str: Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> generate_file_hash(b"test content")
        '6ae8a75555209fd6c44157c0aed8016e763ff435...'
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# ThreatReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThreatReport:
    """Structured security report for a PDF threat analysis.

    Captures the complete analysis result including ML classification,
    LLM-generated threat intelligence, and actionable remediation.

    Attributes:
        timestamp: ISO 8601 timestamp of the analysis.
        file_hash: SHA-256 hash of the analyzed PDF.
        filename: Original filename (sanitized).
        ml_prediction: ML classification result ("Malicious" or "Benign").
        ml_confidence: Classification confidence (0.0 to 1.0).
        risk_severity: LLM-assigned severity ("Critical"/"High"/"Medium"/"Low").
        threat_explanation: LLM-generated analysis of the threat.
        attack_vector: Identified attack technique and methodology.
        suspicious_features: List of (name, value, deviation, description) tuples.
        remediation: Recommended security response actions.
        processing_time_ms: Total analysis pipeline time in milliseconds.
        model_used: LLM model identifier used for analysis.
        raw_llm_response: Complete unprocessed LLM output.
        feature_summary: Complete feature dictionary for reference.
    """

    # --- Core identification ---
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_hash: str = ""
    filename: str = "unknown.pdf"

    # --- ML classification ---
    ml_prediction: str = "Unknown"
    ml_confidence: float = 0.0

    # --- LLM analysis ---
    risk_severity: str = "Unknown"
    threat_explanation: str = ""
    attack_vector: str = ""
    suspicious_features: List[Tuple] = field(default_factory=list)
    remediation: str = ""

    # --- Metadata ---
    processing_time_ms: float = 0.0
    model_used: str = ""
    raw_llm_response: str = ""
    feature_summary: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Export: Markdown
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Generate a formatted Markdown security report.

        Creates a professional, human-readable report suitable for
        download and sharing with security teams.

        Returns:
            str: Complete Markdown-formatted threat report.
        """
        severity_emoji = {
            "Critical": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
            "Unknown": "⚪",
        }
        emoji = severity_emoji.get(self.risk_severity, "⚪")

        # Header
        lines = [
            "# 🛡️ PDF Threat Analysis Report",
            "",
            f"> **Generated by ThreatScope AI** — {self._format_timestamp()}",
            "",
            "---",
            "",
            "## 📋 Summary",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **File** | `{self.filename}` |",
            f"| **SHA-256** | `{self.file_hash[:16]}...{self.file_hash[-8:]}` |" if self.file_hash else "| **SHA-256** | _not available_ |",
            f"| **ML Verdict** | **{self.ml_prediction}** |",
            f"| **Confidence** | {self.ml_confidence * 100:.1f}% |",
            f"| **Risk Severity** | {emoji} **{self.risk_severity}** |",
            f"| **Analysis Time** | {self.processing_time_ms:.0f} ms |",
            f"| **LLM Model** | `{self.model_used}` |" if self.model_used else "",
            "",
            "---",
            "",
        ]

        # Threat explanation
        if self.threat_explanation:
            lines.extend([
                "## 🔍 Threat Analysis",
                "",
                self.threat_explanation,
                "",
                "---",
                "",
            ])

        # Attack vector
        if self.attack_vector:
            lines.extend([
                "## 🎯 Attack Vector",
                "",
                self.attack_vector,
                "",
                "---",
                "",
            ])

        # Suspicious features
        if self.suspicious_features:
            lines.extend([
                "## ⚠️ Suspicious Indicators",
                "",
            ])
            for item in self.suspicious_features:
                if len(item) >= 4:
                    name, value, dev, desc = item[:4]
                    sev = "🔴" if abs(dev) > 5 else "🟠" if abs(dev) > 3 else "🟡"
                    lines.append(
                        f"- {sev} **`{name}`** = `{value:.4f}` "
                        f"(+{dev:.1f}σ) — {desc}"
                    )
                elif len(item) >= 3:
                    name, value, dev = item[:3]
                    lines.append(
                        f"- **`{name}`** = `{value:.4f}` (+{dev:.1f}σ)"
                    )
            lines.extend(["", "---", ""])

        # Remediation
        if self.remediation:
            lines.extend([
                "## 🔧 Recommended Actions",
                "",
                self.remediation,
                "",
                "---",
                "",
            ])

        # Feature summary table
        if self.feature_summary:
            lines.extend([
                "## 📊 Complete Feature Profile",
                "",
                "<details>",
                "<summary>Click to expand full feature table</summary>",
                "",
                "| Feature | Value |",
                "|---------|-------|",
            ])
            for name, value in self.feature_summary.items():
                if isinstance(value, float) and value == int(value):
                    formatted = f"{int(value)}"
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                lines.append(f"| `{name}` | {formatted} |")
            lines.extend(["", "</details>", ""])

        # Footer
        lines.extend([
            "---",
            "",
            f"*Report generated on {self._format_timestamp()} by "
            f"ThreatScope AI (Malicious PDF Detector)*",
            "",
            "*This report is generated by an AI system and should be reviewed "
            "by a qualified security analyst before taking action.*",
        ])

        return "\n".join(line for line in lines if line is not None)

    # ------------------------------------------------------------------
    # Export: JSON
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string.

        Converts all fields to JSON-serializable format, including
        tuples (converted to lists) and datetime objects.

        Args:
            indent: JSON indentation level. Default 2.

        Returns:
            str: JSON-formatted report string.
        """
        data = self._to_serializable_dict()
        return json.dumps(data, indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Export: Dict (for Streamlit)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for Streamlit display.

        All fields are converted to JSON-serializable types.

        Returns:
            dict: Flat dictionary representation of the report.
        """
        return self._to_serializable_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert dataclass to a fully JSON-serializable dictionary."""
        data = asdict(self)
        # Convert tuples in suspicious_features to lists
        if "suspicious_features" in data:
            data["suspicious_features"] = [
                list(item) if isinstance(item, (tuple, list)) else item
                for item in data["suspicious_features"]
            ]
        return data

    def _format_timestamp(self) -> str:
        """Format the ISO timestamp for human-readable display."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, TypeError):
            return self.timestamp

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreatReport":
        """Reconstruct a ThreatReport from a dictionary.

        Args:
            data: Dictionary with ThreatReport field names.

        Returns:
            ThreatReport: Reconstructed report instance.
        """
        # Convert lists back to tuples for suspicious_features
        if "suspicious_features" in data:
            data["suspicious_features"] = [
                tuple(item) if isinstance(item, list) else item
                for item in data["suspicious_features"]
            ]
        
        # Filter out unexpected keys
        valid_fields = {f.name for f in ThreatReport.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "ThreatReport":
        """Reconstruct a ThreatReport from a JSON string.

        Args:
            json_str: JSON-formatted report string.

        Returns:
            ThreatReport: Reconstructed report instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Report collection utilities
# ---------------------------------------------------------------------------

def save_report(report: ThreatReport, path: str, fmt: str = "markdown") -> str:
    """Save a ThreatReport to disk.

    Args:
        report: The report to save.
        path: Output file path.
        fmt: Format — 'markdown' or 'json'.

    Returns:
        str: The path where the report was saved.
    """
    from pathlib import Path
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        content = report.to_json()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".json")
    else:
        content = report.to_markdown()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".md")

    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Report saved to {output_path} ({fmt})")
    return str(output_path)


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ThreatReport — Demo")
    print("=" * 60)

    # Create a sample report
    report = ThreatReport(
        file_hash=generate_file_hash(b"sample malicious pdf content"),
        filename="suspicious_invoice.pdf",
        ml_prediction="Malicious",
        ml_confidence=0.973,
        risk_severity="Critical",
        threat_explanation=(
            "This PDF contains multiple indicators of a JavaScript-based "
            "exploit chain. The presence of `/JS` tags (count=5) combined "
            "with `/OpenAction` (count=1) strongly suggests auto-execution "
            "of malicious JavaScript upon document open. The high "
            "obfuscation count (12) indicates the attacker used hex-encoding "
            "to evade signature-based detection."
        ),
        attack_vector=(
            "**JavaScript Exploit Chain**: `/OpenAction` → `/JS` → "
            "Heap Spray → Shellcode Execution\n\n"
            "MITRE ATT&CK: T1204.002 (User Execution: Malicious File) → "
            "T1059.007 (JavaScript Execution)"
        ),
        suspicious_features=[
            ("js_count", 5.0, 8.2, "/JS tags — JavaScript action references"),
            ("openaction_count", 1.0, 4.5, "/OpenAction — auto-execute on open"),
            ("obfuscation_count", 12.0, 6.1, "Hex-encoded obfuscation patterns"),
        ],
        remediation=(
            "1. **Quarantine** — Isolate this file immediately\n"
            "2. **Do not open** in any PDF reader\n"
            "3. **Scan origin** — Check the email/download source\n"
            "4. **Alert SOC** — Escalate for further investigation\n"
            "5. **Block hash** — Add SHA-256 to blocklist"
        ),
        processing_time_ms=42.5,
        model_used="gemma4:e4b",
        feature_summary={
            "js_count": 5.0,
            "openaction_count": 1.0,
            "obfuscation_count": 12.0,
            "obj_count": 45.0,
            "stream_count": 15.0,
        },
    )

    # Markdown output
    md = report.to_markdown()
    print("\n--- Markdown Report (first 500 chars) ---")
    print(md[:500])
    print(f"\n... ({len(md)} total characters)")

    # JSON output
    js = report.to_json()
    print(f"\n--- JSON Report ({len(js)} chars) ---")
    print(js[:300])

    # Roundtrip test
    restored = ThreatReport.from_json(js)
    print(f"\n--- Roundtrip Test ---")
    print(f"Hash match: {restored.file_hash == report.file_hash}")
    print(f"Prediction match: {restored.ml_prediction == report.ml_prediction}")
    print(f"Features match: {len(restored.suspicious_features) == len(report.suspicious_features)}")

    print("=" * 60)
