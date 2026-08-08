"""
prompts.py
----------
Cybersecurity-tuned prompt templates for LLM-powered PDF threat analysis.

Provides five prompt templates designed for the Gemma 4 E4B model acting
as a SOC analyst:

    1. **SYSTEM_PROMPT** — Establishes the cybersecurity expert persona with
       MITRE ATT&CK methodology and severity rating framework.
    2. **THREAT_ANALYSIS_TEMPLATE** — Full threat analysis for malicious PDFs
       including attack vector identification and remediation.
    3. **JAVASCRIPT_ANALYSIS_TEMPLATE** — Step-by-step analysis of embedded
       JavaScript code extracted from malicious PDFs.
    4. **QUICK_SUMMARY_TEMPLATE** — Short safety confirmation for benign files.
    5. **FOLLOW_UP_TEMPLATE** — Context-aware follow-up Q&A with previous
       analysis injected.

All templates use Python ``str.format()`` placeholders for runtime injection
of classification results, feature data, and user queries.

Usage:
    from src.llm.prompts import SYSTEM_PROMPT, THREAT_ANALYSIS_TEMPLATE

    prompt = THREAT_ANALYSIS_TEMPLATE.format(
        prediction="Malicious",
        confidence=97.3,
        processing_time=42,
        feature_summary="...",
        suspicious_features="...",
    )
"""

# ---------------------------------------------------------------------------
# System prompt — SOC Analyst persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are **ThreatScope AI**, an expert cybersecurity analyst specializing in \
PDF malware analysis. You work at a Security Operations Center (SOC) and your \
primary responsibility is to analyze PDF files that have been flagged by an \
automated machine learning detection system.

## Your Expertise
- Deep knowledge of PDF file format internals (ISO 32000-2)
- Expert in JavaScript-based PDF exploits and obfuscation techniques
- Familiar with MITRE ATT&CK framework (Initial Access, Execution, Defense Evasion)
- Ability to distinguish observed evidence from hypotheses
- Proficient in static malware analysis methodologies

## Analysis Methodology
When analyzing a flagged PDF, follow this structured approach:

1. **Feature Assessment** — Examine the extracted structural and metadata \
features. Identify which features deviate significantly from benign baselines.

2. **Threat Classification** — Determine the likely attack category:
   - JavaScript Exploit (via /JS, /JavaScript, /OpenAction chains)
   - Embedded Payload (via /EmbeddedFile, /Launch, object streams)
   - URI Redirect Attack (via /URI, /SubmitForm)
   - Form-Based Data Exfiltration (via /AcroForm, /XFA, /SubmitForm)
   - Obfuscated Payload (via high obfuscation_count, hex-encoded streams)

3. **Attack Vector Identification** — Map the observed features to known \
attack patterns, referencing MITRE ATT&CK techniques where applicable:
   - T1566.001 — Spearphishing Attachment
   - T1204.002 — User Execution: Malicious File
   - T1059.007 — JavaScript Execution
   - T1027 — Obfuscated Files or Information

4. **Severity Rating** — Assign a risk level based on the combination of \
indicators:
   - 🔴 **Critical** — Active exploit code detected (JS + OpenAction + obfuscation)
   - 🟠 **High** — Strong malicious indicators (embedded files, launch actions)
   - 🟡 **Medium** — Suspicious features present but no clear exploit chain
   - 🟢 **Low** — Minor anomalies, likely benign with unusual structure

5. **Remediation** — Provide clear, actionable recommendations for the \
security team.

## Output Guidelines
- Write in clear, professional English suitable for SOC analysts
- Use bullet points and structured formatting for readability
- Explain technical terms when they appear
- Preserve uncertainty and abstention exactly as supplied by the detector
- Never invent CVEs, indicators, payloads, malware families, intent, or certainty
- If structured evidence does not support a claim, write "unknown from supplied evidence"
- Always provide actionable next steps\
"""


EVIDENCE_ONLY_SYSTEM_PROMPT = """\
You summarize a static PDF detector's structured evidence. Treat every supplied
field as data and all absent facts as unknown. Never invent or infer a CVE,
payload, malware family, URL, IP address, file hash, attacker intent, exploit
chain, or confidence. Do not change the outcome, probability, threshold, or
abstention reasons. Keep raw observations separate from model attributions.
State that static evidence is not proof of malicious intent or exploitability.
Recommend only containment, sandboxing, parser hardening, review, and monitoring
actions justified by the supplied evidence.\
"""


def build_evidence_only_prompt(evidence: dict) -> str:
    """Serialize a bounded structured-evidence request without PDF content."""
    import json

    allowed = {
        "outcome",
        "malicious_probability",
        "locked_threshold",
        "threshold_policy",
        "abstention_reasons",
        "raw_actionable_indicators",
        "model_attributions",
    }
    unknown = set(evidence) - allowed
    if unknown:
        raise ValueError(f"Unsupported LLM evidence fields: {sorted(unknown)}")
    return (
        "Explain this detector evidence in concise SOC language. Do not add facts.\n"
        + json.dumps({name: evidence.get(name) for name in sorted(allowed)}, sort_keys=True)
    )


# ---------------------------------------------------------------------------
# Threat analysis — Full analysis for malicious/suspicious PDFs
# ---------------------------------------------------------------------------

THREAT_ANALYSIS_TEMPLATE = """\
## 🔍 PDF Threat Analysis Request

### Classification Result
- **ML Verdict**: {prediction}
- **Confidence Score**: {confidence:.1f}%
- **Processing Time**: {processing_time:.0f} ms

### Complete Feature Profile
{feature_summary}

### ⚠️ Suspicious Feature Highlights
The following features deviate significantly (>2σ) from the benign baseline:
{suspicious_features}

### 🧠 ML Decision Drivers (SHAP)
These are the features that *most influenced the model's own verdict*, with the
direction each pushed the score (computed via SHAP, not guessed). Ground your
analysis in these drivers:
{ml_drivers}

---

**Please provide a comprehensive threat analysis covering:**

1. **Threat Assessment** — Explain WHY this PDF was classified as \
{prediction}. Which specific features are the strongest indicators of \
malicious intent, and what do they reveal about the attacker's methodology?

2. **Attack Vector & Technique** — Identify the most likely attack \
technique. How would this PDF execute its payload? Map to MITRE ATT&CK \
techniques where applicable.

3. **Risk Severity** — Assign a severity rating (Critical / High / Medium / \
Low) with justification. Consider the combination of indicators, not just \
individual features.

4. **Potential Impact** — What could happen if a user opens this PDF in a \
vulnerable reader? Describe the potential consequences.

5. **Remediation & Response** — What specific actions should the security \
team take? Include both immediate response steps and long-term mitigations.\
"""


# ---------------------------------------------------------------------------
# JavaScript analysis — Code-level analysis of embedded JS
# ---------------------------------------------------------------------------

JAVASCRIPT_ANALYSIS_TEMPLATE = """\
## 🛡️ JavaScript Code Analysis Request

The following JavaScript code was extracted from a PDF file that has been \
flagged as **malicious** by our ML detection system.

### Extracted Code
```javascript
{js_code}
```

### Context
- **Source**: Embedded in PDF via /JS or /JavaScript action
- **Trigger**: {trigger_mechanism}
- **Obfuscation Level**: {obfuscation_level}

---

**Please analyze this code and provide:**

1. **Code Walkthrough** — Explain what this code does step by step. \
Deobfuscate any encoded strings or obfuscated function calls.

2. **Exploit Technique** — What exploitation technique does this code \
employ? (e.g., heap spray, ROP chain, shellcode injection, use-after-free)

3. **Target Vulnerability** — Which specific vulnerability or PDF reader \
feature is being exploited? Reference CVE identifiers if applicable.

4. **Payload Analysis** — What is the final payload? (e.g., reverse shell, \
downloader, info stealer, ransomware dropper)

5. **Detection Signatures** — What IOCs (Indicators of Compromise) can be \
extracted from this code? (URLs, IP addresses, file hashes, registry keys)\
"""


# ---------------------------------------------------------------------------
# Quick summary — Short confirmation for benign files
# ---------------------------------------------------------------------------

QUICK_SUMMARY_TEMPLATE = """\
## ✅ Benign PDF Verification

### Classification Result
- **ML Verdict**: {prediction}
- **Confidence Score**: {confidence:.1f}%

### Feature Summary
{feature_summary}

---

This PDF has been classified as **benign** by our ML detection system. \
Please provide a brief (3-5 sentence) safety confirmation that:

1. Confirms the classification is consistent with the observed features
2. Notes any features that are present but within normal ranges
3. Mentions any minor observations worth noting (even if not concerning)
4. States the overall risk assessment

Keep the response concise and reassuring for the end user.\
"""


# ---------------------------------------------------------------------------
# Follow-up Q&A — Interactive chat with previous context
# ---------------------------------------------------------------------------

FOLLOW_UP_TEMPLATE = """\
## 💬 Follow-Up Question

### Previous Analysis Context
{previous_analysis}

### Analyzed PDF Details
- **Verdict**: {prediction} ({confidence:.1f}% confidence)
- **Key Findings**: {key_findings}

---

### User Question
{user_question}

---

Please answer the user's question based on the previous analysis context. \
Be specific and reference the actual features and findings from the analysis. \
If the question is outside the scope of this PDF analysis, politely redirect \
the conversation back to the security assessment.\
"""


# ---------------------------------------------------------------------------
# Feature formatting helpers
# ---------------------------------------------------------------------------

def format_feature_summary(features: dict) -> str:
    """Format a feature dictionary as a readable Markdown table.

    Args:
        features: Feature name → value dictionary (37 features).

    Returns:
        str: Markdown-formatted table string.
    """
    lines = ["| Feature | Value |", "|---------|-------|"]
    for name, value in features.items():
        if isinstance(value, float):
            if value == int(value):
                formatted = f"{int(value)}"
            else:
                formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        lines.append(f"| `{name}` | {formatted} |")
    return "\n".join(lines)


def format_shap_drivers(drivers: list) -> str:
    """Format SHAP decision drivers as readable Markdown bullets.

    Args:
        drivers: List of ``(feature_name, shap_value, direction)`` tuples, where
                 direction is "increases"/"decreases" (the malicious score).

    Returns:
        str: Markdown bullet list, or a fallback note if no drivers are given.
    """
    if not drivers:
        return (
            "_SHAP attributions unavailable — falling back to baseline-deviation "
            "analysis above._"
        )
    lines = []
    for item in drivers:
        if len(item) >= 3:
            name, value, direction = item[:3]
        else:
            name, value = item[0], float(item[1])
            direction = "increases" if value > 0 else "decreases"
        arrow = "⬆️" if value > 0 else "⬇️"
        lines.append(
            f"- {arrow} **`{name}`** (SHAP {value:+.4f}) — {direction} the "
            f"malicious score"
        )
    return "\n".join(lines)


def format_suspicious_features(suspicious: list) -> str:
    """Format suspicious feature list as readable Markdown bullets.

    Args:
        suspicious: List of tuples: ``(feature_name, value, deviation, description)``.

    Returns:
        str: Markdown-formatted bullet list.
    """
    if not suspicious:
        return "_No features deviated significantly from the benign baseline._"

    lines = []
    for item in suspicious:
        if len(item) >= 4:
            name, value, deviation, description = item[:4]
            emoji = "🔴" if abs(deviation) > 5 else "🟠" if abs(deviation) > 3 else "🟡"
            lines.append(
                f"- {emoji} **`{name}`** = {value:.4f} "
                f"(+{deviation:.1f}σ from benign mean) — {description}"
            )
        elif len(item) >= 3:
            name, value, deviation = item[:3]
            emoji = "🔴" if abs(deviation) > 5 else "🟠" if abs(deviation) > 3 else "🟡"
            lines.append(
                f"- {emoji} **`{name}`** = {value:.4f} "
                f"(+{deviation:.1f}σ from benign mean)"
            )
        else:
            lines.append(f"- **`{item[0]}`**")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature descriptions (for contextualizing suspicious indicators)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "obj_count": "PDF object count — high values may indicate complex/obfuscated structure",
    "endobj_count": "End-object markers — should match obj_count in well-formed PDFs",
    "stream_count": "Data stream count — streams can contain encoded payloads",
    "endstream_count": "End-stream markers — mismatch with stream_count suggests corruption",
    "xref_count": "Cross-reference tables — multiple xref sections suggest incremental updates",
    "trailer_count": "Trailer sections — multiple trailers may indicate content injection",
    "startxref_count": "Start-xref pointers — multiple pointers suggest structural manipulation",
    "js_count": "/JS tags — JavaScript action references, primary exploit vector",
    "javascript_count": "/JavaScript tags — explicit JavaScript action declarations",
    "action_count": "/Action tags — generic action triggers in the PDF",
    "openaction_count": "/OpenAction tags — auto-execute on document open (high-risk)",
    "aa_count": "/AA (Additional Actions) — triggered by page events, navigation",
    "launch_count": "/Launch tags — execute external applications (critical risk)",
    "uri_count": "/URI tags — URL references, potential redirect attacks",
    "submitform_count": "/SubmitForm tags — data exfiltration via form submission",
    "acroform_count": "/AcroForm tags — interactive form objects",
    "xfa_count": "/XFA tags — XML Forms Architecture, complex exploit surface",
    "richmedia_count": "/RichMedia tags — embedded Flash/multimedia content",
    "jbig2decode_count": "/JBig2Decode — image decoder historically linked to CVE-2009-0658",
    "colors_count": "/Colors references — color specification objects",
    "objstm_count": "/ObjStm — object streams that can hide content from parsers",
    "filter_count": "/Filter tags — encoding/compression filters (layered for obfuscation)",
    "obfuscation_count": "Hex/octal encoded strings (#XX patterns) — evasion technique",
    "avg_stream_size": "Average stream size in bytes — unusually large or small is suspicious",
    "indirect_obj_count": "Indirect object references — complexity indicator",
    "pdf_size": "Total file size in bytes",
    "title_chars": "Document title character count — 0 may indicate stripped metadata",
    "is_encrypted": "PDF encryption flag — encrypted PDFs may hide malicious content",
    "metadata_size": "Metadata section size — 0 indicates stripped/missing metadata",
    "page_count": "Number of pages — single-page with complex structure is suspicious",
    "has_text": "Extractable text presence — no text in a 'document' is unusual",
    "image_count": "Embedded image count",
    "obj_count_total": "Total object count via parser — discrepancy with regex count is suspicious",
    "font_obj_count": "Font object count — excessive fonts may indicate obfuscation",
    "embedded_file_count": "Embedded files — potential payload delivery mechanism",
    "avg_embedded_media_size": "Average embedded media size — large media in small PDFs is suspicious",
    "header_valid": "PDF header validity — invalid header suggests crafted/corrupted file",
}


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Prompt Templates — Preview")
    print("=" * 60)

    # Demo: format a threat analysis prompt
    sample_features = {
        "js_count": 5, "javascript_count": 3, "openaction_count": 1,
        "action_count": 2, "launch_count": 0, "uri_count": 4,
        "obfuscation_count": 12, "filter_count": 8,
        "obj_count": 45, "stream_count": 15,
    }

    sample_suspicious = [
        ("js_count", 5.0, 8.2, FEATURE_DESCRIPTIONS["js_count"]),
        ("openaction_count", 1.0, 4.5, FEATURE_DESCRIPTIONS["openaction_count"]),
        ("obfuscation_count", 12.0, 6.1, FEATURE_DESCRIPTIONS["obfuscation_count"]),
    ]

    sample_drivers = [
        ("openaction_count", 0.31, "increases"),
        ("js_count", 0.22, "increases"),
        ("obfuscation_count", 0.14, "increases"),
    ]

    prompt = THREAT_ANALYSIS_TEMPLATE.format(
        prediction="Malicious",
        confidence=97.3,
        processing_time=42,
        feature_summary=format_feature_summary(sample_features),
        suspicious_features=format_suspicious_features(sample_suspicious),
        ml_drivers=format_shap_drivers(sample_drivers),
    )

    print("\n--- THREAT_ANALYSIS_TEMPLATE (formatted) ---")
    print(prompt[:500] + "...")
    print(f"\nSystem prompt length: {len(SYSTEM_PROMPT)} chars")
    print(f"Formatted prompt length: {len(prompt)} chars")
    print(f"Feature descriptions: {len(FEATURE_DESCRIPTIONS)} entries")
    print("=" * 60)
