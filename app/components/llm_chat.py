"""Optional evidence-only local LLM summary.

Only structured model/output evidence enters the prompt.  PDF bytes, extracted
content, cached reports, and guessed vulnerabilities never enter this path.
"""

from __future__ import annotations

import json

import streamlit as st

from app.components.analyzer import AnalysisResult
from src.llm.client import GemmaClient
from src.llm.prompts import EVIDENCE_ONLY_SYSTEM_PROMPT, build_evidence_only_prompt


def _evidence(analysis: AnalysisResult) -> dict:
    return {
        "outcome": analysis.outcome,
        "malicious_probability": analysis.malicious_probability,
        "locked_threshold": analysis.threshold,
        "threshold_policy": analysis.threshold_policy,
        "abstention_reasons": list(analysis.abstention_reasons),
        "raw_actionable_indicators": list(analysis.raw_indicators),
        "model_attributions": list(analysis.model_attributions),
    }


def _deterministic_summary(analysis: AnalysisResult) -> str:
    score = "not produced" if analysis.malicious_probability is None else f"{analysis.malicious_probability:.6f}"
    reasons = ", ".join(analysis.abstention_reasons) or "none"
    return (
        f"Outcome: **{analysis.outcome}**. Malicious-class probability: **{score}**; "
        f"locked threshold: **{analysis.threshold:.6f}**. Abstention reasons: **{reasons}**. "
        "Raw indicators and model attributions below are different evidence layers. "
        "This static analysis does not prove intent, exploitability, malware family, or a CVE."
    )


def render_llm_panel(analysis_result: AnalysisResult) -> None:
    st.markdown("### Evidence-grounded threat summary")
    st.markdown(_deterministic_summary(analysis_result))
    evidence = _evidence(analysis_result)
    st.markdown("#### Raw actionable indicators")
    st.json(evidence["raw_actionable_indicators"])
    st.markdown("#### Model attributions")
    st.json(evidence["model_attributions"])

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = GemmaClient()
    client = st.session_state.llm_client
    online = bool(client.check_health() and client.check_ram())
    if not online:
        st.caption("Optional local LLM unavailable; the deterministic evidence summary remains complete.")
    elif st.button("Generate optional local-language explanation"):
        try:
            response = client.generate(
                build_evidence_only_prompt(evidence),
                system_prompt=EVIDENCE_ONLY_SYSTEM_PROMPT,
                temperature=0.0,
            )
            st.markdown(response)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Optional LLM summary failed: {exc}")

    payload = json.dumps(evidence, indent=2, sort_keys=True)
    st.download_button(
        "Download structured evidence (JSON)",
        data=payload,
        file_name=f"pdf_evidence_{analysis_result.file_hash[:8]}.json",
        mime="application/json",
    )
