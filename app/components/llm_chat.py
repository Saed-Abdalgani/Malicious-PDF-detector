"""
llm_chat.py
-----------
AI Threat Analyst panel.

Designed to be **bulletproof in a live demo**: the panel renders a useful threat
report whether or not Ollama is running.

Resolution order when "Generate AI Threat Report" is clicked:
    1. If Ollama is online and a model fits in RAM -> live Gemma analysis,
       grounded in the model's SHAP decision drivers.
    2. Else, if a cached report matches the uploaded file's hash
       (data/sample_reports/) -> load it.
    3. Else -> run the analyzer's offline fallback (no LLM), still grounded in
       SHAP drivers and suspicious-feature analysis.

ML detection always works independently of the LLM.
"""

import json
from pathlib import Path

import streamlit as st

from src.config import FEATURE_COLUMNS, PROJECT_ROOT
from src.llm.client import GemmaClient
from src.llm.analyzer import ThreatAnalyzer
from src.llm.report_generator import ThreatReport
from app.components.analyzer import AnalysisResult

SAMPLE_REPORTS_DIR = PROJECT_ROOT / "data" / "sample_reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cached_reports() -> dict:
    """Load cached ThreatReports indexed by file hash."""
    cache = {}
    if SAMPLE_REPORTS_DIR.exists():
        for jf in SAMPLE_REPORTS_DIR.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                report = ThreatReport.from_dict(data)
                if report.file_hash:
                    cache[report.file_hash] = report
            except Exception:  # noqa: BLE001
                continue
    return cache


def _compute_shap_drivers(analysis_result: AnalysisResult):
    """Best-effort SHAP decision drivers from the analysis features (cached)."""
    key = f"drivers_{analysis_result.file_hash[:16]}"
    if key in st.session_state:
        return st.session_state[key]

    drivers = []
    try:
        import numpy as np
        from src.features.explain import explain_mlp, top_attributions
        from src.features.vectorizer import load_scaler
        from src.models.mlp import load_mlp
        from src.config import TRAINED_MODELS_DIR

        scaler = load_scaler()
        model = load_mlp(TRAINED_MODELS_DIR / "mlp_best.pt")
        vec = np.array(
            [float(analysis_result.features.get(c, 0.0)) for c in FEATURE_COLUMNS]
        ).reshape(1, -1)
        scaled = scaler.transform(vec)
        sv = explain_mlp(model, scaled, nsamples=100)
        drivers = top_attributions(sv[0], k=6)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"SHAP drivers unavailable ({exc}); using baseline analysis.")

    st.session_state[key] = drivers
    return drivers


def _build_report(analysis_result, client, online, drivers):
    """Produce a ThreatReport using the best available source."""
    # 2. cached report by hash
    cache = _load_cached_reports()
    if not online and analysis_result.file_hash in cache:
        st.info("Ollama offline — showing a cached threat report for this sample.")
        return cache[analysis_result.file_hash]

    # 1 & 3. analyzer (live if online, offline fallback otherwise)
    analyzer = ThreatAnalyzer(client)
    if online:
        client.warmup()
    return analyzer.analyze(
        features=analysis_result.features,
        prediction=analysis_result.prediction,
        confidence=analysis_result.confidence,
        ml_drivers=drivers,
    )


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def render_llm_panel(analysis_result: AnalysisResult):
    """Render the AI Threat Analyst panel for the given analysis result."""
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = GemmaClient()
    client = st.session_state.llm_client

    is_healthy = client.check_health()
    selected_model = client.check_ram() if is_healthy else None
    online = bool(is_healthy and selected_model)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Gemma 4 Threat Intelligence")

    if online:
        st.success(f"🟢 Ollama Connected (Model: {selected_model})")
    else:
        st.warning(
            "🟡 Ollama offline — running in **offline mode**. "
            "The AI report below uses cached / on-device analysis. "
            "Start Ollama (`ollama serve`) for live Gemma 4 generation."
        )

    if st.button("🔍 Generate AI Threat Report"):
        with st.spinner("Computing model decision drivers (SHAP) and analyzing..."):
            try:
                drivers = _compute_shap_drivers(analysis_result)
                report = _build_report(analysis_result, client, online, drivers)
                st.session_state.current_report = report
                st.session_state.chat_history = []
            except Exception as e:  # noqa: BLE001
                st.error(f"Analysis failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    if "current_report" in st.session_state and st.session_state.current_report:
        report = st.session_state.current_report

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Threat Report")
        st.markdown("#### 🔍 Threat Explanation")
        st.write(report.threat_explanation)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Attack Vector")
            st.warning(report.attack_vector or "Not specified")
            st.markdown("#### 🚨 Severity")
            if report.risk_severity in ("Critical", "High"):
                st.error(report.risk_severity)
            else:
                st.info(report.risk_severity)
        with col2:
            st.markdown("#### 🛡️ Remediation")
            st.success(report.remediation or "No action required.")

        # SHAP decision drivers (grounding)
        if report.suspicious_features:
            st.markdown("#### 🧠 Model Decision Drivers & Indicators")
            for item in report.suspicious_features[:6]:
                if len(item) >= 4:
                    name, value, dev, _desc = item[:4]
                    st.caption(f"• {name} = {value:g} ({dev:+.1f}σ from benign)")

        st.markdown("</div>", unsafe_allow_html=True)

        # Follow-up chat — only when the LLM is online
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask Follow-up Questions")
        if not online:
            st.caption("Interactive Q&A requires Ollama. Start it to enable live chat.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask Gemma a question about this analysis..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full = ""
                    try:
                        context = (
                            f"Context: This is a {analysis_result.prediction} PDF. "
                            f"Analysis: {report.threat_explanation}\n\n"
                            f"User Question: {prompt}"
                        )
                        for token in client.generate_stream(context):
                            full += token
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": full}
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Chat error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        render_report_download(report)


def render_report_download(report):
    """Render buttons to download the threat report."""
    st.markdown('<div class="glass-card" style="display: flex; gap: 10px;">', unsafe_allow_html=True)

    st.download_button(
        label="📄 Download Markdown Report",
        data=report.to_markdown(),
        file_name=f"threat_report_{report.file_hash[:8]}.md",
        mime="text/markdown",
    )
    st.download_button(
        label="📦 Download JSON Report",
        data=report.to_json(),
        file_name=f"threat_report_{report.file_hash[:8]}.json",
        mime="application/json",
    )
    st.markdown("</div>", unsafe_allow_html=True)
