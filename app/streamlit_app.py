"""Streamlit UI for the verified Phase 8 deployment bundle."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.analyzer import PDFAnalyzer
from app.components.dashboard import (
    render_feature_radar,
    render_probability_gauge,
    render_scan_history,
    render_verdict,
)
from app.components.llm_chat import render_llm_panel
from app.components.uploader import render_upload_zone
from src.config import PROJECT_ROOT


def _load_css() -> None:
    path = PROJECT_ROOT / "app" / "assets" / "style.css"
    if path.is_file():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _scanner() -> None:
    st.title("Intelligent PDF Malware Detection")
    st.caption(
        "One checksummed bundle performs feature engineering, calibrated scoring, "
        "thresholding, OOD checks, abstention, and local explanation."
    )
    try:
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = PDFAnalyzer()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Deployment bundle unavailable or incompatible. The application has "
            f"failed closed and will not load a legacy model: {exc}"
        )
        return

    uploaded_file = render_upload_zone()
    if uploaded_file is not None:
        with st.spinner("Analyzing PDF structure locally..."):
            try:
                result = st.session_state.analyzer.analyze(uploaded_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed closed: {exc}")
            else:
                st.session_state.last_result = result
                st.session_state.scan_history.append(
                    {
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "filename": Path(uploaded_file.name).name,
                        "outcome": result.outcome,
                        "malicious_probability": result.malicious_probability,
                        "threshold": result.threshold,
                        "time_ms": result.time_ms,
                    }
                )
                left, right = st.columns(2)
                with left:
                    render_verdict(
                        result.outcome,
                        result.malicious_probability,
                        result.threshold,
                        result.time_ms,
                    )
                    render_probability_gauge(
                        result.malicious_probability, result.threshold, result.outcome
                    )
                    if result.abstention_reasons:
                        st.warning("Abstention reasons: " + ", ".join(result.abstention_reasons))
                with right:
                    st.markdown("### Structural feature overview")
                    render_feature_radar(result.features)

                st.markdown("### Raw actionable indicators")
                st.caption("Direct parser observations; these are not model attributions or proof of intent.")
                st.dataframe(pd.DataFrame(result.raw_indicators), hide_index=True, use_container_width=True)
                st.markdown("### Local model attributions")
                st.caption(
                    "Probability deltas from replacing one feature with its real train-only "
                    "reference median; these are model-behavior evidence, not raw indicators."
                )
                st.dataframe(pd.DataFrame(result.model_attributions), hide_index=True, use_container_width=True)
                with st.expander("Complete extracted feature breakdown"):
                    st.dataframe(
                        st.session_state.analyzer.get_feature_breakdown(result.features),
                        hide_index=True,
                        use_container_width=True,
                    )
    render_scan_history()


def _model_dashboard() -> None:
    st.title("Model Performance Dashboard")
    summary_path = PROJECT_ROOT / "reports" / "results" / "experiment_summary.json"
    if not summary_path.is_file():
        st.warning("No active experiment summary exists.")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    st.json(
        {
            "experiment_id": summary.get("experiment", {}).get("experiment_id"),
            "status": summary.get("status"),
            "data_gate_passed": summary.get("data_gate_passed"),
        }
    )
    metrics = summary.get("final_metrics")
    if metrics is None:
        st.warning(
            "No verified final metrics exist. Historical manual measurements are "
            "intentionally excluded from this operational dashboard."
        )
    else:
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)


def _feature_explorer() -> None:
    st.title("Feature Explorer")
    path = PROJECT_ROOT / "reports" / "data" / "feature_dictionary_v2.json"
    if not path.is_file():
        st.warning("Run the gated feature stage to produce the schema catalog.")
        return
    catalog = json.loads(path.read_text(encoding="utf-8"))
    st.markdown("### Base schema")
    st.dataframe(pd.DataFrame(catalog.get("base_schema", {}).get("features", [])), use_container_width=True)
    st.markdown("### Engineered features")
    st.dataframe(pd.DataFrame(catalog.get("engineered_features", [])), use_container_width=True)


def _about() -> None:
    st.title("About Malicious PDF Detector")
    st.write(
        "A local static-analysis application using a versioned feature pipeline, "
        "calibrated model ensemble, validation-selected threshold, explicit "
        "abstention, bounded extraction, and evidence-layered explanation."
    )
    st.info(
        "Performance numbers are never hard-coded here. Verified results appear "
        "only when the active experiment summary contains sealed final metrics."
    )


def main() -> None:
    st.set_page_config(
        page_title="Malicious PDF Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()
    st.session_state.setdefault("scan_history", [])
    st.session_state.setdefault("last_result", None)
    st.sidebar.title("SecurePDF Shield")
    page = st.sidebar.radio(
        "Navigation",
        ("PDF Scanner", "AI Threat Analyst", "Model Dashboard", "Feature Explorer", "About"),
    )
    if page == "PDF Scanner":
        _scanner()
    elif page == "AI Threat Analyst":
        st.title("AI Threat Analyst")
        if st.session_state.last_result is None:
            st.info("Scan a PDF first. The optional LLM never receives uploaded PDF bytes.")
        else:
            render_llm_panel(st.session_state.last_result)
    elif page == "Model Dashboard":
        _model_dashboard()
    elif page == "Feature Explorer":
        _feature_explorer()
    else:
        _about()


if __name__ == "__main__":
    main()
