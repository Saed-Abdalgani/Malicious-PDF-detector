"""Streamlit display helpers for three-way deployment decisions."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_verdict(
    outcome: str,
    malicious_probability: float | None = None,
    threshold: float = 0.5,
    time_ms: float = 0.0,
) -> None:
    styles = {
        "benign": {
            "title": "VERDICT: BENIGN",
            "icon": "✅",
            "badge": "SAFE FILE",
            "bg": "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
            "border": "#22c55e",
            "title_color": "#15803d",
            "badge_bg": "#16a34a",
            "desc": "No malicious indicators detected. Structural features match safe documents.",
        },
        "malicious": {
            "title": "VERDICT: MALICIOUS",
            "icon": "🚨",
            "badge": "THREAT DETECTED",
            "bg": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)",
            "border": "#ef4444",
            "title_color": "#b91c1c",
            "badge_bg": "#dc2626",
            "desc": "High probability of malicious payload, suspicious actions, or evasive structures.",
        },
        "uncertain/abstain": {
            "title": "VERDICT: UNCERTAIN / ABSTAIN",
            "icon": "⚠️",
            "badge": "MANUAL REVIEW REQUIRED",
            "bg": "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)",
            "border": "#f59e0b",
            "title_color": "#b45309",
            "badge_bg": "#d97706",
            "desc": "Fail-closed security triggered. Parser disagreement or out-of-distribution features detected.",
        },
    }
    cfg = styles.get(outcome, styles["uncertain/abstain"])

    html = f"""
    <div style="
        background: {cfg['bg']};
        border: 2px solid {cfg['border']};
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.06);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 2.8rem; line-height: 1;">{cfg['icon']}</span>
                <div>
                    <div style="
                        font-size: 2.2rem;
                        font-weight: 800;
                        color: {cfg['title_color']} !important;
                        letter-spacing: -0.5px;
                        line-height: 1.2;
                    ">{cfg['title']}</div>
                    <div style="
                        margin-top: 4px;
                        font-size: 1rem;
                        color: #334155 !important;
                        font-weight: 500;
                    ">{cfg['desc']}</div>
                </div>
            </div>
            <div style="
                background: {cfg['badge_bg']};
                color: #ffffff !important;
                padding: 8px 22px;
                border-radius: 9999px;
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                text-transform: uppercase;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            ">
                {cfg['badge']}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_probability_gauge(
    malicious_probability: float | None,
    threshold: float,
    outcome: str,
) -> None:
    if malicious_probability is None:
        st.info("The model was not scored because a fail-closed gate requested abstention.")
        return
    color = "#10b981" if outcome == "benign" else "#ef4444" if outcome == "malicious" else "#f59e0b"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=malicious_probability * 100,
            title={"text": "Malicious-class probability"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "threshold": {
                    "line": {"color": "#0f172a", "width": 4},
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(figure, use_container_width=True)


def render_feature_radar(features: Dict[str, float], top_n: int = 8) -> None:
    top = sorted(features.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    top = [item for item in top if item[1] > 0]
    if not top:
        st.info("No non-zero structural features were extracted.")
        return
    labels = [item[0] for item in top]
    values = [item[1] for item in top]
    figure = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Observed features",
        )
    )
    figure.update_layout(showlegend=False, height=400)
    st.plotly_chart(figure, use_container_width=True)


def render_scan_history() -> None:
    history = st.session_state.get("scan_history", [])
    if not history:
        return
    rows = []
    for scan in reversed(history):
        probability = scan.get("malicious_probability")
        rows.append(
            {
                "Time": scan["timestamp"],
                "File": scan["filename"],
                "Verdict": scan["outcome"],
                "Malicious probability": "not scored" if probability is None else f"{probability * 100:.2f}%",
                "Threshold": f"{scan['threshold']:.6f}",
                "Speed": f"{scan['time_ms']:.1f} ms",
            }
        )
    with st.expander("Session History", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
