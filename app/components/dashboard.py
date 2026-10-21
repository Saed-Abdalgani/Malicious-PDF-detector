"""Streamlit display helpers for three-way deployment decisions."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_verdict(
    outcome: str,
    malicious_probability: float | None,
    threshold: float,
    time_ms: float,
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
            "prob_color": "#16a34a",
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
            "prob_color": "#dc2626",
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
            "prob_color": "#d97706",
        },
    }
    cfg = styles.get(outcome, styles["uncertain/abstain"])
    score_str = "Not Scored" if malicious_probability is None else f"{malicious_probability * 100:.2f}%"

    html = f"""
    <div style="
        background: {cfg['bg']};
        border: 2px solid {cfg['border']};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 2.6rem; line-height: 1;">{cfg['icon']}</span>
                <div>
                    <div style="
                        font-size: 2.1rem;
                        font-weight: 800;
                        color: {cfg['title_color']} !important;
                        letter-spacing: -0.5px;
                        line-height: 1.2;
                    ">{cfg['title']}</div>
                    <div style="
                        margin-top: 4px;
                        font-size: 0.95rem;
                        color: #334155 !important;
                        font-weight: 500;
                    ">{cfg['desc']}</div>
                </div>
            </div>
            <div style="
                background: {cfg['badge_bg']};
                color: #ffffff !important;
                padding: 6px 18px;
                border-radius: 9999px;
                font-size: 0.85rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                text-transform: uppercase;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            ">
                {cfg['badge']}
            </div>
        </div>
        
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 18px;
            padding-top: 16px;
            border-top: 1px solid rgba(0, 0, 0, 0.08);
        ">
            <div style="
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 14px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b !important; margin-bottom: 4px;">
                    Malicious Probability
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {cfg['prob_color']} !important;">
                    {score_str}
                </div>
            </div>
            <div style="
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 14px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b !important; margin-bottom: 4px;">
                    Decision Threshold
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a !important;">
                    {threshold:.4f}
                </div>
            </div>
            <div style="
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 14px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b !important; margin-bottom: 4px;">
                    Analysis Speed
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a !important;">
                    {time_ms:.1f} <span style="font-size: 1rem; font-weight: 600; color: #64748b !important;">ms</span>
                </div>
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
    st.markdown("### Session history")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
