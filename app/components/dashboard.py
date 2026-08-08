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
        "benign": ("verdict-safe", "✅", "BENIGN"),
        "malicious": ("verdict-malicious", "⚠", "MALICIOUS"),
        "uncertain/abstain": ("", "⏸", "UNCERTAIN / ABSTAIN"),
    }
    css_class, icon, title = styles[outcome]
    score = "not scored" if malicious_probability is None else f"{malicious_probability * 100:.2f}%"
    st.markdown(
        f"""
        <div class="glass-card {css_class}">
          <h2 style="margin-top:0;">{icon} VERDICT: {title}</h2>
          <p><b>Malicious-class probability:</b> {score}</p>
          <p><b>Locked decision threshold:</b> {threshold:.6f}</p>
          <p><b>Processing time:</b> {time_ms:.1f} ms</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
                    "line": {"color": "white", "width": 4},
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
