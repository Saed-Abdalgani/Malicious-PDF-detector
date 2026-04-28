import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict

def render_verdict(prediction: str, confidence: float, time_ms: float):
    """Render the animated verdict card."""
    is_safe = prediction == "Benign"
    css_class = "verdict-safe" if is_safe else "verdict-malicious"
    icon = "✅" if is_safe else "⚠"
    title = "SAFE" if is_safe else "MALICIOUS"
    
    st.markdown(f'''
    <div class="glass-card {css_class}">
        <h2 style="margin-top: 0;">{icon} VERDICT: {title}</h2>
        <div style="display: flex; justify-content: space-between; margin-top: 20px;">
            <div>
                <p class="metric-label">Confidence</p>
                <p class="metric-value">{(confidence * 100):.1f}%</p>
            </div>
            <div>
                <p class="metric-label">Processing Time</p>
                <p class="metric-value">{time_ms:.1f} <span style="font-size: 1.2rem;">ms</span></p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def render_confidence_gauge(confidence: float, prediction: str):
    """Render a Plotly gauge chart for confidence."""
    color = "#10b981" if prediction == "Benign" else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'color': '#f8fafc', 'size': 18}},
        number = {'suffix': "%", 'font': {'color': '#f8fafc', 'size': 36}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.05)"},
                {'range': [50, 100], 'color': "rgba(255,255,255,0.1)"}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f8fafc", 'family': "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_feature_radar(features: Dict[str, float], top_n: int = 8):
    """Render a Plotly radar chart of the most prominent features."""
    # Sort features by absolute value to find most prominent ones
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    # Filter out zero values to avoid cluttered radar
    top_features = [f for f in top_features if f[1] > 0]
    
    if not top_features:
        st.info("No prominent structural features detected.")
        return

    labels = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='File Features',
        line_color='#3b82f6',
        fillcolor='rgba(59, 130, 246, 0.4)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.2]),
            angularaxis=dict(tickfont=dict(color='#f8fafc', size=12))
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#f8fafc", 'family': "Inter"},
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_scan_history():
    """Render a table of all scans in the current session."""
    if 'scan_history' not in st.session_state or not st.session_state.scan_history:
        return
        
    st.markdown("### 🕒 Session History")
    
    # Format history for dataframe
    history_data = []
    for scan in reversed(st.session_state.scan_history):
        history_data.append({
            "Time": scan["timestamp"],
            "File": scan["filename"],
            "Verdict": scan["prediction"],
            "Confidence": f"{scan['confidence']*100:.1f}%",
            "Speed": f"{scan['time_ms']:.1f}ms"
        })
        
    df = pd.DataFrame(history_data)
    
    # Custom styling for dataframe
    st.dataframe(
        df,
        column_config={
            "Verdict": st.column_config.TextColumn(
                "Verdict",
                help="Malicious or Benign",
            ),
        },
        hide_index=True,
        use_container_width=True
    )
