import streamlit as st
import pandas as pd
import datetime
from pathlib import Path

# Must be the first Streamlit command
st.set_page_config(
    page_title="Malicious PDF Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.components.uploader import render_upload_zone
from app.components.analyzer import PDFAnalyzer
from app.components.dashboard import render_verdict, render_confidence_gauge, render_feature_radar, render_scan_history
from app.components.llm_chat import render_llm_panel
from src.config import PROJECT_ROOT

# Load Custom CSS
def load_css():
    css_path = Path("app/assets/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
load_css()

# Initialize session state
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = PDFAnalyzer()

# Sidebar Navigation
st.sidebar.title("🛡️ SecurePDF Shield")
page = st.sidebar.radio("Navigation", [
    "🔍 PDF Scanner",
    "🤖 AI Threat Analyst",
    "📊 Model Dashboard",
    "📈 Feature Explorer",
    "ℹ️ About"
])

# Page 1: PDF Scanner
if page == "🔍 PDF Scanner":
    st.markdown("<h1>Intelligent PDF Malware Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 1.1rem;'>Upload a PDF to instantly analyze its structure and metadata for malicious patterns using our quantized ML model.</p>", unsafe_allow_html=True)
    
    uploaded_file = render_upload_zone()
    
    if uploaded_file:
        with st.spinner("Analyzing PDF structure..."):
            try:
                result = st.session_state.analyzer.analyze(uploaded_file)
                st.session_state.last_result = result
                
                # Add to history
                st.session_state.scan_history.append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "filename": uploaded_file.name,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "time_ms": result.time_ms
                })
                
                # Dashboard layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    render_verdict(result.prediction, result.confidence, result.time_ms)
                    render_confidence_gauge(result.confidence, result.prediction)
                    
                with col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### 🧬 Structural DNA")
                    render_feature_radar(result.features)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # Feature Breakdown Expander
                with st.expander("View Complete Feature Breakdown"):
                    df_features = st.session_state.analyzer.get_feature_breakdown(result.features)
                    st.dataframe(df_features, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    # Show History
    if st.session_state.scan_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        render_scan_history()

# Page 2: AI Threat Analyst
elif page == "🤖 AI Threat Analyst":
    st.markdown("<h1>AI Threat Analyst</h1>", unsafe_allow_html=True)
    
    if st.session_state.last_result is None:
        st.info("Please scan a PDF in the 'PDF Scanner' page first to generate a threat analysis.")
    else:
        render_llm_panel(st.session_state.last_result)

# Page 3: Model Dashboard
elif page == "📊 Model Dashboard":
    st.markdown("<h1>Model Performance Dashboard</h1>", unsafe_allow_html=True)
    st.info("This dashboard displays the performance metrics of the models trained during Phase 4.")
    
    try:
        report_path = PROJECT_ROOT / "reports" / "results" / "model_comparison.csv"
        if report_path.exists():
            df = pd.read_csv(report_path)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Model Comparison")
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Comparison report not found at {report_path}")
            
    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")

# Page 4: Feature Explorer
elif page == "📈 Feature Explorer":
    st.markdown("<h1>Feature Explorer</h1>", unsafe_allow_html=True)
    st.info("Feature importance charts based on the Random Forest baseline model.")
    
    try:
        importance_path = PROJECT_ROOT / "reports" / "results" / "feature_importance.csv"
        if importance_path.exists():
            df = pd.read_csv(importance_path)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            import plotly.express as px
            fig = px.bar(df, x='Importance', y='Feature', orientation='h', title='Top Discriminative Features')
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#f8fafc", 'family': "Inter"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Feature importance data not found. Run model training phase to generate it.")
    except Exception as e:
        st.error(f"Error: {e}")

# Page 5: About
elif page == "ℹ️ About":
    st.markdown("<h1>About Malicious PDF Detector</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    <h3>Project Overview</h3>
    <p>This application is a lightweight, high-performance detector for malicious PDF files. 
    It extracts 37 structural and metadata features from PDFs and uses a quantized PyTorch MLP 
    model to classify them as Malicious or Benign in real-time on CPU.</p>
    
    <h3>Technology Stack</h3>
    <ul>
        <li><b>Frontend</b>: Streamlit with Custom CSS (Glassmorphism)</li>
        <li><b>ML Backend</b>: PyTorch (INT8 Quantized), Scikit-Learn</li>
        <li><b>LLM Intelligence</b>: Gemma 4 E4B via Ollama</li>
        <li><b>Feature Extraction</b>: pdfminer.six, PyPDF2</li>
    </ul>
    
    <h3>Performance Highlights</h3>
    <ul>
        <li>Inference Time: ~0.5ms (Quantized)</li>
        <li>Memory Footprint: <5MB (Model)</li>
        <li>Detection Accuracy: >95%</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

