import streamlit as st
from src.llm.client import GemmaClient
from src.llm.analyzer import ThreatAnalyzer
from app.components.analyzer import AnalysisResult

def render_llm_panel(analysis_result: AnalysisResult):
    """Render the AI Threat Analyst panel for the given analysis result."""
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = GemmaClient()
        
    client = st.session_state.llm_client
    
    # Check health and memory status
    is_healthy = client.check_health()
    selected_model = client.check_ram()
    
    # Status indicator
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Gemma 4 E4B Threat Intelligence")
    
    if not is_healthy:
        st.error("🔴 Ollama is offline. Please start Ollama to use AI features.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    if selected_model is None:
        st.error("🔴 Insufficient RAM for local LLM inference. Required: ≥3GB free.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    st.success(f"🟢 Ollama Connected (Model: {selected_model})")
    
    # Generate Report Button
    if st.button("🔍 Generate AI Threat Report"):
        with st.spinner("Analyzing threat indicators with Gemma 4..."):
            try:
                # Warmup
                client.warmup()
                
                # Setup analyzer
                analyzer = ThreatAnalyzer(client)
                report = analyzer.analyze(
                    features=analysis_result.features,
                    prediction=analysis_result.prediction,
                    confidence=analysis_result.confidence
                )
                
                # Save to session state
                st.session_state.current_report = report
                st.session_state.chat_history = [] # Reset chat
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Report if available
    if 'current_report' in st.session_state and st.session_state.current_report:
        report = st.session_state.current_report
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Threat Report")
        
        # Display the explanation
        st.markdown("#### 🔍 Threat Explanation")
        st.write(report.threat_explanation)
        
        # Display attack vector and remediation
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Attack Vector")
            st.warning(report.attack_vector)
            st.markdown("#### 🚨 Severity")
            
            if "Critical" in report.risk_severity or "High" in report.risk_severity:
                st.error(report.risk_severity)
            else:
                st.info(report.risk_severity)
                
        with col2:
            st.markdown("#### 🛡️ Remediation")
            st.success(report.remediation)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat interface for follow-ups
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask Follow-up Questions")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask Gemma a question about this analysis..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Construct context
                    context = f"Context: This is a {analysis_result.prediction} PDF. Analysis: {report.threat_explanation}\n\nUser Question: {prompt}"
                    
                    for token in client.generate_stream(context):
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                        
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Chat error: {e}")
                    
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download buttons
        render_report_download(report)

def render_report_download(report):
    """Render buttons to download the threat report."""
    st.markdown('<div class="glass-card" style="display: flex; gap: 10px;">', unsafe_allow_html=True)
    
    # MD Download
    md_content = report.to_markdown()
    st.download_button(
        label="📄 Download Markdown Report",
        data=md_content,
        file_name=f"threat_report_{report.file_hash[:8]}.md",
        mime="text/markdown"
    )
    
    # JSON Download
    json_content = report.to_json()
    st.download_button(
        label="📦 Download JSON Report",
        data=json_content,
        file_name=f"threat_report_{report.file_hash[:8]}.json",
        mime="application/json"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
