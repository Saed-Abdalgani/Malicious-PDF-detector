import streamlit as st
import time
import tempfile
import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from src.features.vectorizer import pdf_to_vector, load_scaler
from src.models.mlp import MaliciousPDFClassifier
from src.config import MODELS_DIR

@dataclass
class AnalysisResult:
    prediction: str
    confidence: float
    features: Dict[str, float]
    time_ms: float
    file_hash: str

class PDFAnalyzer:
    def __init__(self):
        """Initialize the analyzer by loading the quantized model and scaler into session state."""
        if "model" not in st.session_state or "scaler" not in st.session_state:
            with st.spinner("Loading models..."):
                # Load scaler
                try:
                    st.session_state.scaler = load_scaler()
                except Exception as e:
                    st.error(f"Failed to load scaler: {e}")
                    st.session_state.scaler = None

                # Load model
                try:
                    # Try loading quantized model first if it exists
                    quantized_path = MODELS_DIR / "quantized" / "mlp_quantized_dynamic.pt"
                    model = MaliciousPDFClassifier()
                    if quantized_path.exists():
                        # Note: If quantized, need to properly load state dict
                        model.load_state_dict(torch.load(quantized_path, map_location='cpu'))
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    else:
                        # Fallback to trained FP32
                        trained_path = MODELS_DIR / "trained" / "mlp_best.pt"
                        model.load_state_dict(torch.load(trained_path, map_location='cpu'))
                    
                    model.eval()
                    st.session_state.model = model
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.session_state.model = None

    def analyze(self, uploaded_file) -> AnalysisResult:
        """Run the end-to-end inference pipeline on the uploaded file."""
        if not st.session_state.model or not st.session_state.scaler:
            raise RuntimeError("Model or scaler not loaded.")

        start_time = time.perf_counter()
        
        # Save to temp file since extractors need a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        try:
            # 1. Extract features and scale
            scaled_vector, raw_features = pdf_to_vector(tmp_path, scaler=st.session_state.scaler, return_raw=True)
            
            # 2. Inference
            with torch.no_grad():
                X_tensor = torch.tensor(scaled_vector, dtype=torch.float32).unsqueeze(0)
                logit = st.session_state.model(X_tensor)
                prob = torch.sigmoid(logit).item()
                
            # 3. Format result
            confidence = prob if prob >= 0.5 else 1 - prob
            prediction = "Malicious" if prob >= 0.5 else "Benign"
            
            # Calculate hash
            import hashlib
            file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000

            return AnalysisResult(
                prediction=prediction,
                confidence=confidence,
                features=raw_features,
                time_ms=time_ms,
                file_hash=file_hash
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_feature_breakdown(self, features: Dict[str, float]) -> pd.DataFrame:
        """Convert raw features dictionary to a formatted pandas DataFrame for display."""
        # Split into structural and metadata for cleaner display
        df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
        df['Value'] = df['Value'].apply(lambda x: f"{x:g}")
        return df
