import streamlit as st
import time
import tempfile
import os
import torch
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from src.artifacts import verify_deployable_model_artifact
from src.experiment import create_experiment_identity, load_experiment_config
from src.features.vectorizer import (
    FEATURE_PIPELINE_PATH,
    load_feature_pipeline,
    pdf_to_pipeline_vector,
)
from src.models.mlp import MaliciousPDFClassifier
from src.config import DATA_REPORTS_DIR, MODELS_DIR, SPLITS_DIR, SPLIT_SCHEMA_VERSION

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
        if "model" not in st.session_state or "feature_pipeline" not in st.session_state:
            with st.spinner("Loading models..."):
                active_identity = create_experiment_identity(load_experiment_config())
                # Load the same serialized feature pipeline used during training.
                try:
                    st.session_state.feature_pipeline = load_feature_pipeline(
                        FEATURE_PIPELINE_PATH
                    )
                except Exception as e:
                    st.error(f"Failed to load feature pipeline: {e}")
                    st.session_state.feature_pipeline = None

                # Load model
                try:
                    # Try loading quantized model first if it exists
                    quantized_path = MODELS_DIR / "quantized" / "mlp_quantized_dynamic.pt"
                    trained_path = MODELS_DIR / "trained" / "mlp_best.pt"
                    model_path = quantized_path if quantized_path.exists() else trained_path
                    model_metadata = verify_deployable_model_artifact(
                        model_path,
                        active_identity,
                        dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
                        split_manifest_path=(
                            SPLITS_DIR / SPLIT_SCHEMA_VERSION / "split_manifest.json"
                        ),
                        feature_pipeline_path=FEATURE_PIPELINE_PATH,
                    )
                    st.session_state.threshold = float(model_metadata.extra["threshold"])

                    input_dim = len(st.session_state.feature_pipeline.output_feature_names_)
                    model = MaliciousPDFClassifier(input_dim=input_dim)
                    if model_path == quantized_path:
                        # Note: If quantized, need to properly load state dict
                        model.load_state_dict(torch.load(quantized_path, map_location='cpu'))
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                    else:
                        # Fallback to trained FP32
                        model.load_state_dict(torch.load(trained_path, map_location='cpu'))
                    
                    model.eval()
                    st.session_state.model = model
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.session_state.model = None

    def analyze(self, uploaded_file) -> AnalysisResult:
        """Run the end-to-end inference pipeline on the uploaded file."""
        if not st.session_state.model or not st.session_state.feature_pipeline:
            raise RuntimeError("Model or schema-v2 feature pipeline not loaded.")

        start_time = time.perf_counter()
        
        # Save to temp file since extractors need a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        try:
            # 1. Extract and apply the exact serialized training pipeline.
            scaled_vector, raw_features, diagnostics = pdf_to_pipeline_vector(
                tmp_path, pipeline=st.session_state.feature_pipeline
            )
            if diagnostics["abstain_recommended"]:
                return AnalysisResult(
                    prediction="Inconclusive",
                    confidence=0.0,
                    features=raw_features,
                    time_ms=(time.perf_counter() - start_time) * 1000,
                    file_hash=__import__("hashlib").sha256(uploaded_file.getvalue()).hexdigest(),
                )
            
            # 2. Inference
            with torch.no_grad():
                X_tensor = torch.tensor(scaled_vector, dtype=torch.float32).unsqueeze(0)
                logit = st.session_state.model(X_tensor)
                prob = torch.sigmoid(logit).item()
                
            # 3. Format result
            threshold = st.session_state.threshold
            confidence = prob if prob >= threshold else 1 - prob
            prediction = "Malicious" if prob >= threshold else "Benign"
            
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
