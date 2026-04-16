"""
config.py
---------
This module serves as the central configuration hub for the Malicious PDF Detector.
It defines crucial paths (data, models, reports), machine learning specifications 
(feature columns, hyperparameter grids), and sets up the environment variables 
for LLM integration and model inference parameters (e.g. CPU enforcement).
"""

from pathlib import Path

# --- Core Setup ---
RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_PDFS_DIR = DATA_DIR / "sample_pdfs"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
QUANTIZED_MODELS_DIR = MODELS_DIR / "quantized"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

# --- Machine Learning Setup ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15

FEATURE_COLUMNS = [
    "obj_count", "endobj_count", "stream_count", "endstream_count",
    "xref_count", "trailer_count", "startxref_count",
    "js_count", "javascript_count", "action_count", "openaction_count",
    "aa_count", "launch_count", "uri_count", "submitform_count",
    "acroform_count", "xfa_count", "richmedia_count",
    "jbig2decode_count", "colors_count",
    "objstm_count", "filter_count",
    "obfuscation_count",
    "avg_stream_size", "indirect_obj_count",
    "pdf_size", "title_chars", "is_encrypted",
    "metadata_size", "page_count", "has_text",
    "image_count", "obj_count_total",
    "font_obj_count", "embedded_file_count",
    "avg_embedded_media_size", "header_valid"
]

MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    },
    "xgboost": {
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [100, 300],
        "max_depth": [3, 6, 10],
        "tree_method": "hist"
    },
    "lightgbm": {
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [100, 300, 500]
    }
}

QUANTIZATION_BACKEND = 'fbgemm'
DEVICE = 'cpu'

# --- LLM Integration Setup ---
LLM_MODEL = 'gemma4:e4b'
LLM_FALLBACK_MODEL = 'gemma4:e2b'
LLM_BASE_URL = 'http://localhost:11434'
LLM_MAX_CONTEXT = 4096
RAM_THRESHOLD_MB = 5120
