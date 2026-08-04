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

CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENT_CONFIG_PATH = CONFIGS_DIR / "experiment.yaml"
DATA_SOURCES_CONFIG_PATH = CONFIGS_DIR / "data_sources.yaml"

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
DATA_REPORTS_DIR = REPORTS_DIR / "data"
ARCHIVE_REPORTS_DIR = REPORTS_DIR / "archive"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"

# --- Scientific acceptance gates (Professor feedback remediation) ---
FEATURE_SCHEMA_VERSION = "2.0.0"
SPLIT_SCHEMA_VERSION = "2.0.0"
MIN_TRAIN_ROWS = 2_000_000
MIN_VALIDATION_ROWS = 250_000
MIN_TEST_ROWS = 250_000
MIN_BENIGN_PREVALENCE = 0.995
LABEL_COLUMN = "Class"
SAMPLE_ID_COLUMN = "sample_id"
SOURCE_ID_COLUMN = "source_id"
GROUP_ID_COLUMN = "group_id"
FIRST_SEEN_COLUMN = "first_seen_at"

# --- Machine Learning Setup ---
TEST_SIZE = 0.15
VAL_SIZE = 0.15

from src.features.schema_v2 import BASE_FEATURE_COLUMNS  # noqa: E402

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)

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
