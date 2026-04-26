# 📋 Malicious PDF Detector — Master TODO

> **Project**: Lightweight Malicious PDF Detector using Structural Features, Quantization & LLM Analysis
> **Status**: 🟡 Phase 4.5 Complete
> **Last Updated**: 2026-04-26

---

## Phase 0: Project Scaffolding & Environment Setup

- [x] **0.1 — Create directory structure**
  - [x] Create `data/raw/`, `data/processed/`, `data/sample_pdfs/`
  - [x] Create `notebooks/`
  - [x] Create `src/`, `src/data/`, `src/features/`, `src/models/`, `src/optimization/`, `src/llm/`, `src/utils/`
  - [x] Create `models/trained/`, `models/quantized/`
  - [x] Create `reports/figures/`, `reports/results/`
  - [x] Create `app/`, `app/components/`, `app/assets/`
  - [x] Create `tests/`
  - [x] Add all `__init__.py` files in Python packages

- [x] **0.2 — Create `requirements.txt`**
  - [x] Core ML: numpy, pandas, scikit-learn, xgboost, lightgbm, imbalanced-learn
  - [x] PyTorch CPU-only: torch, torchvision (with `--index-url .../cpu`)
  - [x] PDF parsing: pdfminer.six, PyPDF2
  - [x] Visualization: matplotlib, seaborn, plotly
  - [x] Dashboard: streamlit
  - [x] LLM: ollama, httpx
  - [x] Download: requests, beautifulsoup4
  - [x] Utilities: tqdm, joblib, python-magic-bin, psutil
  - [x] Code quality: ruff (PEP 8 linting — NFR-501)

- [x] **0.3 — Create Python virtual environment**
  - [x] Run `python -m venv venv`
  - [x] Activate: `.\venv\Scripts\activate`
  - [x] Install: `pip install -r requirements.txt`
  - [x] Verify all imports work: `python -c "import torch; import sklearn; import streamlit; print('OK')"`

- [x] **0.4 — Create `.gitignore`**
  - [x] Ignore `venv/`, `__pycache__/`, `*.pyc`
  - [x] Ignore `data/raw/` (large CSV), `models/trained/`, `models/quantized/`
  - [x] Ignore `.env`, `*.kaggle`, Jupyter checkpoints

- [x] **0.5 — Create `src/config.py`**
  - [x] Define `RANDOM_SEED = 42`
  - [x] Define `PROJECT_ROOT` using `pathlib.Path`
  - [x] Define all data paths: `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `SAMPLE_PDFS_DIR`
  - [x] Define model paths: `TRAINED_MODELS_DIR`, `QUANTIZED_MODELS_DIR`
  - [x] Define report paths: `FIGURES_DIR`, `RESULTS_DIR`
  - [x] Define `TEST_SIZE = 0.15`, `VAL_SIZE = 0.15`
  - [x] Define `FEATURE_COLUMNS` — list of all 37 CIC feature names
  - [x] Define `MODEL_CONFIGS` — hyperparameter grids for RF, XGB, LGBM
  - [x] Define `QUANTIZATION_BACKEND = 'fbgemm'`
  - [x] Define `DEVICE = 'cpu'`
  - [x] Define LLM config: `LLM_MODEL = 'gemma4:e4b'`, `LLM_FALLBACK_MODEL = 'gemma4:e2b'`
  - [x] Define `LLM_BASE_URL = 'http://localhost:11434'`
  - [x] Define `LLM_MAX_CONTEXT = 4096`
  - [x] Define `RAM_THRESHOLD_MB = 5120`

- [x] **0.6 — Create `src/utils/logger.py`**
  - [x] Configure structured logging with `logging` module
  - [x] Set format: `[%(asctime)s] %(levelname)s - %(name)s - %(message)s`
  - [x] Add file handler → `reports/results/project.log`
  - [x] Add console handler with colored output
  - [x] Create `get_logger(name)` factory function

---

## Phase 1: Data Collection & Preprocessing

- [x] **1.1 — Create `src/data/downloader.py`**
  - [x] Implement `check_existing()` — detect if CSV already exists in `data/raw/`
  - [x] Implement `download_from_cic()` — direct HTTP GET from UNB CIC website
  - [x] Implement `download_from_github()` — fallback to community-hosted GitHub mirrors
  - [x] Implement `manual_fallback()` — auto-open browser to Kaggle dataset page
  - [x] Implement `validate_dataset(csv_path)`:
    - [x] Check row count (~10,025)
    - [x] Check column count (37 features + 1 label)
    - [x] Check data types are numeric
    - [x] Verify no fully empty columns
  - [x] Add SHA-256 hash verification
  - [x] Add progress bar via `tqdm`
  - [x] Add `__main__` block: `python -m src.data.downloader`
  - [x] Test: run downloader and verify `data/raw/pdfmal2022.csv` is created

- [x] **1.2 — Create `src/data/loader.py`**
  - [x] Implement `load_dataset(path=None) -> pd.DataFrame`
    - [x] Default path from `config.py`
    - [x] Read CSV with `pd.read_csv()`
    - [x] Validate column names against `config.FEATURE_COLUMNS`
    - [x] Type coercion: cast all feature columns to `float64`
    - [x] Log shape, null counts, class distribution
  - [x] Implement `get_feature_matrix(df) -> (X, y)` — separate features and label
  - [x] Test: load CSV and print `df.info()`, `df.describe()`

- [x] **1.3 — Create `src/data/cleaner.py`**
  - [x] Implement `remove_duplicates(df) -> df` — deduplicate by all features
    - [x] Log number of duplicates removed
  - [x] Implement `handle_missing(df) -> df`:
    - [x] Drop rows with >50% NaN values
    - [x] Median-impute remaining NaN values per column
    - [x] Log imputation statistics
  - [x] Implement `remove_constant_columns(df) -> df`:
    - [x] Drop columns with zero variance
    - [x] Log removed column names
  - [x] Implement `detect_outliers(df) -> df`:
    - [x] Use IQR method per feature
    - [x] Add `_outlier` flag columns (do NOT remove, just flag)
    - [x] Log outlier counts per feature
  - [x] Implement `clean_pipeline(df) -> df` — run all steps in sequence
    - [x] Save cleaned CSV to `data/processed/cleaned.csv`
    - [x] Log before/after row counts
  - [x] Test: run pipeline, verify `cleaned.csv` exists and has expected shape

- [x] **1.4 — Create `src/data/splitter.py`**
  - [x] Implement `stratified_split(df) -> (train, val, test)`:
    - [x] First split: 85% train_val / 15% test (stratified by label)
    - [x] Second split: train_val → 82.35% train / 17.65% val (≈70/15 overall)
    - [x] Use `sklearn.model_selection.train_test_split` with `random_state=42`
  - [x] Implement `apply_smote(train_df) -> train_df_balanced`:
    - [x] Apply SMOTE from `imblearn.over_sampling`
    - [x] Apply ONLY to training set (never val/test)
    - [x] Log class distribution before/after
  - [x] Implement `save_splits(train, val, test)`:
    - [x] Save `data/processed/train.csv`, `val.csv`, `test.csv`
    - [x] Log split sizes and class ratios
  - [x] Test: verify 3 CSV files exist with correct proportions

- [x] **1.5 — Create `notebooks/01_data_preprocessing.ipynb`**
  - [x] Cell 1: Import modules, display project info
  - [x] Cell 2: Run downloader (or skip if data exists)
  - [x] Cell 3: Load raw dataset, display `.info()`, `.describe()`, `.head()`
  - [x] Cell 4: Run cleaner pipeline, show before/after statistics
  - [x] Cell 5: Visualize data quality heatmap (nulls per feature)
  - [x] Cell 6: Run splitter, show class distribution bar charts
  - [x] Cell 7: Show SMOTE before/after comparison
  - [x] Cell 8: Summary table of all preprocessing steps

---

## Phase 2: Exploratory Data Analysis (EDA)

- [x] **2.1 — Create `src/utils/visualization.py`**
  - [x] Implement `plot_class_distribution(df, save_path)`:
    - [x] Bar chart: malicious vs benign counts
    - [x] Add count labels on bars
    - [x] Dark theme styling
  - [x] Implement `plot_feature_distributions(df, features, save_path)`:
    - [x] Overlaid histograms per class for each feature
    - [x] Grid layout (4-5 features per row)
    - [x] Color: red=malicious, green=benign
  - [x] Implement `plot_correlation_matrix(df, save_path)`:
    - [x] Seaborn heatmap with hierarchical clustering
    - [x] Annotated with correlation values
    - [x] Mask upper triangle
  - [x] Implement `plot_feature_importance(importances, feature_names, save_path)`:
    - [x] Horizontal bar chart, sorted descending
    - [x] Color gradient by importance value
  - [x] Implement `plot_boxplots(df, features, save_path)`:
    - [x] Side-by-side box plots per class
    - [x] Highlight outliers
  - [x] Implement `plot_pairplot(df, top_features, save_path)`:
    - [x] Scatter matrix for top N discriminative features
    - [x] Color by class
  - [x] Implement `plot_roc_curves(models_results, save_path)`:
    - [x] Overlaid ROC curves for multiple models
    - [x] AUC values in legend
  - [x] Implement `plot_confusion_matrix(cm, model_name, save_path)`:
    - [x] Heatmap with counts and percentages
  - [x] Apply consistent styling to all plots (dark theme, Inter font if available)

- [x] **2.2 — Create `notebooks/02_eda.ipynb`**
  - [x] Cell 1: Load cleaned + split data
  - [x] Cell 2: Class distribution analysis
    - [x] Bar chart of malicious vs benign
    - [x] Percentage breakdown
  - [x] Cell 3: Univariate analysis
    - [x] Overlaid histograms for all 37 features
    - [x] Identify top-10 most discriminative features visually
  - [x] Cell 4: Statistical significance testing
    - [x] Mann-Whitney U test for each feature
    - [x] Create p-value table sorted by significance
    - [x] Mark features with p < 0.001
  - [x] Cell 5: Correlation analysis
    - [x] Full correlation matrix heatmap
    - [x] Identify highly correlated pairs (|r| > 0.9)
    - [x] Discussion of redundant features
  - [x] Cell 6: Bivariate analysis
    - [x] Pair plots of top-5 discriminative features
    - [x] Scatter plots with decision boundaries (optional)
  - [x] Cell 7: Outlier analysis
    - [x] Box plots for top-10 features by class
    - [x] IQR outlier count table
  - [x] Cell 8: Key findings summary
    - [x] Markdown table of top-10 features with rationale
    - [x] Conclusions for feature engineering decisions

---

## Phase 3: Feature Engineering (Static PDF Analysis)

- [x] **3.1 — Create `src/features/structural.py`**
  - [x] Implement `extract_structural_features(pdf_path: str) -> dict`:
    - [x] Read raw PDF bytes with `open(pdf_path, 'rb')`
    - [x] Count keyword occurrences via regex:
      - [x] `/JS` → `js_count`
      - [x] `/JavaScript` → `javascript_count`
      - [x] `/OpenAction` → `openaction_count`
      - [x] `/Action` → `action_count`
      - [x] `/AA` → `aa_count`
      - [x] `/Launch` → `launch_count`
      - [x] `/URI` → `uri_count`
      - [x] `/SubmitForm` → `submitform_count`
      - [x] `/AcroForm` → `acroform_count`
      - [x] `/XFA` → `xfa_count`
      - [x] `/RichMedia` → `richmedia_count`
      - [x] `/JBig2Decode` → `jbig2decode_count`
      - [x] `/Colors` → `colors_count`
    - [x] Count structural markers:
      - [x] `obj` / `endobj` → `obj_count`, `endobj_count`
      - [x] `stream` / `endstream` → `stream_count`, `endstream_count`
      - [x] `xref` → `xref_count`
      - [x] `trailer` → `trailer_count`
      - [x] `startxref` → `startxref_count`
      - [x] `/ObjStm` → `objstm_count`
      - [x] `/Filter` → `filter_count`
    - [x] Calculate `avg_stream_size` (total stream bytes / stream count)
    - [x] Count indirect objects → `indirect_obj_count`
    - [x] Detect obfuscation patterns (hex-encoded `#XX` sequences) → `obfuscation_count`
  - [x] Add 30-second timeout for large files
  - [x] Add error handling for corrupted/unreadable PDFs (return zeroed dict)
  - [x] Test: extract features from a known benign and malicious PDF

- [x] **3.2 — Create `src/features/metadata.py`**
  - [x] Implement `extract_metadata_features(pdf_path: str) -> dict`:
    - [x] Use `PyPDF2.PdfReader` to parse PDF
    - [x] Extract `pdf_size` (file size in bytes via `os.path.getsize`)
    - [x] Extract `title_chars` (length of `/Title` metadata, 0 if missing)
    - [x] Extract `is_encrypted` (boolean → int: 0 or 1)
    - [x] Extract `metadata_size` (length of serialized metadata string)
    - [x] Extract `page_count` (number of pages)
    - [x] Extract `has_text` (check if any page has extractable text)
    - [x] Extract `image_count` (count `/XObject` → `/Image` in pages)
    - [x] Extract `obj_count_total` (total objects via `len(reader.objects)` or indirect)
    - [x] Extract `font_obj_count` (count `/Font` references)
    - [x] Extract `embedded_file_count` (count `/EmbeddedFile` in names tree)
    - [x] Extract `avg_embedded_media_size` (average size of embedded files)
    - [x] Extract `header_valid` (check PDF header `%PDF-1.X` presence → 0 or 1)
  - [x] Add error handling for corrupted PDFs
  - [x] Test: extract metadata from sample PDFs

- [x] **3.3 — Create `src/features/vectorizer.py`**
  - [x] Implement `combine_features(structural: dict, metadata: dict) -> np.ndarray`:
    - [x] Merge both dicts into single 37-dimension vector
    - [x] Ensure column order matches `config.FEATURE_COLUMNS` exactly
  - [x] Implement `fit_scaler(X_train) -> StandardScaler`:
    - [x] Fit `sklearn.preprocessing.StandardScaler` on training data
    - [x] Save scaler to `models/scaler.pkl` via `joblib.dump`
  - [x] Implement `transform(X, scaler) -> np.ndarray`:
    - [x] Apply fitted scaler to transform features
  - [x] Implement `load_scaler() -> StandardScaler`:
    - [x] Load from `models/scaler.pkl`
  - [x] Implement `pdf_to_vector(pdf_path: str) -> np.ndarray`:
    - [x] End-to-end: extract structural + metadata → combine → scale
    - [x] Uses saved scaler from training
    - [x] This is the function called by the Streamlit app
  - [x] Test: convert 5 sample PDFs to vectors, verify shape is (37,)

- [x] **3.4 — Create `notebooks/03_feature_engineering.ipynb`**
  - [x] Cell 1: Demo structural extraction on sample PDFs
  - [x] Cell 2: Demo metadata extraction on sample PDFs
  - [x] Cell 3: Show combined feature vectors as DataFrame
  - [x] Cell 4: Fit scaler on training data, show before/after normalization
  - [x] Cell 5: Compare extracted features with CIC ground truth (if sample PDFs available)
  - [x] Cell 6: Summary of feature engineering pipeline

---

## Phase 4: Model Development, Benchmarking & Reporting

- [x] **4.1 — Create `src/models/baseline.py`**
  - [x] Implement `BaselineModel` class:
    - [x] `__init__(self, model_type: str, params: dict)`
    - [x] `train(self, X_train, y_train, X_val, y_val)`:
      - [x] Instantiate model (RF / XGB / LGBM) based on `model_type`
      - [x] Run `GridSearchCV` with 5-fold cross-validation
      - [x] Use `n_jobs=-1` for CPU parallelism
      - [x] Log best hyperparameters
    - [x] `predict(self, X) -> np.ndarray`
    - [x] `predict_proba(self, X) -> np.ndarray`
    - [x] `save(self, path)` — via `joblib.dump` (compression=3)
    - [x] `load(self, path)` — via `joblib.load`
  - [x] Implement Random Forest configuration:
    - [x] `n_estimators=[100, 300, 500]`
    - [x] `max_depth=[10, 20, None]`
    - [x] `min_samples_split=[2, 5]`
  - [x] Implement XGBoost configuration:
    - [x] `learning_rate=[0.01, 0.1, 0.3]`
    - [x] `n_estimators=[100, 300]`
    - [x] `max_depth=[3, 6, 10]`
    - [x] `tree_method='hist'` (CPU-optimized)
  - [x] Implement LightGBM configuration:
    - [x] `learning_rate=[0.01, 0.1]`
    - [x] `num_leaves=[31, 63]`
    - [x] `n_estimators=[100, 300, 500]`
  - [x] Test: train each model on toy data, verify predictions

- [x] **4.2 — Create `src/models/mlp.py`**
  - [x] Implement `MaliciousPDFClassifier(nn.Module)`:
    - [x] `__init__`: define layers:
      - [x] `Linear(37, 128)` -> `BatchNorm1d(128)` -> `ReLU` -> `Dropout(0.3)`
      - [x] `Linear(128, 64)` -> `BatchNorm1d(64)` -> `ReLU` -> `Dropout(0.2)`
      - [x] `Linear(64, 32)` -> `BatchNorm1d(32)` -> `ReLU`
      - [x] `Linear(32, 1)` (output, raw logit)
    - [x] `forward(self, x)`: sequential pass through all layers
  - [x] Implement `PDFDataset(Dataset)`:
    - [x] `__init__`: accept numpy arrays (X, y)
    - [x] `__getitem__`: return `(torch.FloatTensor, torch.FloatTensor)`
    - [x] `__len__`: return sample count
  - [x] Implement training loop function `train_mlp(model, train_loader, val_loader, config)`:
    - [x] Optimizer: `AdamW(weight_decay=1e-4)`
    - [x] Scheduler: `CosineAnnealingLR(T_max=50)`
    - [x] Loss: `BCEWithLogitsLoss`
    - [x] Early stopping: patience=10 on validation loss
    - [x] Device: `torch.device('cpu')` enforced
    - [x] Max epochs: 100
    - [x] Log per-epoch: train_loss, val_loss, val_accuracy
    - [x] Save best checkpoint to `models/trained/mlp_best.pt`
    - [x] Return training history dict
  - [x] Test: train MLP on toy data (100 samples), verify convergence

- [x] **4.3 — Create `src/models/trainer.py`**
  - [x] Implement `TrainingPipeline`:
    - [x] `__init__`: load train/val/test splits from `data/processed/`
    - [x] `train_all_models()`:
      - [x] Train Random Forest -> save to `models/trained/rf_best.pkl`
      - [x] Train XGBoost -> save to `models/trained/xgb_best.pkl`
      - [x] Train LightGBM -> save to `models/trained/lgbm_best.pkl`
      - [x] Train MLP -> save to `models/trained/mlp_best.pt`
      - [x] Log total training time per model
    - [x] `load_all_models() -> dict`: load all saved models
  - [x] Implement timing decorator for training functions
  - [x] Test: run full training pipeline, verify all 4 model files exist

- [x] **4.4 — Create `src/models/evaluator.py`**
  - [x] Implement `evaluate_model(model, X_test, y_test, model_name) -> dict`:
    - [x] Calculate: Accuracy, F1, Precision, Recall, AUC-ROC
    - [x] Generate confusion matrix
    - [x] Generate classification report (per-class metrics)
    - [x] Return dict with all metrics
  - [x] Implement `compare_models(results: list[dict]) -> pd.DataFrame`:
    - [x] Create comparison table DataFrame
    - [x] Sort by F1-score descending
    - [x] Save to `reports/results/model_comparison.csv`
  - [x] Implement `generate_report(results, save_dir)`:
    - [x] Create confusion matrix heatmaps for each model -> `reports/figures/`
    - [x] Create overlaid ROC curves -> `reports/figures/roc_curves.png`
    - [x] Create Precision-Recall curves -> `reports/figures/pr_curves.png`
    - [x] Create feature importance chart (tree models) -> `reports/figures/`
    - [x] Save comparison table as CSV
    - [x] Log summary to console
  - [x] Test: evaluate a trained model, verify all outputs exist

- [x] **4.5 — Create `notebooks/04_model_training.ipynb`**
  - [x] Cell 1: Load preprocessed train/val/test data
  - [x] Cell 2: Train all 4 models (with progress logging)
  - [x] Cell 3: Display hyperparameter tuning results for each model
  - [x] Cell 4: Plot MLP training curves (loss + accuracy over epochs)
  - [x] Cell 5: Run evaluation on test set for all models
  - [x] Cell 6: Display comparison table
  - [x] Cell 7: Display confusion matrices (2x2 grid)
  - [x] Cell 8: Display overlaid ROC curves
  - [x] Cell 9: Display feature importance (top-15 features)
  - [x] Cell 10: Selection rationale — which model to quantize and why

---

## Phase 4.5: LLM Integration — Gemma 4 E4B

- [x] **4.5.1 — Install Ollama**
  - [x] Download Ollama from https://ollama.com/download/windows
  - [x] Install via `winget install Ollama.Ollama`
  - [x] Verify installation: `ollama --version`

- [x] **4.5.2 — Pull Gemma 4 models**
  - [x] Pull primary: `ollama pull gemma4:e4b` (~2.5GB) *(Note: Cancelled locally to save space. Will download later when project is loaded to Google Colab)*
  - [x] Pull fallback: `ollama pull gemma4:e2b` (~1.2GB)
  - [x] Verify: `ollama list` shows both models
  - [x] Quick test: `ollama run gemma4:e4b "Analyze this: /JS count=5, /OpenAction=1"`

- [x] **4.5.3 — Create `src/llm/client.py`**
  - [x] Implement `GemmaClient` class:
    - [x] `__init__(self, model, base_url, max_context)`:
      - [x] Store Ollama config from `config.py`
      - [x] Initialize `ollama.Client`
    - [x] `check_health() -> bool`:
      - [x] Ping Ollama API at `/api/tags`
      - [x] Return True if responsive, False otherwise
    - [x] `check_ram() -> str | None`:
      - [x] Use `psutil.virtual_memory().available`
      - [x] If ≥5GB free → return `'gemma4:e4b'`
      - [x] If ≥3GB and <5GB free → return `'gemma4:e2b'`
      - [x] If <3GB free → return `None` (disable LLM entirely, log warning) — PRD 11.3
      - [x] Log selected model and available RAM
    - [x] `generate(prompt, system_prompt, temperature=0.3) -> str`:
      - [x] Call `ollama.chat()` with messages
      - [x] Handle timeout (30s)
      - [x] Return response text
    - [x] `generate_stream(prompt, system_prompt) -> Generator[str]`:
      - [x] Streaming version for Streamlit UI
      - [x] Yield tokens one at a time
    - [x] `warmup()`:
      - [x] Send a small test prompt to pre-load model into RAM
      - [x] Called once when user first clicks "Analyze with AI"
  - [x] Add connection retry logic (3 attempts, 2s backoff)
  - [x] Add graceful error messages when Ollama is not running
  - [x] Test: connect to Ollama, send test prompt, verify response

- [x] **4.5.4 — Create `src/llm/prompts.py`**
  - [x] Define `SYSTEM_PROMPT` — cybersecurity SOC analyst role
    - [x] Include instructions for threat analysis methodology
    - [x] Include MITRE ATT&CK reference guidance
    - [x] Include severity rating scale (Critical/High/Medium/Low)
  - [x] Define `THREAT_ANALYSIS_TEMPLATE`:
    - [x] Placeholders: `{prediction}`, `{confidence}`, `{processing_time}`
    - [x] `{feature_summary}`: formatted feature table
    - [x] `{suspicious_features}`: features that deviate from benign baseline
    - [x] Request: threat explanation, attack vector, severity, remediation
  - [x] Define `JAVASCRIPT_ANALYSIS_TEMPLATE`:
    - [x] Placeholder: `{js_code}`
    - [x] Request: step-by-step code analysis, exploit technique, vulnerability
  - [x] Define `QUICK_SUMMARY_TEMPLATE`:
    - [x] Shorter prompt for benign files (just confirm safety)
  - [x] Define `FOLLOW_UP_TEMPLATE`:
    - [x] For interactive Q&A: includes previous context + new question
  - [x] Test: format each template with sample data, verify structure

- [x] **4.5.5 — Create `src/llm/analyzer.py`**
  - [x] Implement `ThreatAnalyzer`:
    - [x] `__init__(self, client: GemmaClient)`:
      - [x] Store client reference
      - [x] Load benign baseline statistics (mean/std from training set)
    - [x] `identify_suspicious_features(features: dict) -> list`:
      - [x] Compare each feature against benign baseline
      - [x] Flag features >2σ deviation from benign mean
      - [x] Return sorted list of (feature_name, value, deviation, description)
    - [x] `analyze(features, prediction, confidence) -> ThreatReport`:
      - [x] Build prompt from `THREAT_ANALYSIS_TEMPLATE`
      - [x] Inject suspicious features into prompt
      - [x] Call LLM via client
      - [x] Parse response into `ThreatReport` dataclass
    - [x] `analyze_javascript(js_code: str) -> str`:
      - [x] Build prompt from `JAVASCRIPT_ANALYSIS_TEMPLATE`
      - [x] Return LLM analysis text
    - [x] `quick_summary(features, prediction) -> str`:
      - [x] Short summary for benign files
      - [x] Use `QUICK_SUMMARY_TEMPLATE`
  - [x] Test: analyze sample malicious features, verify report structure

- [x] **4.5.6 — Create `src/llm/report_generator.py`**
  - [x] Implement `ThreatReport` dataclass:
    - [x] Fields: timestamp, file_hash, ml_prediction, ml_confidence
    - [x] Fields: risk_severity, threat_explanation, attack_vector
    - [x] Fields: suspicious_features (list), remediation, processing_time_ms
    - [x] Method `to_markdown() -> str` — formatted Markdown report
    - [x] Method `to_json() -> str` — structured JSON output
    - [x] Method `to_dict() -> dict` — for Streamlit display
  - [x] Implement `generate_file_hash(pdf_bytes: bytes) -> str`:
    - [x] SHA-256 hash of file contents
  - [x] Test: create ThreatReport, verify Markdown and JSON output

- [x] **4.5.7 — Create `notebooks/06_llm_integration.ipynb`**
  - [x] Cell 1: Check Ollama health and model availability
  - [x] Cell 2: Run RAM check, show selected model
  - [x] Cell 3: Demo threat analysis on 3 malicious feature sets
  - [x] Cell 4: Demo threat analysis on 2 benign feature sets
  - [x] Cell 5: Show formatted ThreatReport (Markdown + JSON)
  - [x] Cell 6: Benchmark LLM inference time (5 runs, show mean/std)
  - [x] Cell 7: Demo interactive Q&A with follow-up prompts
  - [x] Cell 8: Compare zero-shot vs. few-shot prompt strategies

---

## Phase 5: Model Optimization via Quantization

- [ ] **5.1 — Create `src/optimization/quantizer.py`**
  - [ ] Implement `ModelQuantizer`:
    - [ ] `__init__(self, model: nn.Module, backend='fbgemm')`:
      - [ ] Store model and set quantization backend
    - [ ] `dynamic_quantize() -> nn.Module`:
      - [ ] Apply `torch.quantization.quantize_dynamic`
      - [ ] Target: `nn.Linear` layers
      - [ ] Weight dtype: `torch.qint8`
      - [ ] Return quantized model
    - [ ] `static_quantize(calibration_loader) -> nn.Module`:
      - [ ] Step 1: Fuse `Linear + BatchNorm + ReLU` layers
        - [ ] Define fuse list based on MLP architecture
        - [ ] Call `torch.quantization.fuse_modules`
      - [ ] Step 2: Set `qconfig` for `fbgemm` backend
      - [ ] Step 3: Insert observers via `torch.quantization.prepare`
      - [ ] Step 4: Run calibration (500 training samples through model)
      - [ ] Step 5: Convert via `torch.quantization.convert`
      - [ ] Return quantized model
    - [ ] `save_quantized(model, path)`:
      - [ ] Save with `torch.save(model.state_dict(), path)`
      - [ ] Also save as TorchScript via `torch.jit.script` for deployment
    - [ ] `load_quantized(path) -> nn.Module`
  - [ ] Test: quantize trained MLP, verify predictions still work

- [ ] **5.2 — Create `src/optimization/benchmark.py`**
  - [ ] Implement `measure_model_size(model_path) -> float`:
    - [ ] Return file size in MB
  - [ ] Implement `measure_inference_time(model, X_sample, n_runs=100) -> dict`:
    - [ ] Run `n_runs` inference passes
    - [ ] Return: mean, std, min, max in milliseconds
    - [ ] Use `time.perf_counter()` for precision
  - [ ] Implement `measure_memory(model) -> float`:
    - [ ] Estimate model RAM usage in MB
  - [ ] Implement `compare_models(fp32_model, int8_model, test_data) -> pd.DataFrame`:
    - [ ] Measure size, speed, memory, accuracy, F1 for both
    - [ ] Calculate percentage change for each metric
    - [ ] Return comparison DataFrame
    - [ ] Save to `reports/results/quantization_comparison.csv`
  - [ ] Implement `benchmark_all_models()`:
    - [ ] Benchmark MLP FP32 vs INT8 (dynamic) vs INT8 (static)
    - [ ] Benchmark tree models (RF, XGB, LGBM) sizes and speeds
    - [ ] Create unified comparison table
  - [ ] Test: run full benchmark, verify CSV output

- [ ] **5.3 — Create `notebooks/05_quantization.ipynb`**
  - [ ] Cell 1: Load trained MLP (FP32) model
  - [ ] Cell 2: Apply dynamic quantization, evaluate accuracy
  - [ ] Cell 3: Apply static quantization with calibration, evaluate accuracy
  - [ ] Cell 4: Model size comparison bar chart (FP32 vs Dynamic INT8 vs Static INT8)
  - [ ] Cell 5: Inference time comparison (box plots from 100 runs)
  - [ ] Cell 6: Accuracy/F1 comparison table — verify <1% drop
  - [ ] Cell 7: Memory usage comparison
  - [ ] Cell 8: Tree model size/speed benchmarks
  - [ ] Cell 9: Final selection: best quantized model for deployment
  - [ ] Cell 10: Save final quantized model to `models/quantized/`

---

## Phase 6: Streamlit Application & Local Deployment

- [ ] **6.1 — Create `app/assets/style.css`**
  - [ ] Define dark theme color variables (CSS custom properties)
  - [ ] Style card components with glassmorphism (backdrop-filter: blur)
  - [ ] Style verdict cards: green gradient (safe), red gradient (malicious)
  - [ ] Style animated scan indicator (CSS keyframes pulse animation)
  - [ ] Import Inter font from Google Fonts
  - [ ] Style LLM chat bubbles (user vs AI messages)
  - [ ] Style metric cards with subtle shadows
  - [ ] Responsive breakpoints for layout

- [ ] **6.2 — Create `app/components/uploader.py`**
  - [ ] Implement `render_upload_zone() -> UploadedFile | None`:
    - [ ] `st.file_uploader` with `type=['pdf']`, `accept_multiple_files=False`
    - [ ] Server-side MIME type validation via `python-magic`
    - [ ] File size check (reject >50MB)
    - [ ] Process via `io.BytesIO` — no disk writes
    - [ ] Sanitize filename for logging (strip path components)
    - [ ] Return uploaded file object or None

- [ ] **6.3 — Create `app/components/analyzer.py`**
  - [ ] Implement `PDFAnalyzer`:
    - [ ] `__init__`: load quantized model + scaler into `st.session_state`
      - [ ] Cache model to avoid reloading on every Streamlit rerun
    - [ ] `analyze(uploaded_file) -> AnalysisResult`:
      - [ ] Extract features from PDF bytes (call `pdf_to_vector`)
      - [ ] Run inference through quantized model
      - [ ] Measure processing time
      - [ ] Return: prediction, confidence, features, time_ms, file_hash
    - [ ] `get_feature_breakdown(features) -> pd.DataFrame`:
      - [ ] Create formatted feature table for display

- [ ] **6.4 — Create `app/components/dashboard.py`**
  - [ ] Implement `render_verdict(result: AnalysisResult)`:
    - [ ] Show large verdict card: "✅ SAFE" or "⚠ MALICIOUS"
    - [ ] Confidence percentage with color coding
    - [ ] Processing time metric
    - [ ] Animated transition on result
  - [ ] Implement `render_confidence_gauge(confidence: float)`:
    - [ ] Plotly gauge chart (0-100%)
    - [ ] Green/yellow/red zones
  - [ ] Implement `render_feature_radar(features, top_n=10)`:
    - [ ] Plotly radar chart of top-10 features
    - [ ] Overlay benign baseline for comparison
  - [ ] Implement `render_scan_history(session_state)`:
    - [ ] Table of all scans in current session
    - [ ] Columns: filename, verdict, confidence, time, timestamp

- [ ] **6.5 — Create `app/components/llm_chat.py`**
  - [ ] Implement `render_llm_panel(analysis_result)`:
    - [ ] Check Ollama health → show status indicator (🟢/🔴)
    - [ ] "Analyze with AI" button (loads LLM on-demand)
    - [ ] Display threat analysis in formatted card
    - [ ] Chat input for follow-up questions
    - [ ] Streaming response display (token-by-token)
    - [ ] "Generate Full Report" button → download as Markdown/JSON
    - [ ] Graceful fallback: if Ollama offline, show message + skip LLM features
  - [ ] Implement `render_report_download(report: ThreatReport)`:
    - [ ] `st.download_button` for Markdown report
    - [ ] `st.download_button` for JSON report
  - [ ] Manage chat history in `st.session_state`

- [ ] **6.6 — Create `app/streamlit_app.py`**
  - [ ] Set page config: title, icon, layout="wide"
  - [ ] Load custom CSS from `app/assets/style.css`
  - [ ] Implement sidebar navigation (5 pages)
  - [ ] **Page 1 — 🔍 PDF Scanner**:
    - [ ] Render upload zone
    - [ ] On upload: run analyzer, show verdict dashboard
    - [ ] Show feature breakdown in expandable section
    - [ ] Download analysis as JSON button
  - [ ] **Page 2 — 🤖 AI Threat Analyst**:
    - [ ] Show LLM panel with threat analysis
    - [ ] Interactive chat interface
    - [ ] Report generation and download
  - [ ] **Page 3 — 📊 Model Dashboard**:
    - [ ] Load and display model comparison table from `reports/results/`
    - [ ] Interactive ROC curves (Plotly)
    - [ ] Confusion matrix heatmaps
    - [ ] Quantization efficiency gains chart
  - [ ] **Page 4 — 📈 Feature Explorer**:
    - [ ] Interactive feature importance chart
    - [ ] Upload PDF to see features vs dataset distribution
    - [ ] Highlight suspicious features
  - [ ] **Page 5 — ℹ️ About**:
    - [ ] Project description and methodology
    - [ ] Architecture diagram (Mermaid or image)
    - [ ] Tech stack, dataset info
    - [ ] LLM model specifications
    - [ ] Performance benchmarks
  - [ ] Test: run `streamlit run app/streamlit_app.py`
  - [ ] Test: upload sample PDF, verify full pipeline works

---

## Phase 7: Testing & Quality Assurance

- [ ] **7.1 — Create `tests/test_features.py`**
  - [ ] Test `extract_structural_features` returns 25-key dict
  - [ ] Test `extract_metadata_features` returns 12-key dict
  - [ ] Test `pdf_to_vector` returns shape (37,) array
  - [ ] Test feature extraction on corrupted file returns zeroed dict
  - [ ] Test feature extraction timeout on large synthetic file
  - [ ] Test scaler save/load roundtrip
  - [ ] Test feature extraction time < 5 seconds for PDFs ≤ 50MB (NFR-105)

- [ ] **7.2 — Create `tests/test_models.py`**
  - [ ] Test `BaselineModel` train/predict cycle on toy data
  - [ ] Test `MaliciousPDFClassifier` forward pass with correct input shape
  - [ ] Test MLP training loop converges on separable toy data
  - [ ] Test model save/load roundtrip for all model types
  - [ ] Test `evaluate_model` returns all expected metric keys
  - [ ] Test training reproducibility: run with RANDOM_SEED=42 twice, verify identical metrics (NFR-204)
  - [ ] Test saved model files contain no raw training data samples (SEC-08)

- [ ] **7.3 — Create `tests/test_quantization.py`**
  - [ ] Test dynamic quantization produces smaller model file
  - [ ] Test static quantization produces smaller model file
  - [ ] Test quantized model predictions match FP32 on 99%+ of test samples
  - [ ] Test quantized model size is <50% of FP32 size
  - [ ] Test quantized model inference is faster than FP32
  - [ ] Test quantized model absolute file size < 1MB (G2)
  - [ ] Test quantized single-sample inference time < 500ms (NFR-102)

- [ ] **7.4 — Create `tests/test_llm.py`**
  - [ ] Test `GemmaClient.check_health()` returns bool
  - [ ] Test `GemmaClient.check_ram()` returns valid model name or None
  - [ ] Test `check_ram()` returns None when <3GB free (PRD 11.3 — LLM disabled)
  - [ ] Test `ThreatReport.to_markdown()` produces valid Markdown
  - [ ] Test `ThreatReport.to_json()` produces valid JSON
  - [ ] Test `identify_suspicious_features` returns non-empty list for malicious features
  - [ ] Test graceful degradation when Ollama is offline

- [ ] **7.5 — Create `tests/test_security.py`** *(NEW — PRD SEC-01 to SEC-08)*
  - [ ] Test analysis pipeline makes zero outbound HTTP requests during scan (SEC-04)
  - [ ] Test PDF parsing does not execute embedded JavaScript (SEC-05)
  - [ ] Test uploaded files are processed in-memory only, not written to disk (SEC-02)
  - [ ] Test file size limit rejects PDFs > 50MB (SEC-03)
  - [ ] Test MIME type validation rejects non-PDF files with .pdf extension (SEC-01)
  - [ ] Test filename sanitization strips path traversal characters (SEC-06)
  - [ ] Test no external API calls are made by LLM module (SEC-07, Ollama is localhost only)

- [ ] **7.6 — Code quality checks** *(NEW — PRD NFR-501, NFR-502)*
  - [ ] Run `ruff check src/` — verify zero PEP 8 violations (NFR-501)
  - [ ] Run `ruff check app/` — verify zero PEP 8 violations (NFR-501)
  - [ ] Verify all public functions and classes have docstrings (NFR-502)
  - [ ] Verify `src/config.py` is the single source for all paths/constants — no hardcoded values elsewhere (NFR-503)

- [ ] **7.7 — Run all tests**
  - [ ] `pytest tests/ -v --tb=short`
  - [ ] Verify all tests pass (test_features, test_models, test_quantization, test_llm, test_security)
  - [ ] Fix any failures

---

## Phase 8: Final Report & Documentation

- [ ] **8.1 — Create `notebooks/07_final_report.ipynb`**
  - [ ] Section 1: Executive summary (key results, 3-4 sentences)
  - [ ] Section 2: Problem statement and motivation
  - [ ] Section 3: Dataset description and preprocessing summary
  - [ ] Section 4: EDA key findings (embed top charts)
  - [ ] Section 5: Feature engineering methodology
  - [ ] Section 6: Model comparison results (embed comparison table + ROC curves)
  - [ ] Section 7: Quantization analysis (embed before/after charts)
  - [ ] Section 8: LLM integration and sample threat reports
  - [ ] Section 9: Streamlit application screenshots
  - [ ] Section 10: Conclusions and future work
  - [ ] Export to PDF: `jupyter nbconvert --to pdf`

- [ ] **8.2 — Create `README.md`**
  - [ ] Project title and description
  - [ ] Architecture diagram
  - [ ] Installation instructions (step-by-step)
  - [ ] Dataset setup instructions
  - [ ] Ollama/LLM setup instructions
  - [ ] Usage guide (how to run training, how to launch app)
  - [ ] Project structure tree
  - [ ] Results summary table
  - [ ] Screenshots of Streamlit app
  - [ ] License and credits

- [ ] **8.3 — Create `setup.py`**
  - [ ] Package metadata: name, version, author, description
  - [ ] `install_requires` from requirements.txt
  - [ ] Entry points for CLI commands

---

## Phase 9: Integration Testing & Final Demo

- [ ] **9.1 — End-to-end pipeline validation**
  - [ ] Run full pipeline from raw data → trained models → quantized models
  - [ ] Verify all model files exist in expected locations
  - [ ] Verify all report files/figures are generated

- [ ] **9.2 — Streamlit app validation**
  - [ ] Launch app: `streamlit run app/streamlit_app.py`
  - [ ] Measure app startup time — verify < 10 seconds without LLM (NFR-104)
  - [ ] Upload 5 known-benign PDFs → verify "Safe" classification
  - [ ] Upload 5 known-malicious PDFs → verify "Malicious" classification
  - [ ] Verify processing time < 3s per file (NFR-101)
  - [ ] Benchmark feature extraction on varied sizes (1KB, 1MB, 10MB, 50MB) — verify < 5s each (NFR-105)
  - [ ] Count user interactions for scan workflow — verify ≤ 3 clicks (NFR-301)
  - [ ] Test AI Threat Analyst page with Ollama running
  - [ ] Test all 5 navigation pages load without errors
  - [ ] Test edge cases: empty PDF, >10MB PDF, corrupted PDF, non-PDF file
  - [ ] Resize browser to 768px width — verify layout doesn't break (PRD 12.3)

- [ ] **9.3 — LLM integration validation**
  - [ ] Verify Ollama auto-detection and model selection
  - [ ] Verify threat analysis generates valid report
  - [ ] Verify chat follow-up works
  - [ ] Verify report download (Markdown + JSON)
  - [ ] Verify graceful degradation when Ollama is stopped
  - [ ] Verify LLM is disabled (not loaded) when <3GB free RAM (PRD 11.3)
  - [ ] Benchmark: LLM response time < 30s on CPU (NFR-103)
  - [ ] Review 3 LLM-generated reports for non-technical readability — no unexplained jargon (NFR-302)

- [ ] **9.4 — Performance validation**
  - [ ] Verify best tree model accuracy ≥ 95% (G1)
  - [ ] Verify all models F1 ≥ 0.94 (G1)
  - [ ] Verify quantized MLP accuracy drop < 2% (G3)
  - [ ] Verify quantized model size reduction ≥ 50% (G3)
  - [ ] Verify quantized model absolute file size < 1MB (G2)
  - [ ] Verify quantized single-sample inference time < 500ms (NFR-102)
  - [ ] Verify quantized inference speedup ≥ 2x vs FP32
  - [ ] Verify fresh `pip install -r requirements.txt` in clean venv succeeds with zero errors (NFR-404)

---

> **Legend**:
> - `[ ]` = Not started
> - `[/]` = In progress
> - `[x]` = Completed
