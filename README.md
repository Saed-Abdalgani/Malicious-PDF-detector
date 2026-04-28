# Malicious PDF Detector

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red)
![Ollama](https://img.shields.io/badge/Ollama-Gemma_4-orange)

A lightweight, high-performance **Malicious PDF Detector** leveraging Structural and Metadata Feature Engineering, PyTorch Model Quantization (INT8), and Local LLM (Gemma 4 via Ollama) for explainable threat intelligence. This tool allows SOC analysts and security researchers to scan PDFs securely entirely locally without exposing sensitive documents to external networks.

## 🚀 Key Features
- **In-Memory Secure Processing**: Never writes uploaded PDFs to disk (PRD SEC-02).
- **Zero Outbound Telemetry**: Air-gapped ML inference + local LLM, guaranteeing zero data leakage.
- **Lightning Fast Inference**: INT8 Post-Training Quantization ensures single-sample detection under 500ms on CPU.
- **AI Threat Analyst**: Explains _why_ a PDF is malicious and provides remediation via cybersecurity-tuned Gemma prompts.
- **Dynamic Dark-Theme Dashboard**: Streamlit interface with radar charts, interactive gauges, and session history.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[User PDF Upload] --> B[Streamlit Uploader]
    B -->|In-Memory Bytes| C[Feature Extractor]
    
    subgraph 🔍 Feature Engineering Pipeline
        C --> D[Structural Keywords /JS, /OpenAction]
        C --> E[PyPDF2 Metadata]
        D --> F[Combined Feature Vector]
        E --> F
        F --> G[Standard Scaler]
    end
    
    subgraph 🧠 ML Inference
        G --> H[Quantized MLP Model INT8]
        H --> I{Verdict: Malicious/Benign}
    end
    
    subgraph 🤖 AI Analyst Layer
        I --> J[Suspicious Feature Identifier]
        J --> K[Ollama: Gemma 4 E4B]
        K --> L[Threat Report & Remediation]
    end
    
    I --> M[Streamlit Dashboard]
    L --> M
```

---

## 📊 Results Summary

The baseline PyTorch MLP model was quantized (Static INT8) to optimize for edge/CPU inference, achieving >50% size reduction with negligible accuracy loss.

| Model Variant | F1-Score | Accuracy | ROC-AUC | Size (MB) | Inference Time |
|---------------|----------|----------|---------|-----------|----------------|
| **MLP (FP32)** | 99.82%   | 99.80%   | 0.999   | 1.25 MB   | ~12ms          |
| **MLP (INT8 Dynamic)**| 99.80% | 99.80% | 0.998 | 0.55 MB | ~8ms           |
| **MLP (INT8 Static)** | 99.80% | 99.78% | 0.998 | **0.55 MB** | **~6ms**     |

---

## 🛠️ Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Saed-Abdalgani/Malicious-PDF-detector.git
cd "Malicious PDF detector"
```

### 2. Setup Python Environment
Ensure you have Python 3.9+ installed.
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Alternatively, install as an editable package:
pip install -e .
```

---

## 🗄️ Dataset Setup

The model is trained on the **CIC Evasive-PDFMal2022** dataset.
To download and preprocess the dataset:
```bash
# Downloads raw data and runs the cleaning pipeline
python -m src.data.downloader
python -m src.data.cleaner
python -m src.data.splitter
```
Alternatively, run the `01_data_preprocessing.ipynb` notebook.

---

## 🤖 Ollama & LLM Setup

To use the **AI Threat Analyst** functionality, you must run Ollama locally.

1. Download and install [Ollama](https://ollama.com/download).
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Open a new terminal and pull the required Gemma 4 models:
   ```bash
   ollama pull gemma4:e4b   # High RAM model (>5GB free RAM required)
   ollama pull gemma4:e2b   # Low RAM fallback model
   ```

_Note: If Ollama is not running, the application gracefully degrades and disables the LLM features while keeping ML detection fully functional._

---

## 💻 Usage Guide

### Launching the Dashboard
Start the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```
Open your browser to `http://localhost:8501`.

### Training the Models
If you want to retrain the models from scratch:
```bash
python -m src.models.trainer
```

---

## 📸 Application Screenshots

### 1. Dashboard & Verdict
![Dashboard Placeholder](app/assets/screenshot_dashboard.png)
*(Drop a PDF into the scanner to receive an instant ML verdict and confidence gauge).*

### 2. AI Threat Analysis
![LLM Placeholder](app/assets/screenshot_llm.png)
*(Click "Analyze with AI" to generate a detailed, cybersecurity-focused threat report).*

---

## 📁 Project Structure

```text
Malicious PDF detector/
│
├── app/                      # Streamlit application UI
│   ├── assets/               # Custom CSS and styling
│   ├── components/           # UI modules (uploader, dashboard, chat)
│   └── streamlit_app.py      # Main entry point
│
├── data/                     # Ignored in git
│   ├── processed/            # Cleaned train/test/val splits
│   └── raw/                  # Downloaded CIC CSV
│
├── models/                   # Saved ML artifacts
│   ├── quantized/            # INT8 optimized models
│   ├── trained/              # FP32 models
│   └── scaler.pkl            # StandardScaler state
│
├── notebooks/                # Jupyter pipelines
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_quantization.ipynb
│   ├── 06_llm_integration.ipynb
│   └── 07_final_report.ipynb # Full project presentation
│
├── reports/                  # Evaluation artifacts
│   ├── figures/              # Heatmaps, ROC curves, Boxplots
│   └── results/              # Evaluation CSVs and logging
│
├── src/                      # Core python module
│   ├── data/                 # Download, clean, split logic
│   ├── features/             # PDF structural & metadata extraction
│   ├── llm/                  # Gemma client and threat analyzer
│   ├── models/               # PyTorch MLP & Scikit-learn models
│   ├── optimization/         # FBGEMM/QNNPACK quantization
│   └── config.py             # Global paths and configurations
│
├── tests/                    # Pytest suite
│   ├── test_features.py
│   ├── test_llm.py
│   ├── test_models.py
│   ├── test_quantization.py
│   └── test_security.py
│
├── requirements.txt          # Package dependencies
└── setup.py                  # Package installer
```

---

## 📜 License & Credits

**License**: MIT License
**Author**: Saed Abdalgani

**Credits**:
- Dataset provided by the [Canadian Institute for Cybersecurity (CIC)](https://www.unb.ca/cic/datasets/pdfmal-2022.html).
- Powered by [PyTorch](https://pytorch.org), [Streamlit](https://streamlit.io), and [Ollama](https://ollama.com).
