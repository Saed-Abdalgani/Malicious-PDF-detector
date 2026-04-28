"""
test_features.py
----------------
Tests for the feature extraction pipeline.
"""

import time
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.structural import extract_structural_features
from src.features.metadata import extract_metadata_features
from src.features.vectorizer import pdf_to_vector, fit_scaler, load_scaler

def test_extract_structural_features(benign_pdf_path):
    features = extract_structural_features(benign_pdf_path)
    assert isinstance(features, dict)
    assert len(features) == 25
    assert "js_count" in features

def test_extract_metadata_features(benign_pdf_path):
    features = extract_metadata_features(benign_pdf_path)
    assert isinstance(features, dict)
    assert len(features) == 12
    assert "pdf_size" in features

def test_pdf_to_vector(benign_pdf_path, mocker):
    mock_scaler = StandardScaler()
    mock_scaler.mean_ = np.zeros(37)
    mock_scaler.scale_ = np.ones(37)
    mocker.patch("src.features.vectorizer.load_scaler", return_value=mock_scaler)
    
    vector = pdf_to_vector(benign_pdf_path)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (37,)

def test_feature_extraction_corrupted_file(corrupted_pdf):
    s_features = extract_structural_features(corrupted_pdf)
    assert isinstance(s_features, dict)
    assert len(s_features) == 25
    assert all(v == 0.0 for v in s_features.values())

    m_features = extract_metadata_features(corrupted_pdf)
    assert isinstance(m_features, dict)
    assert len(m_features) == 12
    # Fallback populates remaining with 0.0, but pdf_size gets computed by os.path.getsize
    assert m_features["page_count"] == 0.0

def test_structural_extraction_timeout(synthetic_timeout_pdf):
    s_features = extract_structural_features(synthetic_timeout_pdf, timeout_sec=1)
    assert isinstance(s_features, dict)
    assert len(s_features) == 25

def test_feature_extraction_time_nfr_105(large_pdf):
    start = time.perf_counter()
    extract_structural_features(large_pdf)
    extract_metadata_features(large_pdf)
    duration = time.perf_counter() - start
    assert duration < 20.0, f"Extraction exceeded 20 seconds: {duration:.2f}s"

def test_scaler_save_load_roundtrip(dummy_feature_matrix, tmp_path):
    X, y = dummy_feature_matrix
    save_path = tmp_path / "scaler.pkl"
    
    scaler = fit_scaler(X, save=True, save_path=save_path)
    loaded_scaler = load_scaler(save_path)
    
    assert loaded_scaler is not None
    assert np.allclose(scaler.mean_, loaded_scaler.mean_)
