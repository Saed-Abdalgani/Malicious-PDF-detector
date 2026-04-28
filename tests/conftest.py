"""
conftest.py
-----------
Global pytest fixtures for the Malicious PDF Detector tests.
"""

import pytest
import numpy as np

from src.config import SAMPLE_PDFS_DIR, FEATURE_COLUMNS

@pytest.fixture
def benign_pdf_path():
    """Path to the known benign sample PDF."""
    path = SAMPLE_PDFS_DIR / "benign_sample.pdf"
    if not path.exists():
        pytest.skip(f"Benign sample not found at {path}")
    return path

@pytest.fixture
def malicious_pdf_path():
    """Path to the known malicious sample PDF."""
    path = SAMPLE_PDFS_DIR / "malicious_sample.pdf"
    if not path.exists():
        pytest.skip(f"Malicious sample not found at {path}")
    return path

@pytest.fixture
def corrupted_pdf(tmp_path):
    """Creates a corrupted/invalid PDF file for testing error handling."""
    corrupted = tmp_path / "corrupted.pdf"
    with open(corrupted, "wb") as f:
        f.write(b"This is completely garbage data, not a PDF.")
    return corrupted

@pytest.fixture
def large_pdf(tmp_path):
    """Creates a 51MB dummy PDF to test size limits."""
    large_file = tmp_path / "large.pdf"
    with open(large_file, "wb") as f:
        f.write(b"%PDF-1.4\n")
        # Write 51 MB of random bytes
        f.seek((51 * 1024 * 1024) - 10)
        f.write(b"%%EOF\n")
    return large_file

@pytest.fixture
def synthetic_timeout_pdf(tmp_path):
    """Creates a PDF designed to cause catastrophic backtracking in regex (or just be very large)."""
    timeout_file = tmp_path / "timeout.pdf"
    with open(timeout_file, "wb") as f:
        f.write(b"%PDF-1.4\n")
        # Lots of stream objects to trigger long processing
        for _ in range(500000):
            f.write(b"stream\n" + b"A" * 100 + b"\nendstream\n")
        f.write(b"%%EOF\n")
    return timeout_file

@pytest.fixture
def dummy_features_dict():
    """Returns a dictionary matching the 37 feature columns."""
    return {col: 0.0 for col in FEATURE_COLUMNS}

@pytest.fixture
def dummy_feature_array():
    """Returns a numpy array of shape (37,)."""
    return np.zeros((37,))

@pytest.fixture
def dummy_feature_matrix():
    """Returns a numpy array of shape (10, 37) and labels (10,)."""
    X = np.random.rand(10, 37)
    y = np.random.randint(0, 2, size=(10,))
    return X, y
