"""
features — PDF Feature Extraction Package
==========================================

Provides three modules for extracting and vectorizing features from
PDF files for malicious PDF detection:

Modules:
    structural: Extract 25 structural features via byte-level regex
    metadata:   Extract 12 metadata features via PyPDF2
    vectorizer: Combine, normalize, and persist feature pipelines
"""

from src.features.structural import extract_structural_features
from src.features.metadata import extract_metadata_features
from src.features.vectorizer import (
    combine_features,
    combine_features_df,
    extract_features_dict,
    fit_scaler,
    load_scaler,
    pdf_to_vector,
    transform,
)

__all__ = [
    "extract_structural_features",
    "extract_metadata_features",
    "combine_features",
    "combine_features_df",
    "extract_features_dict",
    "fit_scaler",
    "load_scaler",
    "pdf_to_vector",
    "transform",
]
