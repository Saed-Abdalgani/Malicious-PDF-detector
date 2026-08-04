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
    "extract_features_record",
    "load_feature_pipeline",
    "pdf_to_pipeline_vector",
]


def __getattr__(name):
    """Load public helpers lazily.

    Lazy imports keep the standalone feature schema importable by ``src.config``
    without recursively importing ``vectorizer`` back into a partially initialized
    configuration module.
    """
    if name == "extract_structural_features":
        from src.features.structural import extract_structural_features

        return extract_structural_features
    if name == "extract_metadata_features":
        from src.features.metadata import extract_metadata_features

        return extract_metadata_features
    if name in {
        "combine_features",
        "combine_features_df",
        "extract_features_dict",
        "fit_scaler",
        "load_scaler",
        "pdf_to_vector",
        "transform",
        "extract_features_record",
        "load_feature_pipeline",
        "pdf_to_pipeline_vector",
    }:
        from src.features import vectorizer

        return getattr(vectorizer, name)
    raise AttributeError(name)
