"""
vectorizer.py
-------------
Feature vectorization and normalization pipeline. Combines the 25
structural features (from ``structural.py``) and 12 metadata features
(from ``metadata.py``) into a unified 37-dimension feature vector,
applies ``StandardScaler`` normalization, and provides the end-to-end
``pdf_to_vector()`` function used by the Streamlit application.

Key responsibilities:
    1. **Combine**: merge structural + metadata dicts → 37-dim vector
    2. **Fit**:    fit a ``StandardScaler`` on training data
    3. **Transform**: normalize feature vectors for inference
    4. **Persist**: save / load the fitted scaler via ``joblib``
    5. **End-to-end**: ``pdf_to_vector(pdf_path)`` — extract + combine + scale

Usage (training time):
    from src.features.vectorizer import fit_scaler, transform
    scaler = fit_scaler(X_train)
    X_train_scaled = transform(X_train, scaler)

Usage (inference time / Streamlit app):
    from src.features.vectorizer import pdf_to_vector
    vector = pdf_to_vector("uploads/suspicious.pdf")
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.artifacts import verify_artifact_compatibility, write_artifact_metadata
from src.config import (
    DATA_REPORTS_DIR,
    FEATURE_COLUMNS,
    MODELS_DIR,
    SPLITS_DIR,
    SPLIT_SCHEMA_VERSION,
)
from src.features.metadata import extract_metadata_features
from src.features.structural import extract_structural_features
from src.utils.logger import get_logger
from src.experiment import create_experiment_identity

logger = get_logger(__name__)

# Path where the fitted scaler is persisted
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Path for the benign baseline statistics (used by LLM analyzer later)
BASELINE_PATH = MODELS_DIR / "benign_baseline.pkl"

# Schema-v2 serialized train/inference feature pipeline.
FEATURE_PIPELINE_PATH = MODELS_DIR / "feature_pipeline_v2.pkl"


# ---------------------------------------------------------------------------
# Feature combination
# ---------------------------------------------------------------------------

def combine_features(
    structural: Dict[str, float],
    metadata: Dict[str, float],
) -> np.ndarray:
    """Merge structural and metadata dicts into a single 37-dim vector.

    The output column order is **guaranteed** to match
    ``config.FEATURE_COLUMNS``, which is critical for model compatibility.
    Any missing keys are filled with ``0.0``.

    Args:
        structural: 25-key dict from ``extract_structural_features``.
        metadata:   12-key dict from ``extract_metadata_features``.

    Returns:
        np.ndarray: Shape ``(37,)`` feature vector (float64).
    """
    merged = {**structural, **metadata}
    vector = np.array(
        [float(merged.get(col, 0.0)) for col in FEATURE_COLUMNS],
        dtype=np.float64,
    )
    return vector


def combine_features_df(
    structural: Dict[str, float],
    metadata: Dict[str, float],
) -> pd.DataFrame:
    """Merge structural and metadata dicts into a single-row DataFrame.

    Useful for inspection, logging, and integration with pandas pipelines.

    Args:
        structural: 25-key dict from ``extract_structural_features``.
        metadata:   12-key dict from ``extract_metadata_features``.

    Returns:
        pd.DataFrame: Single-row DataFrame with 37 columns matching
        ``config.FEATURE_COLUMNS``.
    """
    merged = {**structural, **metadata}
    row = {col: float(merged.get(col, 0.0)) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Scaler fitting, transforming, and persistence
# ---------------------------------------------------------------------------

def fit_scaler(
    X_train: np.ndarray,
    save: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    *,
    partition_name: str,
) -> StandardScaler:
    """Fit a ``StandardScaler`` on training data and optionally save it.

    Args:
        X_train:   Training feature matrix, shape ``(n_samples, 37)``.
        save:      If True (default), persist the scaler to disk.
        save_path: Custom save path. Defaults to ``models/scaler.pkl``.

    Returns:
        StandardScaler: The fitted scaler instance.
    """
    if partition_name != "train":
        raise RuntimeError("Legacy scaler fitting is permitted only on the train partition.")
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.fit_partition_ = "train"
    scaler.feature_schema_version_ = "2.0.0"

    logger.info(
        f"StandardScaler fitted on {X_train.shape[0]} samples, "
        f"{X_train.shape[1]} features"
    )

    if save:
        save_scaler(
            scaler,
            Path(save_path) if save_path else SCALER_PATH,
            partition_name=partition_name,
        )

    return scaler


def transform(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Apply the fitted scaler to transform features.

    Args:
        X:      Feature matrix, shape ``(n_samples, 37)`` or ``(37,)``.
        scaler: A fitted ``StandardScaler``. If None, loads from disk.

    Returns:
        np.ndarray: Scaled feature matrix, same shape as input.
    """
    if scaler is None:
        scaler = load_scaler()

    # Handle single-sample input
    if X.ndim == 1:
        X = X.reshape(1, -1)
        return scaler.transform(X).flatten()

    return scaler.transform(X)


def load_scaler(
    path: Optional[Union[str, Path]] = None,
) -> StandardScaler:
    """Load a fitted ``StandardScaler`` from disk.

    Args:
        path: Path to the saved scaler file.
              Defaults to ``models/scaler.pkl``.

    Returns:
        StandardScaler: The loaded scaler.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    path = Path(path) if path else SCALER_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            f"Run vectorizer.fit_scaler() on training data first."
        )

    verify_artifact_compatibility(path, create_experiment_identity())
    scaler = joblib.load(path)
    if getattr(scaler, "fit_partition_", None) != "train":
        raise RuntimeError("Scaler artifact does not prove train-only fitting.")
    logger.info(f"Scaler loaded from {path}")
    return scaler


def save_scaler(
    scaler: StandardScaler,
    path: Optional[Union[str, Path]] = None,
    *,
    partition_name: str,
) -> Path:
    """Save a fitted ``StandardScaler`` to disk.

    Args:
        scaler: The fitted scaler to save.
        path:   Custom save path. Defaults to ``models/scaler.pkl``.

    Returns:
        Path: The path where the scaler was saved.
    """
    if partition_name != "train" or getattr(scaler, "fit_partition_", None) != "train":
        raise RuntimeError("Scaler artifacts require verified train-only fitting.")
    path = Path(path) if path else SCALER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path, compress=3)
    write_artifact_metadata(
        path,
        "legacy_base_scaler",
        create_experiment_identity(),
        extra={
            "fit_partition": "train",
            "deprecated": True,
            "replacement": "FeaturePipelineV2",
        },
    )
    logger.info(f"Scaler saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Benign baseline statistics (for LLM suspicious-feature detection)
# ---------------------------------------------------------------------------

def compute_benign_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
    *,
    partition_name: str,
) -> Dict[str, Dict[str, float]]:
    """Compute per-feature mean and std for benign samples in the
    training set.

    This baseline is used by the LLM ``ThreatAnalyzer`` to identify
    features that deviate significantly (>2σ) from normal benign
    behavior.

    Args:
        X_train: Training feature matrix, shape ``(n_samples, 37)``.
        y_train: Training labels (0 = benign, 1 = malicious).
        save:    If True (default), persist to ``models/benign_baseline.pkl``.

    Returns:
        dict: ``{feature_name: {"mean": float, "std": float}}``.
    """
    if partition_name != "train":
        raise RuntimeError("Benign baseline may only be estimated from train.")
    # Select benign samples (label == 0)
    benign_mask = (y_train == 0)
    X_benign = X_train[benign_mask]

    baseline = {}
    for i, col_name in enumerate(FEATURE_COLUMNS):
        col_data = X_benign[:, i] if X_benign.ndim == 2 else X_benign[i]
        baseline[col_name] = {
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data)) if len(col_data) > 1 else 0.0,
        }

    if save:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, BASELINE_PATH, compress=3)
        write_artifact_metadata(
            BASELINE_PATH,
            "benign_train_baseline",
            create_experiment_identity(),
            extra={"fit_partition": "train"},
        )
        logger.info(
            f"Benign baseline saved to {BASELINE_PATH} "
            f"(computed from {X_benign.shape[0]} benign samples)"
        )

    return baseline


def load_benign_baseline(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """Load persisted benign baseline statistics.

    Args:
        path: Custom path. Defaults to ``models/benign_baseline.pkl``.

    Returns:
        dict: ``{feature_name: {"mean": float, "std": float}}``.

    Raises:
        FileNotFoundError: If baseline file does not exist.
    """
    path = Path(path) if path else BASELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Benign baseline not found at {path}. "
            f"Run vectorizer.compute_benign_baseline() first."
        )
    verify_artifact_compatibility(path, create_experiment_identity())
    baseline = joblib.load(path)
    logger.info(f"Benign baseline loaded from {path}")
    return baseline


# ---------------------------------------------------------------------------
# End-to-end PDF → Vector pipeline (used by Streamlit app)
# ---------------------------------------------------------------------------

def pdf_to_vector(
    pdf_path: Union[str, Path],
    scaler: Optional[StandardScaler] = None,
    return_raw: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """End-to-end pipeline: PDF file → scaled 37-dim feature vector.

    This is the primary function called by the Streamlit application
    for real-time PDF analysis.

    Steps:
        1. Extract structural features (25) from raw PDF bytes
        2. Extract metadata features (12) via PyPDF2
        3. Combine into 37-dim vector (column order = ``config.FEATURE_COLUMNS``)
        4. Apply ``StandardScaler`` normalization
        5. Return the scaled vector

    Args:
        pdf_path:   Path to the PDF file.
        scaler:     A fitted ``StandardScaler``. If None, loads from disk.
        return_raw: If True, also return the raw (unscaled) feature dict.

    Returns:
        np.ndarray: Scaled feature vector of shape ``(37,)``.
        If ``return_raw=True``, returns ``(scaled_vector, raw_features_dict)``.

    Example:
        >>> vec = pdf_to_vector("uploads/invoice.pdf")
        >>> vec.shape
        (37,)
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Running full extraction pipeline on {pdf_path.name}")

    # Step 1 & 2: Extract features
    structural = extract_structural_features(pdf_path)
    metadata = extract_metadata_features(pdf_path)

    # Step 3: Combine
    raw_vector = combine_features(structural, metadata)
    raw_features = {**structural, **metadata}

    # Step 4: Scale
    scaled_vector = transform(raw_vector, scaler)

    logger.info(f"Pipeline complete for {pdf_path.name}")

    if return_raw:
        return scaled_vector, raw_features
    return scaled_vector


def extract_features_dict(
    pdf_path: Union[str, Path],
) -> Dict[str, float]:
    """Extract raw (unscaled) features as a dictionary.

    Convenience function for inspection, logging, and LLM prompt
    construction. Does NOT apply scaling.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        dict: All 37 features as ``{feature_name: value}``.
    """
    structural = extract_structural_features(pdf_path)
    metadata = extract_metadata_features(pdf_path)
    merged = {**structural, **metadata}

    # Ensure all FEATURE_COLUMNS are present
    return {col: float(merged.get(col, 0.0)) for col in FEATURE_COLUMNS}


def extract_features_record(
    pdf_path: Union[str, Path],
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Extract base schema values and explicit health diagnostics.

    Unlike the legacy dictionary-only API, parser failures and bounded-resource
    exits are observable. They can drive abstention and can only enter a model if
    the serialized training pipeline declares the same status columns.
    """
    from src.features.metadata import extract_metadata_features_with_status
    from src.features.structural import extract_structural_features_with_status

    structural = extract_structural_features_with_status(pdf_path)
    metadata = extract_metadata_features_with_status(pdf_path)
    merged = {**structural.features, **metadata.features}
    base = {name: float(merged.get(name, 0.0)) for name in FEATURE_COLUMNS}
    disagreement = abs(base["obj_count"] - base["obj_count_total"]) / (
        base["obj_count"] + base["obj_count_total"] + 1.0
    )
    structural_status = structural.status.to_dict()
    metadata_status = metadata.status.to_dict()
    numeric_status = {
        name: float(
            structural.status.numeric_features().get(name, 0.0)
            or metadata.status.numeric_features().get(name, 0.0)
        )
        for name in structural.status.numeric_features()
    }
    numeric_status["parser_disagreement"] = float(disagreement > 0.25)
    critical = any(
        numeric_status[name] > 0
        for name in (
            "parse_failure", "extraction_timeout", "file_too_large",
            "extraction_limit_reached", "invalid_header", "invalid_eof",
            "truncated_structure",
        )
    )
    diagnostics: Dict[str, object] = {
        **numeric_status,
        "abstain_recommended": critical,
        "structural_status": structural_status,
        "metadata_status": metadata_status,
        "canonicalization": (
            structural.canonicalization.to_dict()
            if structural.canonicalization is not None else None
        ),
        "object_stream_inspection": (
            structural.object_stream_inspection.to_dict()
            if structural.object_stream_inspection is not None else None
        ),
    }
    return {**base, **numeric_status}, diagnostics


def load_feature_pipeline(
    path: Optional[Union[str, Path]] = None,
):
    """Load the checksummed schema-v2 pipeline for scoring."""
    from src.features.pipeline import FeaturePipelineV2

    return FeaturePipelineV2.load(
        Path(path) if path else FEATURE_PIPELINE_PATH,
        split_manifest_path=SPLITS_DIR / SPLIT_SCHEMA_VERSION / "split_manifest.json",
        dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
        transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
    )


def pdf_to_pipeline_vector(
    pdf_path: Union[str, Path],
    *,
    pipeline=None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, object]]:
    """Run the exact serialized training feature pipeline for one PDF."""
    active_pipeline = pipeline or load_feature_pipeline()
    record, diagnostics = extract_features_record(pdf_path)
    vector = active_pipeline.transform_record(record).to_numpy(dtype=np.float32)[0]
    raw = {name: float(record[name]) for name in FEATURE_COLUMNS}
    return vector, raw, diagnostics


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.features.vectorizer <pdf_path>")
        print("       Extracts and scales features from a PDF file.")
        sys.exit(1)

    path = sys.argv[1]

    # If scaler exists, show scaled; otherwise show raw
    try:
        vec, raw = pdf_to_vector(path, return_raw=True)
        print("=== Raw Features ===")
        print(json.dumps(
            {k: round(v, 4) for k, v in
             zip(FEATURE_COLUMNS, combine_features(
                 extract_structural_features(path),
                 extract_metadata_features(path),
             ))},
            indent=2,
        ))
        print(f"\n=== Scaled Vector (shape: {vec.shape}) ===")
        print(vec)
    except FileNotFoundError as exc:
        print(f"Scaler not found — showing raw features only.\n{exc}\n")
        raw = extract_features_dict(path)
        print(json.dumps({k: round(v, 4) for k, v in raw.items()}, indent=2))
