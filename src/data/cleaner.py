"""Deprecated compatibility shims for the pre-remediation cleaner.

Global imputation, feature-based identity deduplication, and pre-split outlier
thresholds leak information.  Production data must go through
``src.data.validate`` and the train-only preprocessor.
"""
import pandas as pd
from src.config import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by all features."""
    initial_len = len(df)
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # We only deduplicate based on features (ignoring label if multiple labels for same features are present)
    df_clean = df.drop_duplicates(subset=feature_cols).copy()
    num_duplicates = initial_len - len(df_clean)
    
    logger.info(f"Removed {num_duplicates} duplicate rows.")
    return df_clean

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Refuse leakage-prone global imputation."""
    raise RuntimeError(
        "Global median imputation is disabled. Split first, then fit "
        "TrainOnlyPreprocessor on partition_name='train'."
    )

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance."""
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    numeric_df = df[feature_cols]
    
    variances = numeric_df.var()
    constant_cols = variances[variances == 0].index.tolist()
    
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    else:
        logger.info("No constant columns found.")
        
    return df

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows unchanged; unusual high values may be the attack signal."""
    return df.copy()

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Fail closed so legacy callers cannot bypass the versioned data layer."""
    raise RuntimeError(
        "clean_pipeline is disabled. Use validate_registered_source(), then "
        "group_temporal_split(), then TrainOnlyPreprocessor."
    )
