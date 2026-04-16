"""
loader.py
---------
This file is responsible for loading the dataset into a usable format.
It handles file loading, column type coercion, schema validation, 
and separating features from the target label.
"""
import pandas as pd
from typing import Tuple
from src.config import RAW_DATA_DIR, FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(path=None) -> pd.DataFrame:
    """Read CSV, validate columns, and coerce types."""
    if path is None:
        path = RAW_DATA_DIR / "pdfmal2022.csv"
        
    logger.info(f"Loading dataset from {path}...")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Validating column names against config.FEATURE_COLUMNS
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing expected feature columns: {missing_features}")
            
        # Type coercion: cast all feature columns to float64
        feature_cols_present = [col for col in FEATURE_COLUMNS if col in df.columns]
        for col in feature_cols_present:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
        logger.info(f"Null counts total: {df.isnull().sum().sum()}")
        
        label_col = 'Class' if 'Class' in df.columns else df.columns[-1]
        logger.info(f"Class distribution:\n{df[label_col].value_counts(normalize=True)}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and label."""
    label_col = 'Class' if 'Class' in df.columns else df.columns[-1]
    
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    X = df[feature_cols]
    y = df[label_col]
    
    logger.info(f"Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")
    return X, y
