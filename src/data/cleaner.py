"""
cleaner.py
----------
This file is responsible for cleaning the dataset.
It performs necessary operations such as deduplication, handling of 
missing values (dropping / imputing), removing constant columns, 
and flagging outliers using IQR bounds.
"""
import pandas as pd
from src.config import PROCESSED_DATA_DIR, FEATURE_COLUMNS
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
    """Drop rows with >50% NaN and median-impute remaining."""
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Drop rows with >50% NaN values features
    threshold = len(feature_cols) * 0.5
    initial_len = len(df)
    df_clean = df.dropna(thresh=int(len(df.columns) - len(feature_cols) + threshold)).copy()
    dropped_rows = initial_len - len(df_clean)
    logger.info(f"Dropped {dropped_rows} rows with >50% missing values.")
    
    # Median-impute remaining NaN values per column
    imputation_stats = df_clean[feature_cols].isnull().sum()
    imputed_total = imputation_stats.sum()
    
    if imputed_total > 0:
        logger.info(f"Imputing {imputed_total} missing values with column medians...")
        for col in feature_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
    return df_clean

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
    """Use IQR method per feature to flag outliers."""
    logger.info("Detecting outliers using IQR method...")
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_col = f"{col}_outlier"
        
        df[outlier_col] = outliers.astype(int)
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.debug(f"Feature {col} has {outlier_count} outliers.")
            
    logger.info(f"Finished flagging outliers for {len(feature_cols)} features.")
    return df

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in sequence."""
    logger.info(f"Starting cleaning pipeline... Initial shape: {df.shape}")
    
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = remove_constant_columns(df)
    df = detect_outliers(df)
    
    logger.info(f"Cleaning pipeline finished! Final shape: {df.shape}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = PROCESSED_DATA_DIR / "cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned dataset to {cleaned_path}")
    
    return df
