"""
splitter.py
-----------
This file is responsible for splitting the data into subsets.
It creates stratified train, validation, and test splits. It optionally 
applies Synthetic Minority Over-sampling Technique (SMOTE) to the 
training data only to correct class imbalances.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train (70%), val (15%), test (15%)."""
    logger.info("Starting stratified split...")
    label_col = 'Class' if 'Class' in df.columns else df.columns[-1]
    
    # First split: 85% train_val / 15% test
    train_val, test = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        stratify=df[label_col], 
        random_state=RANDOM_SEED
    )
    
    # Second split: train_val -> 82.35% train / 17.65% val (of the 85% remaining)
    # This approximately results in 70/15 overall splits
    val_proportion = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.15 / 0.85 = 0.17647...
    
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        stratify=train_val[label_col],
        random_state=RANDOM_SEED
    )
    
    logger.info(f"Train split size: {len(train)}")
    logger.info(f"Validation split size: {len(val)}")
    logger.info(f"Test split size: {len(test)}")
    
    return train, val, test

def apply_smote(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply SMOTE ONLY to training set."""
    logger.info("Applying SMOTE to balance the training set...")
    label_col = 'Class' if 'Class' in train_df.columns else train_df.columns[-1]
    
    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col]
    
    logger.info(f"Class distribution before SMOTE:\n{y_train.value_counts()}")
    
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Class distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")
    
    train_df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=label_col)], axis=1)
    
    return train_df_balanced

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Save processed splits to disk."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"Saved splits to {PROCESSED_DATA_DIR}")
