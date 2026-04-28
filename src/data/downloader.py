"""
downloader.py
-------------
This file is responsible for downloading the raw PDF malware dataset.
It handles direct downloads, fallback mechanisms (e.g., GitHub mirrors), 
hash verification, and basic dataset structure validation.
"""
import hashlib
import requests
import pandas as pd
import webbrowser
from tqdm import tqdm
from src.config import RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_NAME = "pdfmal2022.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

# Simulated URLs for the dataset
CIC_URL = "https://www.unb.ca/cic/datasets/pdfmal2022.csv"
GITHUB_URL = "https://raw.githubusercontent.com/pdfmalware/dataset/main/pdfmal2022.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/pdfmalware/pdfmal2022"

EXPECTED_SHA256 = None # Skip hash verification if None

def check_existing() -> bool:
    """Check if the dataset already exists."""
    if DATASET_PATH.exists():
        logger.info(f"Dataset already exists at {DATASET_PATH}")
        return True
    return False

def _download_file(url: str, dest_path) -> bool:
    """Helper method to download a file with a progress bar."""
    logger.info(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        return True
    except requests.RequestException as e:
        logger.warning(f"Download failed from {url}: {e}")
        return False

def download_from_cic() -> bool:
    """Direct HTTP GET from UNB CIC website."""
    logger.info("Attempting download from UNB CIC website...")
    return _download_file(CIC_URL, DATASET_PATH)

def download_from_github() -> bool:
    """Fallback to community-hosted GitHub mirrors."""
    logger.info("Attempting download from GitHub mirror...")
    return _download_file(GITHUB_URL, DATASET_PATH)

def manual_fallback():
    """Auto-open browser to Kaggle dataset page."""
    logger.error("Automated downloads failed. Falling back to manual download.")
    logger.info(f"Opening browser to: {KAGGLE_URL}")
    webbrowser.open(KAGGLE_URL)
    logger.info(f"Please save the downloaded CSV to: {DATASET_PATH}")

def verify_hash(csv_path) -> bool:
    """Verify SHA-256 hash of the downloaded dataset."""
    if not EXPECTED_SHA256:
        logger.info("No expected SHA-256 hash provided. Skipping verification.")
        return True
        
    logger.info("Verifying file hash...")
    sha256 = hashlib.sha256()
    try:
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()
        if file_hash == EXPECTED_SHA256:
            logger.info("SHA-256 hash verification passed.")
            return True
        else:
            logger.error(f"Hash mismatch! Expected {EXPECTED_SHA256}, got {file_hash}")
            return False
    except Exception as e:
        logger.error(f"Error during hash verification: {e}")
        return False

def validate_dataset(csv_path) -> bool:
    """Validate dataset structure."""
    logger.info("Validating dataset structure...")
    try:
        df = pd.read_csv(csv_path)
        
        row_count = len(df)
        col_count = len(df.columns)
        
        logger.info(f"Dataset shape: {row_count} rows, {col_count} columns")
        
        # Check row count (~10,025)
        if not (9000 <= row_count <= 11000):
            logger.warning(f"Row count {row_count} is outside expected range (~10,025)")
            
        # Check column count (37 features + 1 label)
        if col_count != 38:
            logger.error(f"Invalid column count: expected 38, got {col_count}")
            return False
            
        # Verify no fully empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.error(f"Found fully empty columns: {empty_cols}")
            return False
            
        # Check data types are numeric (excluding the label if it's text, usually 'Class')
        # We will log if many non-numeric columns exist
        non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
        if len(non_numeric) > 1:
            logger.warning(f"Found multiple non-numeric columns: {non_numeric}")
            
        logger.info("Dataset validation passed!")
        return True
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def main():
    logger.info("Starting dataset download process...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not check_existing():
        if not download_from_cic():
            if not download_from_github():
                manual_fallback()
                return

    if DATASET_PATH.exists():
        verify_hash(DATASET_PATH)
        validate_dataset(DATASET_PATH)

if __name__ == "__main__":
    main()
