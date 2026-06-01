import os
import sys
import logging
from src.config import *
from src.data.cleaner import clean_pipeline
from src.data.loader import load_dataset
from src.data.splitter import stratified_split, apply_smote, save_splits
from src.models.trainer import TrainingPipeline
from src.models.evaluator import compare_models, generate_report
from src.optimization.quantizer import ModelQuantizer
import torch
from src.models.mlp import MaliciousPDFClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase9.1")

def run_pipeline():
    logger.info("Starting pipeline validation...")
    
    # 1. Clean & Split
    logger.info("Loading and cleaning data...")
    df = load_dataset()
    df_clean = clean_pipeline(df)
    train, val, test = stratified_split(df_clean)
    train_balanced = apply_smote(train)
    save_splits(train_balanced, val, test)
    
    # 2. Train Models
    logger.info("Training models...")
    pipeline = TrainingPipeline()
    pipeline.train_all_models()
    
    # 3. Quantize Model
    logger.info("Quantizing MLP...")
    mlp_path = TRAINED_MODELS_DIR / "mlp_best.pt"
    if mlp_path.exists():
        model = MaliciousPDFClassifier()
        model.load_state_dict(torch.load(mlp_path, weights_only=True))
        model.eval()
        quantizer = ModelQuantizer(model)
        quantized_model = quantizer.dynamic_quantize()
        quantizer.save_quantized(quantized_model, QUANTIZED_MODELS_DIR / "mlp_quantized_dynamic.pt")
    else:
        logger.error(f"MLP model not found at {mlp_path}")
        sys.exit(1)
        
    logger.info("Verifying files...")
    expected_files = [
        TRAINED_MODELS_DIR / "rf_best.pkl",
        TRAINED_MODELS_DIR / "xgb_best.pkl",
        TRAINED_MODELS_DIR / "lgbm_best.pkl",
        TRAINED_MODELS_DIR / "mlp_best.pt",
        QUANTIZED_MODELS_DIR / "mlp_quantized_dynamic.pt"
    ]
    for f in expected_files:
        if not f.exists():
            logger.error(f"Missing model file: {f}")
            sys.exit(1)
            
    logger.info("Phase 9.1 validation successful!")

if __name__ == "__main__":
    run_pipeline()
