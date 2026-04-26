"""
models — Malicious PDF Detection Models Package
=================================================

Provides four modules for training, evaluating, and deploying
binary classification models for malicious PDF detection:

Modules:
    baseline:  Tree-based models (RF, XGBoost, LightGBM) with GridSearchCV
    mlp:       PyTorch MLP classifier with early stopping
    trainer:   Unified training pipeline orchestrator
    evaluator: Metrics computation, comparison, and report generation
"""

from src.models.baseline import BaselineModel
from src.models.mlp import (
    MaliciousPDFClassifier,
    PDFDataset,
    create_data_loaders,
    load_mlp,
    train_mlp,
)
from src.models.trainer import TrainingPipeline
from src.models.evaluator import (
    ModelEvaluator,
    compare_models,
    evaluate_model,
    generate_report,
)

__all__ = [
    # Baseline
    "BaselineModel",
    # MLP
    "MaliciousPDFClassifier",
    "PDFDataset",
    "create_data_loaders",
    "load_mlp",
    "train_mlp",
    # Trainer
    "TrainingPipeline",
    # Evaluator
    "ModelEvaluator",
    "compare_models",
    "evaluate_model",
    "generate_report",
]
