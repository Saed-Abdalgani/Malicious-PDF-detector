"""
models — Malicious PDF Detection Models Package
=================================================

Provides legacy compatibility APIs plus the versioned Phase 4–7 workflow.

Modules:
    baseline:  Tree-based models (RF, XGBoost, LightGBM) with GridSearchCV
    mlp:       PyTorch MLP classifier with early stopping
    trainer:   Compatibility facade that delegates to the gated Phase 4 runner
    evaluator: Legacy candidate/debug metrics, never final sealed-test evidence
    phase4:    Required fair training, calibration, thresholds, and selection
    phase5:    One-shot sealed-test evaluation and error analysis
    phase6:    Deep multi-method explainability without reopening sealed test
    phase7:    Safe inert-PDF adversarial and defense evaluation
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
from src.models.bundle import Phase4ModelBundle
from src.models.metrics import metric_definitions, metric_report, select_validation_thresholds
from src.models.tabular_transformer import AsymmetricFocalLoss, FTTransformer

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
    # Remediated Phase 4-5 APIs
    "Phase4ModelBundle",
    "FTTransformer",
    "AsymmetricFocalLoss",
    "metric_definitions",
    "metric_report",
    "select_validation_thresholds",
]
