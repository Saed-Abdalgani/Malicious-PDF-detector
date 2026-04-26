"""
trainer.py
----------
Unified training pipeline that orchestrates the complete model training
workflow: data loading, preprocessing, training all four models (Random
Forest, XGBoost, LightGBM, MLP), saving checkpoints, and collecting
timing statistics.

Provides both a ``TrainingPipeline`` class for programmatic use and a
standalone ``__main__`` entrypoint.

Usage:
    from src.models.trainer import TrainingPipeline
    pipeline = TrainingPipeline()
    results = pipeline.train_all_models()

    # Or from CLI:
    python -m src.models.trainer
"""

import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(func):
    """Decorator that logs the wall-clock execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}...")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{func.__name__} completed in {elapsed:.2f}s"
        )
        return result
    return wrapper


# ---------------------------------------------------------------------------
# TrainingPipeline
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Orchestrates the complete model training workflow.

    Steps:
        1. Load train/val/test splits from ``data/processed/``.
        2. Train Random Forest, XGBoost, LightGBM (via BaselineModel).
        3. Train PyTorch MLP with early stopping.
        4. Save all model checkpoints to ``models/trained/``.
        5. Collect timing and performance statistics.

    Attributes:
        X_train, y_train: Training data arrays.
        X_val, y_val: Validation data arrays.
        X_test, y_test: Test data arrays.
        training_results: Dict of per-model training metadata.
        feature_names: List of feature column names.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the pipeline by loading data splits.

        Args:
            data_dir: Path to processed data directory.
                      Defaults to ``data/processed/``.
            models_dir: Path to save trained models.
                        Defaults to ``models/trained/``.
        """
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DATA_DIR
        self.models_dir = Path(models_dir) if models_dir else TRAINED_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.training_results: Dict[str, Dict[str, Any]] = {}
        self.feature_names: List[str] = []

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self._load_data()

    def _load_data(self):
        """Load train/val/test splits from CSV files."""
        train_path = self.data_dir / "train.csv"
        val_path = self.data_dir / "val.csv"
        test_path = self.data_dir / "test.csv"
        cleaned_path = self.data_dir / "cleaned.csv"

        if train_path.exists() and val_path.exists() and test_path.exists():
            logger.info("Loading train/val/test splits from CSV...")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
        elif cleaned_path.exists():
            logger.warning(
                "Split files not found. Creating splits from cleaned.csv..."
            )
            from src.data.splitter import stratified_split
            cleaned_df = pd.read_csv(cleaned_path)
            train_df, val_df, test_df = stratified_split(cleaned_df)
        else:
            raise FileNotFoundError(
                f"No data found in {self.data_dir}. "
                f"Run Phase 1 data preprocessing first: "
                f"python -m src.data.downloader"
            )

        # Detect label column
        label_col = "Class" if "Class" in train_df.columns else train_df.columns[-1]

        # Get feature columns that are actually present
        self.feature_names = [
            c for c in FEATURE_COLUMNS if c in train_df.columns
        ]

        # Filter out outlier flag columns (from cleaner.py)
        data_cols = self.feature_names + [label_col]

        self.X_train = train_df[self.feature_names].values.astype(np.float64)
        self.y_train = train_df[label_col].values.astype(np.int64)
        self.X_val = val_df[self.feature_names].values.astype(np.float64)
        self.y_val = val_df[label_col].values.astype(np.int64)
        self.X_test = test_df[self.feature_names].values.astype(np.float64)
        self.y_test = test_df[label_col].values.astype(np.int64)

        logger.info(
            f"Data loaded — Train: {self.X_train.shape}, "
            f"Val: {self.X_val.shape}, "
            f"Test: {self.X_test.shape}"
        )
        logger.info(
            f"Features: {len(self.feature_names)}, "
            f"Label column: '{label_col}'"
        )
        logger.info(
            f"Train class distribution: "
            f"{dict(zip(*np.unique(self.y_train, return_counts=True)))}"
        )

    # -------------------------------------------------------------------
    # Individual model training methods
    # -------------------------------------------------------------------

    @timed
    def train_random_forest(self) -> Dict[str, Any]:
        """Train Random Forest with GridSearchCV."""
        from src.models.baseline import BaselineModel

        model = BaselineModel("random_forest")
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        save_path = model.save(self.models_dir / "rf_best.pkl")

        result = {
            "model_type": "random_forest",
            "display_name": "Random Forest",
            "best_params": model.best_params_,
            "best_cv_score": model.cv_results_.get("best_score", 0),
            "training_time_sec": model.training_time_sec,
            "save_path": str(save_path),
            "model_object": model,
        }
        self.training_results["random_forest"] = result
        return result

    @timed
    def train_xgboost(self) -> Dict[str, Any]:
        """Train XGBoost with GridSearchCV."""
        from src.models.baseline import BaselineModel

        model = BaselineModel("xgboost")
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        save_path = model.save(self.models_dir / "xgb_best.pkl")

        result = {
            "model_type": "xgboost",
            "display_name": "XGBoost",
            "best_params": model.best_params_,
            "best_cv_score": model.cv_results_.get("best_score", 0),
            "training_time_sec": model.training_time_sec,
            "save_path": str(save_path),
            "model_object": model,
        }
        self.training_results["xgboost"] = result
        return result

    @timed
    def train_lightgbm(self) -> Dict[str, Any]:
        """Train LightGBM with GridSearchCV."""
        from src.models.baseline import BaselineModel

        model = BaselineModel("lightgbm")
        model.train(self.X_train, self.y_train, self.X_val, self.y_val)

        save_path = model.save(self.models_dir / "lgbm_best.pkl")

        result = {
            "model_type": "lightgbm",
            "display_name": "LightGBM",
            "best_params": model.best_params_,
            "best_cv_score": model.cv_results_.get("best_score", 0),
            "training_time_sec": model.training_time_sec,
            "save_path": str(save_path),
            "model_object": model,
        }
        self.training_results["lightgbm"] = result
        return result

    @timed
    def train_mlp(self, max_epochs: int = 100, batch_size: int = 64) -> Dict[str, Any]:
        """Train PyTorch MLP with early stopping."""
        from src.models.mlp import (
            MaliciousPDFClassifier,
            create_data_loaders,
            train_mlp,
        )

        model = MaliciousPDFClassifier(
            input_dim=len(self.feature_names)
        )

        train_loader, val_loader = create_data_loaders(
            self.X_train.astype(np.float32),
            self.y_train.astype(np.float32),
            self.X_val.astype(np.float32),
            self.y_val.astype(np.float32),
            batch_size=batch_size,
        )

        save_path = self.models_dir / "mlp_best.pt"
        history = train_mlp(
            model, train_loader, val_loader,
            max_epochs=max_epochs,
            save_path=save_path,
        )

        result = {
            "model_type": "mlp",
            "display_name": "MLP (PyTorch)",
            "training_time_sec": history["training_time_sec"],
            "epochs_trained": history["epochs_trained"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "save_path": str(save_path),
            "history": history,
            "model_object": model,
        }
        self.training_results["mlp"] = result
        return result

    # -------------------------------------------------------------------
    # Orchestration
    # -------------------------------------------------------------------

    def train_all_models(
        self,
        include_mlp: bool = True,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 64,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all four models sequentially.

        Args:
            include_mlp: Whether to train the MLP. Default True.
            mlp_epochs: Max MLP training epochs. Default 100.
            mlp_batch_size: MLP batch size. Default 64.

        Returns:
            dict: Per-model training results.
        """
        total_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)

        # Tree-based models
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()

        # Neural network
        if include_mlp:
            self.train_mlp(
                max_epochs=mlp_epochs,
                batch_size=mlp_batch_size,
            )

        total_time = time.perf_counter() - total_start
        logger.info("=" * 60)
        logger.info(f"ALL MODELS TRAINED in {total_time:.2f}s")
        logger.info("=" * 60)

        # Print summary
        for name, result in self.training_results.items():
            logger.info(
                f"  {result['display_name']}: "
                f"{result['training_time_sec']:.2f}s | "
                f"saved: {result['save_path']}"
            )

        return self.training_results

    # -------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------

    def load_all_models(self) -> Dict[str, Any]:
        """Load all saved model files from the models directory.

        Returns:
            dict: ``{model_name: model_object}`` mapping.
        """
        from src.models.baseline import BaselineModel
        from src.models.mlp import load_mlp

        models = {}

        # Tree-based models
        for fname, name in [
            ("rf_best.pkl", "random_forest"),
            ("xgb_best.pkl", "xgboost"),
            ("lgbm_best.pkl", "lightgbm"),
        ]:
            path = self.models_dir / fname
            if path.exists():
                models[name] = BaselineModel.load(path)
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"Model file not found: {path}")

        # MLP
        mlp_path = self.models_dir / "mlp_best.pt"
        if mlp_path.exists():
            models["mlp"] = load_mlp(mlp_path, input_dim=len(self.feature_names))
            logger.info(f"Loaded MLP from {mlp_path}")
        else:
            logger.warning(f"MLP checkpoint not found: {mlp_path}")

        return models

    def get_training_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all training results.

        Returns:
            pd.DataFrame: Training summary table.
        """
        rows = []
        for name, result in self.training_results.items():
            row = {
                "Model": result["display_name"],
                "Training Time (s)": round(result["training_time_sec"], 2),
                "Save Path": Path(result["save_path"]).name,
            }
            if "best_cv_score" in result:
                row["Best CV F1"] = round(result["best_cv_score"], 4)
            if "epochs_trained" in result:
                row["Epochs"] = result["epochs_trained"]
                row["Best Epoch"] = result["best_epoch"] + 1
            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.train_all_models()

    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    print(pipeline.get_training_summary().to_string(index=False))
