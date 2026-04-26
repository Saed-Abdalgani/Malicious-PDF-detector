"""
baseline.py
-----------
Tree-based baseline models for malicious PDF classification.

Provides a unified ``BaselineModel`` wrapper around Random Forest,
XGBoost, and LightGBM classifiers with integrated GridSearchCV
hyperparameter tuning, cross-validated training, prediction, and
model persistence.

Key features:
    - Unified API: ``train()``, ``predict()``, ``predict_proba()``,
      ``save()``, ``load()`` across all three model types.
    - 5-fold stratified cross-validation via ``GridSearchCV``.
    - Full CPU parallelism with ``n_jobs=-1``.
    - Hyperparameter grids sourced from ``config.MODEL_CONFIGS``.
    - Feature importance extraction for tree-based explainability.
    - Compressed model persistence via ``joblib`` (compression=3).

Usage:
    from src.models.baseline import BaselineModel
    model = BaselineModel("random_forest")
    model.train(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)
    model.save("models/trained/rf_best.pkl")
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import (
    MODEL_CONFIGS,
    RANDOM_SEED,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _create_estimator(model_type: str, fixed_params: Optional[Dict] = None):
    """Instantiate a base estimator for the given model type.

    Args:
        model_type: One of 'random_forest', 'xgboost', 'lightgbm'.
        fixed_params: Non-tunable parameters to pass to the constructor.

    Returns:
        A scikit-learn compatible estimator instance.
    """
    if fixed_params is None:
        fixed_params = {}

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **fixed_params,
        )

    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        # Defaults that can be overridden by fixed_params
        xgb_defaults = dict(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
        # fixed_params from config override defaults
        xgb_defaults.update(fixed_params)
        return XGBClassifier(**xgb_defaults)

    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            **fixed_params,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: random_forest, xgboost, lightgbm"
        )


def _get_param_grid(model_type: str) -> Dict[str, List]:
    """Extract the tunable hyperparameter grid from config.

    Separates fixed scalar values (e.g. ``tree_method='hist'``) from
    tunable list values (e.g. ``n_estimators=[100, 300]``).

    Returns:
        (param_grid, fixed_params) tuple.
    """
    raw_config = MODEL_CONFIGS.get(model_type, {})
    param_grid = {}
    fixed_params = {}

    for key, value in raw_config.items():
        if isinstance(value, list):
            param_grid[key] = value
        else:
            fixed_params[key] = value

    return param_grid, fixed_params


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


# ---------------------------------------------------------------------------
# BaselineModel class
# ---------------------------------------------------------------------------

class BaselineModel:
    """Unified wrapper for tree-based classifiers with GridSearchCV tuning.

    Attributes:
        model_type (str): One of 'random_forest', 'xgboost', 'lightgbm'.
        display_name (str): Human-readable model name.
        best_estimator_: The best fitted estimator after ``train()``.
        best_params_ (dict): Best hyperparameters found by GridSearchCV.
        cv_results_ (dict): Full cross-validation results.
        training_time_sec (float): Wall-clock training time in seconds.
        feature_importances_ (np.ndarray): Feature importances from best model.

    Example:
        >>> model = BaselineModel("xgboost")
        >>> model.train(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.save("models/trained/xgb_best.pkl")
    """

    def __init__(
        self,
        model_type: str,
        params: Optional[Dict] = None,
    ):
        """Initialize a BaselineModel.

        Args:
            model_type: One of 'random_forest', 'xgboost', 'lightgbm'.
            params: Optional override for the hyperparameter grid.
                    If None, uses ``config.MODEL_CONFIGS``.
        """
        if model_type not in MODEL_CONFIGS and model_type not in (
            "random_forest", "xgboost", "lightgbm"
        ):
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: random_forest, xgboost, lightgbm"
            )

        self.model_type = model_type
        self.display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)

        # Separate tunable grid from fixed params
        default_grid, default_fixed = _get_param_grid(model_type)
        if params is not None:
            self._param_grid = params
            self._fixed_params = {}
        else:
            self._param_grid = default_grid
            self._fixed_params = default_fixed

        # Will be populated after train()
        self.best_estimator_ = None
        self.best_params_: Dict[str, Any] = {}
        self.cv_results_: Dict[str, Any] = {}
        self.training_time_sec: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None

        logger.info(
            f"BaselineModel initialized: {self.display_name} "
            f"({len(self._param_grid)} tunable params, "
            f"{sum(len(v) for v in self._param_grid.values())} "
            f"total grid points)"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        scoring: str = "f1",
    ) -> "BaselineModel":
        """Train the model with GridSearchCV hyperparameter tuning.

        Args:
            X_train: Training feature matrix, shape ``(n_samples, n_features)``.
            y_train: Training labels, shape ``(n_samples,)``.
            X_val: Validation features (logged for info, not used in CV).
            y_val: Validation labels (logged for info).
            cv_folds: Number of cross-validation folds. Default 5.
            scoring: Scoring metric for GridSearchCV. Default 'f1'.

        Returns:
            self: The fitted BaselineModel instance.
        """
        logger.info(
            f"Training {self.display_name} with {cv_folds}-fold CV "
            f"on {X_train.shape[0]} samples..."
        )

        # Create base estimator
        estimator = _create_estimator(
            self.model_type, self._fixed_params
        )

        # Stratified K-Fold for reproducibility
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED
        )

        # Grid search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self._param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            refit=True,
            return_train_score=True,
        )

        start_time = time.perf_counter()
        grid_search.fit(X_train, y_train)
        self.training_time_sec = time.perf_counter() - start_time

        # Store results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = {
            "mean_test_score": grid_search.cv_results_["mean_test_score"],
            "std_test_score": grid_search.cv_results_["std_test_score"],
            "mean_train_score": grid_search.cv_results_["mean_train_score"],
            "params": grid_search.cv_results_["params"],
            "best_score": grid_search.best_score_,
            "best_index": grid_search.best_index_,
        }

        # Feature importances
        if hasattr(self.best_estimator_, "feature_importances_"):
            self.feature_importances_ = (
                self.best_estimator_.feature_importances_
            )

        logger.info(
            f"{self.display_name} training complete in "
            f"{self.training_time_sec:.2f}s"
        )
        logger.info(f"  Best {scoring}: {grid_search.best_score_:.4f}")
        logger.info(f"  Best params: {self.best_params_}")

        # Validation set evaluation (if provided)
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            from sklearn.metrics import accuracy_score, f1_score
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)
            logger.info(
                f"  Validation — Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.best_estimator_ is None:
            raise RuntimeError(
                "Model not trained yet. Call train() first."
            )
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability predictions.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted probabilities, shape ``(n_samples, 2)``.
        """
        if self.best_estimator_ is None:
            raise RuntimeError(
                "Model not trained yet. Call train() first."
            )
        return self.best_estimator_.predict_proba(X)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the trained model to disk via joblib.

        Args:
            path: File path. Defaults to
                  ``models/trained/{model_type}_best.pkl``.

        Returns:
            Path: The path where the model was saved.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No trained model to save.")

        if path is None:
            path = TRAINED_MODELS_DIR / f"{self.model_type}_best.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the entire BaselineModel object (includes metadata)
        save_data = {
            "model_type": self.model_type,
            "display_name": self.display_name,
            "best_estimator": self.best_estimator_,
            "best_params": self.best_params_,
            "training_time_sec": self.training_time_sec,
            "feature_importances": self.feature_importances_,
        }
        joblib.dump(save_data, path, compress=3)
        logger.info(f"{self.display_name} saved to {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineModel":
        """Load a trained model from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            BaselineModel: The loaded model instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data = joblib.load(path)

        model = cls.__new__(cls)
        model.model_type = save_data["model_type"]
        model.display_name = save_data["display_name"]
        model.best_estimator_ = save_data["best_estimator"]
        model.best_params_ = save_data["best_params"]
        model.training_time_sec = save_data["training_time_sec"]
        model.feature_importances_ = save_data["feature_importances"]
        model._param_grid = {}
        model._fixed_params = {}
        model.cv_results_ = {}

        logger.info(f"{model.display_name} loaded from {path}")
        return model

    def get_feature_importances(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get feature importances as a named dictionary.

        Args:
            feature_names: List of feature names. If None, uses
                           ``config.FEATURE_COLUMNS``.

        Returns:
            dict: ``{feature_name: importance_score}`` sorted descending.
        """
        if self.feature_importances_ is None:
            raise RuntimeError(
                "Feature importances not available. Train the model first."
            )

        if feature_names is None:
            from src.config import FEATURE_COLUMNS
            feature_names = FEATURE_COLUMNS

        imp_dict = dict(zip(feature_names, self.feature_importances_))
        return dict(sorted(imp_dict.items(), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        status = "trained" if self.best_estimator_ else "untrained"
        return f"BaselineModel(type={self.model_type}, status={status})"


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Testing BaselineModel on synthetic data...")

    X, y = make_classification(
        n_samples=500, n_features=37, n_informative=15,
        n_redundant=5, random_state=RANDOM_SEED,
    )

    for model_type in ["random_forest", "xgboost", "lightgbm"]:
        print(f"\n{'='*50}")
        print(f"Training {model_type}...")
        model = BaselineModel(model_type)
        model.train(X[:400], y[:400], X[400:], y[400:])
        preds = model.predict(X[400:])
        acc = (preds == y[400:]).mean()
        print(f"  Test accuracy: {acc:.4f}")
        print(f"  Top-3 important features: "
              f"{list(model.get_feature_importances().items())[:3]}")
