"""
evaluator.py
------------
Comprehensive model evaluation, comparison, and report generation.

Provides three main capabilities:
    1. **evaluate_model** — Compute all metrics (Accuracy, F1, Precision,
       Recall, AUC-ROC) + confusion matrix for a single model.
    2. **compare_models** — Side-by-side comparison DataFrame sorted by F1.
    3. **generate_report** — Publication-quality confusion matrices, overlaid
       ROC curves, Precision-Recall curves, and feature importance charts,
       all using the project's dark visual theme.

Usage:
    from src.models.evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(models, X_test, y_test)
    comparison = evaluator.compare_models(results)
    evaluator.generate_report(results)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    RESULTS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    is_pytorch: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model on the test set.

    Computes Accuracy, F1, Precision, Recall, AUC-ROC, confusion matrix,
    and classification report.

    Args:
        model: Trained model (BaselineModel or PyTorch Module).
        X_test: Test feature matrix.
        y_test: Test labels.
        model_name: Display name for logging and reports.
        is_pytorch: If True, treats ``model`` as a ``nn.Module``.

    Returns:
        dict: Comprehensive evaluation results with keys:
            - ``model_name``, ``accuracy``, ``f1``, ``precision``,
              ``recall``, ``auc_roc``, ``confusion_matrix``,
              ``classification_report``, ``y_pred``, ``y_prob``,
              ``fpr``, ``tpr``, ``pr_precision``, ``pr_recall``,
              ``inference_time_ms``.
    """
    logger.info(f"Evaluating {model_name}...")

    start_time = time.perf_counter()

    if is_pytorch:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test.astype(np.float32))
            logits = model(X_tensor)
            y_prob = torch.sigmoid(logits).squeeze().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
    else:
        # Scikit-learn compatible model (BaselineModel)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]  # probability of positive class
        else:
            y_prob = model.predict(X_test).astype(float)

        y_pred = model.predict(X_test)

    inference_time = (time.perf_counter() - start_time) * 1000  # ms

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_roc = 0.0

    # ROC curve data
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
    except ValueError:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    # Precision-Recall curve data
    try:
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    except ValueError:
        pr_prec, pr_rec = np.array([1, 0]), np.array([0, 1])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    cls_report = classification_report(
        y_test, y_pred,
        target_names=["Benign", "Malicious"],
        zero_division=0,
    )

    result = {
        "model_name": model_name,
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc_roc": float(auc_roc),
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_prec,
        "pr_recall": pr_rec,
        "inference_time_ms": inference_time,
    }

    logger.info(
        f"  {model_name}: Acc={acc:.4f} | F1={f1:.4f} | "
        f"Prec={prec:.4f} | Rec={rec:.4f} | AUC={auc_roc:.4f} | "
        f"Inference: {inference_time:.1f}ms"
    )

    return result


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    results: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Create a comparison table of all evaluated models.

    Args:
        results: List of evaluation result dicts from ``evaluate_model``.
        save_path: Optional CSV save path. Defaults to
                   ``reports/results/model_comparison.csv``.

    Returns:
        pd.DataFrame: Comparison table sorted by F1-score descending.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "model_comparison.csv"

    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "F1-Score": round(r["f1"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "AUC-ROC": round(r["auc_roc"], 4),
            "Inference (ms)": round(r["inference_time_ms"], 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Model comparison saved to {save_path}")

    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: List[Dict[str, Any]],
    feature_importances: Optional[Dict[str, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Generate comprehensive evaluation visualizations.

    Creates:
        - Confusion matrix heatmaps for each model
        - Overlaid ROC curves
        - Precision-Recall curves
        - Feature importance chart (if provided)
        - Model comparison CSV

    Args:
        results: List of evaluation result dicts.
        feature_importances: ``{model_name: importance_array}`` for tree models.
        feature_names: Feature column names for importance chart.
        save_dir: Directory for saving figures. Defaults to ``reports/figures/``.

    Returns:
        dict: ``{figure_name: saved_path}`` mapping.
    """
    if save_dir is None:
        save_dir = FIGURES_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = FEATURE_COLUMNS

    saved_files: Dict[str, Path] = {}

    # Import visualization module
    from src.utils.visualization import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_feature_importance,
    )

    logger.info("Generating evaluation report figures...")

    # --- 1. Confusion matrices ---
    for r in results:
        cm_path = save_dir / f"cm_{r['model_name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_confusion_matrix(
            r["confusion_matrix"],
            r["model_name"],
            save_path=str(cm_path),
        )
        saved_files[f"cm_{r['model_name']}"] = cm_path

    # --- 2. Overlaid ROC curves ---
    roc_data = {}
    for r in results:
        roc_data[r["model_name"]] = {
            "fpr": r["fpr"],
            "tpr": r["tpr"],
            "auc": r["auc_roc"],
        }

    roc_path = save_dir / "roc_curves.png"
    plot_roc_curves(roc_data, save_path=str(roc_path))
    saved_files["roc_curves"] = roc_path

    # --- 3. Precision-Recall curves ---
    try:
        import matplotlib.pyplot as plt
        from src.utils.visualization import _apply_dark_theme, COLORS, GRADIENT_PALETTE, _save_figure

        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(10, 8))

        for idx, r in enumerate(results):
            color = GRADIENT_PALETTE[idx % len(GRADIENT_PALETTE)]
            pr_auc = auc(r["pr_recall"], r["pr_precision"])
            ax.plot(
                r["pr_recall"], r["pr_precision"],
                color=color, linewidth=2.5, alpha=0.9,
                label=f'{r["model_name"]} (AUC={pr_auc:.4f})',
            )

        # Baseline
        baseline_prec = sum(1 for r in results for y in [r.get("y_pred")] if y is not None) / max(len(results), 1)
        ax.axhline(
            y=0.5, color=COLORS["text_muted"],
            linewidth=1, linestyle="--", alpha=0.5,
            label="Baseline",
        )

        ax.set_title("Precision-Recall Curves", pad=15)
        ax.set_xlabel("Recall", labelpad=10)
        ax.set_ylabel("Precision", labelpad=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower left", fontsize=11)

        plt.tight_layout()
        pr_path = save_dir / "pr_curves.png"
        _save_figure(fig, str(pr_path))
        plt.close(fig)
        saved_files["pr_curves"] = pr_path
    except Exception as e:
        logger.warning(f"PR curve generation failed: {e}")

    # --- 4. Feature importance chart ---
    if feature_importances:
        for model_name, importances in feature_importances.items():
            imp_path = save_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
            plot_feature_importance(
                importances,
                feature_names[:len(importances)],
                save_path=str(imp_path),
                title=f"Feature Importance - {model_name}",
            )
            saved_files[f"importance_{model_name}"] = imp_path

    # --- 5. Model comparison CSV ---
    compare_models(results)

    logger.info(f"Report generation complete: {len(saved_files)} figures saved")
    return saved_files


# ---------------------------------------------------------------------------
# ModelEvaluator convenience class
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Convenience class wrapping evaluation, comparison, and reporting.

    Example:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all(models, X_test, y_test)
        comparison = evaluator.get_comparison()
        evaluator.generate_full_report()
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.comparison_df: Optional[pd.DataFrame] = None

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        display_names: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate all models on the test set.

        Args:
            models: ``{model_key: model_object}`` dict.
            X_test: Test features.
            y_test: Test labels.
            display_names: Optional ``{model_key: display_name}`` mapping.

        Returns:
            list: List of evaluation result dicts.
        """
        if display_names is None:
            display_names = {
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM",
                "mlp": "MLP (PyTorch)",
            }

        self.results = []

        for key, model in models.items():
            name = display_names.get(key, key)
            is_pytorch = key == "mlp" or isinstance(model, torch.nn.Module)

            result = evaluate_model(
                model, X_test, y_test, name, is_pytorch=is_pytorch
            )
            self.results.append(result)

        return self.results

    def get_comparison(self) -> pd.DataFrame:
        """Get the comparison DataFrame."""
        self.comparison_df = compare_models(self.results)
        return self.comparison_df

    def generate_full_report(
        self,
        feature_importances: Optional[Dict[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Generate the full visual report."""
        return generate_report(
            self.results,
            feature_importances=feature_importances,
            feature_names=feature_names,
        )

    def get_best_model(self, metric: str = "f1") -> Dict[str, Any]:
        """Get the result dict for the best-performing model.

        Args:
            metric: Metric to rank by. Default 'f1'.

        Returns:
            dict: Evaluation results for the top model.
        """
        if not self.results:
            raise RuntimeError("No evaluation results. Run evaluate_all first.")
        return max(self.results, key=lambda r: r.get(metric, 0))

    def print_summary(self):
        """Print a formatted summary to the console."""
        if self.comparison_df is None:
            self.get_comparison()

        print("\n" + "=" * 72)
        print("  MODEL EVALUATION SUMMARY")
        print("=" * 72)
        print(self.comparison_df.to_string(index=False))
        print("=" * 72)

        best = self.get_best_model()
        print(f"\n  BEST MODEL: {best['model_name']}")
        print(f"  F1-Score:   {best['f1']:.4f}")
        print(f"  AUC-ROC:    {best['auc_roc']:.4f}")
        print(f"  Accuracy:   {best['accuracy']:.4f}")

        print("\n  Classification Reports:")
        print("-" * 72)
        for r in self.results:
            print(f"\n  --- {r['model_name']} ---")
            print(r["classification_report"])


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from src.models.baseline import BaselineModel
    from src.config import RANDOM_SEED

    print("Testing evaluator on synthetic data...")

    X, y = make_classification(
        n_samples=500, n_features=37, n_informative=15,
        n_redundant=5, random_state=RANDOM_SEED,
    )

    # Train a quick model
    model = BaselineModel("random_forest")
    model.train(X[:400], y[:400])

    # Evaluate
    result = evaluate_model(model, X[400:], y[400:], "Random Forest")

    print(f"\nAccuracy: {result['accuracy']:.4f}")
    print(f"F1:       {result['f1']:.4f}")
    print(f"AUC-ROC:  {result['auc_roc']:.4f}")
    print("\nClassification Report:")
    print(result["classification_report"])
