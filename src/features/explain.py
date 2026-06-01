"""
explain.py
----------
SHAP-based explainability for the malicious-PDF models.

Provides global (dataset-level) and per-sample (local) feature attributions for
both the PyTorch MLP and the tree-based baselines. The per-sample attributions
are also used to **ground the LLM threat report**: instead of the LLM guessing
which features mattered, it is told the model's *actual* decision drivers (see
``src/llm/analyzer.py`` and ``src/llm/prompts.py``).

Functions
~~~~~~~~~
- ``explain_mlp``      — SHAP values for the PyTorch MLP (KernelExplainer).
- ``explain_tree``     — SHAP values for RF / XGB / LGBM (TreeExplainer).
- ``top_attributions`` — top-k drivers for a single sample, signed.
- ``global_importance``— mean |SHAP| per feature + bar figure.
- ``explain_pdf``      — convenience: PDF path -> top drivers (for the LLM).

Usage::

    python -m src.features.explain   # self-test on the bundled sample PDFs
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import FEATURE_COLUMNS, FIGURES_DIR, TRAINED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _default_background(n: int = 100, seed: int = 42) -> np.ndarray:
    """Build an approximate background drawn from the model's training domain.

    The CIC features are min-max normalized to ~[0,1], so a uniform[0,1] sample
    is a reasonable, documented stand-in when the real training matrix is not
    available locally. Pass an explicit ``background`` for exact attributions.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, len(FEATURE_COLUMNS))).astype(np.float32)


def _mlp_predict_fn(model):
    """Return a numpy-in/numpy-out P(malicious) function for SHAP."""
    import torch

    def f(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(X), dtype=torch.float32)
            return torch.sigmoid(model(t)).numpy().ravel()

    return f


def explain_mlp(
    model,
    X_explain: np.ndarray,
    background: Optional[np.ndarray] = None,
    nsamples: int = 200,
) -> np.ndarray:
    """Compute SHAP values for the MLP via KernelExplainer.

    Args:
        model: Trained ``MaliciousPDFClassifier`` (eval mode).
        X_explain: Samples to explain, shape ``(n, 37)`` (already scaled like training).
        background: Reference distribution. Defaults to uniform[0,1] approximation.
        nsamples: KernelExplainer coalition samples.

    Returns:
        np.ndarray: SHAP values, shape ``(n, 37)``.
    """
    import shap

    if background is None:
        background = _default_background()
    X_explain = np.atleast_2d(np.asarray(X_explain, dtype=np.float32))

    explainer = shap.KernelExplainer(_mlp_predict_fn(model), shap.sample(background, min(50, len(background))))
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:  # some shap versions return (n, features, outputs)
        sv = sv[..., 0]
    logger.info(f"MLP SHAP computed for {X_explain.shape[0]} sample(s).")
    return sv


def explain_tree(model, X_explain: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a tree model (RF/XGB/LGBM) via TreeExplainer.

    Args:
        model: A fitted tree estimator (or ``BaselineModel`` wrapping one).
        X_explain: Samples to explain.

    Returns:
        np.ndarray: SHAP values for the positive class, shape ``(n, 37)``.
    """
    import shap

    est = getattr(model, "model", model)  # unwrap BaselineModel if needed
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(np.atleast_2d(X_explain))
    if isinstance(sv, list):  # binary classifiers -> [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv


def top_attributions(
    shap_row: np.ndarray,
    feature_names: Optional[List[str]] = None,
    k: int = 6,
) -> List[Tuple[str, float, str]]:
    """Return the top-k signed drivers for a single sample.

    Args:
        shap_row: SHAP values for one sample, shape ``(37,)``.
        feature_names: Feature names. Defaults to ``FEATURE_COLUMNS``.
        k: Number of drivers to return.

    Returns:
        list: ``(feature, shap_value, direction)`` where direction is
              "increases" (pushes toward malicious) or "decreases".
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS
    row = np.asarray(shap_row).ravel()
    order = np.argsort(np.abs(row))[::-1][:k]
    out = []
    for i in order:
        val = float(row[i])
        direction = "increases" if val > 0 else "decreases"
        out.append((feature_names[i], val, direction))
    return out


def global_importance(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    top_n: int = 15,
):
    """Compute mean |SHAP| per feature and optionally save a bar chart.

    Args:
        shap_values: SHAP matrix, shape ``(n, 37)``.
        feature_names: Feature names. Defaults to ``FEATURE_COLUMNS``.
        save_path: Optional figure path. Defaults to
                   ``reports/figures/shap_global_importance.png``.
        top_n: Number of features to display.

    Returns:
        list: ``(feature, mean_abs_shap)`` sorted descending.
    """
    if feature_names is None:
        feature_names = FEATURE_COLUMNS
    mean_abs = np.abs(np.atleast_2d(shap_values)).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    try:
        from src.utils.visualization import plot_feature_importance

        if save_path is None:
            save_path = str(FIGURES_DIR / "shap_global_importance.png")
        plot_feature_importance(
            mean_abs, feature_names,
            save_path=save_path,
            title="Global Feature Importance (mean |SHAP|)",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SHAP importance figure skipped: {exc}")

    return [(f, float(v)) for f, v in ranking[:top_n]]


def explain_pdf(
    pdf_path,
    model=None,
    scaler=None,
    k: int = 6,
) -> List[Tuple[str, float, str]]:
    """Extract a PDF, run the model, and return the top SHAP decision drivers.

    This is the bridge used to ground the LLM threat report.

    Args:
        pdf_path: Path to a PDF file.
        model: MLP model. Loaded from disk if None.
        scaler: Fitted scaler. Loaded from disk if None.
        k: Number of drivers to return.

    Returns:
        list: ``(feature, shap_value, direction)`` tuples.
    """
    from src.features.vectorizer import load_scaler, pdf_to_vector
    from src.models.mlp import load_mlp

    if model is None:
        model = load_mlp(TRAINED_MODELS_DIR / "mlp_best.pt")
    if scaler is None:
        scaler = load_scaler()

    scaled = pdf_to_vector(pdf_path, scaler=scaler).reshape(1, -1)
    sv = explain_mlp(model, scaled)
    return top_attributions(sv[0], k=k)


if __name__ == "__main__":
    from src.config import SAMPLE_PDFS_DIR

    print("=" * 60)
    print("  SHAP explainability — self-test")
    print("=" * 60)
    for pdf in sorted(Path(SAMPLE_PDFS_DIR).glob("*.pdf")):
        drivers = explain_pdf(pdf, k=6)
        print(f"\n{pdf.name} — top decision drivers:")
        for feat, val, direction in drivers:
            print(f"  {feat:20s} SHAP={val:+.4f}  ({direction} malicious score)")
    print("\n" + "=" * 60)
