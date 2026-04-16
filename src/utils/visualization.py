"""
visualization.py
-----------------
Comprehensive visualization module for the Malicious PDF Detector project.
Provides a unified, dark-themed visual style across all plots including:
class distributions, feature histograms, correlation matrices, feature
importance charts, boxplots, pairplots, ROC curves, and confusion matrices.

All functions accept an optional `save_path` to persist figures to disk
and return the matplotlib Figure object for notebook inline display.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Global Style Configuration — Dark Premium Theme
# ─────────────────────────────────────────────────────────────────────

# Color palette
COLORS = {
    "malicious": "#FF4C6A",       # Vibrant red-pink
    "benign": "#00E59B",          # Electric green
    "malicious_light": "#FF8FA3", # Soft red
    "benign_light": "#66FFD0",    # Soft green
    "accent": "#7C4DFF",          # Purple accent
    "accent2": "#00B4D8",         # Cyan accent
    "bg_dark": "#0D1117",         # GitHub dark bg
    "bg_card": "#161B22",         # Card background
    "text": "#E6EDF3",            # Light text
    "text_muted": "#8B949E",      # Muted text
    "grid": "#21262D",            # Grid lines
    "gradient_start": "#7C4DFF",  # Gradient purple
    "gradient_end": "#00B4D8",    # Gradient cyan
}

# Professional gradient palette for multi-series
GRADIENT_PALETTE = [
    "#7C4DFF", "#6366F1", "#3B82F6", "#0EA5E9",
    "#14B8A6", "#22C55E", "#84CC16", "#EAB308",
    "#F97316", "#EF4444", "#EC4899", "#A855F7",
]


def _apply_dark_theme():
    """Apply a consistent dark theme to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg_dark"],
        "axes.facecolor": COLORS["bg_card"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": COLORS["bg_card"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "savefig.facecolor": COLORS["bg_dark"],
        "savefig.edgecolor": COLORS["bg_dark"],
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })

    # Try to use Inter font; fall back gracefully
    try:
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if "Inter" in available_fonts:
            plt.rcParams["font.family"] = "Inter"
        elif "Segoe UI" in available_fonts:
            plt.rcParams["font.family"] = "Segoe UI"
        else:
            plt.rcParams["font.family"] = "sans-serif"
    except Exception:
        plt.rcParams["font.family"] = "sans-serif"


def _save_figure(fig: plt.Figure, save_path: Optional[str] = None):
    """Save figure to disk if path is provided."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────
# 1. Class Distribution
# ─────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    label_col: str = "Class",
    title: str = "Class Distribution — Malicious vs Benign"
) -> plt.Figure:
    """
    Bar chart showing malicious vs benign sample counts.
    Overlays count and percentage labels on each bar with gradient fills.
    """
    _apply_dark_theme()
    logger.info("Generating class distribution plot...")

    if label_col not in df.columns:
        label_col = df.columns[-1]

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()
    labels = [str(c) for c in counts.index]
    values = counts.values

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_colors = [COLORS["benign"], COLORS["malicious"]]
    edge_colors = [COLORS["benign_light"], COLORS["malicious_light"]]

    # Use only as many colors as there are classes
    bar_colors = bar_colors[:len(labels)]
    edge_colors = edge_colors[:len(labels)]

    bars = ax.bar(
        labels, values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.5,
        width=0.55,
        alpha=0.9,
        zorder=3,
    )

    # Add count labels on bars
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color=COLORS["text"],
        )

    ax.set_title(title, pad=20, fontsize=16)
    ax.set_xlabel("Class Label", labelpad=12)
    ax.set_ylabel("Sample Count", labelpad=12)
    ax.set_ylim(0, max(values) * 1.20)

    # Subtle horizontal reference line at 50%
    ax.axhline(y=total / 2, color=COLORS["accent"], linewidth=0.8,
               linestyle="--", alpha=0.5, label=f"50% mark ({total // 2:,})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 2. Feature Distributions (Overlaid Histograms)
# ─────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    label_col: str = "Class",
    cols_per_row: int = 4,
) -> plt.Figure:
    """
    Overlaid histograms per class for each feature in a grid layout.
    Red = Malicious, Green = Benign.
    """
    _apply_dark_theme()
    logger.info("Generating feature distribution plots...")

    if label_col not in df.columns:
        label_col = df.columns[-1]

    if features is None:
        from src.config import FEATURE_COLUMNS
        features = [c for c in FEATURE_COLUMNS if c in df.columns]

    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    classes = sorted(df[label_col].unique())
    class_colors = {classes[0]: COLORS["benign"], classes[-1]: COLORS["malicious"]}
    class_labels = {classes[0]: "Benign", classes[-1]: "Malicious"}

    for idx, feature in enumerate(features):
        ax = axes[idx]
        for cls in classes:
            subset = df[df[label_col] == cls][feature].dropna()
            color = class_colors.get(cls, COLORS["accent"])
            label = class_labels.get(cls, str(cls))
            ax.hist(
                subset, bins=40, alpha=0.55, color=color,
                label=label, edgecolor="none", density=True,
            )
        ax.set_title(feature, fontsize=10, pad=8)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=18, y=1.01)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 3. Correlation Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    features: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
) -> plt.Figure:
    """
    Seaborn heatmap with hierarchical clustering coloring,
    annotation of correlation values, and upper-triangle mask.
    """
    _apply_dark_theme()
    logger.info("Generating correlation matrix heatmap...")

    if features is None:
        from src.config import FEATURE_COLUMNS
        features = [c for c in FEATURE_COLUMNS if c in df.columns]

    corr = df[features].corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Custom diverging colormap — cool blues to warm reds
    cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)

    fig, ax = plt.subplots(figsize=(16, 14))

    hm = sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        square=True, linewidths=0.5,
        linecolor=COLORS["grid"],
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title(title, pad=20, fontsize=16)
    ax.tick_params(labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 4. Feature Importance (Horizontal Bar Chart)
# ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    top_n: int = 20,
    title: str = "Feature Importance — Top Discriminative Features",
) -> plt.Figure:
    """
    Horizontal bar chart sorted descending, with gradient color by importance.
    """
    _apply_dark_theme()
    logger.info("Generating feature importance chart...")

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    # Reverse for horizontal bar (top = most important)
    sorted_importances = sorted_importances[::-1]
    sorted_names = sorted_names[::-1]

    # Create gradient colormap
    norm = plt.Normalize(sorted_importances.min(), sorted_importances.max())
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "importance_cmap",
        [COLORS["gradient_start"], COLORS["gradient_end"]]
    )
    bar_colors = [cmap(norm(v)) for v in sorted_importances]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))

    bars = ax.barh(
        sorted_names, sorted_importances,
        color=bar_colors, edgecolor="none",
        height=0.7, alpha=0.9, zorder=3,
    )

    # Add value labels
    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + max(sorted_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left", va="center", fontsize=9,
            color=COLORS["text_muted"],
        )

    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel("Importance Score", labelpad=10)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 5. Box Plots (Side-by-Side by Class)
# ─────────────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    label_col: str = "Class",
    cols_per_row: int = 4,
    title: str = "Feature Box Plots by Class",
) -> plt.Figure:
    """
    Side-by-side box plots per class with outlier highlighting.
    """
    _apply_dark_theme()
    logger.info("Generating box plots...")

    if label_col not in df.columns:
        label_col = df.columns[-1]

    if features is None:
        from src.config import FEATURE_COLUMNS
        features = [c for c in FEATURE_COLUMNS if c in df.columns][:10]

    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4.5 * n_rows))
    if n_rows == 1 and cols_per_row == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    palette = {
        sorted(df[label_col].unique())[0]: COLORS["benign"],
        sorted(df[label_col].unique())[-1]: COLORS["malicious"],
    }

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            data=df, x=label_col, y=feature, ax=ax,
            palette=palette, width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            linewidth=1.2,
        )
        ax.set_title(feature, fontsize=10, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 6. Pair Plot (Scatter Matrix)
# ─────────────────────────────────────────────────────────────────────

def plot_pairplot(
    df: pd.DataFrame,
    top_features: List[str],
    save_path: Optional[str] = None,
    label_col: str = "Class",
    title: str = "Pair Plot — Top Discriminative Features",
) -> plt.Figure:
    """
    Scatter matrix for top N features, colored by class.
    """
    _apply_dark_theme()
    logger.info("Generating pair plot...")

    if label_col not in df.columns:
        label_col = df.columns[-1]

    classes = sorted(df[label_col].unique())
    palette = {
        classes[0]: COLORS["benign"],
        classes[-1]: COLORS["malicious"],
    }

    plot_df = df[top_features + [label_col]].dropna()

    g = sns.pairplot(
        plot_df, hue=label_col, palette=palette,
        plot_kws={"alpha": 0.4, "s": 12, "edgecolor": "none"},
        diag_kws={"alpha": 0.5, "bins": 30},
        height=2.5,
    )

    g.fig.suptitle(title, y=1.02, fontsize=16,
                   fontweight="bold", color=COLORS["text"])

    # Style the pairplot figure background
    g.fig.set_facecolor(COLORS["bg_dark"])
    for ax_row in g.axes:
        for ax in ax_row:
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
            ax.xaxis.label.set_color(COLORS["text_muted"])
            ax.yaxis.label.set_color(COLORS["text_muted"])
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=150, bbox_inches="tight",
                  facecolor=COLORS["bg_dark"])
        logger.info(f"Pair plot saved to {save_path}")

    return g.fig


# ─────────────────────────────────────────────────────────────────────
# 7. ROC Curves (Overlaid Multi-Model)
# ─────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    models_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "ROC Curves — Model Comparison",
) -> plt.Figure:
    """
    Overlaid ROC curves for multiple models with AUC legend.

    Parameters
    ----------
    models_results : dict
        Keys are model names. Each value dict must contain:
        - 'fpr': array of false positive rates
        - 'tpr': array of true positive rates
        - 'auc': float AUC score
    """
    _apply_dark_theme()
    logger.info("Generating ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (model_name, result) in enumerate(models_results.items()):
        color = GRADIENT_PALETTE[idx % len(GRADIENT_PALETTE)]
        ax.plot(
            result["fpr"], result["tpr"],
            color=color, linewidth=2.5, alpha=0.9,
            label=f'{model_name} (AUC = {result["auc"]:.4f})',
        )

    # Diagonal line (random classifier)
    ax.plot(
        [0, 1], [0, 1],
        color=COLORS["text_muted"], linewidth=1, linestyle="--",
        alpha=0.5, label="Random (AUC = 0.5000)",
    )

    ax.set_title(title, pad=15)
    ax.set_xlabel("False Positive Rate", labelpad=10)
    ax.set_ylabel("True Positive Rate", labelpad=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", fontsize=11)

    # Shade area under best curve
    if models_results:
        best_name = max(models_results, key=lambda k: models_results[k]["auc"])
        best = models_results[best_name]
        ax.fill_between(
            best["fpr"], best["tpr"], alpha=0.08,
            color=GRADIENT_PALETTE[list(models_results.keys()).index(best_name)],
        )

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 8. Confusion Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap confusion matrix with counts and percentages.
    """
    _apply_dark_theme()
    logger.info(f"Generating confusion matrix for {model_name}...")

    if class_names is None:
        class_names = ["Benign", "Malicious"]

    if title is None:
        title = f"Confusion Matrix — {model_name}"

    total = cm.sum()
    cm_pct = cm / total * 100

    # Build annotation text with count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"

    # Custom colormap — dark blues to accent
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cm_cmap",
        ["#161B22", "#1B3A5C", "#2563EB", "#7C4DFF"]
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm, annot=annot, fmt="", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=2, linecolor=COLORS["bg_dark"],
        cbar_kws={"shrink": 0.7, "label": "Count"},
        annot_kws={"size": 14, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel("Predicted Label", labelpad=12, fontsize=12)
    ax.set_ylabel("True Label", labelpad=12, fontsize=12)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 9. Statistical Significance — P-value Heatmap
# ─────────────────────────────────────────────────────────────────────

def plot_pvalue_heatmap(
    pvalue_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Statistical Significance — Mann-Whitney U Test",
) -> plt.Figure:
    """
    Horizontal heatmap showing p-values for each feature.
    Features with p < 0.001 are highlighted.
    """
    _apply_dark_theme()
    logger.info("Generating p-value heatmap...")

    fig, ax = plt.subplots(figsize=(14, max(6, len(pvalue_df) * 0.35)))

    # Create color map: low p-value = bright (significant), high = dim
    sorted_df = pvalue_df.sort_values("p_value", ascending=True)

    # Color by significance
    colors = []
    for p in sorted_df["p_value"]:
        if p < 0.001:
            colors.append(COLORS["malicious"])
        elif p < 0.01:
            colors.append(COLORS["accent"])
        elif p < 0.05:
            colors.append(COLORS["accent2"])
        else:
            colors.append(COLORS["text_muted"])

    # Plot -log10(p-value) for better visual scaling
    neg_log_p = -np.log10(sorted_df["p_value"].clip(lower=1e-300))

    bars = ax.barh(
        sorted_df["feature"], neg_log_p,
        color=colors, edgecolor="none",
        height=0.7, alpha=0.85, zorder=3,
    )

    # Significance threshold lines
    for threshold, label, ls in [(0.05, "p=0.05", "--"), (0.01, "p=0.01", "-."), (0.001, "p=0.001", ":")]:
        ax.axvline(x=-np.log10(threshold), color=COLORS["text_muted"],
                   linewidth=1, linestyle=ls, alpha=0.6, label=label)

    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel("-log₁₀(p-value)", labelpad=10)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# 10. Outlier Count Summary Bar
# ─────────────────────────────────────────────────────────────────────

def plot_outlier_summary(
    outlier_counts: pd.Series,
    save_path: Optional[str] = None,
    top_n: int = 15,
    title: str = "Outlier Counts per Feature (IQR Method)",
) -> plt.Figure:
    """
    Bar chart showing the number of IQR outliers per feature.
    """
    _apply_dark_theme()
    logger.info("Generating outlier summary chart...")

    sorted_counts = outlier_counts.sort_values(ascending=False).head(top_n)

    norm = plt.Normalize(sorted_counts.min(), sorted_counts.max())
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "outlier_cmap",
        [COLORS["accent2"], COLORS["malicious"]]
    )

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.35)))

    bars = ax.barh(
        sorted_counts.index[::-1], sorted_counts.values[::-1],
        color=[cmap(norm(v)) for v in sorted_counts.values[::-1]],
        edgecolor="none", height=0.65, zorder=3,
    )

    for bar, val in zip(bars, sorted_counts.values[::-1]):
        ax.text(
            bar.get_width() + max(sorted_counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}",
            ha="left", va="center", fontsize=9,
            color=COLORS["text_muted"],
        )

    ax.set_title(title, pad=15, fontsize=14)
    ax.set_xlabel("Outlier Count", labelpad=10)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig
