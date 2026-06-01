"""
consistency.py
--------------
CSV-vs-live feature consistency audit.

This module answers a question every deployable ML system must answer: *are
the features produced at inference time drawn from the same distribution the
model was trained on?* For this project the answer (as of Phase 10) is **no**,
and this audit quantifies exactly why.

Background
~~~~~~~~~~
The CIC Evasive-PDFMal2022 **feature CSV** ships its columns min-max normalized
to roughly ``[0, 1]``. The persisted ``StandardScaler`` therefore has a
per-feature ``mean_ ~= 0.5`` and ``scale_ (std) ~= 0.29`` — the tell-tale
signature of a uniform ``[0, 1]`` variable.

The live extractor (``src/features/structural.py`` + ``metadata.py``), however,
emits **raw counts and byte sizes** (e.g. ``pdf_size`` in the hundreds, object
counts as integers). Passing those raw values through a scaler fit on ``[0, 1]``
data maps byte-size features to z-scores in the *hundreds or thousands*,
saturating the network so that every uploaded PDF collapses to a single verdict.

This module makes that mismatch measurable and reproducible.

Usage::

    python -m src.features.consistency
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, RESULTS_DIR, SAMPLE_PDFS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A feature whose live value lands this many sigmas outside the training
# domain is "out of distribution" for the trained model.
_OOD_SIGMA = 10.0


def detect_training_regime(scaler) -> Dict[str, object]:
    """Infer the normalization regime of the data the scaler was fit on.

    Args:
        scaler: A fitted ``StandardScaler`` (exposes ``mean_`` and ``scale_``).

    Returns:
        dict: Summary including whether the training features look min-max
              normalized to [0, 1].
    """
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)

    looks_unit_interval = bool(
        np.all(means > 0.2) and np.all(means < 0.8)
        and np.all(scales > 0.15) and np.all(scales < 0.45)
    )
    return {
        "mean_of_means": float(means.mean()),
        "mean_of_scales": float(scales.mean()),
        "looks_minmax_0_1": looks_unit_interval,
        "interpretation": (
            "Training features appear min-max normalized to ~[0,1] "
            "(mean~=0.5, std~=0.29 per feature)."
            if looks_unit_interval else
            "Training-feature regime is not the classic [0,1] min-max signature."
        ),
    }


def audit_pdf(pdf_path: Path, scaler) -> pd.DataFrame:
    """Extract live features from a PDF and compare them to the training domain.

    Args:
        pdf_path: Path to a PDF file.
        scaler: Fitted ``StandardScaler`` encoding the training domain.

    Returns:
        pd.DataFrame: Per-feature table with raw value, training mean/std, the
        resulting z-score, and an out-of-distribution flag.
    """
    from src.features.vectorizer import extract_features_dict

    raw = extract_features_dict(pdf_path)
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)

    rows = []
    for i, col in enumerate(FEATURE_COLUMNS):
        value = float(raw.get(col, 0.0))
        std = scales[i] if scales[i] > 1e-9 else 1e-9
        z = (value - means[i]) / std
        rows.append({
            "feature": col,
            "live_value": value,
            "train_mean": round(float(means[i]), 4),
            "train_std": round(float(scales[i]), 4),
            "z_score": round(float(z), 2),
            "out_of_distribution": bool(abs(z) > _OOD_SIGMA),
        })

    df = pd.DataFrame(rows)
    df["abs_z"] = df["z_score"].abs()
    return df.sort_values("abs_z", ascending=False).drop(columns="abs_z").reset_index(drop=True)


def run_audit(sample_dir: Path = SAMPLE_PDFS_DIR, save: bool = True) -> Dict[str, object]:
    """Run the full consistency audit over the bundled sample PDFs.

    Args:
        sample_dir: Directory containing sample PDFs.
        save: If True, write ``reports/results/feature_consistency.csv``.

    Returns:
        dict: Audit summary.
    """
    from src.features.vectorizer import load_scaler

    scaler = load_scaler()
    regime = detect_training_regime(scaler)

    logger.info("=" * 70)
    logger.info("CSV-vs-LIVE FEATURE CONSISTENCY AUDIT")
    logger.info("=" * 70)
    logger.info(regime["interpretation"])

    all_tables: List[pd.DataFrame] = []
    per_file_summary = []
    pdfs = sorted(Path(sample_dir).glob("*.pdf"))

    for pdf in pdfs:
        table = audit_pdf(pdf, scaler)
        table.insert(0, "pdf", pdf.name)
        all_tables.append(table)

        n_ood = int(table["out_of_distribution"].sum())
        worst = table.iloc[0]
        per_file_summary.append({
            "pdf": pdf.name,
            "n_features_out_of_distribution": n_ood,
            "worst_feature": worst["feature"],
            "worst_z": worst["z_score"],
        })
        logger.info(
            f"{pdf.name}: {n_ood}/{len(FEATURE_COLUMNS)} features out of training "
            f"distribution (|z|>{_OOD_SIGMA:.0f}); worst = "
            f"{worst['feature']} (z={worst['z_score']})"
        )

    summary = {
        "regime": regime,
        "ood_sigma_threshold": _OOD_SIGMA,
        "per_file": per_file_summary,
        "verdict": (
            "MISMATCH: live raw features fall far outside the model's training "
            "domain. The deployment path needs a consistent representation "
            "(see run_all.py --from-pdfs)."
            if any(s["n_features_out_of_distribution"] > 0 for s in per_file_summary)
            else "Live features are within the training domain."
        ),
    }

    if save and all_tables:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "feature_consistency.csv"
        pd.concat(all_tables, ignore_index=True).to_csv(out, index=False)
        logger.info(f"Per-feature consistency table saved -> {out}")

    logger.info(f"VERDICT: {summary['verdict']}")
    return summary


if __name__ == "__main__":
    run_audit()
