"""
run_all.py
----------
One-command, reproducible end-to-end pipeline for the Malicious PDF Detector.

It chains the full workflow so a grader (or you) can regenerate every artifact
with a single command and a fixed random seed:

    download -> clean -> split (+SMOTE) -> fit scaler -> train -> quantize -> evaluate

Two data sources are supported:

1. **CSV mode (default)** — uses the CIC Evasive-PDFMal2022 feature CSV.
   The bundled ``src/data/downloader.py`` URLs are placeholders, so if the CSV
   is absent this script prints clear, actionable guidance instead of failing
   opaquely.

2. **PDF mode (``--from-pdfs DIR``)** — extracts features *directly from PDF
   files* with the same extractor used at inference time. This is the
   principled fix for the train/inference normalization mismatch documented in
   ``implementation_plan.md`` (Phase 10+): when the model is trained on the
   exact representation produced at inference, the deployment path is
   consistent end-to-end.

   ``DIR`` must contain two sub-folders::

       DIR/benign/*.pdf
       DIR/malicious/*.pdf

Crucially, this pipeline keeps a **single, consistent feature representation**:
the ``StandardScaler`` is fit on the training split and then applied to *all*
splits before training (and saved to ``models/scaler.pkl`` for inference). This
supersedes the earlier flow where the model was trained on unscaled values but
the app applied a scaler at inference.

Usage::

    python -m src.run_all                  # CSV mode
    python -m src.run_all --from-pdfs data/corpus
    python -m src.run_all --skip-train     # only (re)quantize + evaluate existing models
    python -m src.run_all --mlp-epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    QUANTIZED_MODELS_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_COL = "Class"


# ---------------------------------------------------------------------------
# Step 0 — data acquisition / construction
# ---------------------------------------------------------------------------

def _build_csv_from_pdfs(corpus_dir: Path) -> Path:
    """Extract features from a labelled PDF corpus into a CSV.

    Walks ``corpus_dir/benign`` and ``corpus_dir/malicious`` and runs the
    inference-time extractor on every PDF, producing a CSV with the same
    columns the model is trained on. This guarantees train/inference parity.

    Args:
        corpus_dir: Directory containing ``benign/`` and ``malicious/`` subfolders.

    Returns:
        Path: Path to the written raw CSV.
    """
    from src.features.vectorizer import extract_features_dict

    rows = []
    for label_name, label_value in (("benign", 0), ("malicious", 1)):
        subdir = corpus_dir / label_name
        if not subdir.exists():
            logger.warning(f"Corpus sub-folder missing: {subdir}")
            continue
        pdfs = sorted(subdir.glob("*.pdf"))
        logger.info(f"Extracting features from {len(pdfs)} '{label_name}' PDFs...")
        for pdf in pdfs:
            try:
                feats = extract_features_dict(pdf)
                feats[LABEL_COL] = label_value
                rows.append(feats)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad file
                logger.warning(f"  Skipping {pdf.name}: {exc}")

    if not rows:
        raise FileNotFoundError(
            f"No PDFs found under {corpus_dir}/(benign|malicious). "
            f"Populate those folders and retry."
        )

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + [LABEL_COL])
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DATA_DIR / "pdfmal_from_pdfs.csv"
    df.to_csv(out, index=False)
    logger.info(
        f"Built feature CSV from PDFs -> {out} "
        f"({len(df)} rows, {df[LABEL_COL].sum()} malicious)"
    )
    return out


def _ensure_dataset(from_pdfs: Optional[str]) -> Optional[Path]:
    """Locate or build the raw feature CSV.

    Returns the CSV path, or ``None`` if no data is available (caller prints
    guidance and exits gracefully).
    """
    if from_pdfs:
        return _build_csv_from_pdfs(Path(from_pdfs))

    # CSV mode: look for any plausible raw CSV already on disk.
    candidates = list(RAW_DATA_DIR.glob("*.csv"))
    if candidates:
        logger.info(f"Using existing raw dataset: {candidates[0]}")
        return candidates[0]

    # NOTE: we deliberately do NOT invoke the downloader here — its URLs are
    # placeholders and its manual fallback opens a web browser, which is
    # undesirable in an automated run. The caller prints manual guidance.
    return None


def _print_missing_data_help() -> None:
    """Print actionable guidance when no dataset is present."""
    msg = """
============================================================================
  No dataset found — cannot run the full pipeline.
============================================================================
The bundled downloader uses placeholder URLs, so the CIC Evasive-PDFMal2022
data must be provided manually. You have two options:

  OPTION A — Feature CSV (fastest)
    1. Download the CIC PDFMal-2022 feature CSV.
    2. Place it at:  data/raw/pdfmal2022.csv
    3. Re-run:       python -m src.run_all

  OPTION B — Train/infer parity (recommended, fixes the normalization mismatch)
    1. Obtain the dataset's PDF corpus.
    2. Arrange it as:
           data/corpus/benign/*.pdf
           data/corpus/malicious/*.pdf
    3. Run:          python -m src.run_all --from-pdfs data/corpus

Everything else (clean -> split -> scale -> train -> quantize -> evaluate)
then runs automatically and writes results to reports/results/.
============================================================================
"""
    print(msg)


# ---------------------------------------------------------------------------
# Step 1-3 — clean, split, scale
# ---------------------------------------------------------------------------

def _prepare_splits(csv_path: Path) -> tuple:
    """Clean, split, SMOTE, fit+apply scaler. Writes scaled splits to disk.

    Returns:
        (feature_names, scaler) tuple.
    """
    from src.data.cleaner import clean_pipeline
    from src.data.loader import load_dataset
    from src.data.splitter import apply_smote, save_splits, stratified_split
    from src.features.vectorizer import compute_benign_baseline, fit_scaler, transform

    df = load_dataset(csv_path)
    df = clean_pipeline(df)
    train_df, val_df, test_df = stratified_split(df)
    train_df = apply_smote(train_df)

    feature_names = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    label_col = LABEL_COL if LABEL_COL in train_df.columns else train_df.columns[-1]

    X_train = train_df[feature_names].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(np.int64)

    # Fit scaler on TRAIN ONLY, then apply to every split (train==inference).
    scaler = fit_scaler(X_train, save=True)
    compute_benign_baseline(X_train, y_train, save=True)

    def _scale(frame):
        frame = frame.copy()
        frame[feature_names] = transform(
            frame[feature_names].values.astype(np.float64), scaler
        )
        return frame

    save_splits(_scale(train_df), _scale(val_df), _scale(test_df))
    logger.info("Scaled train/val/test splits written to data/processed/")
    return feature_names, scaler


# ---------------------------------------------------------------------------
# Step 5 — quantize
# ---------------------------------------------------------------------------

def _quantize() -> None:
    """Quantize the trained MLP (dynamic INT8) and save the deployable artifact."""
    import torch

    from src.models.mlp import load_mlp
    from src.optimization.quantizer import ModelQuantizer, get_model_size_mb

    mlp_path = TRAINED_MODELS_DIR / "mlp_best.pt"
    if not mlp_path.exists():
        logger.warning(f"No trained MLP at {mlp_path}; skipping quantization.")
        return

    fp32 = load_mlp(mlp_path)
    fp32_mb = get_model_size_mb(fp32)
    quantizer = ModelQuantizer(fp32)
    dyn = quantizer.dynamic_quantize()
    dyn_mb = get_model_size_mb(dyn)
    out = QUANTIZED_MODELS_DIR / "mlp_quantized_dynamic.pt"
    quantizer.save_quantized(dyn, out, save_torchscript=False)

    reduction = (1 - dyn_mb / fp32_mb) * 100 if fp32_mb else 0.0
    logger.info(
        f"Quantization: FP32 {fp32_mb:.4f} MB -> INT8 {dyn_mb:.4f} MB "
        f"({reduction:.1f}% smaller) -> {out}"
    )

    # Persist a small, honest benchmark artifact.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"Variant": "MLP (FP32)", "Size (MB)": round(fp32_mb, 4)},
        {"Variant": "MLP (INT8 dynamic)", "Size (MB)": round(dyn_mb, 4)},
        {"Variant": "Size reduction (%)", "Size (MB)": round(reduction, 1)},
    ]).to_csv(RESULTS_DIR / "quantization_comparison.csv", index=False)


# ---------------------------------------------------------------------------
# Step 6 — evaluate
# ---------------------------------------------------------------------------

def _evaluate() -> None:
    """Evaluate all trained models on the held-out test set and write reports."""
    from src.models.evaluator import ModelEvaluator
    from src.models.trainer import TrainingPipeline

    try:
        pipeline = TrainingPipeline()
    except FileNotFoundError:
        logger.warning(
            "No processed test split found (data/processed/). "
            "Skipping evaluation — run with a dataset to regenerate metrics."
        )
        return

    models = pipeline.load_all_models()
    if not models:
        logger.warning("No trained models found; skipping evaluation.")
        return

    evaluator = ModelEvaluator()
    evaluator.evaluate_all(models, pipeline.X_test, pipeline.y_test)
    evaluator.get_comparison()
    evaluator.print_summary()
    try:
        evaluator.generate_full_report(feature_names=pipeline.feature_names)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Figure generation skipped: {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end reproducible pipeline for the Malicious PDF Detector.",
    )
    parser.add_argument(
        "--from-pdfs", metavar="DIR", default=None,
        help="Build the dataset by extracting features from DIR/{benign,malicious}/*.pdf "
             "(train==inference parity).",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip cleaning/splitting/training; only quantize + evaluate.")
    parser.add_argument("--mlp-epochs", type=int, default=100,
                        help="Max MLP training epochs (default: 100).")
    args = parser.parse_args(argv)

    np.random.seed(RANDOM_SEED)
    logger.info(f"=== run_all (seed={RANDOM_SEED}) ===")

    if not args.skip_train:
        csv_path = _ensure_dataset(args.from_pdfs)
        if csv_path is None:
            _print_missing_data_help()
            return 0  # graceful, not an error

        _prepare_splits(csv_path)

        from src.models.trainer import TrainingPipeline
        TrainingPipeline().train_all_models(mlp_epochs=args.mlp_epochs)

    _quantize()
    _evaluate()

    logger.info("=== run_all complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
