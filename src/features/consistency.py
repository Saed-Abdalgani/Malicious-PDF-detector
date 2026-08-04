"""Schema-v2 live/batch/record parity audit for the serialized feature pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_REPORTS_DIR, SAMPLE_PDFS_DIR
from src.features.vectorizer import (
    extract_features_record,
    load_feature_pipeline,
    pdf_to_pipeline_vector,
)
from src.utils.atomic import atomic_write_json


def audit_pdf(pdf_path: Path, pipeline) -> dict[str, Any]:
    """Prove exact parity between DataFrame, record, and end-to-end PDF paths."""
    record, diagnostics = extract_features_record(pdf_path)
    batch = pipeline.transform(pd.DataFrame([record])).to_numpy(dtype=np.float32)[0]
    record_vector = pipeline.transform_record(record).to_numpy(dtype=np.float32)[0]
    live_vector, _, live_diagnostics = pdf_to_pipeline_vector(
        pdf_path, pipeline=pipeline
    )
    return {
        "pdf": pdf_path.name,
        "batch_record_equal": bool(np.array_equal(batch, record_vector)),
        "batch_live_equal": bool(np.array_equal(batch, live_vector)),
        "output_dimensions": int(len(batch)),
        "abstain_recommended": bool(diagnostics["abstain_recommended"]),
        "diagnostics_stable": diagnostics == live_diagnostics,
    }


def run_audit(
    sample_dir: Path = SAMPLE_PDFS_DIR,
    *,
    pipeline=None,
    minimum_fixtures: int = 100,
    save: bool = True,
) -> dict[str, Any]:
    """Require exact parity on at least 100 safe fixtures/approved rows."""
    active_pipeline = pipeline or load_feature_pipeline()
    pdfs = sorted(Path(sample_dir).glob("*.pdf"))
    if len(pdfs) < minimum_fixtures:
        raise RuntimeError(
            f"Parity audit requires at least {minimum_fixtures} safe PDF fixtures; "
            f"found {len(pdfs)} in {sample_dir}."
        )
    rows = [audit_pdf(pdf, active_pipeline) for pdf in pdfs]
    passed = all(
        row["batch_record_equal"]
        and row["batch_live_equal"]
        and row["diagnostics_stable"]
        for row in rows
    )
    result = {
        "status": "passed" if passed else "failed",
        "fixture_count": len(rows),
        "minimum_fixtures": minimum_fixtures,
        "exact_parity": passed,
        "rows": rows,
    }
    if save:
        DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(DATA_REPORTS_DIR / "feature_parity.json", result)
    if not passed:
        raise RuntimeError("Feature parity audit failed.")
    return result


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
