"""Phase 5 subgroup, calibration, drift, and sanitized error analysis."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import FIRST_SEEN_COLUMN, SAMPLE_ID_COLUMN, SOURCE_ID_COLUMN
from src.models.metrics import metric_report


ACTIVE_CONTENT_COLUMNS = (
    "js_count",
    "javascript_count",
    "openaction_count",
    "action_count",
    "aa_count",
    "launch_count",
    "uri_count",
    "submitform_count",
)
PARSER_STATUS_COLUMNS = (
    "parse_failure",
    "recovery_mode",
    "invalid_eof",
    "extraction_timeout",
    "file_too_large",
    "extraction_limit_reached",
)


def calibration_table(
    y_true: Iterable[int], y_probability: Iterable[float], *, bins: int = 15
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    rows = []
    for index in range(bins):
        mask = assignments == index
        rows.append(
            {
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "rows": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                "observed_malicious_rate": float(labels[mask].mean()) if mask.any() else None,
            }
        )
    return pd.DataFrame(rows)


def _safe_quantile_labels(values: pd.Series, quantiles: int, prefix: str) -> pd.Series:
    ranks = values.rank(method="first")
    actual = min(quantiles, max(1, int(values.nunique(dropna=True))))
    if actual < 2:
        return pd.Series([f"{prefix}_constant"] * len(values), index=values.index)
    return pd.qcut(
        ranks,
        q=actual,
        labels=[f"{prefix}_{index + 1:02d}" for index in range(actual)],
    ).astype(str)


def subgroup_assignments(metadata: pd.DataFrame) -> dict[str, pd.Series]:
    """Return predeclared, deployment-relevant subgroup labels."""
    groups: dict[str, pd.Series] = {}
    if SOURCE_ID_COLUMN in metadata:
        groups["source"] = metadata[SOURCE_ID_COLUMN].astype(str)
    if FIRST_SEEN_COLUMN in metadata:
        timestamps = pd.to_datetime(metadata[FIRST_SEEN_COLUMN], utc=True)
        groups["time_window"] = _safe_quantile_labels(
            timestamps.astype("int64"), 5, "time"
        )
    if "pdf_size" in metadata:
        groups["file_size_decile"] = _safe_quantile_labels(
            pd.to_numeric(metadata["pdf_size"], errors="coerce"), 10, "size"
        )
    if "is_encrypted" in metadata:
        groups["encryption"] = metadata["is_encrypted"].fillna(-1).map(
            {0: "not_encrypted", 1: "encrypted", -1: "unavailable"}
        ).fillna("invalid")
    active = [name for name in ACTIVE_CONTENT_COLUMNS if name in metadata]
    if active:
        groups["active_content"] = (
            metadata.loc[:, active].fillna(0).sum(axis=1).gt(0)
            .map({True: "present", False: "absent"})
        )
    parser = [name for name in PARSER_STATUS_COLUMNS if name in metadata]
    if parser:
        groups["parser_status"] = (
            metadata.loc[:, parser].fillna(0).sum(axis=1).gt(0)
            .map({True: "limited_or_failed", False: "nominal"})
        )
    if "obfuscation_count" in metadata:
        groups["obfuscation_level"] = pd.cut(
            pd.to_numeric(metadata["obfuscation_count"], errors="coerce"),
            bins=[-np.inf, 0, 1, 5, np.inf],
            labels=("none", "one", "two_to_five", "six_or_more"),
        ).astype(str)
    if "missing_feature_count" in metadata:
        groups["missingness_pattern"] = pd.cut(
            metadata["missing_feature_count"],
            bins=[-np.inf, 0, 1, np.inf],
            labels=("none", "one", "two_or_more"),
        ).astype(str)
    return groups


def metrics_by_subgroup(
    metadata: pd.DataFrame,
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    threshold: float,
    minimum_rows: int = 100,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Compute complete metrics for sufficiently populated declared subgroups."""
    labels = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    if len(metadata) != len(labels) or len(labels) != len(probability):
        raise ValueError("Subgroup metadata, labels, and probabilities must align.")
    rows: list[dict[str, Any]] = []
    for group_name, assignments in subgroup_assignments(metadata).items():
        for value in sorted(assignments.dropna().unique()):
            mask = assignments.eq(value).to_numpy()
            if int(mask.sum()) < minimum_rows:
                continue
            report = metric_report(labels[mask], probability[mask], threshold=threshold)
            rows.append(
                {
                    "subgroup_type": group_name,
                    "subgroup_value": str(value),
                    **report,
                }
            )
    unavailable = []
    for name in ("malware_family", "campaign"):
        if name not in metadata:
            unavailable.append(
                {
                    "subgroup": name,
                    "status": "unavailable_in_safe_schema",
                    "reason": "The approved feature contract does not currently supply this field.",
                }
            )
    return pd.DataFrame(rows), unavailable


def sanitized_hard_errors(
    metadata: pd.DataFrame,
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    threshold: float,
    experiment_id: str,
    per_error_type: int = 50,
) -> pd.DataFrame:
    """Return numeric feature profiles for hardest FP/FN with hashed IDs."""
    labels = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(y_probability, dtype=np.float64)
    predicted = probability >= threshold
    if SAMPLE_ID_COLUMN not in metadata:
        raise ValueError("Sanitized error profiles require opaque sample IDs.")
    profile_columns = [
        name
        for name in (
            "pdf_size",
            "is_encrypted",
            "js_count",
            "javascript_count",
            "openaction_count",
            "launch_count",
            "obfuscation_count",
            "missing_feature_count",
        )
        if name in metadata
    ]
    rows: list[dict[str, Any]] = []
    masks = {
        "false_positive": predicted & (labels == 0),
        "false_negative": ~predicted & (labels == 1),
    }
    for error_type, mask in masks.items():
        positions = np.flatnonzero(mask)
        if error_type == "false_positive":
            positions = positions[np.argsort(-probability[positions], kind="mergesort")]
        else:
            positions = positions[np.argsort(probability[positions], kind="mergesort")]
        for position in positions[:per_error_type]:
            sample_value = str(metadata.iloc[position][SAMPLE_ID_COLUMN])
            sample_hash = hashlib.sha256(
                f"{experiment_id}:{sample_value}".encode("utf-8")
            ).hexdigest()
            row: dict[str, Any] = {
                "error_type": error_type,
                "sample_id_sha256": sample_hash,
                "true_label": int(labels[position]),
                "calibrated_probability": float(probability[position]),
                "locked_threshold": float(threshold),
            }
            for name in profile_columns:
                value = metadata.iloc[position][name]
                row[name] = None if pd.isna(value) else float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> tuple[float, float]:
    """Return PSI and Jensen-Shannon divergence using reference quantile bins."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if not len(reference) or not len(current):
        return float("nan"), float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0, 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_count, _ = np.histogram(reference, bins=edges)
    current_count, _ = np.histogram(current, bins=edges)
    reference_probability = np.clip(reference_count / reference_count.sum(), 1e-8, 1.0)
    current_probability = np.clip(current_count / current_count.sum(), 1e-8, 1.0)
    psi = np.sum(
        (current_probability - reference_probability)
        * np.log(current_probability / reference_probability)
    )
    midpoint = (reference_probability + current_probability) / 2.0
    js = 0.5 * np.sum(reference_probability * np.log(reference_probability / midpoint))
    js += 0.5 * np.sum(current_probability * np.log(current_probability / midpoint))
    return float(psi), float(js)


def drift_table(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: Iterable[str],
    *,
    maximum_rows: int = 250_000,
) -> pd.DataFrame:
    """Compute deterministic per-feature train-to-test distribution drift."""
    reference = np.asarray(reference_features)
    current = np.asarray(current_features)
    names = tuple(feature_names)
    if reference.shape[1] != current.shape[1] or reference.shape[1] != len(names):
        raise ValueError("Drift matrices and feature names do not align.")
    reference_index = np.linspace(
        0, len(reference) - 1, min(len(reference), maximum_rows), dtype=np.int64
    )
    current_index = np.linspace(
        0, len(current) - 1, min(len(current), maximum_rows), dtype=np.int64
    )
    rows = []
    for index, name in enumerate(names):
        psi, js = population_stability_index(
            reference[reference_index, index], current[current_index, index]
        )
        rows.append(
            {
                "feature": name,
                "population_stability_index": psi,
                "jensen_shannon_divergence": js,
                "reference_rows": len(reference_index),
                "current_rows": len(current_index),
            }
        )
    return pd.DataFrame(rows)
