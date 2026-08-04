"""Leakage-safe feature diagnostics, ablations, and stability scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.loader import iter_dataset_batches
from src.features.engineered import EngineeredFeatureSpec


@dataclass(frozen=True)
class TrainingFeatureAudit:
    constant_features: tuple[str, ...]
    near_duplicate_pairs: tuple[tuple[str, str, float], ...]
    decision_policy: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ShortcutFeatureAudit:
    """Held-out evidence for suspiciously predictive non-semantic fields."""

    feature: str
    train_rows: int
    validation_rows: int
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    train_auc_raw: float
    train_direction: str
    validation_auc_raw: float
    validation_auc_train_direction: float
    validation_separability: float
    direction_stable: bool
    warning_threshold: float
    high_risk_shortcut: bool
    action: str
    excluded_identity_fields: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binary_auc(values: np.ndarray, labels: np.ndarray, *, partition: str) -> float:
    if not np.isfinite(values).all():
        raise ValueError(f"Shortcut audit found non-finite values in {partition}.")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(
            f"Shortcut audit requires both binary classes in {partition}; got {unique.tolist()}."
        )
    return float(roc_auc_score(labels, values))


def _project_feature_and_label(
    dataset_path: Path,
    *,
    feature: str,
    label_column: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in iter_dataset_batches(
        Path(dataset_path), columns=(feature, label_column), batch_size=batch_size
    ):
        if feature not in batch or label_column not in batch:
            raise ValueError(
                f"Shortcut audit requires {feature!r} and {label_column!r}."
            )
        values.append(batch[feature].to_numpy(dtype=np.float64, copy=False))
        labels.append(batch[label_column].to_numpy(dtype=np.int8, copy=False))
    if not values:
        raise ValueError(f"Shortcut audit dataset is empty: {dataset_path}")
    return np.concatenate(values), np.concatenate(labels)


def audit_file_size_shortcut(
    train_dataset: Path,
    validation_dataset: Path,
    *,
    feature: str = "pdf_size",
    label_column: str = "Class",
    warning_threshold: float = 0.90,
    batch_size: int = 100_000,
) -> ShortcutFeatureAudit:
    """Measure whether file size alone separates labels on held-out validation.

    Direction is selected on the sealed train partition. Validation reports both
    that locked direction and direction-free separability. No test rows are read.
    Creator/producer strings are recorded as structurally unavailable because the
    safe schema deliberately excludes high-cardinality identity metadata.
    """
    if not 0.5 < warning_threshold <= 1.0:
        raise ValueError("warning_threshold must be in (0.5, 1.0].")
    train_values, train_labels = _project_feature_and_label(
        train_dataset,
        feature=feature,
        label_column=label_column,
        batch_size=batch_size,
    )
    validation_values, validation_labels = _project_feature_and_label(
        validation_dataset,
        feature=feature,
        label_column=label_column,
        batch_size=batch_size,
    )
    train_auc_raw = _binary_auc(train_values, train_labels, partition="train")
    validation_auc_raw = _binary_auc(
        validation_values, validation_labels, partition="validation"
    )
    increasing = train_auc_raw >= 0.5
    locked_auc = validation_auc_raw if increasing else 1.0 - validation_auc_raw
    validation_separability = max(validation_auc_raw, 1.0 - validation_auc_raw)
    direction_stable = (validation_auc_raw >= 0.5) == increasing
    high_risk = validation_separability >= warning_threshold
    action = (
        "Block automatic retention: run file-size ablation across seeds and time "
        "windows and justify any inclusion."
        if high_risk
        else "Retain only as a normalization denominator and continue ablation monitoring."
    )
    excluded = {
        name: {
            "status": "not_in_schema",
            "reason": (
                "Raw creator/producer identity strings are excluded from the safe "
                "feature contract because they are high-cardinality provenance "
                "shortcuts and can contain sensitive free text."
            ),
            "derived_replacement": "metadata_size and title_chars only",
        }
        for name in ("creator", "producer")
    }
    return ShortcutFeatureAudit(
        feature=feature,
        train_rows=len(train_labels),
        validation_rows=len(validation_labels),
        train_class_counts={
            str(value): int((train_labels == value).sum()) for value in (0, 1)
        },
        validation_class_counts={
            str(value): int((validation_labels == value).sum()) for value in (0, 1)
        },
        train_auc_raw=train_auc_raw,
        train_direction="increasing" if increasing else "decreasing",
        validation_auc_raw=validation_auc_raw,
        validation_auc_train_direction=float(locked_auc),
        validation_separability=float(validation_separability),
        direction_stable=bool(direction_stable),
        warning_threshold=warning_threshold,
        high_risk_shortcut=bool(high_risk),
        action=action,
        excluded_identity_fields=excluded,
    )


def audit_training_features(
    frame: pd.DataFrame,
    *,
    partition_name: str,
    absolute_correlation_threshold: float = 0.995,
) -> TrainingFeatureAudit:
    """Identify candidates only; correlation never automatically removes a feature."""
    if partition_name != "train":
        raise ValueError("Feature selection audits may only inspect the train partition.")
    numeric = frame.select_dtypes(include=[np.number])
    constants = tuple(name for name in numeric if numeric[name].nunique(dropna=False) <= 1)
    active = numeric.drop(columns=list(constants), errors="ignore")
    correlations = active.corr(method="spearman").abs()
    pairs: list[tuple[str, str, float]] = []
    names = list(correlations.columns)
    for right_index, right in enumerate(names):
        for left in names[:right_index]:
            value = correlations.loc[left, right]
            if np.isfinite(value) and value >= absolute_correlation_threshold:
                pairs.append((left, right, float(value)))
    return TrainingFeatureAudit(
        constants,
        tuple(sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))),
        "Constants may be removed. Correlated pairs require held-out ablation and stability evidence.",
    )


def ablation_feature_sets(
    base_features: Sequence[str], specs: Sequence[EngineeredFeatureSpec]
) -> dict[str, tuple[str, ...]]:
    """Return reproducible base/all/leave-one-family-out comparison sets."""
    base = tuple(base_features)
    engineered = tuple(spec.name for spec in specs)
    families = sorted({spec.family for spec in specs})
    result = {"base_only": base, "all_engineered": (*base, *engineered)}
    for family in families:
        retained = tuple(spec.name for spec in specs if spec.family != family)
        result[f"without_{family}"] = (*base, *retained)
    return result


def importance_rank_stability(
    importances: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute pairwise Spearman rank stability across seeds/time windows."""
    runs = sorted(importances)
    features = sorted({feature for values in importances.values() for feature in values})
    scores: dict[str, float] = {}
    for right_index, right in enumerate(runs):
        for left in runs[:right_index]:
            left_values = [importances[left].get(name, 0.0) for name in features]
            right_values = [importances[right].get(name, 0.0) for name in features]
            correlation = spearmanr(left_values, right_values).statistic
            scores[f"{left}__{right}"] = float(correlation) if np.isfinite(correlation) else 0.0
    return scores
