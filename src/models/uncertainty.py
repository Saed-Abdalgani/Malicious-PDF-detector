"""Group-aware stratified bootstrap uncertainty for locked model decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.metrics import fbeta_from_counts


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower_95: float
    upper_95: float
    replicates: int
    method: str = "group_stratified_percentile"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated(
    y_true: Iterable[int], y_probability: Iterable[float], groups: Iterable[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    group_values = np.asarray(list(groups), dtype=str).reshape(-1)
    if not (len(labels) == len(probabilities) == len(group_values)) or not len(labels):
        raise ValueError("Bootstrap labels, probabilities, and groups must align.")
    if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
        raise ValueError("Bootstrap requires both binary classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Bootstrap probabilities must be finite.")
    unique_groups, inverse = np.unique(group_values, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError("Group-aware bootstrap requires at least two groups.")
    return labels, probabilities, unique_groups, inverse


def _group_strata(
    labels: np.ndarray, inverse: np.ndarray, group_count: int
) -> tuple[np.ndarray, np.ndarray]:
    group_positive = np.zeros(group_count, dtype=bool)
    np.logical_or.at(group_positive, inverse, labels == 1)
    positive_groups = np.flatnonzero(group_positive)
    benign_only_groups = np.flatnonzero(~group_positive)
    if not len(positive_groups) or not len(benign_only_groups):
        # Mixed-label groups can make group stratification degenerate. Falling
        # back to all groups remains cluster-aware and is recorded by callers.
        all_groups = np.arange(group_count)
        return all_groups, np.empty(0, dtype=np.int64)
    return positive_groups, benign_only_groups


def _draw_group_counts(
    rng: np.random.Generator,
    group_count: int,
    first_stratum: np.ndarray,
    second_stratum: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(group_count, dtype=np.int32)
    for stratum in (first_stratum, second_stratum):
        if len(stratum):
            sampled = rng.choice(stratum, size=len(stratum), replace=True)
            counts += np.bincount(sampled, minlength=group_count).astype(np.int32)
    return counts


def _confusion_metrics(counts: np.ndarray) -> dict[str, float]:
    tp, fp, tn, fn = (float(value) for value in counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "precision": precision,
        "recall": recall,
        "f0_5": fbeta_from_counts(tp, fp, fn, beta=0.5),
        "f1": fbeta_from_counts(tp, fp, fn, beta=1.0),
        "f2": fbeta_from_counts(tp, fp, fn, beta=2.0),
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "matthews_correlation_coefficient": (
            (tp * tn - fp * fn) / denominator if denominator else 0.0
        ),
        "false_positives_per_million_benign": fpr * 1_000_000.0,
    }


def group_confusion_bootstrap(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    groups: Iterable[str],
    *,
    threshold: float,
    replicates: int = 1_000,
    random_seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """Return vector-efficient cluster bootstrap CIs for threshold metrics."""
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    labels, probabilities, unique_groups, inverse = _validated(
        y_true, y_probability, groups
    )
    predicted = probabilities >= float(threshold)
    contributions = np.zeros((len(unique_groups), 4), dtype=np.int64)
    np.add.at(contributions[:, 0], inverse, predicted & (labels == 1))
    np.add.at(contributions[:, 1], inverse, predicted & (labels == 0))
    np.add.at(contributions[:, 2], inverse, ~predicted & (labels == 0))
    np.add.at(contributions[:, 3], inverse, ~predicted & (labels == 1))
    first, second = _group_strata(labels, inverse, len(unique_groups))
    point = _confusion_metrics(contributions.sum(axis=0))
    samples = {name: np.empty(replicates, dtype=np.float64) for name in point}
    rng = np.random.default_rng(random_seed)
    for replicate in range(replicates):
        group_weights = _draw_group_counts(
            rng, len(unique_groups), first, second
        )
        metrics = _confusion_metrics(group_weights @ contributions)
        for name, value in metrics.items():
            samples[name][replicate] = value
    return {
        name: BootstrapInterval(
            point=value,
            lower_95=float(np.quantile(samples[name], 0.025)),
            upper_95=float(np.quantile(samples[name], 0.975)),
            replicates=replicates,
        )
        for name, value in point.items()
    }


def group_probability_bootstrap(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    groups: Iterable[str],
    *,
    replicates: int = 200,
    random_seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """Return group-aware CIs for ranking and probability-quality metrics."""
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    labels, probabilities, unique_groups, inverse = _validated(
        y_true, y_probability, groups
    )
    first, second = _group_strata(labels, inverse, len(unique_groups))
    point = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "partial_roc_auc_standardized_fpr_0_001": float(
            roc_auc_score(labels, probabilities, max_fpr=0.001)
        ),
        "partial_roc_auc_standardized_fpr_0_0001": float(
            roc_auc_score(labels, probabilities, max_fpr=0.0001)
        ),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }
    samples = {name: np.empty(replicates, dtype=np.float64) for name in point}
    rng = np.random.default_rng(random_seed)
    completed = 0
    attempts = 0
    while completed < replicates:
        attempts += 1
        if attempts > replicates * 10:
            raise RuntimeError("Unable to produce valid two-class bootstrap replicates.")
        group_weights = _draw_group_counts(
            rng, len(unique_groups), first, second
        )
        weights = group_weights[inverse].astype(np.float64)
        if not np.any(weights[labels == 0]) or not np.any(weights[labels == 1]):
            continue
        samples["roc_auc"][completed] = roc_auc_score(
            labels, probabilities, sample_weight=weights
        )
        samples["pr_auc_average_precision"][completed] = average_precision_score(
            labels, probabilities, sample_weight=weights
        )
        samples["partial_roc_auc_standardized_fpr_0_001"][completed] = roc_auc_score(
            labels, probabilities, sample_weight=weights, max_fpr=0.001
        )
        samples["partial_roc_auc_standardized_fpr_0_0001"][completed] = roc_auc_score(
            labels, probabilities, sample_weight=weights, max_fpr=0.0001
        )
        samples["brier_score"][completed] = float(
            np.average((probabilities - labels) ** 2, weights=weights)
        )
        completed += 1
    return {
        name: BootstrapInterval(
            point=value,
            lower_95=float(np.quantile(samples[name], 0.025)),
            upper_95=float(np.quantile(samples[name], 0.975)),
            replicates=replicates,
        )
        for name, value in point.items()
    }


def paired_group_bootstrap_difference(
    y_true: Iterable[int],
    candidate_probability: Iterable[float],
    reference_probability: Iterable[float],
    groups: Iterable[str],
    *,
    candidate_threshold: float,
    reference_threshold: float,
    metric: str = "f2",
    replicates: int = 1_000,
    random_seed: int = 42,
) -> BootstrapInterval:
    """Paired cluster-bootstrap candidate-minus-reference threshold metric."""
    labels, candidate, unique_groups, inverse = _validated(
        y_true, candidate_probability, groups
    )
    reference = np.asarray(reference_probability, dtype=np.float64).reshape(-1)
    if len(reference) != len(labels) or not np.isfinite(reference).all():
        raise ValueError("Reference probabilities must align and be finite.")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")

    def contributions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
        predicted = probabilities >= threshold
        value = np.zeros((len(unique_groups), 4), dtype=np.int64)
        np.add.at(value[:, 0], inverse, predicted & (labels == 1))
        np.add.at(value[:, 1], inverse, predicted & (labels == 0))
        np.add.at(value[:, 2], inverse, ~predicted & (labels == 0))
        np.add.at(value[:, 3], inverse, ~predicted & (labels == 1))
        return value

    candidate_counts = contributions(candidate, candidate_threshold)
    reference_counts = contributions(reference, reference_threshold)
    point = (
        _confusion_metrics(candidate_counts.sum(axis=0))[metric]
        - _confusion_metrics(reference_counts.sum(axis=0))[metric]
    )
    first, second = _group_strata(labels, inverse, len(unique_groups))
    samples = np.empty(replicates, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    for replicate in range(replicates):
        weights = _draw_group_counts(rng, len(unique_groups), first, second)
        samples[replicate] = (
            _confusion_metrics(weights @ candidate_counts)[metric]
            - _confusion_metrics(weights @ reference_counts)[metric]
        )
    return BootstrapInterval(
        point=float(point),
        lower_95=float(np.quantile(samples, 0.025)),
        upper_95=float(np.quantile(samples, 0.975)),
        replicates=replicates,
        method="paired_group_stratified_percentile",
    )
