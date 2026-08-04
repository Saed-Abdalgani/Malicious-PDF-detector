"""Imbalance-aware binary metrics and validation-only threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


EPSILON = np.finfo(np.float64).eps


class MetricInputError(ValueError):
    """Raised when labels, probabilities, or thresholds are scientifically invalid."""


def metric_definitions() -> dict[str, dict[str, str]]:
    """Return formulas and operational interpretations used in Phase 5."""
    return {
        "positive_class": {
            "formula": "malicious PDF = 1",
            "indicates": "Every positive-class metric treats malicious documents as positive.",
        },
        "precision": {
            "formula": "TP / (TP + FP)",
            "indicates": (
                "Alert quality: among PDFs flagged malicious, the fraction truly malicious. "
                "At 99.5% benign prevalence, small FPR can sharply reduce precision."
            ),
        },
        "recall": {
            "formula": "TP / (TP + FN)",
            "indicates": "Detection coverage: the fraction of malicious PDFs caught.",
        },
        "f1": {
            "formula": "2 * Precision * Recall / (Precision + Recall)",
            "indicates": "Equal-weight harmonic balance of alert quality and detection coverage.",
        },
        "f0_5": {
            "formula": "1.25 * P * R / (0.25 * P + R)",
            "indicates": "Precision-weighted operating view for analyst workload control.",
        },
        "f2": {
            "formula": "5 * P * R / (4 * P + R)",
            "indicates": "Recall-weighted security view; primary objective under the FPR limit.",
        },
        "roc_auc": {
            "formula": "Area under TPR-versus-FPR curve across thresholds",
            "indicates": (
                "Ranking ability, not alert quality. A high value can still yield too many "
                "false alerts under extreme imbalance."
            ),
        },
        "partial_roc_auc": {
            "formula": "Standardized ROC area restricted to FPR <= 0.1% / 0.01%",
            "indicates": "Ranking quality in the operationally plausible false-positive region.",
        },
        "pr_auc_average_precision": {
            "formula": "Average precision across the precision-recall curve",
            "indicates": (
                "Threshold-free precision/recall behavior; its no-skill baseline is malicious "
                "prevalence, not 50%."
            ),
        },
        "specificity": {
            "formula": "TN / (TN + FP)",
            "indicates": "Fraction of benign PDFs correctly left unflagged.",
        },
        "false_positive_rate": {
            "formula": "FP / (FP + TN)",
            "indicates": "Benign false-alert rate and basis for false positives per million.",
        },
        "matthews_correlation_coefficient": {
            "formula": "(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
            "indicates": "Balanced correlation using every confusion-matrix cell.",
        },
        "brier_score": {
            "formula": "mean((probability - label)^2)",
            "indicates": "Probability accuracy/calibration; lower is better.",
        },
        "expected_calibration_error": {
            "formula": "sum(bin_fraction * abs(observed_rate - mean_probability))",
            "indicates": "Average probability-to-observed-rate mismatch; lower is better.",
        },
        "accuracy": {
            "formula": "(TP + TN) / N",
            "indicates": (
                "Reported with a warning only: always-benign already reaches at least 99.5%."
            ),
        },
    }


def _validated_arrays(
    y_true: Iterable[int], y_probability: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != probabilities.size:
        raise MetricInputError("Labels and probabilities must be non-empty and equal-length.")
    if not np.isin(labels, (0, 1)).all():
        raise MetricInputError("The malicious class must be encoded as binary label 1.")
    if not np.isfinite(probabilities).all():
        raise MetricInputError("Probabilities must be finite.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise MetricInputError("Probabilities must lie in the closed interval [0, 1].")
    return labels, probabilities


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fbeta_from_counts(
    true_positive: int | float,
    false_positive: int | float,
    false_negative: int | float,
    *,
    beta: float,
) -> float:
    """Return F-beta directly from confusion counts."""
    if beta <= 0:
        raise ValueError("beta must be positive.")
    beta_squared = beta * beta
    denominator = (
        (1.0 + beta_squared) * true_positive
        + beta_squared * false_negative
        + false_positive
    )
    return _safe_divide((1.0 + beta_squared) * true_positive, denominator)


def expected_calibration_error(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error on natural prevalence."""
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    labels, probabilities = _validated_arrays(y_true, y_probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return float(error)


def confusion_counts(
    y_true: Iterable[int], y_probability: Iterable[float], *, threshold: float
) -> dict[str, int]:
    labels, probabilities = _validated_arrays(y_true, y_probability)
    if not 0.0 < float(threshold) < 1.0:
        raise MetricInputError("An operating threshold must be strictly between 0 and 1.")
    predicted = probabilities >= float(threshold)
    positive = labels == 1
    return {
        "true_positive": int(np.sum(predicted & positive)),
        "false_positive": int(np.sum(predicted & ~positive)),
        "true_negative": int(np.sum(~predicted & ~positive)),
        "false_negative": int(np.sum(~predicted & positive)),
    }


def metric_report(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    threshold: float,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Compute the complete Phase 5 metric contract at one locked threshold."""
    labels, probabilities = _validated_arrays(y_true, y_probability)
    counts = confusion_counts(labels, probabilities, threshold=threshold)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    positive = tp + fn
    negative = tn + fp
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, positive)
    specificity = _safe_divide(tn, negative)
    fpr = _safe_divide(fp, negative)
    fnr = _safe_divide(fn, positive)
    accuracy = _safe_divide(tp + tn, labels.size)
    balanced_accuracy = (recall + specificity) / 2.0
    mcc_denominator = np.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = _safe_divide(tp * tn - fp * fn, mcc_denominator)
    prevalence = float(labels.mean())
    result: dict[str, Any] = {
        "threshold": float(threshold),
        **counts,
        "rows": int(labels.size),
        "malicious_prevalence": prevalence,
        "benign_prevalence": 1.0 - prevalence,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "f0_5": fbeta_from_counts(tp, fp, fn, beta=0.5),
        "f1": fbeta_from_counts(tp, fp, fn, beta=1.0),
        "f2": fbeta_from_counts(tp, fp, fn, beta=2.0),
        "matthews_correlation_coefficient": mcc,
        "balanced_accuracy": balanced_accuracy,
        "false_positives_per_million_benign": fpr * 1_000_000.0,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, bins=calibration_bins
        ),
        "pr_no_skill_baseline": prevalence,
        "accuracy_interpretation": (
            "Accuracy is non-operative under extreme imbalance; compare with the "
            "always-benign baseline and false positives per million."
        ),
    }
    if np.unique(labels).size == 2:
        result.update(
            {
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
            }
        )
    else:
        result.update(
            {
                "roc_auc": None,
                "pr_auc_average_precision": None,
                "partial_roc_auc_standardized_fpr_0_001": None,
                "partial_roc_auc_standardized_fpr_0_0001": None,
            }
        )
    return result


@dataclass(frozen=True)
class ThresholdDecision:
    policy: str
    threshold: float
    selected_on: str
    objective: str
    constraint: str | None
    validation_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _threshold_candidates(
    labels: np.ndarray, probabilities: np.ndarray
) -> list[tuple[float, dict[str, float]]]:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    change = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    indices = np.flatnonzero(change)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    candidates: list[tuple[float, dict[str, float]]] = []
    maximum_score = float(sorted_scores[0])
    if maximum_score < 1.0:
        # The zero-alert operating point is necessary when the empirical low-FPR
        # budget is smaller than one false positive. Calibrated Phase 4 outputs
        # are clipped below one, so this remains a valid strict threshold.
        zero_alert_threshold = float(np.nextafter(maximum_score, 1.0))
        candidates.append(
            (
                zero_alert_threshold,
                {
                    "f1": 0.0,
                    "f2": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "fpr": 0.0,
                },
            )
        )
    for index in indices:
        if sorted_scores[index] <= 0.0:
            # A strict positive threshold cannot include exact-zero scores.
            continue
        threshold = float(min(sorted_scores[index], 1.0 - EPSILON))
        tp = int(cumulative_tp[index])
        fp = int(cumulative_fp[index])
        fn = positives - tp
        candidates.append(
            (
                threshold,
                {
                    "f1": fbeta_from_counts(tp, fp, fn, beta=1.0),
                    "f2": fbeta_from_counts(tp, fp, fn, beta=2.0),
                    "recall": _safe_divide(tp, positives),
                    "precision": _safe_divide(tp, tp + fp),
                    "fpr": _safe_divide(fp, negatives),
                },
            )
        )
    return candidates


def _best_candidate(
    candidates: list[tuple[float, dict[str, float]]],
    *,
    objective: str,
    maximum_fpr: float | None = None,
) -> float:
    eligible = [
        candidate
        for candidate in candidates
        if maximum_fpr is None or candidate[1]["fpr"] <= maximum_fpr + 1e-15
    ]
    if not eligible:
        raise RuntimeError(f"No empirical threshold satisfies FPR <= {maximum_fpr}.")
    # Objective, then lower FPR, then greater recall, then higher threshold.
    return max(
        eligible,
        key=lambda value: (
            value[1][objective],
            -value[1]["fpr"],
            value[1]["recall"],
            value[0],
        ),
    )[0]


def select_validation_thresholds(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    partition_name: str,
) -> dict[str, ThresholdDecision]:
    """Lock all required operating thresholds using validation labels only."""
    if partition_name != "validation_threshold_selection":
        raise RuntimeError(
            "Operating thresholds may only be selected on the dedicated validation "
            "threshold-selection partition."
        )
    labels, probabilities = _validated_arrays(y_true, y_probability)
    if np.unique(labels).size != 2:
        raise MetricInputError("Threshold selection requires both binary classes.")
    candidates = _threshold_candidates(labels, probabilities)
    policies = {
        "fixed_0_5": (0.5, "fixed", None),
        "max_f1": (
            _best_candidate(candidates, objective="f1"),
            "maximize_f1",
            None,
        ),
        "max_f2": (
            _best_candidate(candidates, objective="f2"),
            "maximize_f2",
            None,
        ),
        "fpr_lte_0_001": (
            _best_candidate(candidates, objective="f2", maximum_fpr=0.001),
            "maximize_f2",
            "false_positive_rate <= 0.001",
        ),
        "fpr_lte_0_0001": (
            _best_candidate(candidates, objective="f2", maximum_fpr=0.0001),
            "maximize_f2",
            "false_positive_rate <= 0.0001",
        ),
    }
    return {
        name: ThresholdDecision(
            policy=name,
            threshold=float(threshold),
            selected_on=partition_name,
            objective=objective,
            constraint=constraint,
            validation_metrics=metric_report(
                labels, probabilities, threshold=float(threshold)
            ),
        )
        for name, (threshold, objective, constraint) in policies.items()
    }


def score_development_fold_at_fpr(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    *,
    partition_name: str,
    maximum_fpr: float = 0.001,
) -> ThresholdDecision:
    """Score a train-only temporal fold without creating a deployable threshold."""
    if partition_name != "train_temporal_fold_validation":
        raise RuntimeError("Development tuning may only use a train temporal fold.")
    labels, probabilities = _validated_arrays(y_true, y_probability)
    if np.unique(labels).size != 2:
        raise MetricInputError("Development scoring requires both classes.")
    threshold = _best_candidate(
        _threshold_candidates(labels, probabilities),
        objective="f2",
        maximum_fpr=maximum_fpr,
    )
    return ThresholdDecision(
        policy="development_f2_at_fpr",
        threshold=threshold,
        selected_on=partition_name,
        objective="maximize_f2",
        constraint=f"false_positive_rate <= {maximum_fpr}",
        validation_metrics=metric_report(labels, probabilities, threshold=threshold),
    )
