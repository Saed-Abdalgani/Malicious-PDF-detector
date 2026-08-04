"""Natural-prevalence probability calibration with validation isolation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.metrics import expected_calibration_error


CLIP_EPSILON = 1e-7


def _scores(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Calibration scores must be non-empty and finite.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("Calibration inputs must be probabilities in [0, 1].")
    return array


def _labels(values: Iterable[int], *, expected_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int8).reshape(-1)
    if array.size != expected_rows or not np.isin(array, (0, 1)).all():
        raise ValueError("Calibration labels must be equal-length and binary.")
    if np.unique(array).size != 2:
        raise ValueError("Calibration requires both classes.")
    return array


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, CLIP_EPSILON, 1.0 - CLIP_EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


@dataclass(frozen=True)
class CalibrationSelection:
    selected_method: str
    selected_on: str
    fit_on: str
    candidates: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProbabilityCalibrator:
    """Serializable identity, Platt/sigmoid, or isotonic calibrator."""

    METHODS = {"identity", "sigmoid", "isotonic"}

    def __init__(self, method: str) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown calibration method: {method!r}")
        self.method = method
        self._fitted = False

    def fit(
        self,
        raw_probability: Iterable[float],
        y_true: Iterable[int],
        *,
        partition_name: str,
    ) -> "ProbabilityCalibrator":
        if partition_name != "validation_calibration_fit":
            raise RuntimeError(
                "Probability calibration partition must be validation_calibration_fit."
            )
        if self._fitted:
            raise RuntimeError("ProbabilityCalibrator is immutable after fitting.")
        probabilities = _scores(raw_probability)
        labels = _labels(y_true, expected_rows=len(probabilities))
        if self.method == "sigmoid":
            self.estimator_ = LogisticRegression(
                solver="lbfgs", C=1e6, max_iter=1_000, random_state=0
            ).fit(_logit(probabilities), labels)
        elif self.method == "isotonic":
            self.estimator_ = IsotonicRegression(
                y_min=CLIP_EPSILON,
                y_max=1.0 - CLIP_EPSILON,
                out_of_bounds="clip",
            ).fit(probabilities, labels)
        else:
            self.estimator_ = None
        self.fit_partition_ = partition_name
        self.fit_rows_ = int(len(labels))
        self._fitted = True
        return self

    def predict(self, raw_probability: Iterable[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before use.")
        probabilities = _scores(raw_probability)
        if self.method == "sigmoid":
            calibrated = self.estimator_.predict_proba(_logit(probabilities))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.estimator_.predict(probabilities)
        else:
            calibrated = probabilities
        return np.clip(
            np.asarray(calibrated, dtype=np.float64),
            CLIP_EPSILON,
            1.0 - CLIP_EPSILON,
        )

    def metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted.")
        return {
            "method": self.method,
            "fit_partition": self.fit_partition_,
            "fit_rows": self.fit_rows_,
        }


def fit_and_select_calibrator(
    fit_probability: Iterable[float],
    fit_labels: Iterable[int],
    selection_probability: Iterable[float],
    selection_labels: Iterable[int],
    *,
    fit_partition_name: str,
    selection_partition_name: str,
    methods: tuple[str, ...] = ("identity", "sigmoid", "isotonic"),
) -> tuple[ProbabilityCalibrator, CalibrationSelection]:
    """Fit on one validation slice and select method on a disjoint slice."""
    if fit_partition_name != "validation_calibration_fit":
        raise RuntimeError("Invalid calibration-fit partition name.")
    if selection_partition_name != "validation_calibration_selection":
        raise RuntimeError("Invalid calibration-selection partition name.")
    fit_scores = _scores(fit_probability)
    fit_y = _labels(fit_labels, expected_rows=len(fit_scores))
    selection_scores = _scores(selection_probability)
    selection_y = _labels(selection_labels, expected_rows=len(selection_scores))
    candidates: dict[str, ProbabilityCalibrator] = {}
    evidence: dict[str, dict[str, float]] = {}
    for method in methods:
        calibrator = ProbabilityCalibrator(method).fit(
            fit_scores, fit_y, partition_name=fit_partition_name
        )
        predicted = calibrator.predict(selection_scores)
        candidates[method] = calibrator
        evidence[method] = {
            "brier_score": float(np.mean((predicted - selection_y) ** 2)),
            "expected_calibration_error": expected_calibration_error(
                selection_y, predicted
            ),
        }
    selected_method = min(
        methods,
        key=lambda name: (
            evidence[name]["brier_score"],
            evidence[name]["expected_calibration_error"],
            name,
        ),
    )
    return candidates[selected_method], CalibrationSelection(
        selected_method=selected_method,
        selected_on=selection_partition_name,
        fit_on=fit_partition_name,
        candidates=evidence,
    )
