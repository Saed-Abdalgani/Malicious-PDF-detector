"""Leakage-resistant preprocessing fitted exclusively on the train partition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.artifacts import verify_artifact_compatibility, write_artifact_metadata
from src.experiment import ExperimentIdentity, create_experiment_identity


class PreprocessingStateError(RuntimeError):
    """Raised on leakage-prone fitting or use before fit."""


@dataclass(frozen=True)
class PreprocessingMetadata:
    fit_partition: str
    model_family: str
    input_features: tuple[str, ...]
    active_features: tuple[str, ...]
    constant_features: tuple[str, ...]
    output_features: tuple[str, ...]
    lower_quantile: float
    upper_quantile: float
    imputation: str
    scaler: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}


class TrainOnlyPreprocessor:
    """Median imputation plus family-specific stable transforms.

    Tree models receive imputed raw values and missing indicators. Neural models
    additionally receive train-learned clipping flags and standardized values.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        model_family: str = "neural",
        lower_quantile: float = 0.0001,
        upper_quantile: float = 0.9999,
        add_missing_indicators: bool = True,
        drop_constant: bool = True,
    ) -> None:
        if model_family not in {"neural", "tree"}:
            raise ValueError("model_family must be 'neural' or 'tree'.")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Invalid clipping quantiles.")
        self.feature_names = tuple(feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique.")
        self.model_family = model_family
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.add_missing_indicators = add_missing_indicators
        self.drop_constant = drop_constant
        self._fitted = False

    def _finish_feature_names(self) -> None:
        names = list(self.active_features_)
        if self.add_missing_indicators:
            names.extend(f"{name}__missing" for name in self.feature_names)
        if self.model_family == "neural":
            names.extend(f"{name}__clipped" for name in self.active_features_)
        self.output_features_ = tuple(names)

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Input is missing preprocessing features: {missing}")
        selected = frame.loc[:, self.feature_names]
        bad = [name for name in self.feature_names if not pd.api.types.is_numeric_dtype(selected[name])]
        if bad:
            raise TypeError(f"Preprocessing features must already be numeric: {bad}")
        return selected.astype(np.float64)

    def fit(self, frame: pd.DataFrame, *, partition_name: str) -> "TrainOnlyPreprocessor":
        if partition_name != "train":
            raise PreprocessingStateError(
                "Learned preprocessing may only be fit with partition_name='train'."
            )
        if self._fitted:
            raise PreprocessingStateError("Preprocessor is already fitted and immutable.")
        selected = self._select(frame)
        medians = selected.median(axis=0, skipna=True).fillna(0.0)
        imputed = selected.fillna(medians)
        constant = tuple(name for name in self.feature_names if imputed[name].nunique(dropna=False) <= 1)
        active = tuple(name for name in self.feature_names if not (self.drop_constant and name in constant))
        if not active:
            raise PreprocessingStateError("Every input feature is constant in train.")
        active_frame = imputed.loc[:, active]
        self.medians_ = medians
        self.constant_features_ = constant
        self.active_features_ = active
        self.clip_lower_ = active_frame.quantile(self.lower_quantile)
        self.clip_upper_ = active_frame.quantile(self.upper_quantile)
        self.scaler_ = None
        if self.model_family == "neural":
            clipped = active_frame.clip(self.clip_lower_, self.clip_upper_, axis=1)
            self.scaler_ = StandardScaler().fit(clipped.to_numpy(dtype=np.float64))
        self._finish_feature_names()
        self._fitted = True
        return self

    def fit_batches(
        self,
        batch_factory: Callable[[], Iterable[pd.DataFrame]],
        *,
        partition_name: str,
        quantile_sample_rows: int = 250_000,
        random_seed: int = 42,
    ) -> "TrainOnlyPreprocessor":
        """Fit from repeatable bounded batches using a deterministic reservoir.

        Medians and extreme clipping quantiles are estimated from at most
        ``quantile_sample_rows`` train rows. Standardization moments are then
        calculated over every original train row with ``partial_fit``.
        """
        if partition_name != "train":
            raise PreprocessingStateError(
                "Learned preprocessing may only be fit with partition_name='train'."
            )
        if self._fitted:
            raise PreprocessingStateError("Preprocessor is already fitted and immutable.")
        if quantile_sample_rows < 1:
            raise ValueError("quantile_sample_rows must be positive.")
        reservoir = pd.DataFrame(columns=self.feature_names, dtype=np.float64)
        minima = pd.Series(np.inf, index=self.feature_names, dtype=float)
        maxima = pd.Series(-np.inf, index=self.feature_names, dtype=float)
        observed = 0
        for batch_index, batch in enumerate(batch_factory()):
            selected = self._select(batch)
            observed += len(selected)
            minima = pd.concat([minima, selected.min(skipna=True)], axis=1).min(axis=1)
            maxima = pd.concat([maxima, selected.max(skipna=True)], axis=1).max(axis=1)
            reservoir = pd.concat([reservoir, selected], ignore_index=True)
            if len(reservoir) > quantile_sample_rows:
                reservoir = reservoir.sample(
                    n=quantile_sample_rows,
                    random_state=random_seed + batch_index,
                    ignore_index=True,
                )
        if observed == 0:
            raise PreprocessingStateError("Cannot fit preprocessing on an empty train set.")
        self.medians_ = reservoir.median(axis=0, skipna=True).fillna(0.0)
        constant = tuple(
            name for name in self.feature_names
            if (not np.isfinite(minima[name]) and not np.isfinite(maxima[name]))
            or minima[name] == maxima[name]
        )
        self.constant_features_ = constant
        self.active_features_ = tuple(
            name for name in self.feature_names
            if not (self.drop_constant and name in constant)
        )
        if not self.active_features_:
            raise PreprocessingStateError("Every input feature is constant in train.")
        sample_imputed = reservoir.fillna(self.medians_).loc[:, self.active_features_]
        self.clip_lower_ = sample_imputed.quantile(self.lower_quantile)
        self.clip_upper_ = sample_imputed.quantile(self.upper_quantile)
        self.scaler_ = None
        if self.model_family == "neural":
            self.scaler_ = StandardScaler()
            for batch in batch_factory():
                selected = self._select(batch)
                imputed = selected.fillna(self.medians_).loc[:, self.active_features_]
                clipped = imputed.clip(self.clip_lower_, self.clip_upper_, axis=1)
                self.scaler_.partial_fit(clipped.to_numpy(dtype=np.float64))
        self._finish_feature_names()
        self._fitted = True
        self.quantile_estimation_rows_ = len(reservoir)
        self.fit_rows_ = observed
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise PreprocessingStateError("Preprocessor must be fit before transform.")
        selected = self._select(frame)
        missing = selected.isna().astype(np.float32)
        imputed = selected.fillna(self.medians_).loc[:, self.active_features_]
        pieces: list[np.ndarray] = []
        if self.model_family == "neural":
            below = imputed.lt(self.clip_lower_, axis=1)
            above = imputed.gt(self.clip_upper_, axis=1)
            clipped_flags = (below | above).astype(np.float32)
            clipped = imputed.clip(self.clip_lower_, self.clip_upper_, axis=1)
            pieces.append(self.scaler_.transform(clipped.to_numpy(dtype=np.float64)).astype(np.float32))
        else:
            pieces.append(imputed.to_numpy(dtype=np.float32))
        if self.add_missing_indicators:
            pieces.append(missing.to_numpy(dtype=np.float32))
        if self.model_family == "neural":
            pieces.append(clipped_flags.to_numpy(dtype=np.float32))
        return pd.DataFrame(
            np.concatenate(pieces, axis=1), index=frame.index, columns=self.output_features_
        )

    def fit_transform(self, frame: pd.DataFrame, *, partition_name: str) -> pd.DataFrame:
        return self.fit(frame, partition_name=partition_name).transform(frame)

    def metadata(self) -> PreprocessingMetadata:
        if not self._fitted:
            raise PreprocessingStateError("Preprocessor is not fitted.")
        return PreprocessingMetadata(
            fit_partition="train",
            model_family=self.model_family,
            input_features=self.feature_names,
            active_features=self.active_features_,
            constant_features=self.constant_features_,
            output_features=self.output_features_,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
            imputation="train_median",
            scaler="standard" if self.model_family == "neural" else "none",
        )

    def save(
        self,
        path: Path,
        *,
        identity: ExperimentIdentity | None = None,
    ) -> Path:
        metadata = self.metadata()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)
        write_artifact_metadata(
            output,
            "train_only_preprocessor",
            identity or create_experiment_identity(),
            extra=metadata.to_dict(),
        )
        return output

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        identity: ExperimentIdentity | None = None,
    ) -> "TrainOnlyPreprocessor":
        verify_artifact_compatibility(path, identity or create_experiment_identity())
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Artifact at {path} is not a {cls.__name__}.")
        value.metadata()
        return value
