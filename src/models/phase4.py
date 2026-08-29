"""Phase 4 fair model comparison on verified train/validation partitions only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DATA_REPORTS_DIR,
    FIRST_SEEN_COLUMN,
    GROUP_ID_COLUMN,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SPLITS_DIR,
)
from src.data.access import TrainingSplitAccess, verify_training_split_access
from src.data.validate import verify_validated_dataset
from src.experiment import (
    ExperimentIdentity,
    canonical_json,
    create_experiment_identity,
    load_experiment_config,
    sha256_file,
)
from src.features.pipeline import FeaturePipelineV2
from src.models.bundle import (
    CalibratedMember,
    Phase4ModelBundle,
    positive_class_probability,
)
from src.models.calibration import (
    ProbabilityCalibrator,
    fit_and_select_calibrator,
)
from src.models.matrix import (
    load_model_matrix,
    materialize_model_matrix,
)
from src.models.metrics import (
    metric_report,
    score_development_fold_at_fpr,
    select_validation_thresholds,
)
from src.models.mlp import MaliciousPDFClassifier
from src.models.tabular_transformer import AsymmetricFocalLoss, FTTransformer
from src.models.uncertainty import paired_group_bootstrap_difference
from src.utils.atomic import atomic_write_json, atomic_write_via


PHASE4_VERSION = "1.0.0"
DEFAULT_SEEDS = (42, 1_337, 2_026)


class Phase4GateError(RuntimeError):
    """Raised when Phase 4 would violate a scientific gate."""


@dataclass(frozen=True)
class CandidateSpec:
    model_name: str
    model_family: str
    variant: str
    weighted: bool
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def configuration_id(self) -> str:
        payload = canonical_json(asdict(self)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def variant_key(self) -> str:
        return f"{self.model_name}__{self.variant}"


@dataclass(frozen=True)
class ValidationRoleManifest:
    version: str
    split_manifest_sha256: str
    role_counts: dict[str, int]
    role_group_counts: dict[str, int]
    role_malicious_prevalence: dict[str, float]
    role_time_ranges_utc: dict[str, dict[str, str]]
    group_overlap_zero: bool
    strict_temporal_order: bool
    assignment_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PeakMemoryMonitor:
    """Sample process RSS during a fit without retaining training data."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemoryMonitor":
        process = psutil.Process()

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, process.memory_info().rss
                )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, psutil.Process().memory_info().rss
        )


def hardware_and_packages() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "python": platform.python_version(),
        "packages": packages,
    }


def benchmark_estimator(estimator: Any, features: np.ndarray) -> dict[str, Any]:
    """Measure finalist size, batch throughput, and single-row CPU latency."""
    sample_rows = min(len(features), 8_192)
    sample = features[:sample_rows]
    positive_class_probability(estimator, sample[: min(16, sample_rows)])
    batch_times = []
    for _ in range(3):
        start = time.perf_counter()
        positive_class_probability(estimator, sample)
        batch_times.append(time.perf_counter() - start)
    single_times = []
    single = sample[:1]
    for _ in range(50):
        start = time.perf_counter()
        positive_class_probability(estimator, single)
        single_times.append(time.perf_counter() - start)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".model-size")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if isinstance(estimator, torch.nn.Module):
            torch.save(estimator.state_dict(), temporary)
        else:
            joblib.dump(estimator, temporary, compress=3)
        model_size = temporary.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)
    median_batch = float(np.median(batch_times))
    return {
        "serialized_model_size_bytes": int(model_size),
        "throughput_benchmark_rows": sample_rows,
        "batch_throughput_rows_per_second": (
            float(sample_rows / median_batch) if median_batch else None
        ),
        "single_file_latency_median_ms": float(np.median(single_times) * 1_000.0),
        "single_file_latency_p95_ms": float(np.quantile(single_times, 0.95) * 1_000.0),
    }


def default_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return small, resource-bounded tuning grids for every required model."""
    specs: list[CandidateSpec] = [
        CandidateSpec("always_benign", "dummy", "unweighted", False, {}),
    ]
    for weighted in (False, True):
        variant = "cost_sensitive" if weighted else "unweighted"
        for c_value in (0.1, 1.0):
            specs.append(
                CandidateSpec(
                    "logistic_regression", "linear", variant, weighted,
                    {"C": c_value},
                )
            )
        for depth in (16, 24):
            specs.append(
                CandidateSpec(
                    "extra_trees", "tree", variant, weighted,
                    {
                        "n_estimators": 400,
                        "max_depth": depth,
                        "min_samples_leaf": 2,
                        "max_features": "sqrt",
                    },
                )
            )
        for leaves in (31, 63):
            specs.append(
                CandidateSpec(
                    "lightgbm", "tree", variant, weighted,
                    {
                        "n_estimators": 800,
                        "num_leaves": leaves,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                )
            )
        for depth in (6, 8):
            specs.append(
                CandidateSpec(
                    "xgboost_hist", "tree", variant, weighted,
                    {
                        "n_estimators": 800,
                        "max_depth": depth,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                )
            )
        for dropout in (0.2, 0.3):
            specs.append(
                CandidateSpec(
                    "fully_connected_mlp", "neural", variant, weighted,
                    {
                        "dropout1": dropout,
                        "dropout2": max(0.1, dropout - 0.1),
                        "learning_rate": 1e-3,
                        "tuning_epochs": 8,
                        "final_epochs": 20,
                        "batch_size": 4_096,
                        "loss": "weighted_bce" if weighted else "bce",
                    },
                )
            )
        for token_dimension, blocks in ((32, 2), (64, 3)):
            specs.append(
                CandidateSpec(
                    "ft_transformer", "neural", variant, weighted,
                    {
                        "token_dimension": token_dimension,
                        "blocks": blocks,
                        "heads": 8,
                        "learning_rate": 5e-4,
                        "tuning_epochs": 8,
                        "final_epochs": 20,
                        "batch_size": 2_048,
                        "loss": "asymmetric_focal" if weighted else "bce",
                    },
                )
            )
    return tuple(specs)


def validate_candidate_contract(
    specs: Iterable[CandidateSpec], required_models: Iterable[str]
) -> tuple[CandidateSpec, ...]:
    """Reject programmatic attempts to bypass required models or ablations."""
    candidates = tuple(specs)
    expected = set(required_models)
    observed = {spec.model_name for spec in candidates}
    if observed != expected:
        raise Phase4GateError(
            "Phase 4 candidate inventory differs from the configured required set."
        )
    for model_name in sorted(expected):
        model_specs = [spec for spec in candidates if spec.model_name == model_name]
        variants = {spec.variant for spec in model_specs}
        required_variants = (
            {"unweighted"}
            if model_name == "always_benign"
            else {"unweighted", "cost_sensitive"}
        )
        if variants != required_variants:
            raise Phase4GateError(
                f"Phase 4 weighting ablation is incomplete for {model_name}."
            )
        families = {spec.model_family for spec in model_specs}
        if len(families) != 1:
            raise Phase4GateError(f"Model family is inconsistent for {model_name}.")
        for spec in model_specs:
            if spec.weighted != (spec.variant == "cost_sensitive"):
                raise Phase4GateError(
                    f"Weighting flag/variant mismatch for {spec.variant_key}."
                )
    return candidates


def _positive_weight(labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise Phase4GateError("Training requires both natural-prevalence classes.")
    return float(negatives / positives)


def _create_classical_estimator(
    spec: CandidateSpec, *, seed: int, positive_weight: float
) -> Any:
    parameters = dict(spec.parameters)
    if spec.model_name == "always_benign":
        return DummyClassifier(strategy="constant", constant=0)
    if spec.model_name == "logistic_regression":
        return LogisticRegression(
            C=float(parameters["C"]),
            penalty="l2",
            solver="saga",
            max_iter=300,
            class_weight="balanced" if spec.weighted else None,
            random_state=seed,
            n_jobs=-1,
        )
    if spec.model_name == "extra_trees":
        return ExtraTreesClassifier(
            **parameters,
            class_weight="balanced_subsample" if spec.weighted else None,
            random_state=seed,
            n_jobs=-1,
        )
    if spec.model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **parameters,
            objective="binary",
            scale_pos_weight=positive_weight if spec.weighted else 1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if spec.model_name == "xgboost_hist":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=positive_weight if spec.weighted else 1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Not a classical candidate: {spec.model_name}")


def _create_neural_estimator(spec: CandidateSpec, input_dimension: int) -> nn.Module:
    parameters = spec.parameters
    if spec.model_name == "fully_connected_mlp":
        return MaliciousPDFClassifier(
            input_dim=input_dimension,
            dropout1=float(parameters["dropout1"]),
            dropout2=float(parameters["dropout2"]),
        )
    if spec.model_name == "ft_transformer":
        return FTTransformer(
            input_dimension,
            token_dimension=int(parameters["token_dimension"]),
            blocks=int(parameters["blocks"]),
            heads=int(parameters["heads"]),
        )
    raise ValueError(f"Not a neural candidate: {spec.model_name}")


def _fit_neural(
    spec: CandidateSpec,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    tuning: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = _create_neural_estimator(spec, features.shape[1]).cpu()
    parameters = spec.parameters
    epochs = int(parameters["tuning_epochs" if tuning else "final_epochs"])
    batch_size = int(parameters["batch_size"])
    feature_tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    label_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(feature_tensor, label_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    positive_weight = _positive_weight(np.asarray(labels))
    loss_name = str(parameters["loss"])
    if loss_name == "asymmetric_focal":
        criterion: nn.Module = AsymmetricFocalLoss(
            positive_weight=positive_weight, gamma_positive=1.0, gamma_negative=4.0
        )
    else:
        pos_weight = (
            torch.tensor([positive_weight], dtype=torch.float32)
            if loss_name == "weighted_bce"
            else None
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=1e-4,
    )
    history: list[float] = []
    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).reshape(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            total_rows += len(batch_labels)
        history.append(total_loss / total_rows)
    model.eval()
    return model, {
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": loss_name,
        "final_training_loss": history[-1],
        "loss_history": history,
    }


def fit_candidate(
    spec: CandidateSpec,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    tuning: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Fit one candidate without validation/test labels entering the estimator."""
    y = np.asarray(labels, dtype=np.int8)
    start = time.perf_counter()
    with PeakMemoryMonitor() as memory:
        if spec.model_family == "neural":
            estimator, details = _fit_neural(
                spec, features, y, seed=seed, tuning=tuning
            )
        else:
            estimator = _create_classical_estimator(
                spec, seed=seed, positive_weight=_positive_weight(y)
            )
            estimator.fit(features, y)
            details = {}
    elapsed = time.perf_counter() - start
    benchmark = {} if tuning else benchmark_estimator(estimator, features)
    return estimator, {
        **details,
        **benchmark,
        "wall_time_seconds": elapsed,
        "peak_process_rss_bytes": int(memory.peak_rss_bytes),
        "fit_rows": int(len(y)),
        "fit_partition": "train_temporal_development" if tuning else "train_full",
        "seed": seed,
    }


def select_development_indices(
    metadata: pd.DataFrame,
    *,
    maximum_rows: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select complete groups across source, label, and temporal strata."""
    if maximum_rows < 1:
        raise ValueError("maximum_rows must be positive.")
    if len(metadata) <= maximum_rows:
        return np.arange(len(metadata)), {
            "strategy": "all_train_rows",
            "selected_rows": len(metadata),
            "selected_groups": int(metadata[GROUP_ID_COLUMN].nunique()),
        }
    required = {GROUP_ID_COLUMN, FIRST_SEEN_COLUMN, "Class"}
    if not required.issubset(metadata):
        raise Phase4GateError("Development subset metadata is incomplete.")
    frame = metadata.copy()
    frame[FIRST_SEEN_COLUMN] = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True)
    aggregations: dict[str, tuple[str, str]] = {
        "minimum_time": (FIRST_SEEN_COLUMN, "min"),
        "rows": ("Class", "size"),
        "has_malicious": ("Class", "max"),
    }
    if "source_id" in frame:
        aggregations["source"] = ("source_id", "first")
    groups = frame.groupby(GROUP_ID_COLUMN, sort=False).agg(**aggregations)
    groups["source"] = groups.get("source", pd.Series("unknown", index=groups.index))
    time_rank = groups["minimum_time"].rank(method="first")
    time_bins = min(10, len(groups))
    groups["time_bin"] = pd.qcut(
        time_rank, q=time_bins, labels=False, duplicates="drop"
    ).astype(int)
    groups["stratum"] = list(
        zip(groups["source"].astype(str), groups["has_malicious"], groups["time_bin"])
    )
    groups["selection_hash"] = [
        hashlib.sha256(f"{random_seed}:{group_id}".encode("utf-8")).hexdigest()
        for group_id in groups.index.astype(str)
    ]
    selected: set[str] = set()
    selected_rows = 0
    for _stratum, stratum in groups.groupby("stratum", sort=True):
        target = max(
            1,
            int(round(maximum_rows * float(stratum["rows"].sum()) / len(frame))),
        )
        accumulated = 0
        for group_id, row in stratum.sort_values("selection_hash").iterrows():
            rows = int(row["rows"])
            if selected_rows + rows > maximum_rows:
                continue
            selected.add(str(group_id))
            selected_rows += rows
            accumulated += rows
            if accumulated >= target:
                break
    for group_id, row in groups.sort_values("selection_hash").iterrows():
        group_key = str(group_id)
        rows = int(row["rows"])
        if group_key not in selected and selected_rows + rows <= maximum_rows:
            selected.add(group_key)
            selected_rows += rows
    positions = np.flatnonzero(
        frame[GROUP_ID_COLUMN].astype(str).isin(selected).to_numpy()
    )
    if not len(positions) or np.unique(frame.iloc[positions]["Class"]).size != 2:
        raise Phase4GateError("Development subset does not preserve both classes.")
    return positions, {
        "strategy": "complete_group_source_label_time_stratified_hash_sample",
        "selected_rows": int(len(positions)),
        "selected_groups": len(selected),
        "maximum_rows": maximum_rows,
        "random_seed": random_seed,
        "time_bins": time_bins,
    }


def _strict_boundaries(
    groups: pd.DataFrame, target_fractions: tuple[float, ...]
) -> tuple[int, ...]:
    total_rows = int(groups["rows"].sum())
    cumulative = groups["rows"].cumsum().to_numpy()
    boundaries: list[int] = []
    start = 1
    for fraction in target_fractions:
        target = total_rows * fraction
        preferred = int(np.searchsorted(cumulative, target, side="left") + 1)
        candidates = sorted(
            range(start, len(groups)), key=lambda index: abs(index - preferred)
        )
        selected = None
        for index in candidates:
            left_max = groups.iloc[:index]["maximum_time"].max()
            right_min = groups.iloc[index:]["minimum_time"].min()
            if left_max < right_min:
                selected = index
                break
        if selected is None:
            raise Phase4GateError("No strict group-temporal validation boundary exists.")
        boundaries.append(selected)
        start = selected + 1
    return tuple(boundaries)


def assign_validation_roles(
    metadata: pd.DataFrame, *, split_manifest_sha256: str
) -> tuple[np.ndarray, ValidationRoleManifest]:
    """Create three disjoint group-temporal validation roles."""
    required = {GROUP_ID_COLUMN, FIRST_SEEN_COLUMN, "Class"}
    if not required.issubset(metadata):
        raise Phase4GateError(f"Validation metadata lacks {sorted(required - set(metadata))}.")
    frame = metadata.loc[:, list(required)].copy()
    frame[FIRST_SEEN_COLUMN] = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True)
    groups = (
        frame.groupby(GROUP_ID_COLUMN, sort=False)
        .agg(
            minimum_time=(FIRST_SEEN_COLUMN, "min"),
            maximum_time=(FIRST_SEEN_COLUMN, "max"),
            rows=("Class", "size"),
        )
        .sort_values(["minimum_time", GROUP_ID_COLUMN], kind="mergesort")
    )
    first, second = _strict_boundaries(groups, (0.40, 0.65))
    role_names = (
        "validation_calibration_fit",
        "validation_calibration_selection",
        "validation_threshold_selection",
    )
    group_roles = pd.Series(index=groups.index, dtype="object")
    group_roles.iloc[:first] = role_names[0]
    group_roles.iloc[first:second] = role_names[1]
    group_roles.iloc[second:] = role_names[2]
    roles = frame[GROUP_ID_COLUMN].map(group_roles).to_numpy(dtype=str)
    role_counts: dict[str, int] = {}
    role_group_counts: dict[str, int] = {}
    prevalence: dict[str, float] = {}
    time_ranges: dict[str, dict[str, str]] = {}
    group_sets: list[set[str]] = []
    for role in role_names:
        mask = roles == role
        labels = frame.loc[mask, "Class"].to_numpy(dtype=np.int8)
        if np.unique(labels).size != 2:
            raise Phase4GateError(f"Validation role {role} does not contain both classes.")
        role_counts[role] = int(mask.sum())
        role_groups = set(frame.loc[mask, GROUP_ID_COLUMN].astype(str))
        group_sets.append(role_groups)
        role_group_counts[role] = len(role_groups)
        prevalence[role] = float(labels.mean())
        times = frame.loc[mask, FIRST_SEEN_COLUMN]
        time_ranges[role] = {
            "minimum": times.min().isoformat(),
            "maximum": times.max().isoformat(),
        }
    overlap_zero = all(
        not group_sets[right] & group_sets[left]
        for right in range(len(group_sets))
        for left in range(right)
    )
    strict = all(
        pd.Timestamp(time_ranges[role_names[index]]["maximum"])
        < pd.Timestamp(time_ranges[role_names[index + 1]]["minimum"])
        for index in range(2)
    )
    assignment_hash = hashlib.sha256(
        canonical_json(
            {
                "sample_roles": [
                    [str(group), str(role)]
                    for group, role in zip(frame[GROUP_ID_COLUMN], roles)
                ]
            }
        ).encode("utf-8")
    ).hexdigest()
    return roles, ValidationRoleManifest(
        version="1.0.0",
        split_manifest_sha256=split_manifest_sha256,
        role_counts=role_counts,
        role_group_counts=role_group_counts,
        role_malicious_prevalence=prevalence,
        role_time_ranges_utc=time_ranges,
        group_overlap_zero=overlap_zero,
        strict_temporal_order=strict,
        assignment_sha256=assignment_hash,
    )


def build_train_temporal_folds(
    metadata: pd.DataFrame, *, folds: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding, strict group-temporal folds using train rows only."""
    if folds < 2:
        raise ValueError("At least two temporal folds are required.")
    frame = metadata.loc[:, [GROUP_ID_COLUMN, FIRST_SEEN_COLUMN, "Class"]].copy()
    frame[FIRST_SEEN_COLUMN] = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True)
    groups = (
        frame.groupby(GROUP_ID_COLUMN, sort=False)
        .agg(
            minimum_time=(FIRST_SEEN_COLUMN, "min"),
            maximum_time=(FIRST_SEEN_COLUMN, "max"),
            rows=("Class", "size"),
        )
        .sort_values(["minimum_time", GROUP_ID_COLUMN], kind="mergesort")
    )
    targets = tuple(index / (folds + 1) for index in range(1, folds + 1))
    boundaries = _strict_boundaries(groups, targets)
    blocks = np.split(groups.index.to_numpy(), boundaries)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    group_values = frame[GROUP_ID_COLUMN].to_numpy()
    labels = frame["Class"].to_numpy(dtype=np.int8)
    for index in range(folds):
        train_groups = np.concatenate(blocks[: index + 1])
        validation_groups = blocks[index + 1]
        train_index = np.flatnonzero(np.isin(group_values, train_groups))
        validation_index = np.flatnonzero(np.isin(group_values, validation_groups))
        if np.unique(labels[train_index]).size != 2 or np.unique(
            labels[validation_index]
        ).size != 2:
            raise Phase4GateError("A train temporal fold lacks both classes.")
        result.append((train_index, validation_index))
    return result


def tune_candidate_family(
    specs: Iterable[CandidateSpec],
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    seed: int,
    folds: int = 3,
    maximum_fpr: float = 0.001,
) -> tuple[CandidateSpec, list[dict[str, Any]]]:
    """Select one small configuration through train-only group-temporal folds."""
    candidates = list(specs)
    if not candidates:
        raise ValueError("No candidate configurations supplied.")
    fold_indices = build_train_temporal_folds(metadata, folds=folds)
    records: list[dict[str, Any]] = []
    for spec in candidates:
        fold_scores = []
        for fold_index, (train_index, validation_index) in enumerate(fold_indices):
            estimator, resources = fit_candidate(
                spec,
                features[train_index],
                labels[train_index],
                seed=seed + fold_index,
                tuning=True,
            )
            probability = positive_class_probability(
                estimator, features[validation_index]
            )
            decision = score_development_fold_at_fpr(
                labels[validation_index],
                probability,
                partition_name="train_temporal_fold_validation",
                maximum_fpr=maximum_fpr,
            )
            fold_scores.append(decision.validation_metrics["f2"])
            records.append(
                {
                    "configuration_id": spec.configuration_id,
                    "model_name": spec.model_name,
                    "variant": spec.variant,
                    "fold": fold_index,
                    "train_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "f2_at_fpr": decision.validation_metrics["f2"],
                    "false_positive_rate": decision.validation_metrics[
                        "false_positive_rate"
                    ],
                    "threshold": decision.threshold,
                    **resources,
                }
            )
        records.append(
            {
                "configuration_id": spec.configuration_id,
                "model_name": spec.model_name,
                "variant": spec.variant,
                "fold": "mean",
                "f2_at_fpr": float(np.mean(fold_scores)),
                "f2_std": float(np.std(fold_scores)),
            }
        )
    mean_rows = [record for record in records if record["fold"] == "mean"]
    best_id = max(
        mean_rows,
        key=lambda record: (
            record["f2_at_fpr"],
            -record["f2_std"],
            record["configuration_id"],
        ),
    )["configuration_id"]
    return next(spec for spec in candidates if spec.configuration_id == best_id), records


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    return atomic_write_via(path, lambda temporary: frame.to_csv(temporary, index=False))


def select_validation_champion(
    bundles: dict[str, Phase4ModelBundle],
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_replicates: int,
    random_seed: int,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Select by validation F2/FPR and require neural superiority over best tree."""
    scoreboard: list[dict[str, Any]] = []
    for key, bundle in bundles.items():
        decision = bundle.thresholds[bundle.selected_policy]
        metrics = decision["validation_metrics"]
        member_resources = [member.resource_usage for member in bundle.members]
        latency_values = [
            float(value["single_file_latency_median_ms"])
            for value in member_resources
            if value.get("single_file_latency_median_ms") is not None
        ]
        model_sizes = [
            int(value["serialized_model_size_bytes"])
            for value in member_resources
            if value.get("serialized_model_size_bytes") is not None
        ]
        scoreboard.append(
            {
                "bundle_key": key,
                "model_name": bundle.model_name,
                "model_family": bundle.model_family,
                "variant": bundle.variant,
                "selected_policy": bundle.selected_policy,
                "threshold": decision["threshold"],
                "validation_f2": metrics["f2"],
                "validation_f1": metrics["f1"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_fpr": metrics["false_positive_rate"],
                "validation_brier_score": metrics["brier_score"],
                "validation_expected_calibration_error": metrics[
                    "expected_calibration_error"
                ],
                "ensemble_serialized_size_bytes": sum(model_sizes),
                "member_single_file_latency_median_ms_mean": (
                    float(np.mean(latency_values)) if latency_values else None
                ),
            }
        )
    eligible = [
        row
        for row in scoreboard
        if row["model_family"] != "dummy" and row["validation_fpr"] <= 0.001 + 1e-15
    ]
    if not eligible:
        raise Phase4GateError("No non-dummy finalist satisfies validation FPR <= 0.1%.")
    point_best = max(
        eligible,
        key=lambda row: (
            row["validation_f2"],
            -row["validation_fpr"],
            -row["validation_brier_score"],
            row["validation_precision"],
        ),
    )
    tree_best = max(
        (row for row in eligible if row["model_family"] == "tree"),
        key=lambda row: row["validation_f2"],
        default=None,
    )
    selected = point_best
    paired_evidence = None
    rationale = "Highest validation F2 subject to FPR <= 0.1%."
    if point_best["model_family"] == "neural":
        if tree_best is None:
            raise Phase4GateError("Neural champion comparison requires a tree finalist.")
        candidate_bundle = bundles[point_best["bundle_key"]]
        tree_bundle = bundles[tree_best["bundle_key"]]
        paired = paired_group_bootstrap_difference(
            labels,
            probabilities[point_best["bundle_key"]],
            probabilities[tree_best["bundle_key"]],
            groups,
            candidate_threshold=candidate_bundle.threshold,
            reference_threshold=tree_bundle.threshold,
            metric="f2",
            replicates=bootstrap_replicates,
            random_seed=random_seed,
        )
        paired_evidence = paired.to_dict()
        tree_latency = tree_best["member_single_file_latency_median_ms_mean"]
        neural_latency = point_best["member_single_file_latency_median_ms_mean"]
        calibration_acceptable = point_best["validation_brier_score"] <= (
            tree_best["validation_brier_score"] + 0.005
        )
        latency_acceptable = (
            tree_latency is None
            or neural_latency is None
            or neural_latency <= max(5.0 * tree_latency, tree_latency + 5.0)
        )
        if paired.lower_95 <= 0.0 or not calibration_acceptable or not latency_acceptable:
            selected = tree_best
            rationale = (
                "The neural candidate did not satisfy every predeclared paired-F2, "
                "calibration, and inference-cost condition; the best tree is selected."
            )
        else:
            rationale = (
                "Neural F2 superiority over the best tree is supported by a paired "
                "group-bootstrap 95% interval above zero."
            )
    evidence = {
        "selected_bundle_key": selected["bundle_key"],
        "selection_partition": "validation_threshold_selection",
        "primary_objective": "F2 subject to false_positive_rate <= 0.001",
        "neural_vs_tree_paired_f2": paired_evidence,
        "operational_tradeoff_rule": {
            "calibration": "neural Brier <= tree Brier + 0.005",
            "latency": "neural median <= max(5x tree, tree + 5 ms)",
        },
        "robustness_evidence_status": "phase_7_evaluated_separately_from_phase_4_claim",
        "explanation_stability_status": "phase_6_evaluated_separately_from_phase_4_claim",
        "champion_status": "statistical_champion_with_separate_phase_6_7_evidence",
        "rationale": rationale,
        "test_opened": False,
    }
    return selected["bundle_key"], evidence, scoreboard


class Phase4Runner:
    """Execute fair tuning/refit/calibration without opening sealed test content."""

    def __init__(
        self,
        *,
        split_root: Path | None = None,
        output_root: Path | None = None,
        batch_size: int = 100_000,
    ) -> None:
        config = load_experiment_config()
        self.config = config
        self.identity: ExperimentIdentity = create_experiment_identity(config)
        self.split_root = Path(
            split_root or SPLITS_DIR / str(config["split_version"])
        )
        self.output_root = Path(output_root or MODELS_DIR / "phase4")
        self.batch_size = batch_size
        self.phase4_config = config.get("phase4", {})

    def verify_upstream(
        self,
    ) -> tuple[
        TrainingSplitAccess,
        dict[str, FeaturePipelineV2],
        dict[str, Path],
    ]:
        verify_validated_dataset()
        access = verify_training_split_access(self.split_root)
        gates = self.config["acceptance_gates"]
        if access.row_counts["train"] < int(gates["minimum_train_rows"]):
            raise Phase4GateError("Phase 4 requires the full configured train partition.")
        for partition in ("train", "validation", "test"):
            if access.benign_prevalence[partition] < float(
                gates["minimum_benign_prevalence"]
            ):
                raise Phase4GateError(f"{partition} prevalence gate failed.")
        pipeline_paths = {
            family: MODELS_DIR / f"feature_pipeline_v2_{family}.pkl"
            for family in ("tree", "neural")
        }
        pipelines = {
            family: FeaturePipelineV2.load(
                path,
                split_manifest_path=self.split_root / "split_manifest.json",
                dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
                transformation_manifest_path=DATA_REPORTS_DIR
                / "transformation_manifest.json",
            )
            for family, path in pipeline_paths.items()
        }
        if pipelines["tree"].model_family != "tree" or pipelines[
            "neural"
        ].model_family != "neural":
            raise Phase4GateError("Phase 4 family pipelines are mislabeled.")
        return access, pipelines, pipeline_paths

    def prepare_matrices(
        self,
        access: TrainingSplitAccess,
        pipelines: dict[str, FeaturePipelineV2],
        pipeline_paths: dict[str, Path],
    ) -> dict[str, dict[str, Path]]:
        matrix_root = (
            PROCESSED_DATA_DIR
            / "model_matrices_v2"
            / str(self.config["split_version"])
        )
        paths: dict[str, dict[str, Path]] = {}
        for family in ("tree", "neural"):
            paths[family] = {}
            for partition, source in (
                ("train", Path(access.train_features)),
                ("validation", Path(access.validation_features)),
            ):
                destination = matrix_root / family / partition
                if not destination.exists():
                    materialize_model_matrix(
                        source,
                        destination,
                        pipeline=pipelines[family],
                        pipeline_path=pipeline_paths[family],
                        split_manifest_path=self.split_root / "split_manifest.json",
                        partition_name=partition,
                        batch_size=self.batch_size,
                    )
                paths[family][partition] = destination
        return paths

    def run(self, specs: Iterable[CandidateSpec] | None = None) -> Path:
        access, pipelines, pipeline_paths = self.verify_upstream()
        protected_outputs = [
            self.output_root,
            RESULTS_DIR / "phase4_champion.json",
            RESULTS_DIR / "phase4_tuning.csv",
            RESULTS_DIR / "phase4_finalists.csv",
            RESULTS_DIR / "phase4_thresholds_validation.csv",
            RESULTS_DIR / "phase4_calibration_validation.csv",
            RESULTS_DIR / "phase4_resource_usage.csv",
            RESULTS_DIR / "phase4_seed_stability.csv",
            RESULTS_DIR / "phase4_validation_roles.json",
            RESULTS_DIR / "phase4_development_subset.json",
            RESULTS_DIR / "phase4_environment.json",
        ]
        existing = [str(path) for path in protected_outputs if path.exists()]
        if existing:
            raise Phase4GateError(
                "Phase 4 evidence is immutable under one experiment identity; "
                f"existing outputs require a new experiment version: {existing}"
            )
        matrix_dirs = self.prepare_matrices(access, pipelines, pipeline_paths)
        X_train_by_family: dict[str, np.ndarray] = {}
        X_validation_by_family: dict[str, np.ndarray] = {}
        matrix_manifest_by_family: dict[str, Any] = {}
        y_train = y_validation = None
        train_metadata = validation_metadata = None
        for family in ("tree", "neural"):
            X_train_family, y_train_family, train_metadata_family, train_manifest = (
                load_model_matrix(
                    matrix_dirs[family]["train"],
                    expected_partition="train",
                    feature_pipeline_path=pipeline_paths[family],
                    split_manifest_path=self.split_root / "split_manifest.json",
                )
            )
            (
                X_validation_family,
                y_validation_family,
                validation_metadata_family,
                _,
            ) = load_model_matrix(
                matrix_dirs[family]["validation"],
                expected_partition="validation",
                feature_pipeline_path=pipeline_paths[family],
                split_manifest_path=self.split_root / "split_manifest.json",
            )
            if y_train is None:
                y_train = y_train_family
                y_validation = y_validation_family
                train_metadata = train_metadata_family
                validation_metadata = validation_metadata_family
            elif not np.array_equal(y_train, y_train_family) or not np.array_equal(
                y_validation, y_validation_family
            ):
                raise Phase4GateError("Tree/neural model matrices have different labels.")
            X_train_by_family[family] = X_train_family
            X_validation_by_family[family] = X_validation_family
            matrix_manifest_by_family[family] = train_manifest
        assert y_train is not None and y_validation is not None
        assert train_metadata is not None and validation_metadata is not None
        roles, role_manifest = assign_validation_roles(
            validation_metadata,
            split_manifest_sha256=access.split_manifest_sha256,
        )
        if not role_manifest.group_overlap_zero or not role_manifest.strict_temporal_order:
            raise Phase4GateError("Validation role isolation failed.")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            RESULTS_DIR / "phase4_validation_roles.json", role_manifest.to_dict()
        )

        configured_specs = list(
            validate_candidate_contract(
                specs or default_candidate_specs(),
                self.phase4_config.get("required_models", ()),
            )
        )
        grouped: dict[tuple[str, str], list[CandidateSpec]] = {}
        for spec in configured_specs:
            grouped.setdefault((spec.model_name, spec.variant), []).append(spec)
        development_limit = int(self.phase4_config.get("development_max_rows", 500_000))
        development_indices, development_audit = select_development_indices(
            train_metadata,
            maximum_rows=development_limit,
            random_seed=int(self.config["random_seed"]),
        )
        atomic_write_json(
            RESULTS_DIR / "phase4_development_subset.json",
            {
                **development_audit,
                "partition": "train_only",
                "test_opened": False,
            },
        )
        y_development = y_train[development_indices]
        metadata_development = train_metadata.iloc[development_indices].reset_index(drop=True)

        selected_specs: list[CandidateSpec] = []
        tuning_records: list[dict[str, Any]] = []
        tuning_seed = int(self.config["random_seed"])
        for (_model_name, _variant), candidates in grouped.items():
            if candidates[0].model_family == "dummy":
                selected_specs.append(candidates[0])
                continue
            matrix_family = "neural" if candidates[0].model_family == "neural" else "tree"
            selected, records = tune_candidate_family(
                candidates,
                X_train_by_family[matrix_family][development_indices],
                y_development,
                metadata_development,
                seed=tuning_seed,
                folds=int(self.phase4_config.get("temporal_folds", 3)),
                maximum_fpr=float(self.phase4_config.get("maximum_fpr", 0.001)),
            )
            selected_specs.append(selected)
            tuning_records.extend(records)
        _atomic_csv(RESULTS_DIR / "phase4_tuning.csv", pd.DataFrame(tuning_records))

        fit_mask = roles == "validation_calibration_fit"
        calibration_selection_mask = roles == "validation_calibration_selection"
        threshold_mask = roles == "validation_threshold_selection"
        seeds = tuple(self.phase4_config.get("seeds", DEFAULT_SEEDS))
        common_provenance = {
            "dataset_quality_sha256": sha256_file(
                DATA_REPORTS_DIR / "dataset_quality.json"
            ),
            "split_manifest_sha256": access.split_manifest_sha256,
        }
        environment = hardware_and_packages()
        atomic_write_json(RESULTS_DIR / "phase4_environment.json", environment)
        bundles: dict[str, Phase4ModelBundle] = {}
        threshold_probabilities: dict[str, np.ndarray] = {}
        finalist_paths: dict[str, Path] = {}
        finalist_pipeline_paths: dict[str, Path] = {}
        calibration_rows: list[dict[str, Any]] = []
        resource_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        seed_stability_rows: list[dict[str, Any]] = []
        for spec in selected_specs:
            matrix_family = "neural" if spec.model_family == "neural" else "tree"
            X_train = X_train_by_family[matrix_family]
            X_validation = X_validation_by_family[matrix_family]
            pipeline_path = pipeline_paths[matrix_family]
            provenance = {
                **common_provenance,
                "feature_pipeline_sha256": sha256_file(pipeline_path),
                "train_model_matrix_manifest_sha256": sha256_file(
                    matrix_dirs[matrix_family]["train"] / "matrix_manifest.json"
                ),
                "validation_model_matrix_manifest_sha256": sha256_file(
                    matrix_dirs[matrix_family]["validation"] / "matrix_manifest.json"
                ),
            }
            member_seeds = (int(seeds[0]),) if spec.model_family == "dummy" else tuple(
                int(seed) for seed in seeds
            )
            members: list[CalibratedMember] = []
            member_calibration: dict[str, Any] = {}
            for seed in member_seeds:
                estimator, resource_usage = fit_candidate(
                    spec, X_train, y_train, seed=seed, tuning=False
                )
                raw_validation = positive_class_probability(estimator, X_validation)
                if spec.model_family == "dummy":
                    calibrator = ProbabilityCalibrator("identity").fit(
                        raw_validation[fit_mask],
                        y_validation[fit_mask],
                        partition_name="validation_calibration_fit",
                    )
                    calibration_evidence = {
                        "selected_method": "identity",
                        "fit_on": "validation_calibration_fit",
                        "selected_on": "not_applicable_always_benign",
                    }
                else:
                    calibrator, selection = fit_and_select_calibrator(
                        raw_validation[fit_mask],
                        y_validation[fit_mask],
                        raw_validation[calibration_selection_mask],
                        y_validation[calibration_selection_mask],
                        fit_partition_name="validation_calibration_fit",
                        selection_partition_name="validation_calibration_selection",
                    )
                    calibration_evidence = selection.to_dict()
                members.append(
                    CalibratedMember(
                        seed=seed,
                        estimator=estimator,
                        calibrator=calibrator,
                        training_configuration=asdict(spec),
                        resource_usage=resource_usage,
                    )
                )
                member_calibration[str(seed)] = calibration_evidence
                calibration_rows.append(
                    {
                        "model_name": spec.model_name,
                        "variant": spec.variant,
                        "seed": seed,
                        **calibration_evidence,
                    }
                )
                resource_rows.append(
                    {
                        "model_name": spec.model_name,
                        "variant": spec.variant,
                        "seed": seed,
                        "platform": environment["platform"],
                        "processor": environment["processor"],
                        "logical_cpu_count": environment["logical_cpu_count"],
                        "total_ram_bytes": environment["total_ram_bytes"],
                        "python_version": environment["python"],
                        "package_versions_json": json.dumps(
                            environment["packages"], sort_keys=True
                        ),
                        **resource_usage,
                    }
                )
            ensemble_validation = np.vstack(
                [member.predict_probability(X_validation) for member in members]
            ).mean(axis=0)
            thresholds = {
                name: decision.to_dict()
                for name, decision in select_validation_thresholds(
                    y_validation[threshold_mask],
                    ensemble_validation[threshold_mask],
                    partition_name="validation_threshold_selection",
                ).items()
            }
            selected_policy = (
                "fixed_0_5" if spec.model_family == "dummy" else "fpr_lte_0_001"
            )
            locked_threshold = float(thresholds[selected_policy]["threshold"])
            for member in members:
                member_report = metric_report(
                    y_validation[threshold_mask],
                    member.predict_probability(X_validation)[threshold_mask],
                    threshold=locked_threshold,
                )
                seed_stability_rows.append(
                    {
                        "model_name": spec.model_name,
                        "variant": spec.variant,
                        "seed": member.seed,
                        "partition": "validation_threshold_selection",
                        "ensemble_locked_threshold": locked_threshold,
                        **member_report,
                    }
                )
            bundle = Phase4ModelBundle(
                model_name=spec.model_name,
                model_family=spec.model_family,
                variant=spec.variant,
                feature_names=tuple(
                    matrix_manifest_by_family[matrix_family].feature_names
                ),
                members=members,
                thresholds=thresholds,
                selected_policy=selected_policy,
                provenance=provenance,
                training_evidence={
                    "full_train_rows": int(len(y_train)),
                    "full_train_malicious_prevalence": float(np.mean(y_train)),
                    "fit_partition": "train_full",
                    "configuration": asdict(spec),
                    "hardware": environment,
                },
                calibration_evidence={"members": member_calibration},
            )
            key = spec.variant_key
            path = self.output_root / f"{key}.pkl"
            bundle.save_finalist(
                path,
                identity=self.identity,
                require_three_seeds=spec.model_family != "dummy",
            )
            bundles[key] = bundle
            threshold_probabilities[key] = ensemble_validation[threshold_mask]
            finalist_paths[key] = path
            finalist_pipeline_paths[key] = pipeline_path
            for policy, decision in thresholds.items():
                threshold_rows.append(
                    {
                        "model_name": spec.model_name,
                        "variant": spec.variant,
                        "policy": policy,
                        "threshold": decision["threshold"],
                        **decision["validation_metrics"],
                    }
                )

        selected_key, champion_evidence, scoreboard = select_validation_champion(
            bundles,
            threshold_probabilities,
            y_validation[threshold_mask],
            validation_metadata.loc[threshold_mask, GROUP_ID_COLUMN].astype(str).to_numpy(),
            bootstrap_replicates=int(
                self.phase4_config.get("bootstrap_replicates", 1_000)
            ),
            random_seed=int(self.config["random_seed"]),
        )
        champion_path = self.output_root / "champion_bundle.pkl"
        bundles[selected_key].save_champion(champion_path, identity=self.identity)
        champion_pipeline_path = finalist_pipeline_paths[selected_key]
        champion_manifest = {
            "phase4_version": PHASE4_VERSION,
            "experiment": self.identity.to_dict(),
            **champion_evidence,
            "champion_bundle": str(champion_path.resolve()),
            "champion_bundle_sha256": sha256_file(champion_path),
            "champion_feature_pipeline": str(champion_pipeline_path.resolve()),
            "champion_feature_pipeline_sha256": sha256_file(champion_pipeline_path),
            "finalists": {
                key: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "feature_pipeline_path": str(
                        finalist_pipeline_paths[key].resolve()
                    ),
                    "feature_pipeline_sha256": sha256_file(
                        finalist_pipeline_paths[key]
                    ),
                }
                for key, path in finalist_paths.items()
            },
            "model_matrices": {
                family: {
                    partition: {
                        "path": str(
                            (directory / "matrix_manifest.json").resolve()
                        ),
                        "sha256": sha256_file(directory / "matrix_manifest.json"),
                    }
                    for partition, directory in partitions.items()
                }
                for family, partitions in matrix_dirs.items()
            },
            "full_train_rows": int(len(y_train)),
            "validation_rows": int(len(y_validation)),
            "environment": environment,
            "sealed_test_opened": False,
        }
        champion_manifest_path = RESULTS_DIR / "phase4_champion.json"
        atomic_write_json(champion_manifest_path, champion_manifest)
        _atomic_csv(RESULTS_DIR / "phase4_finalists.csv", pd.DataFrame(scoreboard))
        _atomic_csv(
            RESULTS_DIR / "phase4_thresholds_validation.csv",
            pd.DataFrame(threshold_rows),
        )
        _atomic_csv(
            RESULTS_DIR / "phase4_calibration_validation.csv",
            pd.DataFrame(calibration_rows),
        )
        _atomic_csv(
            RESULTS_DIR / "phase4_resource_usage.csv", pd.DataFrame(resource_rows)
        )
        _atomic_csv(
            RESULTS_DIR / "phase4_seed_stability.csv",
            pd.DataFrame(seed_stability_rows),
        )
        summary_path = RESULTS_DIR / "experiment_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "status": "phase_4_complete_test_still_sealed",
                "data_gate_passed": True,
                "phase_4": {
                    "champion_manifest": str(champion_manifest_path.resolve()),
                    "champion_manifest_sha256": sha256_file(champion_manifest_path),
                    "selection_partition": "validation_threshold_selection",
                    "test_opened": False,
                },
                "final_metrics": None,
            }
        )
        atomic_write_json(summary_path, summary)
        return champion_manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args(argv)
    print(Phase4Runner(batch_size=args.batch_size).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
