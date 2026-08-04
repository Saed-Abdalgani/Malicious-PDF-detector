"""Multi-method, fail-closed explainability primitives for remediation Phase 6.

SHAP is one evidence source, never the conclusion.  Functions in this module
require real train/validation references and expose independent permutation,
ALE, interaction, ablation, stability, faithfulness, local-case, prototype, and
counterfactual evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.bundle import Phase4ModelBundle, positive_class_probability


class ExplainabilityGateError(RuntimeError):
    """Raised when an explanation would violate partition or evidence gates."""


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite two-dimensional matrix.")
    return array


def _labels(values: Iterable[int], rows: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int8).reshape(-1)
    if len(labels) != rows or not np.isin(labels, (0, 1)).all():
        raise ValueError("Explainability labels must align and be binary.")
    return labels


def predict_probability(model: Any, features: np.ndarray) -> np.ndarray:
    """Return calibrated probabilities for bundles and standard probabilities otherwise."""
    matrix = _matrix(features, name="features")
    if isinstance(model, Phase4ModelBundle):
        return model.predict_proba(matrix)
    return positive_class_probability(model, matrix)


def stratified_reference_indices(
    labels: Iterable[int],
    metadata: pd.DataFrame,
    *,
    maximum_rows: int,
    random_seed: int,
) -> np.ndarray:
    """Select a real train reference across class, source, and time strata."""
    y = _labels(labels, len(metadata))
    if maximum_rows < 2:
        raise ValueError("A real explanation background needs at least two rows.")
    frame = pd.DataFrame({"label": y}, index=metadata.index)
    frame["source"] = (
        metadata["source_id"].astype(str) if "source_id" in metadata else "unknown"
    )
    if "first_seen_at" in metadata:
        time = pd.to_datetime(metadata["first_seen_at"], utc=True).astype("int64")
        ranks = time.rank(method="first")
        frame["time_bin"] = pd.qcut(
            ranks, q=min(5, len(ranks)), labels=False, duplicates="drop"
        ).fillna(0).astype(int)
    else:
        frame["time_bin"] = 0
    identifiers = (
        metadata["group_id"].astype(str)
        if "group_id" in metadata
        else metadata.index.astype(str)
    )
    frame["hash"] = [
        hashlib.sha256(f"{random_seed}:{value}".encode()).hexdigest()
        for value in identifiers
    ]
    selected: list[int] = []
    for _stratum, part in frame.groupby(["label", "source", "time_bin"], sort=True):
        quota = max(1, round(maximum_rows * len(part) / len(frame)))
        selected.extend(part.sort_values("hash").index[:quota].tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) > maximum_rows:
        selected = sorted(selected, key=lambda index: frame.loc[index, "hash"])[
            :maximum_rows
        ]
    elif len(selected) < min(maximum_rows, len(frame)):
        remaining = frame.drop(index=selected).sort_values("hash").index
        selected.extend(remaining[: maximum_rows - len(selected)].tolist())
    positions = metadata.index.get_indexer(selected)
    positions = positions[positions >= 0]
    if np.unique(y[positions]).size != 2:
        raise ExplainabilityGateError("Reference background does not contain both classes.")
    return np.sort(positions.astype(np.int64))


def data_explanation_tables(
    features: np.ndarray,
    labels: Iterable[int],
    metadata: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    partition_name: str,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Layer A distributions, missingness, correlations, and train-only MI."""
    if partition_name != "train":
        raise ExplainabilityGateError("Data/proxy discovery may only fit on train.")
    X = _matrix(features, name="training features")
    y = _labels(labels, len(X))
    names = tuple(feature_names)
    if X.shape[1] != len(names) or len(metadata) != len(X):
        raise ValueError("Training explanation inputs do not align.")
    distribution_rows: list[dict[str, Any]] = []
    for column, name in enumerate(names):
        for label in (0, 1):
            values = X[y == label, column]
            distribution_rows.append(
                {
                    "feature": name,
                    "class": label,
                    "rows": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q50": float(np.quantile(values, 0.50)),
                    "q95": float(np.quantile(values, 0.95)),
                }
            )
    correlations = pd.DataFrame(X, columns=names).corr(method="spearman")
    correlation_rows = []
    for right_index, right in enumerate(names):
        for left in names[:right_index]:
            value = correlations.loc[left, right]
            if np.isfinite(value) and abs(value) >= 0.8:
                correlation_rows.append(
                    {"feature_left": left, "feature_right": right, "spearman": value}
                )
    mutual_information = mutual_info_classif(
        X, y, random_state=random_seed, discrete_features=False
    )
    missing_columns = [name for name in metadata if "missing" in name.lower()]
    parser_columns = [
        name
        for name in metadata
        if any(token in name.lower() for token in ("parse", "recovery", "timeout"))
    ]
    pattern_rows = []
    for name in (*missing_columns, *parser_columns):
        values = pd.to_numeric(metadata[name], errors="coerce").fillna(0)
        for label in (0, 1):
            pattern_rows.append(
                {
                    "field": name,
                    "class": label,
                    "rate_nonzero": float(values[y == label].ne(0).mean()),
                }
            )
    return {
        "class_distributions": pd.DataFrame(distribution_rows),
        "correlations": pd.DataFrame(correlation_rows),
        "mutual_information": pd.DataFrame(
            {"feature": names, "mutual_information_train": mutual_information}
        ).sort_values("mutual_information_train", ascending=False),
        "missingness_parser_patterns": pd.DataFrame(pattern_rows),
        "prevalence": pd.DataFrame(
            [{"partition": "train", "rows": len(y), "malicious_prevalence": y.mean()}]
        ),
    }


def permutation_importance_table(
    model: Any,
    features: np.ndarray,
    labels: Iterable[int],
    feature_names: Sequence[str],
    *,
    partition_name: str,
    repeats: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Held-out average-precision permutation importance."""
    if partition_name not in {"validation", "validation_threshold_selection"}:
        raise ExplainabilityGateError("Permutation importance requires held-out validation.")
    X = _matrix(features, name="held-out features")
    y = _labels(labels, len(X))
    names = tuple(feature_names)
    if X.shape[1] != len(names) or np.unique(y).size != 2:
        raise ValueError("Permutation inputs are incomplete.")
    baseline = average_precision_score(y, predict_probability(model, X))
    rng = np.random.default_rng(random_seed)
    rows = []
    working = X.copy()
    for column, name in enumerate(names):
        drops = []
        original = working[:, column].copy()
        for _ in range(repeats):
            working[:, column] = rng.permutation(original)
            drops.append(baseline - average_precision_score(y, predict_probability(model, working)))
        working[:, column] = original
        rows.append(
            {
                "feature": name,
                "method": "heldout_permutation_average_precision_drop",
                "importance": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "baseline_average_precision": float(baseline),
                "partition": partition_name,
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def native_importance_table(model: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Extract averaged model-native importance when the estimator exposes it."""
    estimators = (
        [member.estimator for member in model.members]
        if isinstance(model, Phase4ModelBundle)
        else [model]
    )
    values = []
    for estimator in estimators:
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None and hasattr(estimator, "coef_"):
            raw = np.abs(np.asarray(estimator.coef_)).reshape(-1)
        if raw is not None:
            values.append(np.asarray(raw, dtype=float).reshape(-1))
    if not values:
        return pd.DataFrame(columns=("feature", "method", "importance"))
    importance = np.mean(np.vstack(values), axis=0)
    if len(importance) != len(feature_names):
        raise ValueError("Native importance does not align with feature names.")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "method": "model_native",
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)


def shap_attributions(
    model: Any,
    explain_features: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, str]:
    """TreeSHAP for trees and GradientExplainer for neural seed members."""
    import shap

    X = _matrix(explain_features, name="explanation features")
    reference = _matrix(background, name="real training background")
    if X.shape[1] != reference.shape[1]:
        raise ValueError("SHAP background and explanation columns differ.")
    members = model.members if isinstance(model, Phase4ModelBundle) else None
    estimators = [member.estimator for member in members] if members else [model]
    outputs = []
    methods = []
    for estimator in estimators:
        if isinstance(estimator, torch.nn.Module):
            estimator.eval()
            explainer = shap.GradientExplainer(
                estimator,
                torch.as_tensor(reference, dtype=torch.float32),
            )
            value = explainer.shap_values(torch.as_tensor(X, dtype=torch.float32))
            methods.append("neural_gradient_shap_real_train_background")
        else:
            explainer = shap.TreeExplainer(
                estimator,
                data=reference,
                feature_perturbation="interventional",
            )
            value = explainer.shap_values(X)
            methods.append("tree_shap_real_train_background")
        if isinstance(value, list):
            value = value[-1]
        array = np.asarray(value)
        if array.ndim == 3:
            array = array[..., -1]
        outputs.append(array.reshape(len(X), X.shape[1]))
    return np.mean(np.stack(outputs), axis=0), "+".join(sorted(set(methods)))


def shap_importance_table(
    attributions: np.ndarray, feature_names: Sequence[str], *, method: str
) -> pd.DataFrame:
    values = np.asarray(attributions, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Attributions do not align with feature names.")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "method": method,
            "importance": np.mean(np.abs(values), axis=0),
            "mean_signed_attribution": np.mean(values, axis=0),
        }
    ).sort_values("importance", ascending=False)


def ale_table(
    model: Any,
    features: np.ndarray,
    feature_names: Sequence[str],
    selected_features: Sequence[str],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """First-order accumulated local effects using empirical quantile intervals."""
    X = _matrix(features, name="ALE reference")
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    rows = []
    for name in selected_features:
        column = name_to_index[name]
        edges = np.unique(np.quantile(X[:, column], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        assignments = np.clip(np.digitize(X[:, column], edges[1:-1]), 0, len(edges) - 2)
        local = np.zeros(len(edges) - 1)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for interval in range(len(edges) - 1):
            mask = assignments == interval
            if not mask.any():
                continue
            lower = X[mask].copy()
            upper = X[mask].copy()
            lower[:, column] = edges[interval]
            upper[:, column] = edges[interval + 1]
            local[interval] = np.mean(
                predict_probability(model, upper) - predict_probability(model, lower)
            )
            counts[interval] = int(mask.sum())
        accumulated = np.cumsum(local)
        centered = accumulated - np.average(accumulated, weights=np.maximum(counts, 1))
        for interval, effect in enumerate(centered):
            rows.append(
                {
                    "feature": name,
                    "bin": interval,
                    "lower": float(edges[interval]),
                    "upper": float(edges[interval + 1]),
                    "rows": int(counts[interval]),
                    "ale": float(effect),
                }
            )
    return pd.DataFrame(rows)


def interaction_table(
    model: Any,
    features: np.ndarray,
    feature_names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Empirical finite-difference interaction strength for domain-driven pairs."""
    X = _matrix(features, name="interaction reference")
    index = {name: position for position, name in enumerate(feature_names)}
    baseline = predict_probability(model, X)
    rows = []
    for left, right in pairs:
        if left not in index or right not in index:
            continue
        both = X.copy()
        left_only = X.copy()
        right_only = X.copy()
        left_value = np.quantile(X[:, index[left]], 0.9)
        right_value = np.quantile(X[:, index[right]], 0.9)
        both[:, index[left]] = left_value
        both[:, index[right]] = right_value
        left_only[:, index[left]] = left_value
        right_only[:, index[right]] = right_value
        synergy = (
            predict_probability(model, both)
            - predict_probability(model, left_only)
            - predict_probability(model, right_only)
            + baseline
        )
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "method": "empirical_second_difference_q90",
                "mean_interaction": float(np.mean(synergy)),
                "mean_absolute_interaction": float(np.mean(np.abs(synergy))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_absolute_interaction", ascending=False
    ) if rows else pd.DataFrame()


def importance_stability_table(
    rankings: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Pairwise Spearman stability across seeds, time folds, and bootstraps."""
    runs = sorted(rankings)
    features = sorted({name for values in rankings.values() for name in values})
    rows = []
    for right_index, right in enumerate(runs):
        for left in runs[:right_index]:
            correlation = spearmanr(
                [rankings[left].get(name, 0.0) for name in features],
                [rankings[right].get(name, 0.0) for name in features],
            ).statistic
            rows.append(
                {
                    "run_left": left,
                    "run_right": right,
                    "spearman_rank_correlation": (
                        float(correlation) if np.isfinite(correlation) else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def deletion_insertion_faithfulness(
    model: Any,
    features: np.ndarray,
    background: np.ndarray,
    importance: Mapping[str, float],
    feature_names: Sequence[str],
    *,
    top_k: int = 10,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Verify ranked deletion and insertion outperform random feature orderings."""
    X = _matrix(features, name="faithfulness features")
    reference = _matrix(background, name="faithfulness background")
    median = np.median(reference, axis=0)
    order = sorted(range(len(feature_names)), key=lambda i: importance.get(feature_names[i], 0), reverse=True)
    top = np.array(order[: min(top_k, len(order))], dtype=int)
    baseline = predict_probability(model, X)
    deleted = X.copy()
    deleted[:, top] = median[top]
    top_change = float(np.mean(np.abs(baseline - predict_probability(model, deleted))))
    rng = np.random.default_rng(random_seed)
    random_changes = []
    inserted = np.broadcast_to(median, X.shape).copy()
    inserted[:, top] = X[:, top]
    insertion_change = float(
        np.mean(
            np.abs(
                predict_probability(model, inserted)
                - predict_probability(model, np.broadcast_to(median, X.shape))
            )
        )
    )
    random_insertions = []
    for _ in range(20):
        random_columns = rng.choice(X.shape[1], size=len(top), replace=False)
        random_deleted = X.copy()
        random_deleted[:, random_columns] = median[random_columns]
        random_changes.append(
            float(np.mean(np.abs(baseline - predict_probability(model, random_deleted))))
        )
        random_inserted = np.broadcast_to(median, X.shape).copy()
        random_inserted[:, random_columns] = X[:, random_columns]
        random_insertions.append(
            float(
                np.mean(
                    np.abs(
                        predict_probability(model, random_inserted)
                        - predict_probability(
                            model, np.broadcast_to(median, X.shape)
                        )
                    )
                )
            )
        )
    random_deletion = float(np.mean(random_changes))
    random_insertion = float(np.mean(random_insertions))
    return {
        "method": "deletion_and_insertion_against_random",
        "top_k": len(top),
        "top_feature_mean_absolute_score_change": top_change,
        "random_deletion_mean_absolute_score_change": random_deletion,
        "deletion_ratio_top_to_random": top_change / max(random_deletion, 1e-12),
        "top_feature_insertion_mean_absolute_score_change": insertion_change,
        "random_insertion_mean_absolute_score_change": random_insertion,
        "insertion_ratio_top_to_random": insertion_change
        / max(random_insertion, 1e-12),
        "passed": top_change > random_deletion and insertion_change > random_insertion,
    }


def source_time_shortcut_audit(
    train_metadata: pd.DataFrame,
    train_labels: Iterable[int],
    validation_metadata: pd.DataFrame,
    validation_labels: Iterable[int],
    *,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Measure how well collection source/time alone predicts the label."""
    train_y = _labels(train_labels, len(train_metadata))
    validation_y = _labels(validation_labels, len(validation_metadata))

    def design(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
        source = (
            frame["source_id"].astype(str)
            if "source_id" in frame
            else pd.Series("unknown", index=frame.index)
        )
        if "first_seen_at" in frame:
            timestamp = pd.to_datetime(frame["first_seen_at"], utc=True, errors="coerce")
            month = timestamp.dt.strftime("%Y-%m").fillna("unknown")
        else:
            month = pd.Series("unknown", index=frame.index)
        encoded = pd.get_dummies(
            pd.DataFrame({"source": source, "month": month}), dtype=np.float32
        )
        return encoded if columns is None else encoded.reindex(columns=columns, fill_value=0)

    train_X = design(train_metadata)
    validation_X = design(validation_metadata, tuple(train_X.columns))
    if train_X.shape[1] == 0 or np.unique(train_y).size != 2 or np.unique(validation_y).size != 2:
        return {"available": False, "reason": "insufficient source/time or class coverage"}
    estimator = LogisticRegression(max_iter=300, random_state=random_seed, class_weight="balanced")
    estimator.fit(train_X, train_y)
    probability = estimator.predict_proba(validation_X)[:, 1]
    return {
        "available": True,
        "method": "train_fitted_source_time_only_logistic_regression",
        "validation_roc_auc": float(roc_auc_score(validation_y, probability)),
        "validation_average_precision": float(
            average_precision_score(validation_y, probability)
        ),
        "encoded_columns": int(train_X.shape[1]),
        "warning": "High performance indicates collection/source/time shortcut risk.",
    }


def method_agreement_table(global_importance: pd.DataFrame) -> pd.DataFrame:
    """Pairwise rank agreement and an explicit disagreement flag."""
    if global_importance.empty:
        return pd.DataFrame()
    pivot = global_importance.pivot_table(
        index="feature", columns="method", values="importance", aggfunc="mean", fill_value=0
    )
    rows = []
    methods = sorted(pivot.columns)
    for right_index, right in enumerate(methods):
        for left in methods[:right_index]:
            correlation = spearmanr(pivot[left], pivot[right]).statistic
            value = float(correlation) if np.isfinite(correlation) else 0.0
            rows.append(
                {
                    "method_left": left,
                    "method_right": right,
                    "spearman_rank_correlation": value,
                    "requires_investigation": value < 0.5,
                }
            )
    return pd.DataFrame(rows)


def feature_family_ablation(
    train_and_score: Callable[[tuple[int, ...]], float],
    feature_names: Sequence[str],
    families: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Retrain through a caller-supplied, partition-locked callback per family."""
    index = {name: position for position, name in enumerate(feature_names)}
    all_columns = tuple(range(len(feature_names)))
    baseline = float(train_and_score(all_columns))
    rows = [{"ablation": "none", "heldout_score": baseline, "score_delta": 0.0}]
    for family, members in sorted(families.items()):
        removed = {index[name] for name in members if name in index}
        retained = tuple(column for column in all_columns if column not in removed)
        if not removed or not retained:
            continue
        score = float(train_and_score(retained))
        rows.append(
            {
                "ablation": family,
                "removed_features": len(removed),
                "heldout_score": score,
                "score_delta": score - baseline,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CounterfactualRecord:
    case_index: int
    original_probability: float
    counterfactual_probability: float
    threshold: float
    changed_features: tuple[str, ...]
    prototype_class: str = "observed_benign_train_row"
    feasible: bool = True
    causal_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_features"] = list(self.changed_features)
        return value


def observed_prototype_counterfactuals(
    model: Any,
    cases: np.ndarray,
    train_features: np.ndarray,
    train_labels: Iterable[int],
    feature_names: Sequence[str],
    *,
    threshold: float,
) -> list[CounterfactualRecord]:
    """Use complete observed benign rows, never impossible feature-space edits."""
    X = _matrix(cases, name="counterfactual cases")
    train = _matrix(train_features, name="counterfactual train reference")
    y = _labels(train_labels, len(train))
    benign = train[y == 0]
    if not len(benign):
        raise ExplainabilityGateError("Counterfactuals require observed benign prototypes.")
    scale = np.std(train, axis=0)
    scale[scale < 1e-6] = 1.0
    original = predict_probability(model, X)
    rows = []
    for case_index, case in enumerate(X):
        distances = np.mean(((benign - case) / scale) ** 2, axis=1)
        prototype = benign[int(np.argmin(distances))]
        changed = tuple(
            name
            for name, left, right in zip(feature_names, case, prototype)
            if not np.isclose(left, right, rtol=1e-6, atol=1e-7)
        )
        probability = float(predict_probability(model, prototype.reshape(1, -1))[0])
        rows.append(
            CounterfactualRecord(
                case_index=case_index,
                original_probability=float(original[case_index]),
                counterfactual_probability=probability,
                threshold=float(threshold),
                changed_features=changed,
                feasible=True,
            )
        )
    return rows


def select_local_cases(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float,
    per_type: int,
    abstained: Iterable[bool] | None = None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= threshold
    categories = {
        "tp": (y == 1) & predicted,
        "tn": (y == 0) & ~predicted,
        "fp": (y == 0) & predicted,
        "fn": (y == 1) & ~predicted,
        "high_confidence": np.abs(p - threshold) >= np.quantile(np.abs(p - threshold), 0.9),
        "low_confidence": np.abs(p - threshold) <= np.quantile(np.abs(p - threshold), 0.1),
    }
    if abstained is not None:
        abstention = np.asarray(abstained, dtype=bool).reshape(-1)
        if len(abstention) != len(y):
            raise ValueError("Abstention mask does not align with local cases.")
        categories["abstained"] = abstention
    selected = []
    for name, mask in categories.items():
        positions = np.flatnonzero(mask)
        if name in {"fp", "high_confidence"}:
            positions = positions[np.argsort(-p[positions], kind="mergesort")]
        else:
            positions = positions[np.argsort(p[positions], kind="mergesort")]
        selected.extend(positions[:per_type].tolist())
    return np.array(sorted(set(selected)), dtype=np.int64)


def actionable_conclusions_markdown(
    global_importance: pd.DataFrame,
    stability: pd.DataFrame,
    ablations: pd.DataFrame,
    *,
    minimum_methods: int = 2,
) -> str:
    """Create only multi-method findings tied to an allowed action."""
    frame = global_importance.copy()
    if frame.empty or not {"feature", "method", "importance"}.issubset(frame):
        return "# Actionable conclusions\n\nNo conclusion passed the evidence gate.\n"
    frame["rank"] = frame.groupby("method")["importance"].rank(
        ascending=False, method="min"
    )
    support = (
        frame[frame["rank"] <= 20]
        .groupby("feature")
        .agg(methods=("method", "nunique"), mean_rank=("rank", "mean"))
        .reset_index()
    )
    ablation_support: set[str] = set()
    if not ablations.empty and {"ablation", "score_delta"}.issubset(ablations):
        ablation_support = set(
            ablations.loc[ablations["score_delta"].abs() > 0, "ablation"].astype(str)
        )
    support = support[support["methods"] >= minimum_methods].sort_values("mean_rank")
    median_stability = (
        float(stability["spearman_rank_correlation"].median())
        if not stability.empty and "spearman_rank_correlation" in stability
        else None
    )
    lines = [
        "# Actionable conclusions",
        "",
        "Every conclusion below has independent multi-method support; SHAP alone is insufficient.",
        "",
    ]
    for row in support.head(20).itertuples():
        name = str(row.feature)
        lowered = name.lower()
        if any(token in lowered for token in ("parser", "recovery", "obfuscat", "disagreement")):
            action = "improve extraction"
            detail = "harden canonicalization/parser agreement and route unresolved failures to abstention"
        elif any(token in lowered for token in ("source", "creator", "producer", "pdf_size")) and (median_stability is None or median_stability < 0.7):
            action = "drop/regularize"
            detail = "treat as a possible collection proxy and confirm with source-held-out ablation"
        else:
            action = "keep"
            detail = "retain with ongoing temporal and subgroup stability monitoring"
        lines.append(
            f"- **{action} — `{name}`:** supported by {int(row.methods)} methods; {detail}."
        )
    if support.empty:
        lines.append("No feature met the minimum independent-support requirement.")
    lines.extend(
        [
            "",
            f"Feature-family retraining ablations with measurable effects: {len(ablation_support)}.",
            "Low method agreement must be investigated before operational use.",
            "",
            "Counterfactuals are diagnostic comparisons with observed train rows and are not causal proof.",
        ]
    )
    return "\n".join(lines) + "\n"


# Compatibility helpers for the application-facing local explanation bridge.
def top_attributions(
    attribution_row: np.ndarray, feature_names: Sequence[str], k: int = 6
) -> list[tuple[str, float, str]]:
    row = np.asarray(attribution_row).reshape(-1)
    order = np.argsort(np.abs(row))[::-1][:k]
    return [
        (
            feature_names[index],
            float(row[index]),
            "increases" if row[index] > 0 else "decreases",
        )
        for index in order
    ]


def explain_mlp(
    model: Any,
    X_explain: np.ndarray,
    background: np.ndarray | None = None,
    nsamples: int = 200,
) -> np.ndarray:
    del nsamples
    if background is None:
        raise ExplainabilityGateError("A real train-only background is mandatory.")
    values, _ = shap_attributions(model, X_explain, background)
    return values


def explain_tree(
    model: Any, X_explain: np.ndarray, background: np.ndarray | None = None
) -> np.ndarray:
    if background is None:
        raise ExplainabilityGateError("A real train-only background is mandatory.")
    values, _ = shap_attributions(model, X_explain, background)
    return values
