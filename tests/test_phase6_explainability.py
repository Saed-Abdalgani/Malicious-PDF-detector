from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from src.features.explain import (
    ExplainabilityGateError,
    actionable_conclusions_markdown,
    ale_table,
    data_explanation_tables,
    deletion_insertion_faithfulness,
    feature_family_ablation,
    method_agreement_table,
    observed_prototype_counterfactuals,
    permutation_importance_table,
    select_local_cases,
    source_time_shortcut_audit,
    stratified_reference_indices,
)
from src.models.phase5 import select_phase6_handoff_indices
from src.models.phase6 import Phase6GateError, Phase6Runner, _feature_families


def _data(rows: int = 400) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    labels = (signal + rng.normal(scale=0.25, size=rows) > 0).astype(np.int8)
    features = np.column_stack((signal, noise)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "group_id": [f"g-{index}" for index in range(rows)],
            "source_id": np.where(np.arange(rows) % 2, "source-a", "source-b"),
            "first_seen_at": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "parse_failure": np.arange(rows) % 31 == 0,
        }
    )
    return features, labels, metadata


def test_real_reference_is_deterministic_stratified_and_train_only() -> None:
    features, labels, metadata = _data()
    first = stratified_reference_indices(labels, metadata, maximum_rows=80, random_seed=7)
    second = stratified_reference_indices(labels, metadata, maximum_rows=80, random_seed=7)
    assert np.array_equal(first, second)
    assert len(first) == 80
    assert set(labels[first]) == {0, 1}
    tables = data_explanation_tables(
        features[first], labels[first], metadata.iloc[first].reset_index(drop=True),
        ("signal", "noise"), partition_name="train"
    )
    assert set(tables) == {
        "class_distributions", "correlations", "mutual_information",
        "missingness_parser_patterns", "prevalence",
    }
    with pytest.raises(ExplainabilityGateError):
        data_explanation_tables(
            features[first], labels[first], metadata.iloc[first].reset_index(drop=True),
            ("signal", "noise"), partition_name="test"
        )


def test_independent_methods_identify_signal_and_shap_alone_is_not_a_conclusion() -> None:
    features, labels, _ = _data()
    model = ExtraTreesClassifier(n_estimators=60, random_state=42).fit(features[:300], labels[:300])
    permutation = permutation_importance_table(
        model, features[300:], labels[300:], ("signal", "noise"),
        partition_name="validation", repeats=4
    )
    assert permutation.iloc[0]["feature"] == "signal"
    native = pd.DataFrame(
        {"feature": ["signal", "noise"], "method": ["native", "native"], "importance": [0.9, 0.1]}
    )
    shap_only = pd.DataFrame(
        {"feature": ["signal"], "method": ["shap"], "importance": [1.0]}
    )
    rejected = actionable_conclusions_markdown(shap_only, pd.DataFrame(), pd.DataFrame())
    assert "No feature met" in rejected
    accepted = actionable_conclusions_markdown(
        pd.concat([permutation, native], ignore_index=True), pd.DataFrame(), pd.DataFrame()
    )
    assert "`signal`" in accepted
    agreement = method_agreement_table(pd.concat([permutation, native], ignore_index=True))
    assert not agreement.empty


def test_ale_faithfulness_ablation_and_feasible_counterfactuals() -> None:
    features, labels, _ = _data()
    model = ExtraTreesClassifier(n_estimators=80, random_state=9).fit(features[:300], labels[:300])
    ale = ale_table(model, features[300:], ("signal", "noise"), ("signal",), bins=6)
    assert not ale.empty and set(ale["feature"]) == {"signal"}
    faithfulness = deletion_insertion_faithfulness(
        model, features[300:], features[:300], {"signal": 1.0, "noise": 0.0},
        ("signal", "noise"), top_k=1
    )
    assert faithfulness["method"] == "deletion_and_insertion_against_random"
    scores = feature_family_ablation(
        lambda columns: 0.9 if len(columns) == 2 else 0.6,
        ("signal", "noise"), {"signal_family": ["signal"]}
    )
    assert scores.loc[scores["ablation"] == "signal_family", "score_delta"].iloc[0] < 0
    records = observed_prototype_counterfactuals(
        model, features[labels == 1][:3], features[:300], labels[:300],
        ("signal", "noise"), threshold=0.5
    )
    assert records and all(record.feasible and not record.causal_claim for record in records)
    assert all(record.prototype_class == "observed_benign_train_row" for record in records)


def test_source_time_audit_local_abstention_and_feature_families() -> None:
    _, labels, metadata = _data()
    audit = source_time_shortcut_audit(
        metadata.iloc[:300], labels[:300], metadata.iloc[300:], labels[300:]
    )
    assert audit["available"] and 0 <= audit["validation_roc_auc"] <= 1
    probabilities = np.linspace(0.01, 0.99, len(labels))
    selected = select_local_cases(
        labels, probabilities, threshold=0.5, per_type=2,
        abstained=metadata["parse_failure"]
    )
    assert set(np.flatnonzero(metadata["parse_failure"])) & set(selected)
    families = _feature_families(("javascript_count", "parse_failure", "x__missing", "pdf_size"))
    assert set(families) >= {"active_content", "parser_status", "missingness", "document_structure"}


def test_phase5_handoff_selection_is_deterministic_and_representative() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 8, dtype=np.int8)
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9] * 8)
    metadata = pd.DataFrame({"sample_id": [f"id-{index}" for index in range(len(labels))]})
    first = select_phase6_handoff_indices(
        labels, probabilities, metadata, threshold=0.5, maximum_rows=32, random_seed=42
    )
    second = select_phase6_handoff_indices(
        labels, probabilities, metadata, threshold=0.5, maximum_rows=32, random_seed=42
    )
    assert np.array_equal(first, second)
    outcomes = {
        (int(labels[index]), int(probabilities[index] >= 0.5)) for index in first
    }
    assert outcomes == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_phase6_fails_closed_before_upstream_phase5(tmp_path) -> None:
    runner = Phase6Runner()
    runner.output_root = tmp_path / "explainability"
    with pytest.raises(Phase6GateError, match="completed Phase 5"):
        runner.run()
    assert not runner.output_root.exists()
