from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.filters import inspect_bounded_object_streams
from src.features.vectorizer import extract_features_record
from src.models.phase7 import Phase7GateError, Phase7Runner
from src.security.adversarial import (
    FORBIDDEN_FIXTURE_MARKERS,
    MUTATIONS,
    build_inert_pdf,
    defense_matrix,
    evaluate_fixture_corpus,
    hex_escape_names,
    validate_inert_pdf,
)


class _FixturePipeline:
    def transform_record(self, record):
        risk = sum(
            float(record.get(name, 0.0))
            for name in (
                "js_count",
                "javascript_count",
                "openaction_count",
                "action_count",
                "aa_count",
            )
        )
        return pd.DataFrame([[risk, float(record.get("pdf_size", 0.0)) / 10_000.0]])


class _FixtureModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        values = np.asarray(features, dtype=float)
        probability = 1.0 / (1.0 + np.exp(-(values[:, 0] - 0.5)))
        return np.column_stack((1.0 - probability, probability))


def test_all_mutations_are_inert_valid_and_non_rendered() -> None:
    source = build_inert_pdf(security_marker=True, serial=7)
    assert validate_inert_pdf(source).valid
    assert len(MUTATIONS) >= 12
    digests = set()
    for name, mutation in MUTATIONS.items():
        mutated = mutation(source)
        validation = validate_inert_pdf(mutated)
        assert validation.valid, (name, validation.reason)
        assert validation.inert and validation.parseable and validation.pages >= 1
        assert validation.bytes <= 2 * 1024 * 1024
        digests.add(validation.sha256)
        assert all(marker.lower() not in mutated.lower() for marker in FORBIDDEN_FIXTURE_MARKERS)
    assert len(digests) == len(MUTATIONS)


def test_name_escaping_is_canonicalized_and_object_stream_is_bounded(tmp_path) -> None:
    source = build_inert_pdf(security_marker=True)
    escaped = hex_escape_names(source)
    assert b"/J#61vaScript" in escaped
    path = tmp_path / "escaped.pdf"
    path.write_bytes(escaped)
    features, diagnostics = extract_features_record(path)
    assert features["javascript_count"] >= 1
    assert features["openaction_count"] >= 1
    assert not diagnostics["abstain_recommended"]
    stream_mutation = MUTATIONS["object_stream_filter_chain"](source)
    inspection = inspect_bounded_object_streams(stream_mutation)
    assert inspection.object_streams_seen == 1
    assert inspection.object_streams_validated == 1
    assert not inspection.limit_reached


def test_validator_rejects_network_or_launch_behavior() -> None:
    source = build_inert_pdf(security_marker=False)
    unsafe = source.replace(b"/Type /Catalog", b"/Type /Catalog /URI (https://example.test)")
    result = validate_inert_pdf(unsafe)
    assert not result.valid
    assert not result.inert
    assert "forbidden_active_marker" in result.reason


def test_safe_fixture_evaluation_meets_eight_family_gate() -> None:
    scored, validity, metrics = evaluate_fixture_corpus(
        _FixturePipeline(), _FixtureModel(), threshold=0.5, fixtures_per_class=2
    )
    valid_families = validity.loc[
        (validity["mutation"] != "none") & validity["valid"], "mutation"
    ].nunique()
    assert valid_families >= 8
    assert "adaptive_query_selected_worst_case" in set(scored["mutation"])
    assert not scored["raw_pdf_persisted"].any()
    assert not metrics["benchmark_labels_are_malware_ground_truth"].any()
    assert {
        "literal_token_baseline",
        "canonical_multiview_indicator",
        "champion_model",
        "champion_plus_abstention",
    } == set(metrics["defense_view"])


def test_defense_matrix_separates_demonstrated_and_future_work() -> None:
    matrix = defense_matrix()
    assert (matrix["status"] == "demonstrated").sum() >= 4
    future = matrix[matrix["status"].str.startswith("future")]
    assert not future.empty
    assert set(future["pre"]) == {"not_run"}
    assert set(future["post"]) == {"not_run"}


def test_phase7_fails_closed_without_phase6(tmp_path) -> None:
    runner = Phase7Runner()
    runner.output_root = tmp_path / "adversarial"
    with pytest.raises(Phase7GateError, match="completed Phase 6"):
        runner.run()
    assert not runner.output_root.exists()
