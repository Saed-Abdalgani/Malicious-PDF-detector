from __future__ import annotations

import numpy as np
import pandas as pd

from src.validated_bundle import ManuallyValidatedFeaturePipeline, _nonnegative_numeric, _split_indices
from tests.phase_helpers import canonical_frame


def test_legacy_numeric_cleaning_preserves_only_supported_nonnegative_values():
    values = pd.Series(["1", "29(2)", "-1", "pdfid.py", ">", " 3.5 "])
    parsed = _nonnegative_numeric(values)
    assert parsed.iloc[0] == 1.0
    assert parsed.iloc[1] == 29.0
    assert np.isnan(parsed.iloc[2])
    assert np.isnan(parsed.iloc[3])
    assert np.isnan(parsed.iloc[4])
    assert parsed.iloc[5] == 3.5


def test_validated_partitions_are_disjoint_complete_and_stratified():
    labels = np.r_[np.zeros(4_468, dtype=np.int8), np.ones(5_555, dtype=np.int8)]
    partitions = _split_indices(labels)
    combined = np.concatenate(list(partitions.values()))
    assert len(combined) == len(labels)
    assert len(np.unique(combined)) == len(labels)
    assert set(combined) == set(range(len(labels)))
    assert len(partitions["train"]) == 7_016
    for indices in partitions.values():
        assert set(np.unique(labels[indices])) == {0, 1}


def test_validated_pipeline_masks_source_unavailable_live_features():
    frame = canonical_frame(120)
    pipeline = ManuallyValidatedFeaturePipeline(unavailable_features=("uri_count",)).fit(
        frame.assign(uri_count=np.nan), partition_name="train"
    )
    first = frame.iloc[0].to_dict()
    second = dict(first)
    first["uri_count"] = 0.0
    second["uri_count"] = 999.0
    assert np.array_equal(
        pipeline.transform_record(first).to_numpy(),
        pipeline.transform_record(second).to_numpy(),
    )
