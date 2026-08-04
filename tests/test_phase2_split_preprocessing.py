import numpy as np
import pytest

from src.data.preprocessing import PreprocessingStateError, TrainOnlyPreprocessor
from src.data.splitter import (
    SplitRequirements,
    SplitGateError,
    build_frozen_splits_from_dataset,
    group_temporal_split,
    verify_frozen_splits,
    write_frozen_splits,
)
from tests.phase_helpers import canonical_frame


def test_group_temporal_split_has_no_sample_or_group_overlap(tmp_path):
    frame = canonical_frame(300)
    requirements = SplitRequirements(1, 1, 1, 0.995, 0.2, 0.2, "test-v2")
    train, validation, test, audit = group_temporal_split(
        frame, requirements=requirements
    )
    assert audit["gates_passed"]
    assert max(train["first_seen_at"]) < min(validation["first_seen_at"])
    assert max(validation["first_seen_at"]) < min(test["first_seen_at"])
    manifest = write_frozen_splits(
        train, validation, test, audit,
        requirements=requirements,
        output_root=tmp_path / "split-v2",
    )
    assert manifest.gates_passed
    assert (tmp_path / "split-v2" / "SEALED").is_file()
    verified = verify_frozen_splits(
        tmp_path / "split-v2", requirements=requirements, batch_size=19
    )
    assert verified.manifest_hash == manifest.manifest_hash
    with pytest.raises(FileExistsError):
        write_frozen_splits(
            train, validation, test, audit,
            requirements=requirements,
            output_root=tmp_path / "split-v2",
        )


def test_preprocessing_is_train_only_and_validation_cannot_change_statistics():
    train = canonical_frame(20)[["pdf_size", "obj_count"]]
    validation = train.copy()
    validation["pdf_size"] = 1e12
    processor = TrainOnlyPreprocessor(
        ["pdf_size", "obj_count"], model_family="neural", drop_constant=False
    )
    with pytest.raises(PreprocessingStateError):
        processor.fit(validation, partition_name="validation")
    processor.fit(train, partition_name="train")
    mean_before = processor.scaler_.mean_.copy()
    transformed = processor.transform(validation)
    assert np.array_equal(mean_before, processor.scaler_.mean_)
    assert processor.metadata().fit_partition == "train"
    assert any(name.endswith("__clipped") for name in transformed.columns)


def test_streaming_split_writes_separate_feature_and_identity_layers(tmp_path):
    source = tmp_path / "clean"
    source.mkdir()
    canonical_frame(90).to_parquet(source / "part-000000.parquet", index=False)
    requirements = SplitRequirements(1, 1, 1, 0.995, 0.2, 0.2, "stream-v2")
    manifest = build_frozen_splits_from_dataset(
        source,
        requirements=requirements,
        output_root=tmp_path / "stream-v2",
        batch_size=17,
    )
    assert manifest.gates_passed
    assert (tmp_path / "stream-v2" / "train" / "features").is_dir()
    assert (tmp_path / "stream-v2" / "train" / "ids").is_dir()
    verify_frozen_splits(
        tmp_path / "stream-v2", requirements=requirements, batch_size=11
    )
    first_part = next(
        (tmp_path / "stream-v2" / "test" / "features").glob("*.parquet")
    )
    first_part.write_bytes(first_part.read_bytes() + b"tamper")
    with pytest.raises(SplitGateError, match="Checksum mismatch"):
        verify_frozen_splits(
            tmp_path / "stream-v2", requirements=requirements, batch_size=11
        )
