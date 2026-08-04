import json

import pytest

from src.artifacts import (
    ArtifactCompatibilityError,
    verify_deployable_model_artifact,
    verify_artifact_compatibility,
    write_artifact_metadata,
    write_model_artifact_metadata,
)
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file


def test_artifact_sidecar_binds_identity_and_checksum(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model-v2")
    identity = create_experiment_identity(
        load_experiment_config(), code_commit="test-commit"
    )
    sidecar = write_artifact_metadata(
        artifact, "test_model", identity, extra={"threshold": 0.73}
    )
    assert json.loads(sidecar.read_text())["extra"]["threshold"] == 0.73
    verify_artifact_compatibility(artifact, identity)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ArtifactCompatibilityError, match="checksum"):
        verify_artifact_compatibility(artifact, identity)


def test_deployable_model_requires_locked_threshold_and_upstream_hashes(tmp_path):
    identity = create_experiment_identity(
        load_experiment_config(), code_commit="test-commit"
    )
    model = tmp_path / "model.bin"
    quality = tmp_path / "dataset_quality.json"
    split = tmp_path / "split_manifest.json"
    pipeline = tmp_path / "feature_pipeline.pkl"
    model.write_bytes(b"model")
    quality.write_bytes(b"quality")
    split.write_bytes(b"split")
    pipeline.write_bytes(b"pipeline")
    write_model_artifact_metadata(
        model,
        identity,
        model_type="test",
        threshold=0.73,
        dataset_quality_sha256=sha256_file(quality),
        split_manifest_sha256=sha256_file(split),
        feature_pipeline_sha256=sha256_file(pipeline),
    )
    verify_deployable_model_artifact(
        model,
        identity,
        dataset_quality_path=quality,
        split_manifest_path=split,
        feature_pipeline_path=pipeline,
    )
    split.write_bytes(b"changed")
    with pytest.raises(ArtifactCompatibilityError, match="provenance mismatch"):
        verify_deployable_model_artifact(
            model,
            identity,
            dataset_quality_path=quality,
            split_manifest_path=split,
            feature_pipeline_path=pipeline,
        )
