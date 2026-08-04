"""Immutable memory-mapped model matrices derived from the sealed feature pipeline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    FIRST_SEEN_COLUMN,
    GROUP_ID_COLUMN,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
)
from src.data.loader import dataset_row_count, iter_dataset_batches
from src.experiment import sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.utils.atomic import atomic_write_json


SUBGROUP_COLUMNS = (
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
    GROUP_ID_COLUMN,
    FIRST_SEEN_COLUMN,
    LABEL_COLUMN,
    "pdf_size",
    "is_encrypted",
    "js_count",
    "javascript_count",
    "openaction_count",
    "action_count",
    "aa_count",
    "launch_count",
    "uri_count",
    "submitform_count",
    "obfuscation_count",
    "parse_failure",
    "recovery_mode",
    "invalid_eof",
    "extraction_timeout",
    "file_too_large",
    "extraction_limit_reached",
)


@dataclass(frozen=True)
class ModelMatrixManifest:
    version: str
    partition: str
    rows: int
    columns: int
    feature_names: tuple[str, ...]
    label_used_as_feature: bool
    input_path: str
    input_parts: tuple[dict[str, Any], ...]
    x_path: str
    x_sha256: str
    y_path: str
    y_sha256: str
    metadata_parts: tuple[dict[str, Any], ...]
    feature_pipeline_sha256: str
    split_manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        value["input_parts"] = list(self.input_parts)
        value["metadata_parts"] = list(self.metadata_parts)
        return value


def _safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe model-matrix manifest path: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Model-matrix path escapes root: {relative}") from exc
    return path


def materialize_model_matrix(
    dataset_path: Path,
    output_dir: Path,
    *,
    pipeline: FeaturePipelineV2,
    pipeline_path: Path,
    split_manifest_path: Path,
    partition_name: str,
    batch_size: int = 100_000,
    allow_test: bool = False,
) -> Path:
    """Transform a sealed partition once into an immutable float32 memmap."""
    if partition_name == "test" and not allow_test:
        raise RuntimeError("Sealed test matrix requires a live Phase 5 access claim.")
    if partition_name not in {"train", "validation", "test"}:
        raise ValueError("Unknown matrix partition.")
    source = Path(dataset_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Model matrix output is immutable: {destination}")
    if not Path(pipeline_path).is_file() or not Path(split_manifest_path).is_file():
        raise FileNotFoundError("Model matrix provenance artifacts are missing.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = dataset_row_count(source)
    if rows < 1:
        raise RuntimeError(f"Cannot materialize empty partition: {source}")
    feature_names = tuple(pipeline.output_feature_names_)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    x_path = temporary / "X.npy"
    y_path = temporary / "y.npy"
    metadata_parts = temporary / "metadata"
    metadata_parts.mkdir()
    matrix = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, len(feature_names)),
    )
    labels = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.int8, shape=(rows,)
    )
    position = 0
    try:
        for part_index, batch in enumerate(
            iter_dataset_batches(source, batch_size=batch_size)
        ):
            transformed = pipeline.transform(batch)
            if tuple(transformed.columns) != feature_names:
                raise RuntimeError("Pipeline output columns changed during matrix build.")
            values = transformed.to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(values).all():
                raise RuntimeError("Pipeline emitted non-finite model inputs.")
            end = position + len(batch)
            matrix[position:end] = values
            labels[position:end] = batch[LABEL_COLUMN].to_numpy(
                dtype=np.int8, copy=False
            )
            # Preserve every safe numeric base input for later local explanations.
            # This remains a feature-only table: no raw PDF content, text, URLs,
            # payloads, or model outputs are admitted.
            metadata_columns = list(
                dict.fromkeys(
                    [
                        *[name for name in SUBGROUP_COLUMNS if name in batch],
                        *[
                            name
                            for name in pipeline.required_input_columns
                            if name in batch and name != LABEL_COLUMN
                        ],
                    ]
                )
            )
            metadata = batch.loc[:, metadata_columns].copy()
            base_features = batch.loc[:, pipeline.required_input_columns].copy()
            metadata["missing_feature_count"] = base_features.isna().sum(axis=1).astype(
                np.int16
            )
            metadata.to_parquet(
                metadata_parts / f"part-{part_index:06d}.parquet",
                index=False,
                compression="zstd",
            )
            position = end
        if position != rows:
            raise RuntimeError(f"Matrix row mismatch: wrote {position}, expected {rows}.")
        matrix.flush()
        labels.flush()
        matrix = None
        labels = None
        metadata_entries = tuple(
            {
                "path": path.relative_to(temporary).as_posix(),
                "rows": int(len(pd.read_parquet(path, columns=[LABEL_COLUMN]))),
                "sha256": sha256_file(path),
            }
            for path in sorted(metadata_parts.glob("*.parquet"))
        )
        input_parts = tuple(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in sorted(source.glob("*.parquet"))
        )
        manifest = ModelMatrixManifest(
            version="1.0.0",
            partition=partition_name,
            rows=rows,
            columns=len(feature_names),
            feature_names=feature_names,
            label_used_as_feature=False,
            input_path=str(source.resolve()),
            input_parts=input_parts,
            x_path="X.npy",
            x_sha256=sha256_file(x_path),
            y_path="y.npy",
            y_sha256=sha256_file(y_path),
            metadata_parts=metadata_entries,
            feature_pipeline_sha256=sha256_file(Path(pipeline_path)),
            split_manifest_sha256=sha256_file(Path(split_manifest_path)),
        )
        atomic_write_json(temporary / "matrix_manifest.json", manifest.to_dict())
        os.replace(temporary, destination)
        return destination / "matrix_manifest.json"
    except Exception:
        matrix = None
        labels = None
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_model_matrix(
    matrix_dir: Path,
    *,
    expected_partition: str,
    feature_pipeline_path: Path,
    split_manifest_path: Path,
    load_metadata: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, ModelMatrixManifest]:
    """Verify and memory-map a materialized model matrix."""
    root = Path(matrix_dir)
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Matrix manifest is missing: {manifest_path}")
    try:
        manifest = ModelMatrixManifest(**json.loads(manifest_path.read_text("utf-8")))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("Model matrix manifest is malformed.") from exc
    if manifest.partition != expected_partition or manifest.label_used_as_feature:
        raise RuntimeError("Model matrix partition/label contract is invalid.")
    x_path = _safe_relative(root, manifest.x_path)
    y_path = _safe_relative(root, manifest.y_path)
    expected_hashes = {
        x_path: manifest.x_sha256,
        y_path: manifest.y_sha256,
        Path(feature_pipeline_path): manifest.feature_pipeline_sha256,
        Path(split_manifest_path): manifest.split_manifest_sha256,
    }
    for path, digest in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Model matrix provenance/checksum mismatch: {path}")
    listed_metadata: list[Path] = []
    metadata_rows = 0
    for entry in manifest.metadata_parts:
        path = _safe_relative(root, entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Model matrix metadata mismatch: {path}")
        listed_metadata.append(path)
        metadata_rows += int(entry["rows"])
    actual_metadata = {
        path.resolve() for path in (root / "metadata").glob("*.parquet")
    }
    if set(listed_metadata) != actual_metadata or metadata_rows != manifest.rows:
        raise RuntimeError("Model matrix metadata part inventory is invalid.")
    input_root = Path(manifest.input_path)
    if not input_root.is_dir():
        raise RuntimeError("Model matrix input dataset is missing.")
    listed_inputs: set[Path] = set()
    for entry in manifest.input_parts:
        path = Path(entry["path"]).resolve()
        try:
            path.relative_to(input_root.resolve())
        except ValueError as exc:
            raise RuntimeError("Model matrix input part escapes input dataset.") from exc
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Model matrix input part mismatch: {path}")
        listed_inputs.add(path)
    if listed_inputs != {path.resolve() for path in input_root.glob("*.parquet")}:
        raise RuntimeError("Model matrix input part inventory is invalid.")
    features = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path, mmap_mode="r")
    metadata = (
        pd.concat((pd.read_parquet(path) for path in listed_metadata), ignore_index=True)
        if load_metadata
        else pd.DataFrame(index=pd.RangeIndex(manifest.rows))
    )
    if (
        features.shape != (manifest.rows, manifest.columns)
        or labels.shape != (manifest.rows,)
        or len(metadata) != manifest.rows
        or len(manifest.feature_names) != manifest.columns
    ):
        raise RuntimeError("Model matrix shapes differ from its manifest.")
    return features, labels, metadata, manifest
