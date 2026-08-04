"""Strict, scalable readers for approved PDF feature tables."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from src.config import FEATURE_COLUMNS, LABEL_COLUMN
from src.data.manifest import DataSourceSpec, load_source_registry
from src.data.schema import FeatureTableSchema
from src.features.schema_v2 import BASE_FEATURE_SPECS
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FEATURE_TABLE_SUFFIXES = {".csv", ".parquet", ".jsonl"}


def _require_supported(path: Path) -> str:
    if path.is_dir():
        # Canonical layers are partitioned Parquet datasets.
        return ".parquet"
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FEATURE_TABLE_SUFFIXES:
        raise ValueError(
            f"Unsupported feature-table format {suffix!r}; expected one of "
            f"{sorted(SUPPORTED_FEATURE_TABLE_SUFFIXES)}."
        )
    return suffix


def iter_dataset_batches(
    path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield bounded pandas batches from CSV, Parquet, or JSONL.

    Parquet scanning uses Arrow predicate/column projection infrastructure. CSV
    and JSONL use pandas chunking because incoming source dtypes must be inspected
    before any Arrow casting can occur.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Feature table not found: {source_path}")
    suffix = _require_supported(source_path)
    if suffix == ".csv":
        yield from pd.read_csv(
            source_path,
            usecols=list(columns) if columns else None,
            chunksize=batch_size,
            low_memory=False,
        )
        return
    if suffix == ".jsonl":
        yield from pd.read_json(
            source_path,
            lines=True,
            chunksize=batch_size,
        )
        return

    dataset = pads.dataset(source_path, format="parquet")
    scanner = dataset.scanner(columns=list(columns) if columns else None, batch_size=batch_size)
    for record_batch in scanner.to_batches():
        yield record_batch.to_pandas(types_mapper=None)


def dataset_row_count(path: Path) -> int:
    """Count rows without materializing a Parquet table when possible."""
    source_path = Path(path)
    suffix = _require_supported(source_path)
    if suffix == ".parquet":
        return int(pads.dataset(source_path, format="parquet").count_rows())
    return sum(len(batch) for batch in iter_dataset_batches(source_path))


def load_dataset(
    path: Optional[Path] = None,
    *,
    schema: Optional[FeatureTableSchema] = None,
    strict: bool = True,
    allow_extra: bool = False,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a complete feature table for analysis or small benchmarks.

    Multi-million-row training code should use :func:`iter_dataset_batches` or
    memory-mapped outputs instead.  No numeric field is silently coerced.
    """
    if path is None:
        raise ValueError(
            "An explicit approved feature-table path is required; no legacy dataset "
            "is selected automatically."
        )
    source_path = Path(path)
    frames: list[pd.DataFrame] = []
    consumed = 0
    for batch in iter_dataset_batches(source_path):
        if max_rows is not None:
            remaining = max_rows - consumed
            if remaining <= 0:
                break
            batch = batch.iloc[:remaining].copy()
        frames.append(batch)
        consumed += len(batch)
    if not frames:
        raise ValueError(f"Feature table is empty: {source_path}")
    frame = pd.concat(frames, ignore_index=True)
    active_schema = schema or FeatureTableSchema()
    if strict:
        active_schema.validate_frame(frame, allow_extra=allow_extra)
    logger.info("Loaded %s rows and %s columns from %s", *frame.shape, source_path)
    return frame


def load_registered_source(
    source_id: str,
    *,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load and canonicalize a configured local source under schema v2."""
    registry = load_source_registry()
    source: DataSourceSpec = registry.require(source_id)
    path = source.resolved_local_path()
    if path is None:
        raise ValueError(f"Source {source_id!r} has no configured local_path.")
    raw = load_dataset(path, strict=False, max_rows=max_rows)
    schema = FeatureTableSchema()
    canonical = schema.canonicalize_source_frame(
        raw,
        feature_overrides=source.column_mapping,
        metadata_mapping=source.metadata_mapping,
    )
    schema.validate_frame(canonical)
    return canonical


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    schema: Optional[FeatureTableSchema] = None,
    compression: str = "zstd",
) -> Path:
    """Write a canonical feature frame with Arrow schema metadata."""
    active_schema = schema or FeatureTableSchema()
    active_schema.validate_frame(frame)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``from_pandas`` with an explicit schema is an intentional, validated cast.
    import pyarrow as pa

    canonical = frame.loc[:, active_schema.expected_columns()].copy()
    canonical["first_seen_at"] = pd.to_datetime(
        canonical["first_seen_at"], utc=True
    )
    canonical["Class"] = canonical["Class"].astype(np.int8)
    canonical["label_confidence"] = canonical["label_confidence"].astype(np.float32)
    for spec in BASE_FEATURE_SPECS:
        if spec.kind == "boolean":
            canonical[spec.name] = canonical[spec.name].astype("Int8")
        elif spec.kind == "count":
            canonical[spec.name] = canonical[spec.name].astype("Int32")
        else:
            canonical[spec.name] = canonical[spec.name].astype(np.float32)
    table = pa.Table.from_pandas(
        canonical,
        schema=active_schema.arrow_schema(),
        preserve_index=False,
        safe=True,
    )
    pq.write_table(table, output, compression=compression)
    return output


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the canonical base-feature matrix and malicious label."""
    FeatureTableSchema().validate_frame(df, allow_extra=True)
    X = df.loc[:, FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(np.int8, copy=False)
    logger.info("Feature matrix shape: %s, label shape: %s", X.shape, y.shape)
    return X, y
