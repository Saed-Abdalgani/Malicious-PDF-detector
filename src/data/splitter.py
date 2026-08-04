"""Deterministic, leakage-resistant group-temporal dataset splitting."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    FIRST_SEEN_COLUMN,
    GROUP_ID_COLUMN,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SPLITS_DIR,
)
from src.data.loader import write_parquet
from src.data.schema import FeatureTableSchema
from src.experiment import canonical_json, load_experiment_config, sha256_file
from src.utils.atomic import atomic_write_json


class SplitGateError(RuntimeError):
    """Raised when a candidate split violates a scientific acceptance gate."""


@dataclass(frozen=True)
class SplitRequirements:
    minimum_train_rows: int
    minimum_validation_rows: int
    minimum_test_rows: int
    minimum_benign_prevalence: float
    validation_fraction: float
    test_fraction: float
    split_version: str

    @classmethod
    def from_experiment(cls) -> "SplitRequirements":
        config = load_experiment_config()
        gates, splitting = config["acceptance_gates"], config["splitting"]
        return cls(
            minimum_train_rows=int(gates["minimum_train_rows"]),
            minimum_validation_rows=int(gates["minimum_validation_rows"]),
            minimum_test_rows=int(gates["minimum_test_rows"]),
            minimum_benign_prevalence=float(gates["minimum_benign_prevalence"]),
            validation_fraction=float(splitting["validation_fraction"]),
            test_fraction=float(splitting["test_fraction"]),
            split_version=str(config["split_version"]),
        )


@dataclass(frozen=True)
class SplitManifest:
    split_version: str
    strategy: str
    row_counts: dict[str, int]
    benign_prevalence: dict[str, float]
    time_ranges_utc: dict[str, dict[str, str]]
    group_counts: dict[str, int]
    sample_overlap: dict[str, int]
    group_overlap: dict[str, int]
    files: dict[str, Any]
    requirements: dict[str, Any]
    gates_passed: bool
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nearest_time_cut(cohorts: pd.DataFrame, target_rows: int) -> pd.Timestamp:
    cumulative = cohorts["rows"].cumsum()
    position = (cumulative - target_rows).abs().idxmin()
    return cohorts.loc[position, FIRST_SEEN_COLUMN]


def _overlap_count(left: pd.Series, right: pd.Series) -> int:
    return int(pd.Index(left.unique()).intersection(pd.Index(right.unique())).size)


def _partition_stats(partitions: dict[str, pd.DataFrame]) -> tuple[dict, dict, dict]:
    prevalence: dict[str, float] = {}
    ranges: dict[str, dict[str, str]] = {}
    group_counts: dict[str, int] = {}
    for name, frame in partitions.items():
        times = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True)
        prevalence[name] = float((frame[LABEL_COLUMN] == 0).mean()) if len(frame) else 0.0
        ranges[name] = {
            "minimum": times.min().isoformat() if len(times) else "",
            "maximum": times.max().isoformat() if len(times) else "",
        }
        group_counts[name] = int(frame[GROUP_ID_COLUMN].nunique())
    return prevalence, ranges, group_counts


def group_temporal_split(
    frame: pd.DataFrame,
    *,
    requirements: SplitRequirements | None = None,
    enforce_gates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Assign complete groups to ordered windows without resampling any row."""
    FeatureTableSchema().validate_frame(frame)
    req = requirements or SplitRequirements.from_experiment()
    if req.validation_fraction <= 0 or req.test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if req.validation_fraction + req.test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one.")

    work = frame.copy()
    work[FIRST_SEEN_COLUMN] = pd.to_datetime(work[FIRST_SEEN_COLUMN], utc=True)
    group_times = (
        work.groupby(GROUP_ID_COLUMN, sort=False)
        .agg(**{FIRST_SEEN_COLUMN: (FIRST_SEEN_COLUMN, "min"), "rows": (SAMPLE_ID_COLUMN, "size")})
        .reset_index()
    )
    cohorts = (
        group_times.groupby(FIRST_SEEN_COLUMN, as_index=False)["rows"].sum()
        .sort_values(FIRST_SEEN_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(cohorts) < 3:
        raise SplitGateError(
            "Group-temporal splitting needs at least three distinct first-seen cohorts."
        )
    total = len(work)
    train_target = round(total * (1 - req.validation_fraction - req.test_fraction))
    validation_end_target = round(total * (1 - req.test_fraction))
    train_cut = _nearest_time_cut(cohorts.iloc[:-2], train_target)
    later = cohorts[cohorts[FIRST_SEEN_COLUMN] > train_cut]
    if len(later) < 2:
        raise SplitGateError("No distinct validation and test cohorts remain.")
    validation_cut = _nearest_time_cut(
        cohorts[cohorts[FIRST_SEEN_COLUMN] < cohorts[FIRST_SEEN_COLUMN].max()],
        validation_end_target,
    )
    if validation_cut <= train_cut:
        validation_cut = later.iloc[0][FIRST_SEEN_COLUMN]

    group_times["partition"] = "test"
    group_times.loc[group_times[FIRST_SEEN_COLUMN] <= validation_cut, "partition"] = "validation"
    group_times.loc[group_times[FIRST_SEEN_COLUMN] <= train_cut, "partition"] = "train"
    work = work.merge(
        group_times[[GROUP_ID_COLUMN, "partition"]], on=GROUP_ID_COLUMN,
        how="left", validate="many_to_one",
    )
    partitions = {
        name: work.loc[work["partition"].eq(name)].drop(columns="partition").reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    prevalence, ranges, group_counts = _partition_stats(partitions)
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    sample_overlap = {
        f"{left}_{right}": _overlap_count(
            partitions[left][SAMPLE_ID_COLUMN], partitions[right][SAMPLE_ID_COLUMN]
        )
        for left, right in pairs
    }
    group_overlap = {
        f"{left}_{right}": _overlap_count(
            partitions[left][GROUP_ID_COLUMN], partitions[right][GROUP_ID_COLUMN]
        )
        for left, right in pairs
    }
    gate_checks: dict[str, bool] = {
        "train_minimum_rows": len(partitions["train"]) >= req.minimum_train_rows,
        "validation_minimum_rows": len(partitions["validation"]) >= req.minimum_validation_rows,
        "test_minimum_rows": len(partitions["test"]) >= req.minimum_test_rows,
        **{
            f"{name}_minimum_benign_prevalence": value >= req.minimum_benign_prevalence
            for name, value in prevalence.items()
        },
        "sample_overlap_zero": not any(sample_overlap.values()),
        "group_overlap_zero": not any(group_overlap.values()),
        "strict_temporal_order": (
            ranges["train"]["maximum"] < ranges["validation"]["minimum"]
            and ranges["validation"]["maximum"] < ranges["test"]["minimum"]
        ),
    }
    audit = {
        "strategy": "group_temporal",
        "row_counts": {name: len(value) for name, value in partitions.items()},
        "benign_prevalence": prevalence,
        "time_ranges_utc": ranges,
        "group_counts": group_counts,
        "sample_overlap": sample_overlap,
        "group_overlap": group_overlap,
        "gate_checks": gate_checks,
        "gates_passed": all(gate_checks.values()),
    }
    if enforce_gates and not audit["gates_passed"]:
        failed = [name for name, passed in gate_checks.items() if not passed]
        raise SplitGateError(f"Candidate split failed gates: {failed}")
    return partitions["train"], partitions["validation"], partitions["test"], audit


def write_frozen_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    audit: dict[str, Any],
    *,
    requirements: SplitRequirements | None = None,
    output_root: Path | None = None,
) -> SplitManifest:
    """Write immutable split tables, row-ID manifests, and a sealed manifest."""
    req = requirements or SplitRequirements.from_experiment()
    root = Path(output_root or SPLITS_DIR / req.split_version)
    seal_path = root / "SEALED"
    if root.exists():
        raise FileExistsError(
            f"Split version {root} is already materialized; create a new split version."
        )
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}
    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        partition_root = root / name
        partition_root.mkdir()
        (partition_root / "features").mkdir()
        (partition_root / "ids").mkdir()
        data_path = write_parquet(frame, partition_root / "features" / "part-000000.parquet")
        ids_path = partition_root / "ids" / "part-000000.parquet"
        frame[[SAMPLE_ID_COLUMN, GROUP_ID_COLUMN]].to_parquet(ids_path, index=False)
        files[name] = {
            "features": [
                {
                    "path": data_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(data_path),
                }
            ],
            "sample_ids": [
                {
                    "path": ids_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(ids_path),
                }
            ],
        }
    payload = {
        "split_version": req.split_version,
        "strategy": "group_temporal",
        "row_counts": audit["row_counts"],
        "benign_prevalence": audit["benign_prevalence"],
        "time_ranges_utc": audit["time_ranges_utc"],
        "group_counts": audit["group_counts"],
        "sample_overlap": audit["sample_overlap"],
        "group_overlap": audit["group_overlap"],
        "files": files,
        "requirements": asdict(req),
        "gates_passed": bool(audit["gates_passed"]),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    manifest = SplitManifest(**payload, manifest_hash=digest)
    manifest_path = root / "split_manifest.json"
    atomic_write_json(manifest_path, manifest.to_dict())
    atomic_write_json(
        seal_path,
        {"manifest_sha256": sha256_file(manifest_path), "manifest_hash": digest},
    )
    return manifest


def _lookup_partitions(
    connection: sqlite3.Connection, group_ids: list[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    unique = list(dict.fromkeys(group_ids))
    for start in range(0, len(unique), 800):
        chunk = unique[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT group_id, partition_name FROM groups WHERE group_id IN ({placeholders})",
            chunk,
        ).fetchall()
        result.update((str(group_id), str(partition)) for group_id, partition in rows)
    return result


def build_frozen_splits_from_dataset(
    dataset_path: Path,
    *,
    requirements: SplitRequirements | None = None,
    output_root: Path | None = None,
    batch_size: int = 100_000,
) -> SplitManifest:
    """Stream a canonical Parquet dataset into sealed group-temporal partitions."""
    from src.data.loader import iter_dataset_batches

    req = requirements or SplitRequirements.from_experiment()
    root = Path(output_root or SPLITS_DIR / req.split_version)
    if root.exists():
        raise FileExistsError(f"Split output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-building-", dir=root.parent))
    database = temporary / "split-index.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE groups (
            group_id TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            partition_name TEXT
        );
        CREATE TABLE sample_ids (
            sample_id TEXT PRIMARY KEY,
            partition_name TEXT NOT NULL
        );
        """
    )
    try:
        input_rows = 0
        metadata_columns = [GROUP_ID_COLUMN, FIRST_SEEN_COLUMN, SAMPLE_ID_COLUMN]
        for batch in iter_dataset_batches(
            Path(dataset_path), columns=metadata_columns, batch_size=batch_size
        ):
            input_rows += len(batch)
            batch[FIRST_SEEN_COLUMN] = pd.to_datetime(batch[FIRST_SEEN_COLUMN], utc=True)
            grouped = (
                batch.groupby(GROUP_ID_COLUMN, sort=False)
                .agg(first_seen=(FIRST_SEEN_COLUMN, "min"), row_count=(SAMPLE_ID_COLUMN, "size"))
                .reset_index()
            )
            records = [
                (str(row[GROUP_ID_COLUMN]), row["first_seen"].isoformat(), int(row["row_count"]))
                for _, row in grouped.iterrows()
            ]
            with connection:
                connection.executemany(
                    """
                    INSERT INTO groups(group_id, first_seen, row_count) VALUES (?, ?, ?)
                    ON CONFLICT(group_id) DO UPDATE SET
                        first_seen = MIN(first_seen, excluded.first_seen),
                        row_count = row_count + excluded.row_count
                    """,
                    records,
                )
        cohort_rows = connection.execute(
            "SELECT first_seen, SUM(row_count) FROM groups GROUP BY first_seen ORDER BY first_seen"
        ).fetchall()
        cohorts = pd.DataFrame(cohort_rows, columns=[FIRST_SEEN_COLUMN, "rows"])
        cohorts[FIRST_SEEN_COLUMN] = pd.to_datetime(cohorts[FIRST_SEEN_COLUMN], utc=True)
        if len(cohorts) < 3:
            raise SplitGateError("At least three distinct time cohorts are required.")
        train_target = round(input_rows * (1 - req.validation_fraction - req.test_fraction))
        validation_target = round(input_rows * (1 - req.test_fraction))
        train_cut = _nearest_time_cut(cohorts.iloc[:-2], train_target)
        validation_cut = _nearest_time_cut(cohorts.iloc[:-1], validation_target)
        if validation_cut <= train_cut:
            later = cohorts[cohorts[FIRST_SEEN_COLUMN] > train_cut]
            if len(later) < 2:
                raise SplitGateError("No distinct validation and test cohorts remain.")
            validation_cut = later.iloc[0][FIRST_SEEN_COLUMN]
        train_cut_text = train_cut.isoformat()
        validation_cut_text = validation_cut.isoformat()
        with connection:
            connection.execute(
                "UPDATE groups SET partition_name = CASE "
                "WHEN first_seen <= ? THEN 'train' "
                "WHEN first_seen <= ? THEN 'validation' ELSE 'test' END",
                (train_cut_text, validation_cut_text),
            )

        rows_by_partition: Counter[str] = Counter()
        labels_by_partition: dict[str, Counter[int]] = {
            name: Counter() for name in ("train", "validation", "test")
        }
        time_min: dict[str, pd.Timestamp | None] = {name: None for name in labels_by_partition}
        time_max: dict[str, pd.Timestamp | None] = {name: None for name in labels_by_partition}
        part_numbers: Counter[str] = Counter()
        duplicate_sample_ids = 0
        for name in labels_by_partition:
            (temporary / name).mkdir()
            (temporary / name / "features").mkdir()
            (temporary / name / "ids").mkdir()
        for batch in iter_dataset_batches(Path(dataset_path), batch_size=batch_size):
            mapping = _lookup_partitions(
                connection, batch[GROUP_ID_COLUMN].astype(str).tolist()
            )
            assigned = batch[GROUP_ID_COLUMN].astype(str).map(mapping)
            if assigned.isna().any():
                raise SplitGateError("A group was absent from the split index.")
            for name in labels_by_partition:
                partition = batch.loc[assigned.eq(name)].copy()
                if partition.empty:
                    continue
                number = part_numbers[name]
                path = temporary / name / "features" / f"part-{number:06d}.parquet"
                ids_path = temporary / name / "ids" / f"part-{number:06d}.parquet"
                write_parquet(partition, path)
                partition[[SAMPLE_ID_COLUMN, GROUP_ID_COLUMN]].to_parquet(ids_path, index=False)
                part_numbers[name] += 1
                rows_by_partition[name] += len(partition)
                labels_by_partition[name].update(partition[LABEL_COLUMN].astype(int).tolist())
                times = pd.to_datetime(partition[FIRST_SEEN_COLUMN], utc=True)
                low, high = times.min(), times.max()
                time_min[name] = low if time_min[name] is None else min(time_min[name], low)
                time_max[name] = high if time_max[name] is None else max(time_max[name], high)
                records = zip(partition[SAMPLE_ID_COLUMN].astype(str), [name] * len(partition))
                with connection:
                    before = connection.total_changes
                    connection.executemany(
                        "INSERT OR IGNORE INTO sample_ids(sample_id, partition_name) VALUES (?, ?)",
                        records,
                    )
                    inserted = connection.total_changes - before
                duplicate_sample_ids += len(partition) - inserted

        prevalence = {
            name: (
                labels_by_partition[name][0] / rows_by_partition[name]
                if rows_by_partition[name] else 0.0
            )
            for name in labels_by_partition
        }
        ranges = {
            name: {
                "minimum": time_min[name].isoformat() if time_min[name] is not None else "",
                "maximum": time_max[name].isoformat() if time_max[name] is not None else "",
            }
            for name in labels_by_partition
        }
        group_counts = dict(
            connection.execute(
                "SELECT partition_name, COUNT(*) FROM groups GROUP BY partition_name"
            ).fetchall()
        )
        gate_checks = {
            "train_minimum_rows": rows_by_partition["train"] >= req.minimum_train_rows,
            "validation_minimum_rows": rows_by_partition["validation"] >= req.minimum_validation_rows,
            "test_minimum_rows": rows_by_partition["test"] >= req.minimum_test_rows,
            **{
                f"{name}_minimum_benign_prevalence": value >= req.minimum_benign_prevalence
                for name, value in prevalence.items()
            },
            "sample_overlap_zero": duplicate_sample_ids == 0,
            "group_overlap_zero": True,
            "strict_temporal_order": (
                ranges["train"]["maximum"] < ranges["validation"]["minimum"]
                and ranges["validation"]["maximum"] < ranges["test"]["minimum"]
            ),
        }
        if not all(gate_checks.values()):
            failed = [name for name, passed in gate_checks.items() if not passed]
            raise SplitGateError(f"Streaming candidate split failed gates: {failed}")

        connection.close()
        database.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(database) + suffix).unlink(missing_ok=True)
        files: dict[str, Any] = {}
        for name in labels_by_partition:
            feature_parts = sorted((temporary / name / "features").glob("*.parquet"))
            id_parts = sorted((temporary / name / "ids").glob("*.parquet"))
            files[name] = {
                "features": [
                    {"path": path.relative_to(temporary).as_posix(), "sha256": sha256_file(path)}
                    for path in feature_parts
                ],
                "sample_ids": [
                    {"path": path.relative_to(temporary).as_posix(), "sha256": sha256_file(path)}
                    for path in id_parts
                ],
            }
        payload = {
            "split_version": req.split_version,
            "strategy": "group_temporal_streaming",
            "row_counts": dict(rows_by_partition),
            "benign_prevalence": prevalence,
            "time_ranges_utc": ranges,
            "group_counts": {name: int(group_counts.get(name, 0)) for name in labels_by_partition},
            "sample_overlap": {"all_partitions": duplicate_sample_ids},
            "group_overlap": {"all_partitions": 0},
            "files": files,
            "requirements": asdict(req),
            "gates_passed": True,
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        manifest = SplitManifest(**payload, manifest_hash=digest)
        manifest_path = temporary / "split_manifest.json"
        atomic_write_json(manifest_path, manifest.to_dict())
        atomic_write_json(
            temporary / "SEALED",
            {"manifest_sha256": sha256_file(manifest_path), "manifest_hash": digest},
        )
        shutil.move(str(temporary), str(root))
        return manifest
    except Exception:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _safe_manifest_parts(
    root: Path,
    manifest: dict[str, Any],
    partition: str,
    kind: str,
) -> list[Path]:
    entries = manifest.get("files", {}).get(partition, {}).get(kind)
    if not isinstance(entries, list) or not entries:
        raise SplitGateError(f"Manifest has no {partition}/{kind} parts.")
    paths: list[Path] = []
    root_resolved = root.resolve()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SplitGateError(f"Malformed {partition}/{kind} manifest entry.")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SplitGateError(f"Unsafe split-manifest path: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise SplitGateError(f"Split part escapes sealed root: {relative}") from exc
        if not path.is_file():
            raise SplitGateError(f"Sealed split part is missing: {path}")
        expected_hash = str(entry.get("sha256", ""))
        if sha256_file(path) != expected_hash:
            raise SplitGateError(f"Checksum mismatch for sealed split part: {path}")
        paths.append(path)
    return paths


def verify_frozen_splits(
    split_root: Path,
    *,
    requirements: SplitRequirements | None = None,
    batch_size: int = 100_000,
) -> SplitManifest:
    """Independently prove a sealed split's hashes, IDs, groups, time, and gates."""
    from src.data.loader import iter_dataset_batches

    root = Path(split_root)
    manifest_path = root / "split_manifest.json"
    seal_path = root / "SEALED"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise SplitGateError(f"Split root is not sealed: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SplitGateError("Split manifest or seal is unreadable.") from exc
    if seal.get("manifest_sha256") != sha256_file(manifest_path):
        raise SplitGateError("SEALED manifest checksum does not match split_manifest.json.")
    stored_hash = manifest.get("manifest_hash")
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    calculated_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if stored_hash != calculated_hash or seal.get("manifest_hash") != calculated_hash:
        raise SplitGateError("Logical split manifest hash is invalid.")
    req = requirements or SplitRequirements.from_experiment()
    if manifest.get("split_version") != req.split_version:
        raise SplitGateError(
            f"Split version mismatch: {manifest.get('split_version')} != {req.split_version}"
        )
    if not manifest.get("gates_passed"):
        raise SplitGateError("Manifest does not claim passed gates.")

    database_dir = Path(tempfile.mkdtemp(prefix="split-verification-"))
    connection = sqlite3.connect(database_dir / "verification.sqlite")
    connection.executescript(
        """
        CREATE TABLE feature_rows (
            sample_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            partition_name TEXT NOT NULL
        );
        CREATE TABLE id_rows (
            sample_id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            partition_name TEXT NOT NULL
        );
        CREATE TABLE groups (
            group_id TEXT PRIMARY KEY,
            partition_name TEXT NOT NULL
        );
        """
    )
    partitions = ("train", "validation", "test")
    observed_rows: Counter[str] = Counter()
    benign_rows: Counter[str] = Counter()
    time_min: dict[str, pd.Timestamp | None] = {name: None for name in partitions}
    time_max: dict[str, pd.Timestamp | None] = {name: None for name in partitions}
    sample_conflicts = group_conflicts = 0
    try:
        for partition in partitions:
            feature_paths = _safe_manifest_parts(root, manifest, partition, "features")
            id_paths = _safe_manifest_parts(root, manifest, partition, "sample_ids")
            expected_feature_paths = {
                path.resolve() for path in (root / partition / "features").glob("*.parquet")
            }
            expected_id_paths = {
                path.resolve() for path in (root / partition / "ids").glob("*.parquet")
            }
            if set(feature_paths) != expected_feature_paths:
                raise SplitGateError(f"Unlisted or missing feature parts in {partition}.")
            if set(id_paths) != expected_id_paths:
                raise SplitGateError(f"Unlisted or missing ID parts in {partition}.")

            for path in feature_paths:
                for batch in iter_dataset_batches(
                    path,
                    columns=[
                        SAMPLE_ID_COLUMN,
                        GROUP_ID_COLUMN,
                        FIRST_SEEN_COLUMN,
                        LABEL_COLUMN,
                    ],
                    batch_size=batch_size,
                ):
                    observed_rows[partition] += len(batch)
                    benign_rows[partition] += int(batch[LABEL_COLUMN].eq(0).sum())
                    times = pd.to_datetime(batch[FIRST_SEEN_COLUMN], utc=True)
                    low, high = times.min(), times.max()
                    time_min[partition] = (
                        low if time_min[partition] is None else min(time_min[partition], low)
                    )
                    time_max[partition] = (
                        high if time_max[partition] is None else max(time_max[partition], high)
                    )
                    with connection:
                        before = connection.total_changes
                        connection.executemany(
                            "INSERT OR IGNORE INTO feature_rows VALUES (?, ?, ?)",
                            zip(
                                batch[SAMPLE_ID_COLUMN].astype(str),
                                batch[GROUP_ID_COLUMN].astype(str),
                                [partition] * len(batch),
                            ),
                        )
                        sample_conflicts += len(batch) - (connection.total_changes - before)
                        unique_groups = batch[GROUP_ID_COLUMN].astype(str).unique().tolist()
                        before = connection.total_changes
                        connection.executemany(
                            "INSERT OR IGNORE INTO groups VALUES (?, ?)",
                            ((group_id, partition) for group_id in unique_groups),
                        )
                        inserted_groups = connection.total_changes - before
                        if inserted_groups != len(unique_groups):
                            for group_id in unique_groups:
                                existing = connection.execute(
                                    "SELECT partition_name FROM groups WHERE group_id = ?",
                                    (group_id,),
                                ).fetchone()
                                if existing and existing[0] != partition:
                                    group_conflicts += 1
            for path in id_paths:
                for batch in iter_dataset_batches(path, batch_size=batch_size):
                    if set(batch.columns) != {SAMPLE_ID_COLUMN, GROUP_ID_COLUMN}:
                        raise SplitGateError(f"Invalid ID-manifest schema: {path}")
                    with connection:
                        before = connection.total_changes
                        connection.executemany(
                            "INSERT OR IGNORE INTO id_rows VALUES (?, ?, ?)",
                            zip(
                                batch[SAMPLE_ID_COLUMN].astype(str),
                                batch[GROUP_ID_COLUMN].astype(str),
                                [partition] * len(batch),
                            ),
                        )
                        if connection.total_changes - before != len(batch):
                            raise SplitGateError("Duplicate sample ID in ID manifests.")

        feature_minus_ids = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT * FROM feature_rows EXCEPT SELECT * FROM id_rows
            )
            """
        ).fetchone()[0]
        ids_minus_features = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT * FROM id_rows EXCEPT SELECT * FROM feature_rows
            )
            """
        ).fetchone()[0]
        if feature_minus_ids or ids_minus_features:
            raise SplitGateError("Feature rows and row/group-ID manifests do not match exactly.")
        if sample_conflicts or group_conflicts:
            raise SplitGateError(
                f"Overlap detected: sample={sample_conflicts}, group={group_conflicts}."
            )
        if any(int(value) != 0 for value in manifest.get("sample_overlap", {}).values()):
            raise SplitGateError("Manifest records nonzero sample overlap.")
        if any(int(value) != 0 for value in manifest.get("group_overlap", {}).values()):
            raise SplitGateError("Manifest records nonzero group overlap.")
        actual_group_counts = dict(
            connection.execute(
                "SELECT partition_name, COUNT(*) FROM groups GROUP BY partition_name"
            ).fetchall()
        )
        prevalence = {
            name: benign_rows[name] / observed_rows[name] if observed_rows[name] else 0.0
            for name in partitions
        }
        minimums = {
            "train": req.minimum_train_rows,
            "validation": req.minimum_validation_rows,
            "test": req.minimum_test_rows,
        }
        for name in partitions:
            if observed_rows[name] < minimums[name]:
                raise SplitGateError(f"{name} has only {observed_rows[name]} rows.")
            if prevalence[name] < req.minimum_benign_prevalence:
                raise SplitGateError(f"{name} benign prevalence is only {prevalence[name]:.8f}.")
            if int(manifest["row_counts"].get(name, -1)) != observed_rows[name]:
                raise SplitGateError(f"{name} row count differs from manifest.")
            if int(manifest["group_counts"].get(name, -1)) != int(
                actual_group_counts.get(name, 0)
            ):
                raise SplitGateError(f"{name} group count differs from manifest.")
            if abs(float(manifest["benign_prevalence"].get(name, -1)) - prevalence[name]) > 1e-12:
                raise SplitGateError(f"{name} prevalence differs from manifest.")
            actual_range = {
                "minimum": time_min[name].isoformat() if time_min[name] is not None else "",
                "maximum": time_max[name].isoformat() if time_max[name] is not None else "",
            }
            if manifest["time_ranges_utc"].get(name) != actual_range:
                raise SplitGateError(f"{name} time range differs from manifest.")
        if not (
            time_max["train"] < time_min["validation"]
            and time_max["validation"] < time_min["test"]
        ):
            raise SplitGateError("Temporal windows are not strictly ordered.")
        return SplitManifest(**manifest)
    finally:
        connection.close()
        shutil.rmtree(database_dir, ignore_errors=True)


def stratified_split(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Random row splitting is disabled; use group_temporal_split().")


def apply_smote(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("SMOTE is prohibited in the production pipeline.")


def save_splits(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("Use write_frozen_splits() to create checksummed, sealed splits.")
