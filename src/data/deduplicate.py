"""Conflict-aware, streaming deduplication for canonical feature tables.

Provider/sample identifiers define identity.  A feature-row fingerprint is
tracked only as a secondary shortcut warning: two files can legitimately have
the same feature vector and are never merged for that reason alone.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, LABEL_COLUMN, SAMPLE_ID_COLUMN


@dataclass(frozen=True)
class DeduplicationStats:
    observed_rows: int
    unique_sample_ids: int
    duplicate_sample_id_rows: int
    conflicting_sample_ids: int
    conflicting_rows: int
    repeated_feature_rows: int
    conflicting_feature_hashes: int
    conflicting_feature_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def feature_row_fingerprints(frame: pd.DataFrame) -> pd.Series:
    """Return deterministic secondary fingerprints for the 37 raw features."""
    # pandas' vectorized hash is substantially more scalable than serializing
    # millions of rows in Python.  It is not used as document identity.
    hashed = pd.util.hash_pandas_object(
        frame.loc[:, FEATURE_COLUMNS], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return pd.Series([f"{value:016x}" for value in hashed], index=frame.index)


class StreamingDeduplicator:
    """Two-pass SQLite index that removes ID duplicates and all contradictions."""

    def __init__(self, database_path: Path | str = ":memory:") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                first_label INTEGER NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 1,
                conflict INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS feature_rows (
                feature_hash TEXT PRIMARY KEY,
                first_label INTEGER NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 1,
                conflict INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS emitted (
                sample_id TEXT PRIMARY KEY
            );
            """
        )
        self._observed_rows = 0

    def observe_batch(self, frame: pd.DataFrame) -> None:
        """Index one schema-valid batch without deciding which rows to emit."""
        identifiers = frame[SAMPLE_ID_COLUMN].astype(str).tolist()
        labels = frame[LABEL_COLUMN].astype(int).tolist()
        feature_hashes = feature_row_fingerprints(frame).tolist()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO samples(sample_id, first_label) VALUES (?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    row_count = row_count + 1,
                    conflict = MAX(conflict, first_label <> excluded.first_label)
                """,
                zip(identifiers, labels),
            )
            self.connection.executemany(
                """
                INSERT INTO feature_rows(feature_hash, first_label) VALUES (?, ?)
                ON CONFLICT(feature_hash) DO UPDATE SET
                    row_count = row_count + 1,
                    conflict = MAX(conflict, first_label <> excluded.first_label)
                """,
                zip(feature_hashes, labels),
            )
        self._observed_rows += len(frame)

    def selection_reasons(self, frame: pd.DataFrame) -> pd.Series:
        """Mark first non-conflicting IDs as keep and all others with a reason."""
        reasons: list[str] = []
        cursor = self.connection.cursor()
        with self.connection:
            for sample_id in frame[SAMPLE_ID_COLUMN].astype(str):
                record = cursor.execute(
                    "SELECT conflict FROM samples WHERE sample_id = ?", (sample_id,)
                ).fetchone()
                if record is None:
                    reasons.append("dedup_index_missing")
                    continue
                if bool(record[0]):
                    reasons.append("conflicting_label_for_sample_id")
                    continue
                inserted = cursor.execute(
                    "INSERT OR IGNORE INTO emitted(sample_id) VALUES (?)", (sample_id,)
                ).rowcount
                reasons.append("keep" if inserted else "duplicate_sample_id")
        return pd.Series(reasons, index=frame.index, dtype="string")

    def stats(self) -> DeduplicationStats:
        unique_ids, duplicates, conflicts, conflict_rows = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(row_count - 1), 0),
                   COALESCE(SUM(conflict), 0),
                   COALESCE(SUM(CASE WHEN conflict = 1 THEN row_count ELSE 0 END), 0)
            FROM samples
            """
        ).fetchone()
        repeated_features, conflicting_features, conflicting_feature_rows = self.connection.execute(
            """
            SELECT COALESCE(SUM(row_count - 1), 0), COALESCE(SUM(conflict), 0),
                   COALESCE(SUM(CASE WHEN conflict = 1 THEN row_count ELSE 0 END), 0)
            FROM feature_rows
            """
        ).fetchone()
        return DeduplicationStats(
            observed_rows=self._observed_rows,
            unique_sample_ids=int(unique_ids),
            duplicate_sample_id_rows=int(duplicates),
            conflicting_sample_ids=int(conflicts),
            conflicting_rows=int(conflict_rows),
            repeated_feature_rows=int(repeated_features),
            conflicting_feature_hashes=int(conflicting_features),
            conflicting_feature_rows=int(conflicting_feature_rows),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StreamingDeduplicator":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
