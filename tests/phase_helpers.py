from __future__ import annotations

import pandas as pd

from src.features.schema_v2 import BASE_FEATURE_SPECS


def canonical_frame(rows: int = 12, *, malicious_indices=()) -> pd.DataFrame:
    malicious = set(malicious_indices)
    values = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for index in range(rows):
        row = {}
        for offset, spec in enumerate(BASE_FEATURE_SPECS):
            if spec.kind == "boolean":
                value = float((index + offset) % 2)
            elif spec.name == "pdf_size":
                value = float(1_000 + index * 17)
            else:
                value = float((index + offset) % 7)
            row[spec.name] = value
        row.update(
            sample_id=f"sample-{index:06d}",
            Class=1 if index in malicious else 0,
            source_id="approved-source-a" if index % 2 else "approved-source-b",
            group_id=f"group-{index:06d}",
            first_seen_at=start + pd.Timedelta(days=index),
            label_confidence=1.0,
        )
        values.append(row)
    return pd.DataFrame(values)
