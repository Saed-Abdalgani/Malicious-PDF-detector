"""Dataset-card and source-shortcut diagnostics for sanitized feature rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import FIRST_SEEN_COLUMN, LABEL_COLUMN, SOURCE_ID_COLUMN


@dataclass(frozen=True)
class SourceShortcutAudit:
    status: str
    roc_auc: float | None
    sampled_rows: int
    source_count: int
    unusually_predictive: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_label_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    table = (
        frame.groupby([SOURCE_ID_COLUMN, LABEL_COLUMN], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    return table.to_dict(orient="records")


def source_only_diagnostic(
    frame: pd.DataFrame,
    *,
    random_seed: int = 42,
    max_rows: int = 250_000,
    warning_auc: float = 0.80,
) -> SourceShortcutAudit:
    """Measure whether source/time metadata alone can recover the label."""
    if frame.empty or frame[LABEL_COLUMN].nunique() < 2:
        return SourceShortcutAudit(
            "not_estimable", None, len(frame), frame[SOURCE_ID_COLUMN].nunique(), False,
            "Both labels are required for a source-only ROC AUC.",
        )
    sampled = frame[[SOURCE_ID_COLUMN, FIRST_SEEN_COLUMN, LABEL_COLUMN]].copy()
    if len(sampled) > max_rows:
        sampled, _ = train_test_split(
            sampled,
            train_size=max_rows,
            stratify=sampled[LABEL_COLUMN],
            random_state=random_seed,
        )
    timestamp = pd.to_datetime(sampled[FIRST_SEEN_COLUMN], utc=True)
    sampled["first_seen_month"] = timestamp.dt.strftime("%Y-%m")
    X = sampled[[SOURCE_ID_COLUMN, "first_seen_month"]]
    y = sampled[LABEL_COLUMN].astype(int)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=random_seed
        )
        encoder = ColumnTransformer(
            [("metadata", OneHotEncoder(handle_unknown="ignore"), list(X.columns))]
        )
        model = make_pipeline(
            encoder,
            LogisticRegression(max_iter=250, class_weight="balanced", random_state=random_seed),
        )
        model.fit(X_train, y_train)
        auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    except ValueError as exc:
        return SourceShortcutAudit(
            "not_estimable", None, len(sampled), sampled[SOURCE_ID_COLUMN].nunique(),
            False, f"Diagnostic could not form a valid holdout: {exc}",
        )
    warning = bool(np.isfinite(auc) and auc >= warning_auc)
    return SourceShortcutAudit(
        "warning" if warning else "passed",
        auc,
        len(sampled),
        sampled[SOURCE_ID_COLUMN].nunique(),
        warning,
        (
            "Source/time metadata predicts labels unusually well; use source balancing "
            "or a sealed external-source holdout."
            if warning
            else "No unusually strong source/time-only shortcut was detected."
        ),
    )


def coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    times = pd.to_datetime(frame[FIRST_SEEN_COLUMN], utc=True)
    labels = frame[LABEL_COLUMN].value_counts().sort_index()
    return {
        "rows": int(len(frame)),
        "source_count": int(frame[SOURCE_ID_COLUMN].nunique()),
        "sources": sorted(frame[SOURCE_ID_COLUMN].astype(str).unique().tolist()),
        "first_seen_min_utc": times.min().isoformat() if len(times) else None,
        "first_seen_max_utc": times.max().isoformat() if len(times) else None,
        "label_counts": {str(int(k)): int(v) for k, v in labels.items()},
        "benign_prevalence": float((frame[LABEL_COLUMN] == 0).mean()),
        "source_label_contingency": source_label_table(frame),
    }
