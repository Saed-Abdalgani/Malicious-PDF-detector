"""Strict schema validation for safe PDF feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from src.config import (
    FIRST_SEEN_COLUMN,
    GROUP_ID_COLUMN,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
)
from src.features.schema_v2 import (
    BASE_FEATURE_COLUMNS,
    BASE_FEATURE_SPECS,
    FEATURE_SCHEMA_VERSION,
    FeatureSchemaError,
    resolve_source_columns,
)

LABEL_CONFIDENCE_COLUMN = "label_confidence"
MAX_INT32_COUNT = 2_147_483_647
REQUIRED_METADATA_COLUMNS = (
    SAMPLE_ID_COLUMN,
    LABEL_COLUMN,
    SOURCE_ID_COLUMN,
    GROUP_ID_COLUMN,
    FIRST_SEEN_COLUMN,
    LABEL_CONFIDENCE_COLUMN,
)

DISALLOWED_CONTENT_COLUMN_TOKENS = (
    "raw_bytes",
    "file_bytes",
    "payload",
    "script_body",
    "javascript_body",
    "document_text",
    "raw_text",
    "sample_url",
    "download_url",
    "sample_path",
)


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    message: str
    column: Optional[str] = None
    row_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "row_count": self.row_count,
        }


class DatasetSchemaError(ValueError):
    """Raised when a feature table violates the canonical schema."""

    def __init__(self, issues: Iterable[SchemaIssue]):
        self.issues = list(issues)
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(summary)


class FeatureTableSchema:
    """Schema-v2 validator with explicit source-column canonicalization."""

    version = FEATURE_SCHEMA_VERSION
    feature_columns = BASE_FEATURE_COLUMNS
    metadata_columns = REQUIRED_METADATA_COLUMNS

    def expected_columns(self) -> tuple[str, ...]:
        return (*self.metadata_columns, *self.feature_columns)

    def canonicalize_source_frame(
        self,
        frame: pd.DataFrame,
        *,
        feature_overrides: Optional[Mapping[str, str]] = None,
        metadata_mapping: Optional[Mapping[str, str]] = None,
        keep_extra_metadata: bool = False,
    ) -> pd.DataFrame:
        """Rename an approved source explicitly into the canonical contract."""
        feature_mapping = resolve_source_columns(
            frame.columns, overrides=feature_overrides
        )
        rename = {source: target for target, source in feature_mapping.items()}
        metadata_map = dict(metadata_mapping or {})
        for canonical in self.metadata_columns:
            source = metadata_map.get(canonical, canonical)
            if source not in frame.columns:
                raise FeatureSchemaError(
                    f"Required metadata column {canonical!r} is absent "
                    f"(expected source field {source!r})."
                )
            rename[source] = canonical
        canonical = frame.rename(columns=rename)
        if keep_extra_metadata:
            selected = list(dict.fromkeys([*self.expected_columns(), *canonical.columns]))
            selected = [column for column in selected if column in canonical.columns]
        else:
            extras = sorted(set(canonical.columns) - set(self.expected_columns()))
            if extras:
                unsafe = [
                    column
                    for column in extras
                    if any(
                        token in column.lower()
                        for token in DISALLOWED_CONTENT_COLUMN_TOKENS
                    )
                ]
                if unsafe:
                    raise FeatureSchemaError(
                        f"Raw content/payload columns are forbidden: {unsafe}"
                    )
                raise FeatureSchemaError(
                    f"Unexpected source columns require an explicit schema decision: {extras}"
                )
            selected = list(self.expected_columns())
        return canonical.loc[:, selected].copy()

    def arrow_schema(self) -> pa.Schema:
        fields = [
            pa.field(SAMPLE_ID_COLUMN, pa.string(), nullable=False),
            pa.field(LABEL_COLUMN, pa.int8(), nullable=False),
            pa.field(SOURCE_ID_COLUMN, pa.string(), nullable=False),
            pa.field(GROUP_ID_COLUMN, pa.string(), nullable=False),
            pa.field(FIRST_SEEN_COLUMN, pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field(LABEL_CONFIDENCE_COLUMN, pa.float32(), nullable=False),
        ]
        for spec in BASE_FEATURE_SPECS:
            if spec.kind == "boolean":
                arrow_type = pa.int8()
            elif spec.kind == "count":
                arrow_type = pa.int32()
            else:
                arrow_type = pa.float32()
            fields.append(pa.field(spec.name, arrow_type, nullable=spec.allow_missing))
        return pa.schema(fields, metadata={b"feature_schema_version": self.version.encode()})

    def validate_columns(
        self,
        columns: Iterable[str],
        *,
        allow_extra: bool = False,
    ) -> list[SchemaIssue]:
        column_list = list(columns)
        column_set = set(column_list)
        issues: list[SchemaIssue] = []
        duplicate_columns = sorted(
            {column for column in column_list if column_list.count(column) > 1}
        )
        if duplicate_columns:
            issues.append(
                SchemaIssue(
                    "duplicate_columns",
                    f"Duplicate column names are forbidden: {duplicate_columns}",
                )
            )
        missing = sorted(set(self.expected_columns()) - column_set)
        if missing:
            issues.append(
                SchemaIssue("missing_columns", f"Required columns are absent: {missing}")
            )
        extras = sorted(column_set - set(self.expected_columns()))
        unsafe_extras = [
            column
            for column in extras
            if any(token in column.lower() for token in DISALLOWED_CONTENT_COLUMN_TOKENS)
        ]
        if unsafe_extras:
            issues.append(
                SchemaIssue(
                    "unsafe_content_columns",
                    f"Raw content/payload columns are forbidden: {unsafe_extras}",
                )
            )
        if extras and not allow_extra:
            issues.append(
                SchemaIssue("unexpected_columns", f"Unexpected columns: {extras}")
            )
        return issues

    def validate_frame(
        self,
        frame: pd.DataFrame,
        *,
        allow_extra: bool = False,
        raise_on_error: bool = True,
    ) -> list[SchemaIssue]:
        """Validate a canonical frame without silently coercing its values."""
        issues = self.validate_columns(frame.columns, allow_extra=allow_extra)
        if any(issue.code == "missing_columns" for issue in issues):
            if raise_on_error:
                raise DatasetSchemaError(issues)
            return issues

        label = frame[LABEL_COLUMN]
        if not pd.api.types.is_numeric_dtype(label):
            issues.append(
                SchemaIssue("label_type", "Class must be a numeric 0/1 column.", LABEL_COLUMN)
            )
        else:
            invalid_labels = ~label.isin([0, 1]) | label.isna()
            if invalid_labels.any():
                issues.append(
                    SchemaIssue(
                        "invalid_label",
                        "Class contains null, ambiguous, or non-binary values.",
                        LABEL_COLUMN,
                        int(invalid_labels.sum()),
                    )
                )

        for column in (SAMPLE_ID_COLUMN, SOURCE_ID_COLUMN, GROUP_ID_COLUMN):
            values = frame[column]
            invalid = values.isna() | values.astype(str).str.strip().eq("")
            if invalid.any():
                issues.append(
                    SchemaIssue(
                        "empty_identifier",
                        f"{column} contains null or blank identifiers.",
                        column,
                        int(invalid.sum()),
                    )
                )
            if column == SAMPLE_ID_COLUMN:
                looks_like_location = values.astype(str).str.contains(
                    r"(?:https?://|[\\/])", case=False, regex=True
                )
                if looks_like_location.any():
                    issues.append(
                        SchemaIssue(
                            "non_opaque_sample_id",
                            "sample_id must be opaque and must not expose a URL or path.",
                            column,
                            int(looks_like_location.sum()),
                        )
                    )

        parsed_time = pd.to_datetime(frame[FIRST_SEEN_COLUMN], errors="coerce", utc=True)
        invalid_time = parsed_time.isna()
        if invalid_time.any():
            issues.append(
                SchemaIssue(
                    "invalid_first_seen",
                    "first_seen_at contains missing or unparseable timestamps.",
                    FIRST_SEEN_COLUMN,
                    int(invalid_time.sum()),
                )
            )

        confidence = frame[LABEL_CONFIDENCE_COLUMN]
        if not pd.api.types.is_numeric_dtype(confidence):
            issues.append(
                SchemaIssue(
                    "label_confidence_type",
                    "label_confidence must be numeric in [0, 1].",
                    LABEL_CONFIDENCE_COLUMN,
                )
            )
        else:
            invalid_confidence = confidence.isna() | ~confidence.between(0.0, 1.0)
            if invalid_confidence.any():
                issues.append(
                    SchemaIssue(
                        "invalid_label_confidence",
                        "label_confidence must be finite and in [0, 1].",
                        LABEL_CONFIDENCE_COLUMN,
                        int(invalid_confidence.sum()),
                    )
                )

        for spec in BASE_FEATURE_SPECS:
            series = frame[spec.name]
            if not pd.api.types.is_numeric_dtype(series):
                issues.append(
                    SchemaIssue(
                        "feature_type",
                        f"{spec.name} must be numeric; silent coercion is forbidden.",
                        spec.name,
                    )
                )
                continue
            numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
            infinite = np.isinf(numeric)
            if infinite.any():
                issues.append(
                    SchemaIssue(
                        "non_finite_feature",
                        f"{spec.name} contains infinite values.",
                        spec.name,
                        int(infinite.sum()),
                    )
                )
            if not spec.allow_missing:
                missing = np.isnan(numeric)
                if missing.any():
                    issues.append(
                        SchemaIssue(
                            "missing_required_feature",
                            f"{spec.name} may not be missing.",
                            spec.name,
                            int(missing.sum()),
                        )
                    )
            finite_values = numeric[np.isfinite(numeric)]
            if spec.kind in {"count", "boolean"}:
                non_integral = ~np.isclose(
                    finite_values, np.rint(finite_values), rtol=0, atol=1e-6
                )
                if non_integral.any():
                    issues.append(
                        SchemaIssue(
                            f"non_integral_{spec.kind}",
                            f"{spec.name} must contain integral {spec.kind} values.",
                            spec.name,
                            int(non_integral.sum()),
                        )
                    )
            if spec.minimum is not None and (finite_values < spec.minimum).any():
                issues.append(
                    SchemaIssue(
                        "feature_below_minimum",
                        f"{spec.name} has values below {spec.minimum}.",
                        spec.name,
                        int((finite_values < spec.minimum).sum()),
                    )
                )
            if spec.maximum is not None and (finite_values > spec.maximum).any():
                issues.append(
                    SchemaIssue(
                        "feature_above_maximum",
                        f"{spec.name} has values above {spec.maximum}.",
                        spec.name,
                        int((finite_values > spec.maximum).sum()),
                    )
                )
            if spec.kind == "count" and (finite_values > MAX_INT32_COUNT).any():
                issues.append(
                    SchemaIssue(
                        "count_overflow",
                        f"{spec.name} exceeds the int32 storage contract.",
                        spec.name,
                        int((finite_values > MAX_INT32_COUNT).sum()),
                    )
                )

        if issues and raise_on_error:
            raise DatasetSchemaError(issues)
        return issues
