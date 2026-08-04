"""Deterministic, deployment-safe engineered features with explicit lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.schema_v2 import BASE_FEATURE_COLUMNS, BASE_FEATURE_SPECS


@dataclass(frozen=True)
class EngineeredFeatureSpec:
    name: str
    family: str
    unit: str
    minimum: float | None
    maximum: float | None
    formula: str
    rationale: str
    lineage: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / (denominator.astype(float).clip(lower=0) + 1.0)


class EngineeredFeatureBuilder:
    """Pure feature transform; it learns no values and never reads the label."""

    OPTIONAL_STATUS_COLUMNS = (
        "parse_failure", "recovery_mode", "invalid_eof", "extraction_timeout",
        "file_too_large", "extraction_limit_reached",
    )

    def __init__(self, *, optional_status_columns: Sequence[str] = ()) -> None:
        unknown = sorted(set(optional_status_columns) - set(self.OPTIONAL_STATUS_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown optional extraction statuses: {unknown}")
        self.optional_status_columns = tuple(optional_status_columns)
        self._specs: list[EngineeredFeatureSpec] = []

    def _add(
        self,
        output: pd.DataFrame,
        name: str,
        values: pd.Series | np.ndarray,
        *,
        family: str,
        unit: str,
        formula: str,
        rationale: str,
        lineage: Sequence[str],
        minimum: float | None = 0.0,
        maximum: float | None = None,
    ) -> None:
        output[name] = np.asarray(values, dtype=np.float32)
        self._specs.append(
            EngineeredFeatureSpec(
                name, family, unit, minimum, maximum, formula, rationale,
                tuple(lineage),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(BASE_FEATURE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Base schema-v2 features are missing: {missing}")
        base = frame.loc[:, BASE_FEATURE_COLUMNS].astype(np.float64)
        output = base.astype(np.float32).copy()
        self._specs = []

        for spec in BASE_FEATURE_SPECS:
            if spec.kind == "boolean":
                continue
            name = f"log1p__{spec.name}"
            self._add(
                output, name, np.log1p(base[spec.name].clip(lower=0)),
                family="scale_stable", unit="log1p",
                formula=f"log(1 + max({spec.name}, 0))",
                rationale="Reduces leverage from heavy-tailed counts and byte sizes.",
                lineage=(spec.name,),
            )

        action_columns = (
            "js_count", "javascript_count", "openaction_count", "action_count",
            "aa_count", "launch_count", "uri_count", "submitform_count",
        )
        action_total = base.loc[:, action_columns].sum(axis=1)
        script_total = base["js_count"] + base["javascript_count"]
        density_specs = (
            ("active_actions_per_object", action_total, base["obj_count_total"], action_columns + ("obj_count_total",), "Active action tokens / (objects + 1)."),
            ("javascript_per_object", script_total, base["obj_count_total"], ("js_count", "javascript_count", "obj_count_total"), "JavaScript tokens / (objects + 1)."),
            ("javascript_per_kib", script_total * 1024.0, base["pdf_size"], ("js_count", "javascript_count", "pdf_size"), "JavaScript tokens per KiB using a +1 byte denominator."),
            ("streams_per_object", base["stream_count"], base["obj_count_total"], ("stream_count", "obj_count_total"), "Streams / (objects + 1)."),
            ("embedded_files_per_page", base["embedded_file_count"], base["page_count"], ("embedded_file_count", "page_count"), "Embedded files / (pages + 1)."),
            ("filters_per_stream", base["filter_count"], base["stream_count"], ("filter_count", "stream_count"), "Filters / (streams + 1)."),
            ("obfuscations_per_object", base["obfuscation_count"], base["obj_count_total"], ("obfuscation_count", "obj_count_total"), "Name escapes / (objects + 1)."),
            ("images_per_page", base["image_count"], base["page_count"], ("image_count", "page_count"), "Images / (pages + 1)."),
            ("fonts_per_page", base["font_obj_count"], base["page_count"], ("font_obj_count", "page_count"), "Fonts / (pages + 1)."),
            ("text_richness_per_page", base["has_text"], base["page_count"], ("has_text", "page_count"), "Text-presence / (pages + 1)."),
        )
        for name, numerator, denominator, lineage, formula in density_specs:
            self._add(
                output, name, _ratio(numerator, denominator), family="density",
                unit="ratio", formula=formula,
                rationale="Normalizes security-relevant counts by document scale.",
                lineage=lineage,
            )

        gap_specs = (
            ("object_end_gap", "obj_count", "endobj_count"),
            ("stream_end_gap", "stream_count", "endstream_count"),
            ("xref_startxref_gap", "xref_count", "startxref_count"),
            ("xref_trailer_gap", "xref_count", "trailer_count"),
            ("scanner_parser_object_gap", "obj_count", "obj_count_total"),
        )
        for prefix, left, right in gap_specs:
            gap = (base[left] - base[right]).abs()
            self._add(
                output, f"abs__{prefix}", gap, family="structural_consistency",
                unit="occurrences", formula=f"abs({left} - {right})",
                rationale="Highlights disagreement between paired structural views.",
                lineage=(left, right),
            )
            self._add(
                output, f"normalized__{prefix}", gap / (base[left] + base[right] + 1.0),
                family="structural_consistency", unit="ratio",
                formula=f"abs({left} - {right}) / ({left} + {right} + 1)",
                rationale="Makes structural disagreement comparable across file sizes.",
                lineage=(left, right), maximum=1.0,
            )

        def presence(name: str) -> pd.Series:
            return base[name].gt(0).astype(float)
        interactions = (
            ("openaction_x_javascript", presence("openaction_count") * script_total.gt(0), ("openaction_count", "js_count", "javascript_count")),
            ("aa_x_javascript", presence("aa_count") * script_total.gt(0), ("aa_count", "js_count", "javascript_count")),
            ("launch_x_embedded", presence("launch_count") * presence("embedded_file_count"), ("launch_count", "embedded_file_count")),
            ("xfa_acroform_x_script", ((base["xfa_count"] + base["acroform_count"]) > 0) * script_total.gt(0), ("xfa_count", "acroform_count", "js_count", "javascript_count")),
            ("filters_x_obfuscation", presence("filter_count") * presence("obfuscation_count"), ("filter_count", "obfuscation_count")),
            ("encryption_x_active_content", base["is_encrypted"].gt(0) * action_total.gt(0), ("is_encrypted", *action_columns)),
        )
        for name, values, lineage in interactions:
            self._add(
                output, name, values, family="interaction", unit="boolean",
                formula="1 when every named risk condition is present, else 0.",
                rationale="Represents combinations that are more informative than isolated tokens.",
                lineage=lineage, maximum=1.0,
            )

        self._add(
            output, "object_stream_density", _ratio(base["objstm_count"], base["obj_count_total"]),
            family="complexity", unit="ratio",
            formula="objstm_count / (obj_count_total + 1)",
            rationale="Measures the share of objects hidden behind object streams.",
            lineage=("objstm_count", "obj_count_total"),
        )
        self._add(
            output, "metadata_file_ratio", _ratio(base["metadata_size"], base["pdf_size"]),
            family="complexity", unit="ratio",
            formula="metadata_size / (pdf_size + 1)",
            rationale="Detects unusually metadata-heavy documents.",
            lineage=("metadata_size", "pdf_size"),
        )
        embedded_bytes = base["embedded_file_count"] * base["avg_embedded_media_size"]
        self._add(
            output, "embedded_media_share", _ratio(embedded_bytes, base["pdf_size"]),
            family="complexity", unit="ratio",
            formula="embedded_file_count * avg_embedded_media_size / (pdf_size + 1)",
            rationale="Estimates how much of a file is accounted for by embedded media.",
            lineage=("embedded_file_count", "avg_embedded_media_size", "pdf_size"),
        )
        entropy_columns = list(action_columns) + [
            "stream_count", "objstm_count", "filter_count", "embedded_file_count"
        ]
        matrix = base.loc[:, entropy_columns].fillna(0).clip(lower=0).to_numpy(float)
        totals = matrix.sum(axis=1)
        probabilities = np.divide(
            matrix, totals[:, None], out=np.zeros_like(matrix), where=totals[:, None] > 0
        )
        entropy = -np.sum(
            np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-30), 0),
            axis=1,
        ) / np.log(len(entropy_columns))
        concentration = np.divide(
            matrix.max(axis=1), totals, out=np.zeros_like(totals), where=totals > 0
        )
        self._add(
            output, "structural_token_entropy", entropy, family="complexity",
            unit="normalized_entropy", formula="Normalized Shannon entropy over selected structural counts.",
            rationale="Separates concentrated single-mechanism files from structurally diverse ones.",
            lineage=entropy_columns, maximum=1.0,
        )
        self._add(
            output, "structural_count_concentration", concentration, family="complexity",
            unit="ratio", formula="Maximum selected count / sum of selected counts.",
            rationale="Complements entropy with a dominant-token measure.",
            lineage=entropy_columns, maximum=1.0,
        )

        parser_disagreement = (base["obj_count"] - base["obj_count_total"]).abs() / (
            base["obj_count"] + base["obj_count_total"] + 1.0
        )
        health = (
            ("truncated_structure", ((base["obj_count"] != base["endobj_count"]) | (base["stream_count"] != base["endstream_count"])).astype(float), ("obj_count", "endobj_count", "stream_count", "endstream_count"), "Paired start/end markers disagree."),
            ("invalid_header", (1.0 - base["header_valid"].clip(0, 1)), ("header_valid",), "Header-valid is false."),
            ("invalid_xref", ((base["startxref_count"] == 0) | ((base["xref_count"] == 0) & (base["objstm_count"] == 0))).astype(float), ("startxref_count", "xref_count", "objstm_count"), "No supported xref representation is visible."),
            ("parser_disagreement", parser_disagreement, ("obj_count", "obj_count_total"), "Normalized parser/scanner object disagreement."),
        )
        for name, values, lineage, formula in health:
            self._add(
                output, name, values, family="parser_health", unit="boolean" if name != "parser_disagreement" else "ratio",
                formula=formula,
                rationale="Supports abstention and quality-aware modeling; it is not a malicious label.",
                lineage=lineage, maximum=1.0,
            )
        for name in self.optional_status_columns:
            if name not in frame.columns:
                raise ValueError(
                    f"Serialized pipeline requires status column {name!r}; it cannot be defaulted to zero."
                )
            self._add(
                output, name, frame[name], family="parser_health", unit="boolean",
                formula=f"Typed extractor status `{name}`.",
                rationale="Makes extraction limitations observable and available for abstention.",
                lineage=(name,), maximum=1.0,
            )
        return output

    def feature_specs(self) -> tuple[EngineeredFeatureSpec, ...]:
        if not self._specs:
            # Build a single dummy row to materialize the deterministic catalog.
            dummy = pd.DataFrame([{name: 0.0 for name in BASE_FEATURE_COLUMNS}])
            for name in self.optional_status_columns:
                dummy[name] = 0.0
            self.transform(dummy)
        return tuple(self._specs)

    def lineage(self) -> Mapping[str, tuple[str, ...]]:
        return {spec.name: spec.lineage for spec in self.feature_specs()}
