"""Two-pass validation, quarantine, deduplication, and data-card reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    DATA_REPORTS_DIR,
    FEATURE_COLUMNS,
    FIRST_SEEN_COLUMN,
    GROUP_ID_COLUMN,
    LABEL_COLUMN,
    PROCESSED_DATA_DIR,
    SAMPLE_ID_COLUMN,
    SOURCE_ID_COLUMN,
)
from src.data.audit import source_only_diagnostic
from src.data.deduplicate import StreamingDeduplicator
from src.data.loader import iter_dataset_batches, write_parquet
from src.data.manifest import DataSourceSpec, load_source_registry
from src.data.downloader import validate_source_approval, verify_source_file
from src.data.schema import (
    FeatureTableSchema,
    LABEL_CONFIDENCE_COLUMN,
    MAX_INT32_COUNT,
)
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.schema_v2 import BASE_FEATURE_SPECS
from src.utils.atomic import atomic_write_json


class DatasetGateError(RuntimeError):
    """Raised after reports are written when scientific data gates fail."""


def _safe_reported_path(path_value: str, root: Path) -> Path:
    path = Path(path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetGateError(f"Reported output escapes its immutable layer: {path}") from exc
    return path


@dataclass(frozen=True)
class DatasetQualityReport:
    status: str
    source_id: str
    input_path: str
    input_verification: dict[str, Any]
    clean_output: str
    quarantine_output: str
    schema_version: str
    row_flow: dict[str, int]
    quarantine_reasons: dict[str, int]
    deduplication: dict[str, int]
    label_counts: dict[str, int]
    benign_prevalence: float | None
    feature_quality: dict[str, dict[str, Any]]
    source_time_coverage: dict[str, Any]
    source_only_diagnostic: dict[str, Any]
    label_noise_audit: dict[str, Any]
    gates: dict[str, Any]
    output_files: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _FeatureQualityAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {
            name: {
                "non_missing": 0,
                "missing": 0,
                "minimum": None,
                "maximum": None,
                "first": None,
                "constant": True,
            }
            for name in FEATURE_COLUMNS
        }

    def update(self, frame: pd.DataFrame) -> None:
        for name in FEATURE_COLUMNS:
            series = frame[name]
            state = self.values[name]
            state["missing"] += int(series.isna().sum())
            finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
            state["non_missing"] += int(len(finite))
            if finite.empty:
                continue
            low, high = float(finite.min()), float(finite.max())
            state["minimum"] = low if state["minimum"] is None else min(state["minimum"], low)
            state["maximum"] = high if state["maximum"] is None else max(state["maximum"], high)
            if state["first"] is None:
                state["first"] = float(finite.iloc[0])
            if not np.allclose(finite.to_numpy(dtype=float), state["first"], rtol=0, atol=0):
                state["constant"] = False

    def report(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, state in self.values.items():
            total = state["non_missing"] + state["missing"]
            result[name] = {
                "missing_rows": state["missing"],
                "missing_fraction": state["missing"] / total if total else None,
                "minimum": state["minimum"],
                "maximum": state["maximum"],
                "constant": bool(state["constant"] and state["non_missing"] > 0),
            }
        return result


def _append_reason(reasons: pd.Series, mask: pd.Series | np.ndarray, code: str) -> None:
    mask_array = np.asarray(mask, dtype=bool)
    if not mask_array.any():
        return
    current = reasons.loc[mask_array]
    reasons.loc[mask_array] = np.where(current.eq(""), code, current + ";" + code)


def row_validation_reasons(frame: pd.DataFrame, schema: FeatureTableSchema) -> pd.Series:
    """Return stable reason codes without silently coercing source values."""
    reasons = pd.Series("", index=frame.index, dtype="string")
    column_issues = schema.validate_columns(frame.columns)
    if column_issues:
        code = ";".join(sorted({issue.code for issue in column_issues}))
        reasons[:] = code
        return reasons

    label = frame[LABEL_COLUMN]
    if not pd.api.types.is_numeric_dtype(label):
        _append_reason(reasons, np.ones(len(frame), dtype=bool), "label_type")
    else:
        _append_reason(reasons, label.isna() | ~label.isin([0, 1]), "invalid_label")

    for name in (SAMPLE_ID_COLUMN, SOURCE_ID_COLUMN, GROUP_ID_COLUMN):
        value = frame[name]
        invalid = value.isna() | value.astype(str).str.strip().eq("")
        _append_reason(reasons, invalid, f"empty_{name}")
    sample_ids = frame[SAMPLE_ID_COLUMN].astype(str)
    _append_reason(
        reasons,
        sample_ids.str.contains(r"(?:https?://|[\\/])", case=False, regex=True),
        "non_opaque_sample_id",
    )
    parsed_time = pd.to_datetime(frame[FIRST_SEEN_COLUMN], errors="coerce", utc=True)
    _append_reason(reasons, parsed_time.isna(), "invalid_first_seen_at")

    confidence = frame[LABEL_CONFIDENCE_COLUMN]
    if not pd.api.types.is_numeric_dtype(confidence):
        _append_reason(reasons, np.ones(len(frame), dtype=bool), "label_confidence_type")
    else:
        values = confidence.to_numpy(dtype=float, na_value=np.nan)
        _append_reason(
            reasons, ~np.isfinite(values) | (values < 0) | (values > 1),
            "invalid_label_confidence",
        )

    for spec in BASE_FEATURE_SPECS:
        series = frame[spec.name]
        if not pd.api.types.is_numeric_dtype(series):
            _append_reason(reasons, np.ones(len(frame), dtype=bool), f"feature_type:{spec.name}")
            continue
        values = series.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        _append_reason(reasons, np.isinf(values), f"non_finite:{spec.name}")
        if not spec.allow_missing:
            _append_reason(reasons, np.isnan(values), f"missing_required:{spec.name}")
        if spec.minimum is not None:
            _append_reason(reasons, finite & (values < spec.minimum), f"below_minimum:{spec.name}")
        if spec.maximum is not None:
            _append_reason(reasons, finite & (values > spec.maximum), f"above_maximum:{spec.name}")
        if spec.kind in {"count", "boolean"}:
            _append_reason(
                reasons,
                finite & ~np.isclose(values, np.rint(values), rtol=0, atol=1e-6),
                f"non_integral_{spec.kind}:{spec.name}",
            )
        if spec.kind == "count":
            _append_reason(
                reasons,
                finite & (values > MAX_INT32_COUNT),
                f"count_overflow:{spec.name}",
            )
    return reasons


def _canonical_types(frame: pd.DataFrame) -> pd.DataFrame:
    typed = frame.copy()
    typed[FIRST_SEEN_COLUMN] = pd.to_datetime(typed[FIRST_SEEN_COLUMN], utc=True)
    typed[LABEL_COLUMN] = typed[LABEL_COLUMN].astype(np.int8)
    typed[LABEL_CONFIDENCE_COLUMN] = typed[LABEL_CONFIDENCE_COLUMN].astype(np.float32)
    typed.loc[:, FEATURE_COLUMNS] = typed.loc[:, FEATURE_COLUMNS].astype(np.float32)
    return typed


def _opaque_quarantine_id(value: object) -> str:
    return "q-" + hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _sanitized_quarantine(
    frame: pd.DataFrame,
    reasons: pd.Series | str,
    schema: FeatureTableSchema,
) -> pd.DataFrame:
    """Keep only canonical non-payload fields and hash all identifiers."""
    selected = [name for name in schema.expected_columns() if name in frame.columns]
    safe = frame.loc[:, selected].copy()
    for name in (SAMPLE_ID_COLUMN, SOURCE_ID_COLUMN, GROUP_ID_COLUMN):
        if name in safe:
            safe[name] = safe[name].map(_opaque_quarantine_id)
    if isinstance(reasons, str):
        safe["quarantine_reason"] = reasons
    else:
        safe["quarantine_reason"] = reasons.reindex(frame.index).astype(str).values
    return safe


def _layer_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.glob("*.parquet")):
        metadata = pq.ParquetFile(path).metadata
        files.append(
            {
                "path": str(path.resolve()),
                "rows": int(metadata.num_rows),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _write_markdown(report: DatasetQualityReport, path: Path) -> None:
    failed = [name for name, value in report.gates.items() if not value.get("passed", False)]
    lines = [
        "# Dataset quality report",
        "",
        f"Status: **{report.status}**",
        f"Source: `{report.source_id}`",
        f"Schema: `{report.schema_version}`",
        "",
        "## Row flow",
        "",
        *[f"- {name}: {value:,}" for name, value in report.row_flow.items()],
        "",
        "## Prevalence",
        "",
        f"- Benign: {report.label_counts.get('0', 0):,}",
        f"- Malicious: {report.label_counts.get('1', 0):,}",
        f"- Benign prevalence: {report.benign_prevalence!s}",
        "",
        "## Gate failures",
        "",
        *([f"- {name}: {report.gates[name]['actual']}" for name in failed] or ["- None"]),
        "",
        "## Source-shortcut diagnostic",
        "",
        f"- {report.source_only_diagnostic.get('interpretation')}",
        "",
        "## Label-noise audit",
        "",
        f"- Sample-ID conflict fraction: {report.label_noise_audit.get('sample_id_conflict_fraction')}",
        f"- Feature-row label disagreement fraction: {report.label_noise_audit.get('feature_hash_label_disagreement_fraction')}",
        f"- Sanitized rows selected for review: {report.label_noise_audit.get('manual_review_rows')}",
        "",
        "Raw files were not inspected or retrieved; this audit uses sanitized feature rows only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_feature_table(
    input_path: Path,
    *,
    source_id: str,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    feature_mapping: Mapping[str, str] | None = None,
    metadata_mapping: Mapping[str, str] | None = None,
    input_verification: Mapping[str, Any] | None = None,
    batch_size: int = 100_000,
    enforce_gates: bool = True,
) -> DatasetQualityReport:
    """Validate an immutable source table and write clean/quarantined Parquet."""
    source_path = Path(input_path)
    clean_root = Path(output_dir or PROCESSED_DATA_DIR / "schema_v2" / source_id)
    quarantine_root = clean_root.parent / f"{clean_root.name}_quarantine"
    reports = Path(report_dir or DATA_REPORTS_DIR)
    for root in (clean_root, quarantine_root):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"Immutable output layer already exists and is non-empty: {root}. "
                "Choose a new dataset version/output directory."
            )
    clean_root.mkdir(parents=True, exist_ok=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    schema = FeatureTableSchema()
    reason_counts: Counter[str] = Counter()
    received = schema_valid_rows = label_valid = fully_valid_rows = 0
    quality = _FeatureQualityAccumulator()

    with tempfile.TemporaryDirectory(prefix="pdf-feature-qc-") as temporary:
        temporary_root = Path(temporary)
        valid_root = temporary_root / "valid"
        valid_root.mkdir()
        database = temporary_root / "dedup.sqlite"
        deduplicator = StreamingDeduplicator(database)
        valid_part = quarantine_part = 0

        for raw in iter_dataset_batches(source_path, batch_size=batch_size):
            received += len(raw)
            try:
                canonical = schema.canonicalize_source_frame(
                    raw,
                    feature_overrides=feature_mapping,
                    metadata_mapping=metadata_mapping,
                )
                reasons = row_validation_reasons(canonical, schema)
            except Exception as exc:
                reason = f"fatal_schema_mapping:{type(exc).__name__}"
                reason_counts[reason] += len(raw)
                quarantined = _sanitized_quarantine(raw, reason, schema)
                quarantined.to_parquet(
                    quarantine_root / f"part-{quarantine_part:06d}.parquet",
                    index=False,
                )
                quarantine_part += 1
                continue

            valid_mask = reasons.eq("")
            schema_mask = reasons.map(
                lambda combined: not any(
                    code
                    for code in str(combined).split(";")
                    if code and not (
                        code.startswith("label_")
                        or code.startswith("invalid_label")
                    )
                )
            )
            label_mask = (
                canonical[LABEL_COLUMN].isin([0, 1])
                if pd.api.types.is_numeric_dtype(canonical[LABEL_COLUMN])
                else pd.Series(False, index=canonical.index)
            )
            schema_valid_rows += int(schema_mask.sum())
            label_valid += int((schema_mask & label_mask).sum())
            valid = _canonical_types(canonical.loc[valid_mask].copy())
            fully_valid_rows += len(valid)
            if not valid.empty:
                quality.update(valid)
                write_parquet(valid, valid_root / f"part-{valid_part:06d}.parquet")
                deduplicator.observe_batch(valid)
                valid_part += 1
            invalid = canonical.loc[~valid_mask].copy()
            if not invalid.empty:
                invalid = _sanitized_quarantine(
                    invalid, reasons.loc[~valid_mask], schema
                )
                invalid.to_parquet(
                    quarantine_root / f"part-{quarantine_part:06d}.parquet", index=False
                )
                quarantine_part += 1
                for combined in invalid["quarantine_reason"]:
                    for code in str(combined).split(";"):
                        reason_counts[code] += 1

        dedup_stats = deduplicator.stats()
        clean_rows = 0
        label_counts: Counter[int] = Counter()
        source_counts: Counter[str] = Counter()
        source_label_counts: Counter[tuple[str, int]] = Counter()
        first_seen_min = first_seen_max = None
        diagnostic_frame = pd.DataFrame(
            columns=[SOURCE_ID_COLUMN, FIRST_SEEN_COLUMN, LABEL_COLUMN]
        )
        review_candidates = pd.DataFrame(
            columns=[*schema.expected_columns(), "review_reason"]
        )
        clean_part = 0
        valid_batches = (
            iter_dataset_batches(valid_root, batch_size=batch_size)
            if valid_part
            else ()
        )
        for batch in valid_batches:
            reasons = deduplicator.selection_reasons(batch)
            keep = reasons.eq("keep")
            clean = batch.loc[keep].copy()
            rejected = batch.loc[~keep].copy()
            if not rejected.empty:
                rejected = _sanitized_quarantine(
                    rejected, reasons.loc[~keep], schema
                )
                rejected.to_parquet(
                    quarantine_root / f"part-{quarantine_part:06d}.parquet", index=False
                )
                quarantine_part += 1
                reason_counts.update(rejected["quarantine_reason"].astype(str))
                conflicts = batch.loc[
                    ~keep & reasons.eq("conflicting_label_for_sample_id")
                ].copy()
                if not conflicts.empty:
                    conflicts["review_reason"] = "sample_id_label_conflict"
                    review_candidates = pd.concat(
                        [review_candidates, conflicts[[*schema.expected_columns(), "review_reason"]]],
                        ignore_index=True,
                    ).iloc[:1000]
            if clean.empty:
                continue
            write_parquet(clean, clean_root / f"part-{clean_part:06d}.parquet")
            clean_part += 1
            clean_rows += len(clean)
            label_counts.update(clean[LABEL_COLUMN].astype(int).tolist())
            source_counts.update(clean[SOURCE_ID_COLUMN].astype(str).tolist())
            source_label_counts.update(
                zip(clean[SOURCE_ID_COLUMN].astype(str), clean[LABEL_COLUMN].astype(int))
            )
            times = pd.to_datetime(clean[FIRST_SEEN_COLUMN], utc=True)
            batch_min, batch_max = times.min(), times.max()
            first_seen_min = batch_min if first_seen_min is None else min(first_seen_min, batch_min)
            first_seen_max = batch_max if first_seen_max is None else max(first_seen_max, batch_max)
            diagnostic_frame = pd.concat(
                [
                    diagnostic_frame,
                    clean[[SOURCE_ID_COLUMN, FIRST_SEEN_COLUMN, LABEL_COLUMN]],
                ],
                ignore_index=True,
            )
            if len(diagnostic_frame) > 250_000:
                diagnostic_frame = diagnostic_frame.sample(
                    n=250_000,
                    random_state=42 + clean_part,
                    ignore_index=True,
                )
            low_confidence = clean[clean[LABEL_CONFIDENCE_COLUMN] < 0.80].copy()
            if not low_confidence.empty and len(review_candidates) < 1000:
                low_confidence["review_reason"] = "low_label_confidence"
                review_candidates = pd.concat(
                    [
                        review_candidates,
                        low_confidence[[*schema.expected_columns(), "review_reason"]],
                    ],
                    ignore_index=True,
                ).iloc[:1000]
        deduplicator.close()

    benign_prevalence = label_counts[0] / clean_rows if clean_rows else None
    config = load_experiment_config()
    gates_cfg = config["acceptance_gates"]
    gate_values = {
        "minimum_input_rows": (received, int(gates_cfg["minimum_input_rows"]), ">="),
        "minimum_clean_rows": (clean_rows, int(gates_cfg["minimum_clean_rows"]), ">="),
        "minimum_benign_prevalence": (
            benign_prevalence if benign_prevalence is not None else 0.0,
            float(gates_cfg["minimum_benign_prevalence"]),
            ">=",
        ),
    }
    gates = {
        name: {
            "passed": bool(actual >= required),
            "actual": actual,
            "required": required,
            "operator": operator,
        }
        for name, (actual, required, operator) in gate_values.items()
    }
    shortcut = source_only_diagnostic(
        diagnostic_frame, random_seed=int(config["random_seed"])
    )
    review_path = reports / "sanitized_label_review.parquet"
    if not review_candidates.empty:
        review_candidates.to_parquet(review_path, index=False, compression="zstd")
    label_noise = {
        "sample_id_conflict_fraction": (
            dedup_stats.conflicting_rows / dedup_stats.observed_rows
            if dedup_stats.observed_rows else 0.0
        ),
        "feature_hash_label_disagreement_fraction": (
            dedup_stats.conflicting_feature_rows / dedup_stats.observed_rows
            if dedup_stats.observed_rows else 0.0
        ),
        "manual_review_rows": int(len(review_candidates)),
        "manual_review_path": str(review_path.resolve()) if review_path.exists() else None,
        "review_policy": "Sanitized feature rows only; no raw sample retrieval.",
    }
    coverage = {
        "source_count": len(source_counts),
        "source_rows": dict(sorted(source_counts.items())),
        "source_label_contingency": [
            {"source_id": source, "Class": label, "rows": rows}
            for (source, label), rows in sorted(source_label_counts.items())
        ],
        "first_seen_min_utc": first_seen_min.isoformat() if first_seen_min is not None else None,
        "first_seen_max_utc": first_seen_max.isoformat() if first_seen_max is not None else None,
    }
    passed = all(value["passed"] for value in gates.values())
    verified_input = dict(input_verification or {})
    if not verified_input:
        verified_input = {
            "path": str(source_path.resolve()),
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
    output_files = {
        "clean": _layer_files(clean_root),
        "quarantine": _layer_files(quarantine_root),
    }
    report = DatasetQualityReport(
        status="passed" if passed else "failed_acceptance_gates",
        source_id=source_id,
        input_path=str(source_path.resolve()),
        input_verification=verified_input,
        clean_output=str(clean_root.resolve()),
        quarantine_output=str(quarantine_root.resolve()),
        schema_version=schema.version,
        row_flow={
            "received": received,
            "schema_valid": schema_valid_rows,
            "label_valid": label_valid,
            "fully_valid": fully_valid_rows,
            "deduplicated": clean_rows,
            "split_eligible": clean_rows,
        },
        quarantine_reasons=dict(sorted(reason_counts.items())),
        deduplication=dedup_stats.to_dict(),
        label_counts={"0": label_counts[0], "1": label_counts[1]},
        benign_prevalence=benign_prevalence,
        feature_quality=quality.report(),
        source_time_coverage=coverage,
        source_only_diagnostic=shortcut.to_dict(),
        label_noise_audit=label_noise,
        gates=gates,
        output_files=output_files,
    )
    json_path = reports / "dataset_quality.json"
    atomic_write_json(json_path, report.to_dict())
    _write_markdown(report, reports / "dataset_quality.md")
    transformation_manifest = {
        "experiment": create_experiment_identity().to_dict(),
        "source_id": source_id,
        "feature_schema_version": schema.version,
        "input": verified_input,
        "outputs": output_files,
        "dataset_quality_path": str(json_path.resolve()),
        "dataset_quality_sha256": sha256_file(json_path),
        "immutable_layers": True,
        "raw_content_persisted": False,
    }
    atomic_write_json(reports / "transformation_manifest.json", transformation_manifest)
    if enforce_gates and not passed:
        failures = [name for name, value in gates.items() if not value["passed"]]
        raise DatasetGateError(
            f"Dataset quality report was written, but gates failed: {failures}"
        )
    return report


def validate_registered_source(source_id: str, *, enforce_gates: bool = True) -> DatasetQualityReport:
    registry = load_source_registry()
    source: DataSourceSpec = registry.require(source_id)
    try:
        validate_source_approval(source, registry.policy)
    except Exception as exc:
        raise DatasetGateError(str(exc)) from exc
    path = source.resolved_local_path()
    if path is None:
        raise DatasetGateError(f"Source {source_id!r} has no local feature-table path.")
    verification = verify_source_file(path, source)
    return validate_feature_table(
        path,
        source_id=source_id,
        feature_mapping=source.column_mapping,
        metadata_mapping=source.metadata_mapping,
        input_verification=verification,
        enforce_gates=enforce_gates,
    )


def verify_validated_dataset(
    report_path: Path | None = None,
    transformation_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Re-prove source/clean/quarantine hashes and every configured Phase 1 gate."""
    quality_path = Path(report_path or DATA_REPORTS_DIR / "dataset_quality.json")
    transformation_path = Path(
        transformation_manifest_path
        or DATA_REPORTS_DIR / "transformation_manifest.json"
    )
    if not quality_path.is_file() or not transformation_path.is_file():
        raise DatasetGateError("Dataset quality or transformation manifest is missing.")
    try:
        report = json.loads(quality_path.read_text(encoding="utf-8"))
        transformation = json.loads(transformation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetGateError("Dataset quality evidence is unreadable.") from exc
    if report.get("status") != "passed" or not all(
        gate.get("passed") for gate in report.get("gates", {}).values()
    ):
        raise DatasetGateError("Dataset quality gates are not all passed.")
    if transformation.get("dataset_quality_sha256") != sha256_file(quality_path):
        raise DatasetGateError("Transformation manifest does not bind dataset_quality.json.")
    if transformation.get("raw_content_persisted") is not False:
        raise DatasetGateError("Transformation manifest does not prove feature-only outputs.")
    input_record = report.get("input_verification", {})
    input_path = Path(input_record.get("path", ""))
    if not input_path.is_file() or sha256_file(input_path) != input_record.get("sha256"):
        raise DatasetGateError("Approved source bytes no longer match the quality report.")
    if transformation.get("input", {}).get("sha256") != input_record.get("sha256"):
        raise DatasetGateError("Transformation manifest source hash differs from quality report.")
    expected_roots = {
        "clean": Path(report["clean_output"]),
        "quarantine": Path(report["quarantine_output"]),
    }
    for layer, root in expected_roots.items():
        entries = report.get("output_files", {}).get(layer, [])
        reported_paths: set[Path] = set()
        total_rows = 0
        for entry in entries:
            path = _safe_reported_path(entry["path"], root)
            if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                raise DatasetGateError(f"{layer} output checksum mismatch: {path}")
            metadata = pq.ParquetFile(path).metadata
            if int(entry.get("rows", -1)) != metadata.num_rows:
                raise DatasetGateError(f"{layer} output row count mismatch: {path}")
            reported_paths.add(path)
            total_rows += metadata.num_rows
        actual_paths = {path.resolve() for path in root.glob("*.parquet")}
        if reported_paths != actual_paths:
            raise DatasetGateError(f"Unlisted or missing files in {layer} output layer.")
        if layer == "clean" and total_rows != int(report["row_flow"]["split_eligible"]):
            raise DatasetGateError("Clean-layer row count differs from row-flow evidence.")
    config = load_experiment_config()
    gates = config["acceptance_gates"]
    row_flow = report["row_flow"]
    if int(row_flow["received"]) < int(gates["minimum_input_rows"]):
        raise DatasetGateError("Verified input row count is below the active gate.")
    if int(row_flow["split_eligible"]) < int(gates["minimum_clean_rows"]):
        raise DatasetGateError("Verified clean row count is below the active gate.")
    if float(report["benign_prevalence"]) < float(gates["minimum_benign_prevalence"]):
        raise DatasetGateError("Verified benign prevalence is below the active gate.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument("--no-enforce-gates", action="store_true")
    args = parser.parse_args()
    report = validate_registered_source(
        args.source_id, enforce_gates=not args.no_enforce_gates
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
