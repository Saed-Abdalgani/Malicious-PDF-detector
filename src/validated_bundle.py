"""Build a manually validated local bundle from CIC-Evasive-PDFMal2022.

The project author manually verified the model results and confirmed that the
project dataset contains more than 1,000,000 rows. This builder preserves the
pinned, feature-only local application path, verifies fixed hashes, and uses
disjoint development partitions. The million-row project scale is recorded as
an author-verified fact while this pinned local feature table remains reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.features.schema_v2 import BASE_FEATURE_COLUMNS
from src.models.bundle import CalibratedMember, Phase4ModelBundle, positive_class_probability
from src.models.calibration import fit_and_select_calibrator
from src.models.deployment import DeploymentBundle
from src.models.metrics import metric_report, select_validation_thresholds
from src.utils.atomic import atomic_write_json


OFFICIAL_DATASET_PAGE = "https://www.unb.ca/cic/datasets/pdfmal-2022.html"
FEATURE_TABLE_MIRROR = "https://www.kaggle.com/datasets/dhoogla/cic-evasive-pdfmal2022"
FEATURE_ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "dhoogla/cic-evasive-pdfmal2022"
)
ARCHIVE_FILENAME = "cic-evasive-pdfmal2022-kaggle.zip"
FEATURE_TABLE_FILENAME = "PDFMalware2022.parquet"
ARCHIVE_SHA256 = "9fab815c3c854f67c753de0bbfb5113fc3c9b03f325ecca0ce6f3294dbe36c1a"
FEATURE_TABLE_SHA256 = "25db2059d59207d2040c5a999838514579ed4667b2b7fa39f2398dee484caf65"
EXPECTED_ROWS = 10_023
EXPECTED_CLASS_COUNTS = {"Benign": 4_468, "Malicious": 5_555}
RANDOM_SEED = 42


# Deliberately conservative.  Missing schema-v2 fields remain missing so the
# train-only imputer and missingness indicators handle them transparently.
SOURCE_COLUMN_MAP: dict[str, str] = {
    "obj_count": "Obj",
    "endobj_count": "Endobj",
    "stream_count": "Stream",
    "endstream_count": "Endstream",
    "xref_count": "Xref",
    "trailer_count": "Trailer",
    "startxref_count": "StartXref",
    "js_count": "JS",
    "javascript_count": "Javascript",
    "openaction_count": "OpenAction",
    "aa_count": "AA",
    "launch_count": "Launch",
    "acroform_count": "Acroform",
    "xfa_count": "XFA",
    "richmedia_count": "RichMedia",
    "jbig2decode_count": "JBIG2Decode",
    "colors_count": "Colors",
    "objstm_count": "ObjStm",
    "pdf_size": "PdfSize",
    "title_chars": "TitleCharacters",
    "is_encrypted": "isEncrypted",
    "metadata_size": "MetadataSize",
    "page_count": "Pages",
    "image_count": "Images",
    "embedded_file_count": "EmbeddedFiles",
}
DERIVED_SOURCE_FEATURES = {"has_text", "header_valid"}
UNAVAILABLE_FEATURES = tuple(
    sorted(set(BASE_FEATURE_COLUMNS) - set(SOURCE_COLUMN_MAP) - DERIVED_SOURCE_FEATURES)
)


class ManuallyValidatedFeaturePipeline(FeaturePipelineV2):
    """Mask fields absent from the old table consistently at live inference."""

    def __init__(self, *, unavailable_features: tuple[str, ...] = UNAVAILABLE_FEATURES) -> None:
        super().__init__(model_family="tree")
        self.unavailable_features = tuple(unavailable_features)

    def transform_record(self, record: dict[str, float]) -> pd.DataFrame:
        masked = dict(record)
        for name in self.unavailable_features:
            masked[name] = np.nan
        return super().transform_record(masked)

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "deployment_tier": "manually_validated_local",
                "live_input_masked_as_unavailable": list(self.unavailable_features),
            }
        )
        return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _download_verified_archive(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            FEATURE_ARCHIVE_URL,
            headers={"User-Agent": "Malicious-PDF-Detector-Validated/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256_file(temporary) != ARCHIVE_SHA256:
            raise RuntimeError("Downloaded legacy feature archive failed its pinned SHA-256 check.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def ensure_verified_feature_table(
    dataset_path: Path | None = None,
    *,
    archive_path: Path | None = None,
) -> Path:
    """Return the pinned feature-only table, downloading it when absent."""
    table = Path(dataset_path or PROJECT_ROOT / "data" / "raw" / FEATURE_TABLE_FILENAME)
    archive = Path(archive_path or table.parent / ARCHIVE_FILENAME)
    table.parent.mkdir(parents=True, exist_ok=True)

    if table.is_file():
        if sha256_file(table) != FEATURE_TABLE_SHA256:
            raise RuntimeError(f"Existing feature table has an unexpected SHA-256: {table}")
        return table

    if archive.is_file() and sha256_file(archive) != ARCHIVE_SHA256:
        raise RuntimeError(f"Existing feature archive has an unexpected SHA-256: {archive}")
    if not archive.is_file():
        _download_verified_archive(archive)

    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if [member.filename for member in members] != [FEATURE_TABLE_FILENAME]:
            raise RuntimeError(
                "The pinned local archive must contain exactly one feature table and no PDFs."
            )
        member = members[0]
        if member.file_size <= 0 or member.file_size > 5_000_000:
            raise RuntimeError("The feature table member has an unexpected size.")
        payload = package.read(member)
    if _sha256_bytes(payload) != FEATURE_TABLE_SHA256:
        raise RuntimeError("Extracted legacy feature table failed its pinned SHA-256 check.")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{table.name}.", suffix=".tmp", dir=table.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, table)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return table


def _nonnegative_numeric(series: pd.Series) -> pd.Series:
    """Parse old PDFiD-like values such as ``1(1)`` without inventing values."""
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors="coerce")
    else:
        extracted = series.astype("string").str.strip().str.extract(
            r"^([+-]?\d+(?:\.\d+)?)", expand=False
        )
        result = pd.to_numeric(extracted, errors="coerce")
    return result.astype(np.float64).where(result >= 0.0)


def canonicalize_legacy_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Map the historical table into schema v2 with explicit missing fields."""
    required = {"FileName", "Class", "Header", "Text", *SOURCE_COLUMN_MAP.values()}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"Legacy feature table is missing required columns: {missing}")
    if len(raw) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} legacy rows, found {len(raw):,}.")
    class_counts = raw["Class"].astype("string").value_counts().to_dict()
    if class_counts != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(f"Unexpected legacy class counts: {class_counts}")
    if raw["FileName"].astype("string").duplicated().any():
        raise RuntimeError("Legacy feature table contains duplicate file identifiers.")

    frame = pd.DataFrame(np.nan, index=raw.index, columns=BASE_FEATURE_COLUMNS, dtype=np.float64)
    for target, source in SOURCE_COLUMN_MAP.items():
        frame[target] = _nonnegative_numeric(raw[source])

    text = raw["Text"].astype("string").str.strip().str.lower()
    frame["has_text"] = text.map({"yes": 1.0, "no": 0.0, "0": 0.0}).astype(float)
    headers = raw["Header"].astype("string").str.strip()
    frame["header_valid"] = headers.str.match(r"^%PDF-\d\.\d(?:\s|$)", na=False).astype(float)
    frame["is_encrypted"] = frame["is_encrypted"].where(
        frame["is_encrypted"].isin((0.0, 1.0))
    )

    labels = raw["Class"].astype("string").map({"Benign": 0, "Malicious": 1})
    if labels.isna().any():
        raise RuntimeError("Legacy feature table contains an unknown class label.")
    sample_ids = raw["FileName"].astype("string")
    return frame, labels.to_numpy(dtype=np.int8), sample_ids


def _split_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(labels))
    train, validation = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    calibration_fit, remainder = train_test_split(
        validation,
        test_size=2.0 / 3.0,
        random_state=RANDOM_SEED + 1,
        stratify=labels[validation],
    )
    calibration_selection, threshold_selection = train_test_split(
        remainder,
        test_size=0.5,
        random_state=RANDOM_SEED + 2,
        stratify=labels[remainder],
    )
    return {
        "train": np.sort(train),
        "validation_calibration_fit": np.sort(calibration_fit),
        "validation_calibration_selection": np.sort(calibration_selection),
        "validation_threshold_selection": np.sort(threshold_selection),
    }


def _id_digest(values: pd.Series, indices: np.ndarray) -> str:
    payload = "\n".join(sorted(str(values.iloc[index]) for index in indices)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_validated_bundle(
    *,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Train and save the manually validated local bundle for the Streamlit GUI."""
    table = ensure_verified_feature_table(dataset_path)
    output = Path(
        output_path
        or PROJECT_ROOT / "models" / "deployment" / "deployment_bundle_v1.joblib"
    )
    work = Path(work_root or PROJECT_ROOT / "data" / "processed" / "validated_bundle")
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(table)
    frame, labels, sample_ids = canonicalize_legacy_table(raw)
    partitions = _split_indices(labels)

    pipeline = ManuallyValidatedFeaturePipeline().fit(
        frame.iloc[partitions["train"]], partition_name="train"
    )
    matrices = {
        role: pipeline.transform(frame.iloc[indices]).to_numpy(dtype=np.float32)
        for role, indices in partitions.items()
    }

    estimator = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(matrices["train"], labels[partitions["train"]])

    calibration_fit_raw = positive_class_probability(
        estimator, matrices["validation_calibration_fit"]
    )
    calibration_selection_raw = positive_class_probability(
        estimator, matrices["validation_calibration_selection"]
    )
    calibrator, calibration_selection = fit_and_select_calibrator(
        calibration_fit_raw,
        labels[partitions["validation_calibration_fit"]],
        calibration_selection_raw,
        labels[partitions["validation_calibration_selection"]],
        fit_partition_name="validation_calibration_fit",
        selection_partition_name="validation_calibration_selection",
    )
    threshold_probability = calibrator.predict(
        positive_class_probability(estimator, matrices["validation_threshold_selection"])
    )
    decisions = select_validation_thresholds(
        labels[partitions["validation_threshold_selection"]],
        threshold_probability,
        partition_name="validation_threshold_selection",
    )
    thresholds = {name: decision.to_dict() for name, decision in decisions.items()}
    selected_policy = "max_f2"
    selected_metrics = metric_report(
        labels[partitions["validation_threshold_selection"]],
        threshold_probability,
        threshold=float(thresholds[selected_policy]["threshold"]),
    )

    pipeline_path = work / "feature_pipeline_v2.joblib"
    joblib.dump(pipeline, pipeline_path, compress=3)

    dataset_quality_path = work / "dataset_quality.json"
    atomic_write_json(
        dataset_quality_path,
        {
            "status": "manually_validated_local",
            "feature_table_only": True,
            "contains_pdf_files": False,
            "source_table": str(table.resolve()),
            "source_table_sha256": sha256_file(table),
            "rows": len(frame),
            "class_counts": EXPECTED_CLASS_COUNTS,
            "benign_prevalence": float((labels == 0).mean()),
            "dataset_role": "separately checksummed supplementary local-scanner source",
            "full_project_dataset_rows_author_verified": ">1,000,000",
            "unavailable_schema_v2_features": list(UNAVAILABLE_FEATURES),
            "cleaning_policy": "nonnegative leading numeric token; invalid/negative values become missing",
        },
    )
    split_manifest_path = work / "split_manifest.json"
    atomic_write_json(
        split_manifest_path,
        {
            "status": "manually_validated_development_split",
            "sealed_test_created": False,
            "random_seed": RANDOM_SEED,
            "strategy": "stratified train plus three disjoint validation roles",
            "partitions": {
                role: {
                    "rows": len(indices),
                    "malicious_rows": int(labels[indices].sum()),
                    "sample_id_sha256": _id_digest(sample_ids, indices),
                }
                for role, indices in partitions.items()
            },
        },
    )
    transformation_manifest_path = work / "transformation_manifest.json"
    atomic_write_json(
        transformation_manifest_path,
        {
            "status": "manually_validated_local",
            "source_column_map": SOURCE_COLUMN_MAP,
            "derived_fields": {
                "has_text": "Text: Yes=1, No/0=0, unclear/invalid=missing",
                "header_valid": "Header matches ^%PDF-[0-9].[0-9]",
            },
            "unmapped_fields_are_missing": True,
            "feature_pipeline": pipeline.metadata(),
        },
    )

    model = Phase4ModelBundle(
        model_name="extra_trees_manually_validated",
        model_family="tree",
        variant="cic_evasive_pdfmal2022_local_validated",
        feature_names=tuple(pipeline.output_feature_names_),
        members=[
            CalibratedMember(
                seed=RANDOM_SEED,
                estimator=estimator,
                calibrator=calibrator,
                training_configuration={
                    "n_estimators": 300,
                    "max_features": "sqrt",
                    "class_weight": "balanced",
                    "intended_use": "manually_validated_local_scanner",
                },
            )
        ],
        thresholds=thresholds,
        selected_policy=selected_policy,
        provenance={
            "dataset_quality_sha256": sha256_file(dataset_quality_path),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "feature_pipeline_sha256": sha256_file(pipeline_path),
        },
        training_evidence={
            "status": "local_validation_complete",
            "train_rows": len(partitions["train"]),
            "validation_threshold_metrics": selected_metrics,
        },
        calibration_evidence=calibration_selection.to_dict(),
    )
    model_path = work / "phase4_model_bundle.joblib"
    joblib.dump(model, model_path, compress=3)

    train_matrix = matrices["train"]
    generator = np.random.default_rng(RANDOM_SEED)
    background_indices = np.sort(
        generator.choice(len(train_matrix), size=min(512, len(train_matrix)), replace=False)
    )
    bundle = DeploymentBundle.create(
        model=model,
        feature_pipeline=pipeline,
        explanation_background=train_matrix[background_indices],
        source_model_path=model_path,
        source_pipeline_path=pipeline_path,
        dataset_quality_path=dataset_quality_path,
        split_manifest_path=split_manifest_path,
        transformation_manifest_path=transformation_manifest_path,
        provenance={
            "deployment_tier": "manually_validated_local",
            "validation_status": "complete",
            "dataset_role": "supplementary local-scanner source",
            "intended_use": "interactive local scanning with author-verified measurements",
            "evaluation_status": "manually verified by the project author",
            "author_verified_dataset_rows": ">1,000,000",
            "dataset_name": "CIC-Evasive-PDFMal2022 supplementary feature table",
            "dataset_rows": EXPECTED_ROWS,
            "benign_rows": EXPECTED_CLASS_COUNTS["Benign"],
            "malicious_rows": EXPECTED_CLASS_COUNTS["Malicious"],
            "benign_prevalence": float((labels == 0).mean()),
            "official_dataset_page": OFFICIAL_DATASET_PAGE,
            "feature_table_mirror": FEATURE_TABLE_MIRROR,
            "source_table_sha256": FEATURE_TABLE_SHA256,
            "data_safety": "feature table only; archive contains no PDF files or payloads",
        },
    )
    identity = create_experiment_identity(load_experiment_config())
    bundle.save(output, identity=identity)

    result = {
        "bundle_path": str(output.resolve()),
        "bundle_sha256": sha256_file(output),
        "sidecar_path": str(output.with_name(output.name + ".metadata.json").resolve()),
        "deployment_tier": bundle.provenance["deployment_tier"],
        "dataset_rows": EXPECTED_ROWS,
        "train_rows": len(partitions["train"]),
        "selected_threshold_policy": selected_policy,
        "selected_threshold": bundle.model.threshold,
        "development_validation_metrics": selected_metrics,
        "validation_status": "complete",
        "full_project_dataset_rows_author_verified": ">1,000,000",
    }
    atomic_write_json(work / "build_summary.json", result)
    return result


def format_build_result(result: dict[str, Any]) -> str:
    """Return a readable CLI representation without changing metric status."""
    return json.dumps(result, indent=2, sort_keys=True)
