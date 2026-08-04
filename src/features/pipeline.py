"""The single serialized schema-v2 feature pipeline for train and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from src.artifacts import verify_artifact_compatibility, write_artifact_metadata
from src.data.preprocessing import TrainOnlyPreprocessor
from src.experiment import ExperimentIdentity, canonical_json, create_experiment_identity
from src.experiment import sha256_file
from src.features.engineered import EngineeredFeatureBuilder
from src.features.schema_v2 import (
    BASE_FEATURE_COLUMNS,
    BASE_FEATURE_SPECS,
    FEATURE_SCHEMA_VERSION,
)
from src.utils.atomic import atomic_write_json


class FeaturePipelineV2:
    """Pure engineering followed by immutable train-only learned preprocessing."""

    version = "2.0.0"

    def __init__(
        self,
        *,
        model_family: str = "neural",
        optional_status_columns: Sequence[str] = (),
        lower_quantile: float = 0.0001,
        upper_quantile: float = 0.9999,
    ) -> None:
        self.model_family = model_family
        self.builder = EngineeredFeatureBuilder(
            optional_status_columns=optional_status_columns
        )
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self._fitted = False

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        return (*BASE_FEATURE_COLUMNS, *self.builder.optional_status_columns)

    def fit(self, frame: pd.DataFrame, *, partition_name: str) -> "FeaturePipelineV2":
        engineered = self.builder.transform(frame)
        self.preprocessor = TrainOnlyPreprocessor(
            engineered.columns,
            model_family=self.model_family,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
            add_missing_indicators=True,
            drop_constant=True,
        ).fit(engineered, partition_name=partition_name)
        self.engineered_feature_names_ = tuple(engineered.columns)
        self.output_feature_names_ = self.preprocessor.output_features_
        self.engineering_specs_ = tuple(spec.to_dict() for spec in self.builder.feature_specs())
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeaturePipelineV2 must be fit before transform.")
        engineered = self.builder.transform(frame)
        if tuple(engineered.columns) != self.engineered_feature_names_:
            raise RuntimeError("Engineered feature order changed after fitting.")
        return self.preprocessor.transform(engineered)

    def fit_transform(self, frame: pd.DataFrame, *, partition_name: str) -> pd.DataFrame:
        return self.fit(frame, partition_name=partition_name).transform(frame)

    def fit_dataset(
        self,
        dataset_path: Path,
        *,
        partition_name: str,
        sealed_split_root: Path,
        dataset_quality_path: Path,
        transformation_manifest_path: Path,
        batch_size: int = 100_000,
        quantile_sample_rows: int = 250_000,
    ) -> "FeaturePipelineV2":
        """Fit on a partitioned Parquet train layer without full materialization."""
        from src.data.loader import iter_dataset_batches
        from src.data.splitter import verify_frozen_splits
        from src.data.validate import verify_validated_dataset

        if partition_name != "train":
            raise RuntimeError("A serialized feature pipeline can only fit the sealed train split.")
        split_root = Path(sealed_split_root).resolve()
        expected_dataset = split_root / "train" / "features"
        if Path(dataset_path).resolve() != expected_dataset.resolve():
            raise RuntimeError(
                f"Training dataset must be the sealed train/features layer: {expected_dataset}"
            )
        split_manifest = verify_frozen_splits(split_root, batch_size=batch_size)
        verify_validated_dataset(
            dataset_quality_path, transformation_manifest_path
        )
        self.split_manifest_sha256_ = sha256_file(split_root / "split_manifest.json")
        self.split_manifest_hash_ = split_manifest.manifest_hash
        self.dataset_quality_sha256_ = sha256_file(Path(dataset_quality_path))
        self.transformation_manifest_sha256_ = sha256_file(
            Path(transformation_manifest_path)
        )
        self.train_dataset_path_ = str(expected_dataset)

        def engineered_batches():
            for batch in iter_dataset_batches(dataset_path, batch_size=batch_size):
                yield self.builder.transform(batch)

        first = next(engineered_batches(), None)
        if first is None:
            raise RuntimeError(f"Training dataset is empty: {dataset_path}")
        self.preprocessor = TrainOnlyPreprocessor(
            first.columns,
            model_family=self.model_family,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
            add_missing_indicators=True,
            drop_constant=True,
        ).fit_batches(
            engineered_batches,
            partition_name=partition_name,
            quantile_sample_rows=quantile_sample_rows,
        )
        self.engineered_feature_names_ = tuple(first.columns)
        self.output_feature_names_ = self.preprocessor.output_features_
        self.engineering_specs_ = tuple(spec.to_dict() for spec in self.builder.feature_specs())
        self._fitted = True
        return self

    def transform_record(self, record: Mapping[str, float]) -> pd.DataFrame:
        missing = sorted(set(self.required_input_columns) - set(record))
        if missing:
            raise ValueError(f"Feature record is missing required values: {missing}")
        return self.transform(pd.DataFrame([dict(record)]))

    def metadata(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("FeaturePipelineV2 is not fitted.")
        specs_hash = hashlib.sha256(
            canonical_json({"specs": list(self.engineering_specs_)}).encode("utf-8")
        ).hexdigest()
        return {
            "pipeline_version": self.version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_family": self.model_family,
            "required_input_columns": list(self.required_input_columns),
            "engineered_feature_count": len(self.engineered_feature_names_),
            "output_feature_count": len(self.output_feature_names_),
            "output_feature_names": list(self.output_feature_names_),
            "engineering_spec_sha256": specs_hash,
            "preprocessing": self.preprocessor.metadata().to_dict(),
            "output_feature_specs": self.output_feature_specs(),
            "fit_partition": "train",
            "split_manifest_sha256": getattr(self, "split_manifest_sha256_", None),
            "split_manifest_hash": getattr(self, "split_manifest_hash_", None),
            "dataset_quality_sha256": getattr(self, "dataset_quality_sha256_", None),
            "transformation_manifest_sha256": getattr(
                self, "transformation_manifest_sha256_", None
            ),
            "train_dataset_path": getattr(self, "train_dataset_path_", None),
        }

    def output_feature_specs(self) -> list[dict[str, Any]]:
        """Document every model input, including learned flags and units."""
        if not self._fitted:
            raise RuntimeError("FeaturePipelineV2 is not fitted.")
        source_specs = {spec.name: spec.to_dict() for spec in BASE_FEATURE_SPECS}
        source_specs.update(
            {spec.name: spec.to_dict() for spec in self.builder.feature_specs()}
        )
        output: list[dict[str, Any]] = []
        for feature_index, name in enumerate(self.preprocessor.active_features_):
            source = source_specs[name]
            if self.model_family == "neural":
                formula = (
                    f"standardize(clip(impute_train_median({name}), "
                    "train_quantile_bounds), train_mean, train_std)"
                )
                unit = "standard deviations"
                mean = float(self.preprocessor.scaler_.mean_[feature_index])
                scale = float(self.preprocessor.scaler_.scale_[feature_index])
                minimum = float((self.preprocessor.clip_lower_[name] - mean) / scale)
                maximum = float((self.preprocessor.clip_upper_[name] - mean) / scale)
                range_basis = "train-learned clipped and standardized bounds"
            else:
                formula = f"impute_train_median({name})"
                unit = source["unit"]
                minimum, maximum = source.get("minimum"), source.get("maximum")
                range_basis = "schema domain; null maximum means intentionally unbounded"
            output.append(
                {
                    "name": name,
                    "family": "preprocessed_numeric",
                    "unit": unit,
                    "minimum": minimum,
                    "maximum": maximum,
                    "range_basis": range_basis,
                    "formula": formula,
                    "rationale": (
                        "Uses train-only learned values while preserving identical "
                        "batch and inference transformation."
                    ),
                    "lineage": [name],
                }
            )
        if self.preprocessor.add_missing_indicators:
            for name in self.preprocessor.feature_names:
                output.append(
                    {
                        "name": f"{name}__missing",
                        "family": "missingness",
                        "unit": "boolean",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "range_basis": "closed boolean domain",
                        "formula": f"1 if {name} is unavailable before imputation, else 0",
                        "rationale": "Distinguishes unavailable extraction from a true zero.",
                        "lineage": [name],
                    }
                )
        if self.model_family == "neural":
            for name in self.preprocessor.active_features_:
                output.append(
                    {
                        "name": f"{name}__clipped",
                        "family": "numeric_stability",
                        "unit": "boolean",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "range_basis": "closed boolean domain",
                        "formula": (
                            f"1 if imputed {name} is outside train-learned "
                            "quantile bounds, else 0"
                        ),
                        "rationale": (
                            "Makes neural stability clipping observable instead of "
                            "silently erasing extreme security signals."
                        ),
                        "lineage": [name],
                    }
                )
        if [spec["name"] for spec in output] != list(self.output_feature_names_):
            raise RuntimeError("Output feature documentation order is inconsistent.")
        return output

    def lineage(self) -> dict[str, list[str]]:
        return {name: list(values) for name, values in self.builder.lineage().items()}

    def save(
        self,
        path: Path,
        *,
        identity: ExperimentIdentity | None = None,
    ) -> Path:
        metadata = self.metadata()
        provenance_fields = (
            "split_manifest_sha256",
            "split_manifest_hash",
            "dataset_quality_sha256",
            "transformation_manifest_sha256",
        )
        missing_provenance = [name for name in provenance_fields if not metadata.get(name)]
        if missing_provenance:
            raise RuntimeError(
                "Feature pipeline cannot be serialized without verified Phase 1/2 "
                f"provenance: {missing_provenance}"
            )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)
        write_artifact_metadata(
            output, "feature_pipeline_v2", identity or create_experiment_identity(),
            extra=metadata,
        )
        return output

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        identity: ExperimentIdentity | None = None,
        split_manifest_path: Path | None = None,
        dataset_quality_path: Path | None = None,
        transformation_manifest_path: Path | None = None,
    ) -> "FeaturePipelineV2":
        active = identity or create_experiment_identity()
        metadata = verify_artifact_compatibility(path, active)
        if metadata.extra.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise RuntimeError("Serialized feature pipeline has an incompatible schema.")
        upstream = {
            "split_manifest_sha256": split_manifest_path,
            "dataset_quality_sha256": dataset_quality_path,
            "transformation_manifest_sha256": transformation_manifest_path,
        }
        for field, upstream_path in upstream.items():
            expected_hash = metadata.extra.get(field)
            if not expected_hash:
                raise RuntimeError(f"Serialized feature pipeline lacks {field}.")
            if upstream_path is not None:
                source = Path(upstream_path)
                if not source.is_file() or sha256_file(source) != expected_hash:
                    raise RuntimeError(
                        f"Serialized feature pipeline upstream mismatch: {field}."
                    )
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Artifact at {path} is not FeaturePipelineV2.")
        value.metadata()
        return value


def materialize_engineered_layer(
    input_path: Path,
    output_dir: Path,
    *,
    builder: EngineeredFeatureBuilder | None = None,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Write immutable float32 engineered Parquet parts separately from raw features."""
    from src.data.loader import iter_dataset_batches

    source = Path(input_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Engineered output already exists: {destination}")
    destination.mkdir(parents=True)
    active_builder = builder or EngineeredFeatureBuilder()
    rows = 0
    parts: list[dict[str, Any]] = []
    for part_index, batch in enumerate(iter_dataset_batches(source, batch_size=batch_size)):
        engineered = active_builder.transform(batch)
        # Preserve split identity/label metadata beside the model features.
        metadata = [
            name for name in (
                "sample_id", "Class", "source_id", "group_id", "first_seen_at",
                "label_confidence",
            )
            if name in batch.columns
        ]
        output = pd.concat(
            [batch.loc[:, metadata].reset_index(drop=True), engineered.reset_index(drop=True)],
            axis=1,
        )
        path = destination / f"part-{part_index:06d}.parquet"
        output.to_parquet(path, index=False, compression="zstd")
        rows += len(output)
        parts.append(
            {
                "path": path.name,
                "rows": len(output),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    specs = [spec.to_dict() for spec in active_builder.feature_specs()]
    input_parts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(source.glob("*.parquet"))
    ]
    manifest = {
        "input": str(source.resolve()),
        "output": str(destination.resolve()),
        "rows": rows,
        "parts": parts,
        "feature_count": len(active_builder.feature_specs()) + len(BASE_FEATURE_COLUMNS),
        "lineage": {name: list(value) for name, value in active_builder.lineage().items()},
        "input_parts": input_parts,
        "engineering_spec_sha256": hashlib.sha256(
            canonical_json({"specs": specs}).encode("utf-8")
        ).hexdigest(),
        "label_used_as_feature": False,
    }
    manifest_path = destination / "engineered_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
