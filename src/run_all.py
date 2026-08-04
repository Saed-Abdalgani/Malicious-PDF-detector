"""Fail-closed Phase 0-6 remediation workflow.

Phase 4 cannot begin until an approved feature-only source passes the 2.5M-row
quality gate and the sealed train split contains at least 2M rows at natural
prevalence. Phase 5 additionally requires explicit one-shot test confirmation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.artifacts import archive_legacy_results, initialize_experiment_summary
from src.config import DATA_REPORTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, SPLITS_DIR
from src.data.manifest import load_source_registry
from src.data.splitter import build_frozen_splits_from_dataset, verify_frozen_splits
from src.data.validate import validate_registered_source, verify_validated_dataset
from src.experiment import create_experiment_identity, load_experiment_config
from src.features.engineered import EngineeredFeatureBuilder
from src.features.pipeline import FeaturePipelineV2, materialize_engineered_layer
from src.features.schema_v2 import schema_dictionary
from src.features.selection import audit_file_size_shortcut
from src.utils.logger import get_logger
from src.utils.atomic import atomic_write_json

logger = get_logger(__name__)


def _phase_zero() -> None:
    archive_legacy_results()
    initialize_experiment_summary()
    logger.info("Phase 0 complete: legacy evidence frozen; active experiment initialized.")


def _available_sources() -> str:
    registry = load_source_registry()
    lines = []
    for source in registry.sources.values():
        lines.append(
            f"- {source.source_id}: enabled={source.enabled}, "
            f"approval={source.approval_status}, role={source.role}"
        )
    return "\n".join(lines)


def _phase_three(
    split_root: Path,
    *,
    split_version: str,
    compatibility_model_family: str,
    batch_size: int,
) -> dict[str, Path]:
    builder = EngineeredFeatureBuilder()
    shortcut_audit = audit_file_size_shortcut(
        split_root / "train" / "features",
        split_root / "validation" / "features",
        batch_size=batch_size,
    )
    atomic_write_json(
        DATA_REPORTS_DIR / "feature_shortcut_audit.json",
        shortcut_audit.to_dict(),
    )
    engineered_root = PROCESSED_DATA_DIR / "engineered_v2" / split_version
    manifests = {}
    # The test partition is deliberately absent here. Deterministic test
    # transformation happens inside Phase 5 only after the exclusive ledger
    # claim; Phase 3 and Phase 4 must not parse sealed-test feature content.
    for partition in ("train", "validation"):
        manifests[partition] = materialize_engineered_layer(
            split_root / partition / "features",
            engineered_root / partition,
            builder=builder,
            batch_size=batch_size,
        )
    pipelines: dict[str, FeaturePipelineV2] = {}
    pipeline_paths: dict[str, Path] = {}
    for family in ("tree", "neural"):
        pipeline = FeaturePipelineV2(model_family=family)
        pipeline.fit_dataset(
            split_root / "train" / "features",
            partition_name="train",
            sealed_split_root=split_root,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
            batch_size=batch_size,
        )
        pipelines[family] = pipeline
        pipeline_paths[family] = pipeline.save(
            MODELS_DIR / f"feature_pipeline_v2_{family}.pkl"
        )
    # Temporary inference compatibility path until Phase 4 selects a champion.
    compatibility_pipeline = pipelines[compatibility_model_family]
    pipeline_paths["compatibility_default"] = compatibility_pipeline.save(
        MODELS_DIR / "feature_pipeline_v2.pkl"
    )
    DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = {
        "base_schema": schema_dictionary(),
        "engineered_features": [spec.to_dict() for spec in builder.feature_specs()],
        "lineage": {name: list(value) for name, value in builder.lineage().items()},
        "pipelines": {
            family: pipeline.metadata() for family, pipeline in pipelines.items()
        },
        "model_input_features": {
            family: pipeline.output_feature_specs()
            for family, pipeline in pipelines.items()
        },
        "shortcut_audit": shortcut_audit.to_dict(),
        "materialized_layers": manifests,
    }
    atomic_write_json(DATA_REPORTS_DIR / "feature_dictionary_v2.json", catalog)
    return pipeline_paths


def _update_summary(
    *,
    quality: dict,
    split_manifest: dict,
    pipeline_paths: dict[str, Path],
) -> None:
    identity = create_experiment_identity(load_experiment_config())
    summary = {
        "experiment": identity.to_dict(),
        "status": "phase_3_complete_ready_for_phase_4",
        "data_gate_passed": True,
        "phase_1_dataset_quality": quality,
        "phase_2_split": split_manifest,
        "phase_3_feature_pipelines": {
            name: str(path.resolve()) for name, path in pipeline_paths.items()
        },
        "final_metrics": None,
        "notes": [
            "No model metrics are final until the Phase 5 sealed evaluation closes.",
            "Raw PDF or executable payload datasets were not admitted.",
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RESULTS_DIR / "experiment_summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-id",
        help="Enabled, approved, feature-only source ID from configs/data_sources.yaml.",
    )
    parser.add_argument("--through-phase", type=int, choices=range(0, 7), default=3)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--model-family", choices=("neural", "tree"), default="neural")
    parser.add_argument(
        "--confirm-sealed-test-evaluation",
        action="store_true",
        help=(
            "Required for Phase 5. Claims and opens the sealed test exactly once; "
            "rerunning under the same experiment is prohibited."
        ),
    )
    args = parser.parse_args(argv)

    _phase_zero()
    if args.through_phase == 0:
        return 0
    if not args.source_id:
        parser.error(
            "--source-id is required for Phase 1+. No source is selected automatically.\n"
            + _available_sources()
        )

    quality = validate_registered_source(args.source_id, enforce_gates=True)
    verify_validated_dataset()
    logger.info("Phase 1 complete: %s approved rows survived QC.", quality.row_flow["split_eligible"])
    if args.through_phase == 1:
        return 0

    config = load_experiment_config()
    split_version = str(config["split_version"])
    split_root = SPLITS_DIR / split_version
    split_manifest = build_frozen_splits_from_dataset(
        Path(quality.clean_output), output_root=split_root, batch_size=args.batch_size
    )
    split_manifest = verify_frozen_splits(
        split_root, batch_size=args.batch_size
    )
    logger.info("Phase 2 complete: sealed split manifest %s.", split_manifest.manifest_hash)
    if args.through_phase == 2:
        return 0

    pipeline_paths = _phase_three(
        split_root,
        split_version=split_version,
        compatibility_model_family=args.model_family,
        batch_size=args.batch_size,
    )
    _update_summary(
        quality=quality.to_dict(),
        split_manifest=split_manifest.to_dict(),
        pipeline_paths=pipeline_paths,
    )
    logger.info("Phase 3 complete: serialized family pipelines at %s.", pipeline_paths)
    if args.through_phase == 3:
        return 0

    from src.models.phase4 import Phase4Runner

    phase4_manifest = Phase4Runner(
        split_root=split_root, batch_size=args.batch_size
    ).run()
    logger.info("Phase 4 complete: champion manifest at %s.", phase4_manifest)
    if args.through_phase == 4:
        return 0
    if not args.confirm_sealed_test_evaluation:
        parser.error(
            "Phase 5 irreversibly opens the sealed test. Re-run with "
            "--confirm-sealed-test-evaluation after reviewing Phase 4 evidence."
        )

    from src.models.phase5 import Phase5Runner

    phase5_manifest = Phase5Runner(
        split_root=split_root,
        phase4_manifest_path=phase4_manifest,
        batch_size=args.batch_size,
    ).run()
    logger.info("Phase 5 complete: locked evaluation at %s.", phase5_manifest)
    if args.through_phase == 5:
        return 0

    from src.models.phase6 import Phase6Runner

    phase6_manifest = Phase6Runner(split_root=split_root).run()
    logger.info("Phase 6 complete: explainability manifest at %s.", phase6_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
