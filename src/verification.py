"""Stage-aware Phase 9 release verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.artifacts import metadata_path
from src.config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SPLITS_DIR
from src.data.splitter import verify_frozen_splits
from src.data.validate import verify_validated_dataset
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.models.deployment import DeploymentBundle
from src.utils.atomic import atomic_write_json


PHASE9_VERSION = "1.0.0"


class VerificationGateError(RuntimeError):
    """Raised when a release contract is absent, stale, or incompatible."""


def _json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationGateError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationGateError(f"{description} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationGateError(f"{description} must be a JSON object.")
    return value


def _check(path: Path, expected: str, description: str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise VerificationGateError(f"{description} is missing: {target}")
    actual = sha256_file(target)
    if actual != expected:
        raise VerificationGateError(f"{description} checksum mismatch.")
    return {"path": str(target.resolve()), "sha256": actual, "passed": True}


def verify_release(*, write_report: bool = True) -> dict[str, Any]:
    """Verify Phase 0-10 contracts without changing any trained artifact."""
    config = load_experiment_config()
    identity = create_experiment_identity(config)
    summary_path = RESULTS_DIR / "experiment_summary.json"
    summary = _json(summary_path, "Experiment summary")
    if summary.get("status") != "phase_8_complete_deployment_bundle_verified":
        raise VerificationGateError("Release verification requires completed Phase 8 evidence.")
    if summary.get("final_metrics") is None:
        raise VerificationGateError("Release verification requires sealed Phase 5 final metrics.")

    checks: dict[str, Any] = {
        "experiment_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
            "passed": True,
        }
    }
    quality = verify_validated_dataset()
    split_root = SPLITS_DIR / str(config["split_version"])
    split = verify_frozen_splits(split_root)
    gates = config["acceptance_gates"]
    if int(quality["row_flow"]["split_eligible"]) < int(gates["minimum_clean_rows"]):
        raise VerificationGateError("Verified clean row count is below the release gate.")
    for partition, minimum in (
        ("train", int(gates["minimum_train_rows"])),
        ("validation", int(gates["minimum_validation_rows"])),
        ("test", int(gates["minimum_test_rows"])),
    ):
        if int(split.row_counts[partition]) < minimum:
            raise VerificationGateError(f"Verified {partition} rows are below the release gate.")
        if float(split.benign_prevalence[partition]) < float(gates["minimum_benign_prevalence"]):
            raise VerificationGateError(f"Verified {partition} prevalence is below the release gate.")
    checks["data_and_split_gates"] = {
        "clean_rows": int(quality["row_flow"]["split_eligible"]),
        "partition_rows": {
            name: int(rows) for name, rows in split.row_counts.items()
        },
        "partition_benign_prevalence": {
            name: float(prevalence)
            for name, prevalence in split.benign_prevalence.items()
        },
        "passed": True,
    }

    phase4_entry = summary.get("phase_4", {})
    checks["phase_4_manifest"] = _check(
        Path(str(phase4_entry.get("champion_manifest", ""))),
        str(phase4_entry.get("champion_manifest_sha256", "")),
        "Phase 4 champion manifest",
    )
    phase_manifests: dict[int, tuple[Path, dict[str, Any]]] = {}
    for phase in range(5, 9):
        entry = summary.get(f"phase_{phase}")
        if not isinstance(entry, dict):
            raise VerificationGateError(f"Experiment summary lacks Phase {phase} provenance.")
        phase_path = Path(str(entry.get("manifest", "")))
        checks[f"phase_{phase}_manifest"] = _check(
            phase_path,
            str(entry.get("manifest_sha256", "")),
            f"Phase {phase} manifest",
        )
        phase_manifests[phase] = (phase_path, _json(phase_path, f"Phase {phase} manifest"))

    for phase in (5, 6, 7):
        manifest_path, manifest = phase_manifests[phase]
        for name, entry in manifest.get("outputs", {}).items():
            output_path = (
                Path(entry["path"])
                if entry.get("path")
                else manifest_path.parent / name
            )
            checks[f"phase_{phase}_output:{name}"] = _check(
                output_path,
                str(entry.get("sha256", "")),
                f"Phase {phase} output {name}",
            )

    phase8_manifest = phase_manifests[8][1]
    phase8_root = MODELS_DIR / "deployment"
    checks["phase_8_internal_manifest"] = _check(
        phase8_root / "manifest.json",
        sha256_file(phase_manifests[8][0]),
        "Phase 8 internal manifest",
    )
    checks["phase_8_golden_predictions"] = _check(
        phase8_root / "golden_predictions.json",
        str(phase8_manifest.get("golden_predictions_sha256", "")),
        "Phase 8 golden predictions",
    )
    checks["phase_8_resource_gate"] = _check(
        phase8_root / "resource_gate.json",
        str(phase8_manifest.get("resource_gate_sha256", "")),
        "Phase 8 resource gate",
    )

    bundle_path = Path(summary["phase_8"]["bundle"])
    checks["deployment_bundle"] = _check(
        bundle_path,
        str(summary["phase_8"]["bundle_sha256"]),
        "Deployment bundle",
    )
    bundle = DeploymentBundle.load(bundle_path, identity=identity)
    checks["deployment_bundle"].update(
        {
            "schema_sha256": bundle.schema_sha256,
            "threshold": bundle.model.threshold,
            "threshold_policy": bundle.model.selected_policy,
            "sidecar_sha256": sha256_file(metadata_path(bundle_path)),
        }
    )

    docs_manifest_path = RESULTS_DIR / "documentation_sync_manifest.json"
    docs_manifest = _json(docs_manifest_path, "Documentation sync manifest")
    if docs_manifest.get("experiment_summary_sha256") != sha256_file(summary_path):
        raise VerificationGateError("Generated documentation is stale relative to the experiment summary.")
    for name, entry in docs_manifest.get("outputs", {}).items():
        checks[f"documentation:{name}"] = _check(
            PROJECT_ROOT / entry["path"], entry["sha256"], f"Generated documentation {name}"
        )

    deployment_root = MODELS_DIR / "deployment"
    retained_pdfs = sorted(deployment_root.rglob("*.pdf")) if deployment_root.exists() else []
    if retained_pdfs:
        raise VerificationGateError(f"Deployment artifacts retain PDF bytes: {retained_pdfs}")
    checks["uploaded_pdf_retention"] = {"retained_pdf_files": 0, "passed": True}

    report = {
        "phase9_version": PHASE9_VERSION,
        "experiment": identity.to_dict(),
        "checks": checks,
        "all_checks_passed": True,
        "scope": [
            "manifest_and_checksum",
            "schema_and_bundle_compatibility",
            "sealed_metrics_presence",
            "phase5_to_phase8_provenance",
            "documentation_sync",
            "uploaded_pdf_non_retention",
        ],
    }
    if write_report:
        atomic_write_json(RESULTS_DIR / "phase9_verification.json", report)
    return report


def main() -> int:
    print(json.dumps(verify_release(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
