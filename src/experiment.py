"""Experiment configuration and deterministic scientific identity.

Every generated artifact is tied to the dataset, feature schema, split schema,
code revision, and seed that produced it.  This prevents stale models and
scalers from being loaded merely because their filenames happen to match.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from src.config import EXPERIMENT_CONFIG_PATH, PROJECT_ROOT


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_experiment_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and validate the YAML experiment configuration."""
    config_path = Path(path or EXPERIMENT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ExperimentConfigurationError("Experiment config must be a mapping.")

    required = {
        "project_name",
        "random_seed",
        "dataset_version",
        "feature_schema_version",
        "split_version",
        "acceptance_gates",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ExperimentConfigurationError(
            f"Experiment config is missing required keys: {missing}"
        )

    gates = config["acceptance_gates"]
    benign = float(gates.get("minimum_benign_prevalence", 0.0))
    if not 0.99 < benign < 1.0:
        raise ExperimentConfigurationError(
            "minimum_benign_prevalence must be greater than 0.99 and less than 1."
        )
    if int(gates.get("minimum_train_rows", 0)) < 2_000_000:
        raise ExperimentConfigurationError(
            "minimum_train_rows must satisfy the professor's 2,000,000-row gate."
        )
    phase4 = config.get("phase4", {})
    seeds = phase4.get("seeds", [])
    if len(set(seeds)) < 3:
        raise ExperimentConfigurationError(
            "Phase 4 requires at least three distinct finalist seeds."
        )
    if not 0.0 < float(phase4.get("maximum_fpr", 1.0)) <= 0.001:
        raise ExperimentConfigurationError(
            "Phase 4 maximum_fpr must be at most the predeclared 0.1% limit."
        )
    required_models = set(phase4.get("required_models", []))
    expected_models = {
        "always_benign",
        "logistic_regression",
        "extra_trees",
        "lightgbm",
        "xgboost_hist",
        "fully_connected_mlp",
        "ft_transformer",
    }
    if required_models != expected_models:
        raise ExperimentConfigurationError(
            "Phase 4 required_models must contain the complete professor model set."
        )
    phase5 = config.get("phase5", {})
    if int(phase5.get("sealed_test_maximum_evaluations", 0)) != 1:
        raise ExperimentConfigurationError("The sealed test evaluation limit must be one.")
    for field in ("bootstrap_replicates", "probability_bootstrap_replicates"):
        if int(phase5.get(field, 0)) < 100:
            raise ExperimentConfigurationError(
                f"Phase 5 {field} must request at least 100 replicates."
            )
    phase6 = config.get("phase6", {})
    if int(phase6.get("background_rows", 0)) < 100:
        raise ExperimentConfigurationError("Phase 6 requires at least 100 background rows.")
    if int(phase6.get("bootstrap_replicates", 0)) < 100:
        raise ExperimentConfigurationError(
            "Phase 6 stability requires at least 100 bootstrap replicates."
        )
    if int(phase6.get("minimum_independent_support_methods", 0)) < 2:
        raise ExperimentConfigurationError(
            "Phase 6 conclusions require at least two independent support methods."
        )
    phase7 = config.get("phase7", {})
    if int(phase7.get("minimum_valid_mutation_families", 0)) < 8:
        raise ExperimentConfigurationError(
            "Phase 7 requires at least eight valid PDF mutation families."
        )
    if int(phase7.get("fixtures_per_benchmark_class", 0)) < 2:
        raise ExperimentConfigurationError(
            "Phase 7 requires at least two inert fixtures per benchmark class."
        )
    if bool(phase7.get("persist_fixture_pdfs", True)):
        raise ExperimentConfigurationError(
            "Phase 7 may not persist generated PDF fixtures in report artifacts."
        )
    phase8 = config.get("phase8", {})
    if int(phase8.get("explanation_background_rows", 0)) < 100:
        raise ExperimentConfigurationError(
            "Phase 8 requires at least 100 real train-only explanation rows."
        )
    if int(phase8.get("golden_fixture_count", 0)) < 100:
        raise ExperimentConfigurationError(
            "Phase 8 requires at least 100 golden inference fixtures."
        )
    maximum_pdf_bytes = int(phase8.get("maximum_pdf_bytes", 0))
    if not 0 < maximum_pdf_bytes <= 100 * 1024 * 1024:
        raise ExperimentConfigurationError("Phase 8 maximum_pdf_bytes is invalid.")
    if bool(phase8.get("persist_uploaded_pdfs", True)):
        raise ExperimentConfigurationError(
            "Phase 8 may not retain uploaded PDF bytes."
        )
    return config


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a mapping deterministically for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(project_root: Path = PROJECT_ROOT) -> str:
    """Return the current commit, or ``uncommitted`` outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"


@dataclass(frozen=True)
class ExperimentIdentity:
    """Immutable identity shared by all artifacts from one experiment."""

    experiment_id: str
    project_name: str
    dataset_version: str
    feature_schema_version: str
    split_version: str
    code_commit: str
    random_seed: int
    config_sha256: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentIdentity":
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def create_experiment_identity(
    config: Optional[Mapping[str, Any]] = None,
    *,
    created_at: Optional[datetime] = None,
    code_commit: Optional[str] = None,
) -> ExperimentIdentity:
    """Create an identity whose stable ID excludes the creation timestamp."""
    cfg = dict(config or load_experiment_config())
    commit = code_commit or current_git_commit()
    stable_payload = {
        "project_name": cfg["project_name"],
        "dataset_version": cfg["dataset_version"],
        "feature_schema_version": cfg["feature_schema_version"],
        "split_version": cfg["split_version"],
        "code_commit": commit,
        "random_seed": int(cfg["random_seed"]),
        "config": cfg,
    }
    config_hash = hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()
    experiment_id = hashlib.sha256(
        canonical_json(stable_payload).encode("utf-8")
    ).hexdigest()[:20]
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return ExperimentIdentity(
        experiment_id=experiment_id,
        project_name=str(cfg["project_name"]),
        dataset_version=str(cfg["dataset_version"]),
        feature_schema_version=str(cfg["feature_schema_version"]),
        split_version=str(cfg["split_version"]),
        code_commit=commit,
        random_seed=int(cfg["random_seed"]),
        config_sha256=config_hash,
        created_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
    )
