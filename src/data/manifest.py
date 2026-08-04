"""Typed source registry for safe, feature-only dataset acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from src.config import DATA_SOURCES_CONFIG_PATH, PROJECT_ROOT


class SourceManifestError(ValueError):
    """Raised when a data-source registry violates the safety contract."""


@dataclass(frozen=True)
class SafeDataPolicy:
    allowed_formats: tuple[str, ...]
    disallowed_archive_members: tuple[str, ...]
    require_https: bool = True
    require_sha256: bool = True
    feature_only: bool = True


@dataclass(frozen=True)
class DataSourceSpec:
    source_id: str
    title: str
    provider: str
    version: str
    approval_status: str
    enabled: bool
    feature_only: bool
    file_format: str
    url: Optional[str]
    local_path: Optional[str]
    sha256: Optional[str]
    expected_size_bytes: Optional[int]
    expected_rows: Optional[int]
    license: str
    role: str
    authoritative_page: Optional[str] = None
    column_mapping: Mapping[str, str] = field(default_factory=dict)
    metadata_mapping: Mapping[str, str] = field(default_factory=dict)

    def resolved_local_path(self) -> Optional[Path]:
        if not self.local_path:
            return None
        path = Path(self.local_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class SourceRegistry:
    registry_version: int
    policy: SafeDataPolicy
    sources: Mapping[str, DataSourceSpec]

    def require(self, source_id: str) -> DataSourceSpec:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise SourceManifestError(f"Unknown dataset source: {source_id}") from exc


def _as_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise SourceManifestError("Policy extension lists must be arrays of strings.")
    return tuple(value.lower() for value in values)


def load_source_registry(path: Optional[Path] = None) -> SourceRegistry:
    """Load the source registry and validate all safety-relevant fields."""
    registry_path = Path(path or DATA_SOURCES_CONFIG_PATH)
    if not registry_path.exists():
        raise FileNotFoundError(f"Data-source registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), dict):
        raise SourceManifestError("Registry must contain a `sources` mapping.")

    policy_raw = raw.get("policy", {})
    policy = SafeDataPolicy(
        allowed_formats=_as_tuple(policy_raw.get("allowed_formats", [])),
        disallowed_archive_members=_as_tuple(
            policy_raw.get("disallowed_archive_members", [])
        ),
        require_https=bool(policy_raw.get("require_https", True)),
        require_sha256=bool(policy_raw.get("require_sha256", True)),
        feature_only=bool(policy_raw.get("feature_only", True)),
    )
    sources: dict[str, DataSourceSpec] = {}
    required = {
        "title",
        "provider",
        "version",
        "approval_status",
        "enabled",
        "feature_only",
        "file_format",
        "license",
        "role",
    }
    for source_id, source_raw in raw["sources"].items():
        if not isinstance(source_raw, dict):
            raise SourceManifestError(f"Source {source_id!r} must be a mapping.")
        missing = sorted(required - set(source_raw))
        if missing:
            raise SourceManifestError(
                f"Source {source_id!r} is missing required fields: {missing}"
            )
        file_format = str(source_raw["file_format"]).lower()
        if file_format not in policy.allowed_formats:
            raise SourceManifestError(
                f"Source {source_id!r} uses disallowed format {file_format!r}."
            )
        sources[source_id] = DataSourceSpec(
            source_id=source_id,
            title=str(source_raw["title"]),
            provider=str(source_raw["provider"]),
            version=str(source_raw["version"]),
            approval_status=str(source_raw["approval_status"]),
            enabled=bool(source_raw["enabled"]),
            feature_only=bool(source_raw["feature_only"]),
            file_format=file_format,
            url=source_raw.get("url"),
            local_path=source_raw.get("local_path"),
            sha256=source_raw.get("sha256"),
            expected_size_bytes=source_raw.get("expected_size_bytes"),
            expected_rows=source_raw.get("expected_rows"),
            license=str(source_raw["license"]),
            role=str(source_raw["role"]),
            authoritative_page=source_raw.get("authoritative_page"),
            column_mapping=dict(source_raw.get("column_mapping", {})),
            metadata_mapping=dict(source_raw.get("metadata_mapping", {})),
        )
    return SourceRegistry(
        registry_version=int(raw.get("registry_version", 1)),
        policy=policy,
        sources=sources,
    )
