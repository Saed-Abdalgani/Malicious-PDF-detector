"""Manifest-driven, fail-closed acquisition of safe feature-only datasets.

The old implementation contained placeholder URLs, disabled checksums, and a
browser-opening fallback.  This replacement downloads only explicitly approved,
feature-only CSV/Parquet/JSONL tables over HTTPS and verifies their declared
checksum and size.  It never extracts archives or retrieves raw PDF/malware files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

from src.config import MANIFESTS_DIR, RAW_DATA_DIR
from src.data.manifest import (
    DataSourceSpec,
    SafeDataPolicy,
    SourceManifestError,
    load_source_registry,
)
from src.experiment import sha256_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnsafeDataSourceError(RuntimeError):
    """Raised when acquisition would violate the feature-only safety policy."""


def inspect_archive_members(
    archive_path: Path,
    disallowed_extensions: Iterable[str],
) -> list[str]:
    """Inspect ZIP member names without extracting and return unsafe members."""
    path = Path(archive_path)
    if not zipfile.is_zipfile(path):
        return []
    disallowed = {extension.lower() for extension in disallowed_extensions}
    unsafe: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            suffix = Path(member.filename).suffix.lower()
            if suffix in disallowed or not suffix:
                unsafe.append(member.filename)
    return unsafe


def validate_source_approval(
    source: DataSourceSpec,
    policy: SafeDataPolicy,
) -> None:
    """Fail unless a source is enabled, approved, feature-only, and checksummed."""
    problems: list[str] = []
    if not source.enabled:
        problems.append("source is disabled")
    if source.approval_status != "approved":
        problems.append(f"approval_status={source.approval_status!r}")
    if policy.feature_only and not source.feature_only:
        problems.append("source is not feature-only")
    if source.file_format.lower() not in policy.allowed_formats:
        problems.append(f"format {source.file_format!r} is not allowlisted")
    if policy.require_sha256:
        digest = source.sha256 or ""
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            problems.append("a valid SHA-256 is required")
    if source.url and policy.require_https:
        parsed = urlparse(source.url)
        if parsed.scheme.lower() != "https":
            problems.append("dataset URL must use HTTPS")
    if not source.url and not source.local_path:
        problems.append("neither URL nor local_path is configured")
    if problems:
        raise UnsafeDataSourceError(
            f"Dataset source {source.source_id!r} is not eligible: " + "; ".join(problems)
        )


def verify_source_file(path: Path, source: DataSourceSpec) -> dict:
    """Verify extension, size, checksum, and absence of archive payloads."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source_path}")
    if source_path.suffix.lower() != source.file_format.lower():
        raise UnsafeDataSourceError(
            f"Dataset extension mismatch: expected {source.file_format}, got {source_path.suffix}."
        )
    if zipfile.is_zipfile(source_path):
        raise UnsafeDataSourceError(
            "Archive containers are not accepted as primary feature tables. "
            "Provide an approved, checksummed CSV/Parquet/JSONL file directly."
        )
    size = source_path.stat().st_size
    if source.expected_size_bytes is not None and size != int(source.expected_size_bytes):
        raise UnsafeDataSourceError(
            f"Dataset size mismatch: expected {source.expected_size_bytes}, got {size}."
        )
    digest = sha256_file(source_path)
    if source.sha256 and digest.lower() != source.sha256.lower():
        raise UnsafeDataSourceError(
            f"Dataset checksum mismatch: expected {source.sha256}, got {digest}."
        )
    return {"path": str(source_path.resolve()), "size_bytes": size, "sha256": digest}


def _download_to_temporary_file(
    source: DataSourceSpec,
    destination_dir: Path,
    *,
    timeout_seconds: int = 60,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not source.url:
        raise SourceManifestError(f"Source {source.source_id!r} has no URL.")
    suffix = source.file_format
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{source.source_id}-", suffix=f"{suffix}.part", dir=destination_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    maximum = int(source.expected_size_bytes * 1.01) if source.expected_size_bytes else None
    downloaded = 0
    try:
        with requests.get(
            source.url,
            stream=True,
            allow_redirects=True,
            timeout=(15, timeout_seconds),
            headers={"User-Agent": "MaliciousPDFDetector-AcademicDataFetcher/2.0"},
        ) as response:
            response.raise_for_status()
            final_url = urlparse(response.url)
            if final_url.scheme.lower() != "https":
                raise UnsafeDataSourceError("Redirected dataset URL is not HTTPS.")
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise UnsafeDataSourceError("Dataset endpoint returned HTML, not a feature table.")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if maximum is not None and downloaded > maximum:
                        raise UnsafeDataSourceError(
                            "Download exceeded the approved expected-size envelope."
                        )
                    handle.write(chunk)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def materialize_approved_source(
    source_id: str,
    *,
    destination_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Verify or download one approved feature table and write its provenance."""
    registry = load_source_registry()
    source = registry.require(source_id)
    validate_source_approval(source, registry.policy)

    if source.local_path:
        local = source.resolved_local_path()
        assert local is not None
        verification = verify_source_file(local, source)
        final_path = local
    else:
        temporary = _download_to_temporary_file(source, destination_dir)
        candidate = temporary.with_suffix(source.file_format)
        try:
            # Verify against the approved spec before giving the file its final name.
            temporary.replace(candidate)
            verification = verify_source_file(candidate, source)
            final_path = destination_dir / f"{source.source_id}-{source.version}{source.file_format}"
            if final_path.exists() and candidate.resolve() != final_path.resolve():
                raise FileExistsError(
                    f"Immutable acquired dataset already exists: {final_path}"
                )
            if candidate.resolve() != final_path.resolve():
                shutil.move(str(candidate), str(final_path))
        except Exception:
            temporary.unlink(missing_ok=True)
            candidate.unlink(missing_ok=True)
            raise

    manifest_dir = MANIFESTS_DIR / source.source_id / source.version
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "acquisition.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_id": source.source_id,
                "title": source.title,
                "provider": source.provider,
                "version": source.version,
                "approval_status": source.approval_status,
                "feature_only": source.feature_only,
                "license": source.license,
                "role": source.role,
                "verified_file": verification,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Approved feature table verified: %s", final_path)
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List source IDs and approval state.")
    verify_parser = subparsers.add_parser("verify", help="Verify a configured local source.")
    verify_parser.add_argument("source_id")
    acquire_parser = subparsers.add_parser("acquire", help="Acquire/verify an approved source.")
    acquire_parser.add_argument("source_id")
    args = parser.parse_args()

    registry = load_source_registry()
    if args.command == "list":
        for source in registry.sources.values():
            print(
                f"{source.source_id}: status={source.approval_status}, "
                f"enabled={source.enabled}, role={source.role}, format={source.file_format}"
            )
        return 0
    source = registry.require(args.source_id)
    if args.command == "verify":
        validate_source_approval(source, registry.policy)
        path = source.resolved_local_path()
        if path is None:
            raise UnsafeDataSourceError("The selected source has no local_path.")
        print(json.dumps(verify_source_file(path, source), indent=2))
        return 0
    print(materialize_approved_source(args.source_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
