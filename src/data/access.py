"""Phase-specific access controls for the sealed train/validation/test split."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.experiment import ExperimentIdentity, canonical_json, sha256_file
from src.utils.atomic import atomic_write_json


class SplitAccessError(RuntimeError):
    """Raised when a phase attempts unauthorized or repeated split access."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_listed_path(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SplitAccessError(f"Unsafe split path in manifest: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SplitAccessError(f"Split path escapes root: {relative}") from exc
    return path


@dataclass(frozen=True)
class TrainingSplitAccess:
    split_root: str
    train_features: str
    validation_features: str
    split_manifest_sha256: str
    split_manifest_hash: str
    row_counts: dict[str, int]
    benign_prevalence: dict[str, float]
    test_content_opened: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_training_split_access(split_root: Path) -> TrainingSplitAccess:
    """Verify the sealed control plane without parsing sealed-test Parquet content."""
    root = Path(split_root)
    manifest_path = root / "split_manifest.json"
    seal_path = root / "SEALED"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise SplitAccessError("Training requires a sealed Phase 2 split.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SplitAccessError("Split manifest/seal is unreadable.") from exc
    manifest_sha = sha256_file(manifest_path)
    if seal.get("manifest_sha256") != manifest_sha:
        raise SplitAccessError("SEALED checksum does not bind split_manifest.json.")
    logical_payload = dict(manifest)
    logical_payload.pop("manifest_hash", None)
    logical_hash = hashlib.sha256(
        canonical_json(logical_payload).encode("utf-8")
    ).hexdigest()
    if logical_hash != manifest.get("manifest_hash") or logical_hash != seal.get(
        "manifest_hash"
    ):
        raise SplitAccessError("Sealed split logical manifest hash is invalid.")
    if not manifest.get("gates_passed"):
        raise SplitAccessError("Sealed split does not claim passed Phase 2 gates.")
    if any(int(value) for value in manifest.get("sample_overlap", {}).values()):
        raise SplitAccessError("Sealed split records sample overlap.")
    if any(int(value) for value in manifest.get("group_overlap", {}).values()):
        raise SplitAccessError("Sealed split records group overlap.")

    for partition in ("train", "validation", "test"):
        for kind, directory in (("features", "features"), ("sample_ids", "ids")):
            entries = manifest.get("files", {}).get(partition, {}).get(kind)
            if not isinstance(entries, list) or not entries:
                raise SplitAccessError(f"Manifest lacks {partition}/{kind} parts.")
            listed: set[Path] = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise SplitAccessError(
                        f"Malformed manifest entry in {partition}/{kind}."
                    )
                path = _safe_listed_path(root, str(entry.get("path", "")))
                if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                    raise SplitAccessError(f"Sealed split part mismatch: {path}")
                listed.add(path)
            actual = {
                path.resolve()
                for path in (root / partition / directory).glob("*.parquet")
            }
            if listed != actual:
                raise SplitAccessError(f"Unlisted/missing {partition}/{kind} parts.")

    train = (root / "train" / "features").resolve()
    validation = (root / "validation" / "features").resolve()
    return TrainingSplitAccess(
        split_root=str(root.resolve()),
        train_features=str(train),
        validation_features=str(validation),
        split_manifest_sha256=manifest_sha,
        split_manifest_hash=logical_hash,
        row_counts={key: int(value) for key, value in manifest["row_counts"].items()},
        benign_prevalence={
            key: float(value) for key, value in manifest["benign_prevalence"].items()
        },
    )


class SealedTestLedger:
    """Exclusive one-shot authorization ledger for Phase 5 test access."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._claim_token: str | None = None

    def claim(
        self,
        *,
        identity: ExperimentIdentity,
        split_root: Path,
        champion_bundle_path: Path,
    ) -> str:
        """Atomically claim the one permitted sealed-test evaluation."""
        if self._claim_token is not None:
            raise SplitAccessError("This ledger object already holds a test claim.")
        root = Path(split_root)
        champion = Path(champion_bundle_path)
        if not champion.is_file():
            raise SplitAccessError("Champion bundle is missing before test claim.")
        training_access = verify_training_split_access(root)
        token = uuid.uuid4().hex
        value = {
            "status": "claimed_test_must_not_be_reopened",
            "claim_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "claimed_at_utc": _utc_now(),
            "experiment": identity.to_dict(),
            "split_manifest_sha256": training_access.split_manifest_sha256,
            "champion_bundle_sha256": sha256_file(champion),
            "completed_at_utc": None,
            "output_artifacts": None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise SplitAccessError(
                "The sealed test has already been claimed for this experiment. "
                "A rerun requires a new experiment/split version."
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._claim_token = token
        return token

    def authorized_test_features(self, split_root: Path, *, token: str) -> Path:
        """Return the test path only to the live claimant."""
        if token != self._claim_token:
            raise SplitAccessError("Invalid live sealed-test claim token.")
        return Path(split_root).resolve() / "test" / "features"

    def complete(self, *, token: str, output_artifacts: dict[str, str]) -> Path:
        """Close the one-shot ledger with checksummed final outputs."""
        if token != self._claim_token:
            raise SplitAccessError("Invalid live sealed-test claim token.")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        expected_token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if value.get("claim_token_sha256") != expected_token_hash:
            raise SplitAccessError("Sealed-test ledger claim token was tampered.")
        if value.get("status") != "claimed_test_must_not_be_reopened":
            raise SplitAccessError("Sealed-test ledger is not in claimed state.")
        checked: dict[str, dict[str, str]] = {}
        for name, raw_path in output_artifacts.items():
            path = Path(raw_path)
            if not path.is_file():
                raise SplitAccessError(f"Required Phase 5 output is missing: {path}")
            try:
                display = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                display = str(path.resolve())
            checked[name] = {"path": display, "sha256": sha256_file(path)}
        value.update(
            {
                "status": "completed_test_closed",
                "completed_at_utc": _utc_now(),
                "output_artifacts": checked,
            }
        )
        atomic_write_json(self.path, value)
        self._claim_token = None
        return self.path
