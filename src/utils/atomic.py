"""Atomic writers for manifests and scientific control-plane artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


def atomic_write_text(path: Path, value: str, *, encoding: str = "utf-8") -> Path:
    """Replace a text file atomically after flushing it to stable storage."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return atomic_write_text(
        path,
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
    )


def atomic_write_via(path: Path, writer: Callable[[Path], None]) -> Path:
    """Atomically publish a binary/tabular artifact produced by ``writer``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise RuntimeError("Atomic writer did not create its temporary artifact.")
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
