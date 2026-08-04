"""Bounded static structural extraction with PDF-name canonicalization."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from src.features.canonicalize import CanonicalizationReport, canonicalize_pdf_syntax
from src.features.filters import ObjectStreamInspection, inspect_bounded_object_streams
from src.features.status import ExtractionStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_STREAMS = 100_000

_KEYWORD_PATTERNS: Dict[str, re.Pattern[bytes]] = {
    "js_count": re.compile(rb"/JS\b"),
    "javascript_count": re.compile(rb"/JavaScript\b"),
    "openaction_count": re.compile(rb"/OpenAction\b"),
    "action_count": re.compile(rb"/Action\b"),
    "aa_count": re.compile(rb"/AA\b"),
    "launch_count": re.compile(rb"/Launch\b"),
    "uri_count": re.compile(rb"/URI\b"),
    "submitform_count": re.compile(rb"/SubmitForm\b"),
    "acroform_count": re.compile(rb"/AcroForm\b"),
    "xfa_count": re.compile(rb"/XFA\b"),
    "richmedia_count": re.compile(rb"/RichMedia\b"),
    "jbig2decode_count": re.compile(rb"/JBig2Decode\b", re.IGNORECASE),
    "colors_count": re.compile(rb"/Colors\b"),
}
_STRUCT_PATTERNS: Dict[str, re.Pattern[bytes]] = {
    "obj_count": re.compile(rb"\b\d+\s+\d+\s+obj\b"),
    "endobj_count": re.compile(rb"\bendobj\b"),
    "stream_count": re.compile(rb"\bstream\b"),
    "endstream_count": re.compile(rb"\bendstream\b"),
    "xref_count": re.compile(rb"\bxref\b"),
    "trailer_count": re.compile(rb"\btrailer\b"),
    "startxref_count": re.compile(rb"\bstartxref\b"),
}
_MISC_PATTERNS: Dict[str, re.Pattern[bytes]] = {
    "objstm_count": re.compile(rb"/ObjStm\b"),
    "filter_count": re.compile(rb"/Filter\b"),
}
_OBFUSCATION_PATTERN = re.compile(rb"#[0-9A-Fa-f]{2}")
_INDIRECT_OBJ_PATTERN = re.compile(rb"\b\d+\s+\d+\s+R\b")
_STREAM_CONTENT_PATTERN = re.compile(
    rb"\bstream[\x20\x09]*(?:\r\n|\n|\r)(.*?)(?:\r\n|\n|\r)?endstream\b",
    re.DOTALL,
)
_HEADER_PATTERN = re.compile(rb"%PDF-\d+\.\d+")


class _TimeoutError(Exception):
    pass


@dataclass(frozen=True)
class StructuralExtraction:
    features: dict[str, float]
    status: ExtractionStatus
    canonicalization: CanonicalizationReport | None
    object_stream_inspection: ObjectStreamInspection | None = None


def _run_with_timeout(func, *, args: tuple = (), timeout_sec: int = 30):
    result: list[object | None] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = func(*args)
        except BaseException as exc:  # preserve typed extraction failure
            error[0] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise _TimeoutError(f"Structural extraction exceeded {timeout_sec}s.")
    if error[0] is not None:
        raise error[0]
    return result[0]


def _zeroed_features() -> dict[str, float]:
    keys = [
        *_KEYWORD_PATTERNS,
        *_STRUCT_PATTERNS,
        *_MISC_PATTERNS,
        "obfuscation_count",
        "avg_stream_size",
        "indirect_obj_count",
    ]
    return {name: 0.0 for name in keys}


def _extract(
    raw_bytes: bytes,
) -> tuple[dict[str, float], CanonicalizationReport, bool, ObjectStreamInspection]:
    canonical, report = canonicalize_pdf_syntax(raw_bytes, max_scan_bytes=MAX_PDF_BYTES)
    features: dict[str, float] = {}
    for name, pattern in _KEYWORD_PATTERNS.items():
        features[name] = float(len(pattern.findall(canonical)))
    for name, pattern in _STRUCT_PATTERNS.items():
        features[name] = float(len(pattern.findall(canonical)))
    for name, pattern in _MISC_PATTERNS.items():
        features[name] = float(len(pattern.findall(canonical)))
    # Count the original evasive spelling, not its canonical replacement.
    features["obfuscation_count"] = float(len(_OBFUSCATION_PATTERN.findall(raw_bytes)))
    features["indirect_obj_count"] = float(len(_INDIRECT_OBJ_PATTERN.findall(canonical)))
    sizes: list[int] = []
    for index, match in enumerate(_STREAM_CONTENT_PATTERN.finditer(raw_bytes)):
        if index >= MAX_STREAMS:
            break
        sizes.append(len(match.group(1)))
    features["avg_stream_size"] = float(sum(sizes) / len(sizes)) if sizes else 0.0
    has_xref_stream = re.search(rb"/Type\s*/XRef\b", canonical) is not None
    object_streams = inspect_bounded_object_streams(raw_bytes)
    return features, report, has_xref_stream, object_streams


def _read_bounded(path: Path) -> tuple[bytes, bool]:
    size = path.stat().st_size
    if size > MAX_PDF_BYTES:
        return b"", True
    with path.open("rb") as handle:
        return handle.read(MAX_PDF_BYTES + 1), False


def extract_structural_features_with_status(
    pdf_path: Union[str, Path],
    timeout_sec: int = 30,
) -> StructuralExtraction:
    """Extract 25 features plus typed health status; never execute content."""
    path = Path(pdf_path)
    if not path.is_file():
        return StructuralExtraction(
            _zeroed_features(),
            ExtractionStatus(ok=False, parse_failure=True, error_code="file_not_found"),
            None,
        )
    try:
        raw, too_large = _read_bounded(path)
        if too_large:
            return StructuralExtraction(
                _zeroed_features(),
                ExtractionStatus(
                    ok=False, file_too_large=True, limit_reached=True,
                    error_code="file_size_limit",
                ),
                None,
            )
        features, report, has_xref_stream, object_streams = _run_with_timeout(
            _extract, args=(raw,), timeout_sec=timeout_sec
        )
        invalid_header = _HEADER_PATTERN.search(raw[:1024]) is None
        invalid_eof = b"%%EOF" not in raw[-4096:]
        truncated_structure = (
            features["obj_count"] != features["endobj_count"]
            or features["stream_count"] != features["endstream_count"]
            or report.truncated
        )
        invalid_xref = (
            features["startxref_count"] == 0
            or (features["xref_count"] == 0 and not has_xref_stream)
        )
        status = ExtractionStatus(
            ok=not (invalid_header or invalid_eof or truncated_structure),
            truncated_structure=truncated_structure,
            invalid_header=invalid_header,
            invalid_eof=invalid_eof,
            invalid_xref=invalid_xref,
            limit_reached=(
                report.truncated
                or features["stream_count"] > MAX_STREAMS
                or object_streams.limit_reached
                or object_streams.decode_failures > 0
            ),
            error_code="none" if not (invalid_header or invalid_eof) else "invalid_pdf_envelope",
        )
        return StructuralExtraction(features, status, report, object_streams)
    except _TimeoutError:
        return StructuralExtraction(
            _zeroed_features(),
            ExtractionStatus(ok=False, timeout=True, error_code="timeout"),
            None,
        )
    except Exception as exc:
        logger.warning("Structural extraction failed for %s: %s", path.name, exc)
        return StructuralExtraction(
            _zeroed_features(),
            ExtractionStatus(ok=False, parse_failure=True, error_code=type(exc).__name__),
            None,
        )


def extract_structural_features(
    pdf_path: Union[str, Path], timeout_sec: int = 30
) -> dict[str, float]:
    """Compatibility API returning only the stable 25-feature dictionary."""
    return extract_structural_features_with_status(pdf_path, timeout_sec).features


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.features.structural <pdf_path>")
    result = extract_structural_features_with_status(sys.argv[1])
    print(json.dumps({"features": result.features, "status": result.status.to_dict()}, indent=2))
