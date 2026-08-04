"""Bounded metadata extraction without rendering or embedded-stream decoding."""

from __future__ import annotations

import io
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

from src.features.status import ExtractionStatus
from src.features.structural import MAX_PDF_BYTES
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_PAGES_INSPECTED = 100
MAX_RESOURCE_ENTRIES = 10_000

_META_FEATURE_KEYS = (
    "pdf_size", "title_chars", "is_encrypted", "metadata_size", "page_count",
    "has_text", "image_count", "obj_count_total", "font_obj_count",
    "embedded_file_count", "avg_embedded_media_size", "header_valid",
)
_OBJECT = re.compile(rb"\b\d+\s+\d+\s+obj\b")
_FONT = re.compile(rb"/Font\b")
_IMAGE = re.compile(rb"/Image\b")
_EMBEDDED = re.compile(rb"/EmbeddedFile\b")
_HEADER = re.compile(rb"%PDF-\d+\.\d+")


@dataclass(frozen=True)
class MetadataExtraction:
    features: dict[str, float]
    status: ExtractionStatus


class _MetadataTimeout(Exception):
    pass


def _zeroed_features() -> Dict[str, float]:
    return {name: 0.0 for name in _META_FEATURE_KEYS}


def _raw_fallback(raw: bytes, size: int) -> dict[str, float]:
    features = _zeroed_features()
    features.update(
        pdf_size=float(size),
        obj_count_total=float(len(_OBJECT.findall(raw))),
        font_obj_count=float(len(_FONT.findall(raw))),
        image_count=float(len(_IMAGE.findall(raw))),
        embedded_file_count=float(len(_EMBEDDED.findall(raw))),
        has_text=float(bool(_FONT.search(raw))),
        header_valid=float(bool(_HEADER.search(raw[:1024]))),
    )
    return features


def _resolve(value):
    return value.get_object() if hasattr(value, "get_object") else value


def _declared_stream_length(value) -> int | None:
    try:
        resolved = _resolve(value)
        length = resolved.get("/Length") if hasattr(resolved, "get") else None
        length = _resolve(length)
        if isinstance(length, (int, float)) and 0 <= int(length) <= MAX_PDF_BYTES:
            return int(length)
    except Exception:
        return None
    return None


def _parse(raw: bytes, size: int) -> dict[str, float]:
    from PyPDF2 import PdfReader

    features = _raw_fallback(raw, size)
    reader = PdfReader(io.BytesIO(raw), strict=False)
    features["is_encrypted"] = float(reader.is_encrypted)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return features
    try:
        metadata = reader.metadata
        features["title_chars"] = float(len(str(metadata.title))) if metadata and metadata.title else 0.0
        features["metadata_size"] = float(len(str(metadata))) if metadata else 0.0
    except Exception:
        pass
    try:
        page_count = len(reader.pages)
        features["page_count"] = float(page_count)
    except Exception:
        page_count = 0
    try:
        xref_objects = sum(
            len(objects) for objects in getattr(reader, "xref", {}).values()
        )
        object_stream_members = len(getattr(reader, "xref_objStm", {}))
        parsed_object_count = xref_objects + object_stream_members
        if parsed_object_count > 0:
            features["obj_count_total"] = float(parsed_object_count)
    except Exception:
        # The raw scanner fallback remains visible for parser disagreement.
        pass

    fonts = images = 0
    for page_index in range(min(page_count, MAX_PAGES_INSPECTED)):
        try:
            page = reader.pages[page_index]
            resources = _resolve(page.get("/Resources") or {})
            font_map = _resolve(resources.get("/Font") or {}) if hasattr(resources, "get") else {}
            xobjects = _resolve(resources.get("/XObject") or {}) if hasattr(resources, "get") else {}
            fonts += min(len(font_map), MAX_RESOURCE_ENTRIES) if hasattr(font_map, "__len__") else 0
            if hasattr(xobjects, "items"):
                for _, value in list(xobjects.items())[:MAX_RESOURCE_ENTRIES]:
                    obj = _resolve(value)
                    if hasattr(obj, "get") and str(obj.get("/Subtype")) == "/Image":
                        images += 1
        except Exception:
            continue
    if page_count <= MAX_PAGES_INSPECTED:
        features["font_obj_count"] = float(fonts)
        features["image_count"] = float(images)
    # Static text-likelihood only; no content stream is decompressed.
    features["has_text"] = float(fonts > 0 or bool(_FONT.search(raw)))

    embedded_count = 0
    declared_sizes: list[int] = []
    try:
        root = _resolve(reader.trailer.get("/Root"))
        names = _resolve(root.get("/Names")) if hasattr(root, "get") else None
        embedded = _resolve(names.get("/EmbeddedFiles")) if hasattr(names, "get") else None
        name_list = _resolve(embedded.get("/Names")) if hasattr(embedded, "get") else None
        if isinstance(name_list, (list, tuple)):
            for index in range(1, min(len(name_list), MAX_RESOURCE_ENTRIES * 2), 2):
                embedded_count += 1
                file_spec = _resolve(name_list[index])
                ef = _resolve(file_spec.get("/EF")) if hasattr(file_spec, "get") else None
                stream = ef.get("/F") if hasattr(ef, "get") else None
                length = _declared_stream_length(stream)
                if length is not None:
                    declared_sizes.append(length)
    except Exception:
        pass
    if embedded_count:
        features["embedded_file_count"] = float(embedded_count)
    features["avg_embedded_media_size"] = (
        float(sum(declared_sizes) / len(declared_sizes)) if declared_sizes else 0.0
    )
    return features


def _parse_with_timeout(raw: bytes, size: int, timeout_sec: int) -> dict[str, float]:
    result: list[dict[str, float] | None] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = _parse(raw, size)
        except BaseException as exc:
            error[0] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        raise _MetadataTimeout
    if error[0] is not None:
        raise error[0]
    assert result[0] is not None
    return result[0]


def extract_metadata_features_with_status(
    pdf_path: Union[str, Path],
    *,
    timeout_sec: int = 15,
) -> MetadataExtraction:
    path = Path(pdf_path)
    if not path.is_file():
        return MetadataExtraction(
            _zeroed_features(),
            ExtractionStatus(ok=False, parse_failure=True, error_code="file_not_found"),
        )
    size = path.stat().st_size
    if size > MAX_PDF_BYTES:
        features = _zeroed_features()
        features["pdf_size"] = float(size)
        return MetadataExtraction(
            features,
            ExtractionStatus(
                ok=False, file_too_large=True, limit_reached=True,
                error_code="file_size_limit",
            ),
        )
    try:
        raw = path.read_bytes()
        features = _parse_with_timeout(raw, size, timeout_sec)
        invalid_header = features["header_valid"] == 0
        invalid_eof = b"%%EOF" not in raw[-4096:]
        status = ExtractionStatus(
            ok=not (invalid_header or invalid_eof),
            invalid_header=invalid_header,
            invalid_eof=invalid_eof,
            recovery_mode=False,
            limit_reached=features["page_count"] > MAX_PAGES_INSPECTED,
            error_code="none" if not invalid_header else "invalid_header",
        )
        return MetadataExtraction(features, status)
    except _MetadataTimeout:
        raw = path.read_bytes()
        return MetadataExtraction(
            _raw_fallback(raw, size),
            ExtractionStatus(ok=False, timeout=True, error_code="timeout"),
        )
    except Exception as exc:
        raw = path.read_bytes()
        logger.info("Metadata parser fallback for %s: %s", path.name, exc)
        return MetadataExtraction(
            _raw_fallback(raw, size),
            ExtractionStatus(
                ok=False, parse_failure=True, recovery_mode=True,
                error_code=type(exc).__name__,
            ),
        )


def extract_metadata_features(pdf_path: Union[str, Path]) -> dict[str, float]:
    """Compatibility API returning the stable 12 metadata features."""
    return extract_metadata_features_with_status(pdf_path).features


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.features.metadata <pdf_path>")
    result = extract_metadata_features_with_status(sys.argv[1])
    print(json.dumps({"features": result.features, "status": result.status.to_dict()}, indent=2))
