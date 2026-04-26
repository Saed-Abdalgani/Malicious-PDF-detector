"""
metadata.py
-----------
Extracts 12 general / metadata features from PDF files using PyPDF2
for high-level document properties and ``os.path`` for file-system
attributes.

These features complement the low-level structural features extracted by
``structural.py``, capturing document-level characteristics like page count,
encryption status, font usage, and embedded file counts that help
differentiate benign documents from malicious ones.

Features extracted:
    File-level (1):       pdf_size
    Metadata (3):         title_chars, is_encrypted, metadata_size
    Document stats (4):   page_count, has_text, image_count, obj_count_total
    Object analysis (3):  font_obj_count, embedded_file_count,
                          avg_embedded_media_size
    Header validation (1): header_valid

Usage:
    from src.features.metadata import extract_metadata_features
    features = extract_metadata_features("path/to/file.pdf")
"""

import os
import re
from pathlib import Path
from typing import Dict, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Default feature dictionary — returned on unrecoverable errors
# ---------------------------------------------------------------------------

_META_FEATURE_KEYS = [
    "pdf_size",
    "title_chars",
    "is_encrypted",
    "metadata_size",
    "page_count",
    "has_text",
    "image_count",
    "obj_count_total",
    "font_obj_count",
    "embedded_file_count",
    "avg_embedded_media_size",
    "header_valid",
]


def _zeroed_features() -> Dict[str, float]:
    """Return a feature dict with all 12 metadata features set to 0.0."""
    return {k: 0.0 for k in _META_FEATURE_KEYS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_header(raw_bytes: bytes) -> int:
    """Validate the PDF header.

    Checks for the standard ``%PDF-1.X`` or ``%PDF-2.X`` signature
    in the first 1024 bytes.

    Returns:
        1 if a valid PDF header is found, 0 otherwise.
    """
    header_region = raw_bytes[:1024]
    return 1 if re.search(rb"%PDF-\d+\.\d+", header_region) else 0


def _count_pattern_in_bytes(raw_bytes: bytes, pattern: bytes) -> int:
    """Count occurrences of a byte pattern in raw bytes."""
    return len(re.findall(pattern, raw_bytes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_metadata_features(
    pdf_path: Union[str, Path],
) -> Dict[str, float]:
    """Extract 12 general / metadata features from a PDF file.

    Uses ``PyPDF2.PdfReader`` for parsed metadata and page-level
    inspection, with a raw-byte fallback for header validation and
    object counting.  Handles corrupted, encrypted, and malformed PDFs
    gracefully by returning a zeroed feature dict on failure.

    Args:
        pdf_path: Absolute or relative path to a PDF file.

    Returns:
        dict: A dictionary mapping 12 feature names to their numeric
        (float) values.

    Example:
        >>> feats = extract_metadata_features("data/sample_pdfs/benign.pdf")
        >>> len(feats)
        12
        >>> feats["page_count"]
        3.0
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return _zeroed_features()

    features: Dict[str, float] = {}

    try:
        # --- File-system level ---
        features["pdf_size"] = float(os.path.getsize(pdf_path))

        # Read raw bytes for header validation and fallback counting
        raw_bytes = pdf_path.read_bytes()

        # Header validity
        features["header_valid"] = float(_check_header(raw_bytes))

        # --- PyPDF2 parsing ---
        try:
            from PyPDF2 import PdfReader
            from PyPDF2.errors import PdfReadError
        except ImportError:
            logger.error(
                "PyPDF2 is not installed. "
                "Run: pip install PyPDF2>=3.0.0"
            )
            features.update({k: 0.0 for k in _META_FEATURE_KEYS
                             if k not in features})
            return features

        try:
            reader = PdfReader(str(pdf_path))
        except (PdfReadError, Exception) as exc:
            logger.warning(
                f"PyPDF2 could not parse {pdf_path.name}: {exc}. "
                f"Falling back to raw-byte heuristics."
            )
            # Fallback: fill remaining features with byte-level heuristics
            features["title_chars"] = 0.0
            features["is_encrypted"] = 0.0
            features["metadata_size"] = 0.0
            features["page_count"] = 0.0
            features["has_text"] = 0.0
            features["image_count"] = 0.0
            features["obj_count_total"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"\b\d+\s+\d+\s+obj\b")
            )
            features["font_obj_count"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"/Font\b")
            )
            features["embedded_file_count"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"/EmbeddedFile\b")
            )
            features["avg_embedded_media_size"] = 0.0
            return features

        # --- Encryption status ---
        features["is_encrypted"] = 1.0 if reader.is_encrypted else 0.0

        # If encrypted, try decrypting with empty password for metadata access
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass  # continue with whatever is accessible

        # --- Title characters ---
        try:
            metadata = reader.metadata
            if metadata and metadata.title:
                features["title_chars"] = float(len(str(metadata.title)))
            else:
                features["title_chars"] = 0.0
        except Exception:
            features["title_chars"] = 0.0

        # --- Metadata size ---
        try:
            if metadata:
                meta_str = str(metadata)
                features["metadata_size"] = float(len(meta_str))
            else:
                features["metadata_size"] = 0.0
        except Exception:
            features["metadata_size"] = 0.0

        # --- Page count ---
        try:
            features["page_count"] = float(len(reader.pages))
        except Exception:
            features["page_count"] = 0.0

        # --- Has extractable text ---
        try:
            has_text = False
            # Check first 5 pages (or fewer) for text content
            max_pages = min(len(reader.pages), 5)
            for i in range(max_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text and text.strip():
                    has_text = True
                    break
            features["has_text"] = 1.0 if has_text else 0.0
        except Exception:
            features["has_text"] = 0.0

        # --- Image count ---
        try:
            image_count = 0
            for page in reader.pages:
                if "/XObject" in (page.get("/Resources") or {}):
                    x_objects = page["/Resources"]["/XObject"].get_object()
                    for obj_name in x_objects:
                        x_obj = x_objects[obj_name].get_object()
                        if x_obj.get("/Subtype") == "/Image":
                            image_count += 1
            features["image_count"] = float(image_count)
        except Exception:
            # Fallback: count /Image patterns in raw bytes
            features["image_count"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"/Image\b")
            )

        # --- Total object count ---
        try:
            # Use the raw byte pattern for reliability
            features["obj_count_total"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"\b\d+\s+\d+\s+obj\b")
            )
        except Exception:
            features["obj_count_total"] = 0.0

        # --- Font object count ---
        try:
            font_count = 0
            for page in reader.pages:
                resources = page.get("/Resources")
                if resources and "/Font" in resources:
                    fonts = resources["/Font"].get_object()
                    font_count += len(fonts)
            features["font_obj_count"] = float(font_count)
        except Exception:
            features["font_obj_count"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"/Font\b")
            )

        # --- Embedded file count ---
        try:
            embedded_count = 0
            embedded_sizes = []

            # Check the document's names tree for embedded files
            if reader.trailer and "/Root" in reader.trailer:
                root = reader.trailer["/Root"].get_object()
                if "/Names" in root:
                    names = root["/Names"].get_object()
                    if "/EmbeddedFiles" in names:
                        ef = names["/EmbeddedFiles"].get_object()
                        if "/Names" in ef:
                            name_list = ef["/Names"]
                            # Name list is [name, ref, name, ref, ...]
                            for i in range(1, len(name_list), 2):
                                embedded_count += 1
                                try:
                                    file_spec = name_list[i].get_object()
                                    if "/EF" in file_spec:
                                        ef_dict = file_spec["/EF"].get_object()
                                        if "/F" in ef_dict:
                                            stream = ef_dict["/F"].get_object()
                                            if hasattr(stream, "get_data"):
                                                embedded_sizes.append(
                                                    len(stream.get_data())
                                                )
                                except Exception:
                                    pass

            # Fallback to byte counting if we got zero
            if embedded_count == 0:
                embedded_count = _count_pattern_in_bytes(
                    raw_bytes, rb"/EmbeddedFile\b"
                )

            features["embedded_file_count"] = float(embedded_count)

            if embedded_sizes:
                features["avg_embedded_media_size"] = float(
                    sum(embedded_sizes) / len(embedded_sizes)
                )
            else:
                features["avg_embedded_media_size"] = 0.0

        except Exception:
            features["embedded_file_count"] = float(
                _count_pattern_in_bytes(raw_bytes, rb"/EmbeddedFile\b")
            )
            features["avg_embedded_media_size"] = 0.0

        logger.info(
            f"Metadata extraction complete for {pdf_path.name} — "
            f"{sum(1 for v in features.values() if v > 0)} / 12 "
            f"features non-zero"
        )
        return features

    except Exception as exc:
        logger.error(
            f"Metadata extraction failed for {pdf_path.name}: {exc}",
            exc_info=True,
        )
        return _zeroed_features()


# ---------------------------------------------------------------------------
# Module entrypoint for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.features.metadata <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_metadata_features(path)
    print(json.dumps(result, indent=2))
