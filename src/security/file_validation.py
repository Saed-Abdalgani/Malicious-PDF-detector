"""Small, dependency-free file-envelope checks for uploaded PDFs."""

from __future__ import annotations

import re
from dataclasses import dataclass


_PDF_HEADER = re.compile(rb"%PDF-(?:1\.[0-7]|2\.0)(?:\s|%)")


@dataclass(frozen=True)
class PDFEnvelopeValidation:
    valid: bool
    mime_type: str
    reason: str


def validate_pdf_envelope(data: bytes, *, maximum_bytes: int = 50 * 1024 * 1024) -> PDFEnvelopeValidation:
    """Validate size/header/EOF markers without parsing or executing content."""
    if not data:
        return PDFEnvelopeValidation(False, "application/octet-stream", "empty_file")
    if len(data) > maximum_bytes:
        return PDFEnvelopeValidation(False, "application/pdf", "file_size_limit")
    if _PDF_HEADER.search(data[:1024]) is None:
        return PDFEnvelopeValidation(False, "application/octet-stream", "invalid_pdf_header")
    if b"%%EOF" not in data[-4096:]:
        return PDFEnvelopeValidation(False, "application/pdf", "missing_pdf_eof")
    return PDFEnvelopeValidation(True, "application/pdf", "ok")
