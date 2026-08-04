"""Typed, model-visible status emitted by bounded static PDF extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractionStatus:
    ok: bool = True
    parse_failure: bool = False
    recovery_mode: bool = False
    truncated_structure: bool = False
    invalid_header: bool = False
    invalid_eof: bool = False
    invalid_xref: bool = False
    timeout: bool = False
    parser_disagreement: bool = False
    file_too_large: bool = False
    limit_reached: bool = False
    error_code: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def numeric_features(self) -> dict[str, float]:
        return {
            "parse_failure": float(self.parse_failure),
            "recovery_mode": float(self.recovery_mode),
            "truncated_structure": float(self.truncated_structure),
            "invalid_header": float(self.invalid_header),
            "invalid_eof": float(self.invalid_eof),
            "invalid_xref": float(self.invalid_xref),
            "extraction_timeout": float(self.timeout),
            "parser_disagreement": float(self.parser_disagreement),
            "file_too_large": float(self.file_too_large),
            "extraction_limit_reached": float(self.limit_reached),
        }
