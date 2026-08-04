"""Bounded PDF lexical canonicalization that never decodes stream bodies."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_WHITESPACE = re.compile(rb"[\x00\x09\x0a\x0c\x0d\x20]+")
_STREAM_START = re.compile(
    rb">>[\x00\x09\x0a\x0c\x0d\x20]+stream[\x20\x09]*(?:\r\n|\n|\r)"
)
_STREAM_END = re.compile(rb"(?:\r\n|\n|\r)?endstream\b")
_DELIMITERS = set(b"\x00\x09\x0a\x0c\x0d ()<>[]{}/%")
_HEX = set(b"0123456789abcdefABCDEF")


@dataclass(frozen=True)
class CanonicalizationReport:
    input_bytes: int
    scanned_bytes: int
    output_bytes: int
    decoded_name_escapes: int
    comments_removed: int
    stream_bodies_skipped: int
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _canonicalize_non_stream(value: bytes) -> tuple[bytes, int, int]:
    """Canonicalize lexical syntax while keeping strings opaque."""
    output = bytearray()
    escapes = comments = 0
    index = 0
    while index < len(value):
        current = value[index]
        if current == ord("%"):
            comments += 1
            index += 1
            while index < len(value) and value[index] not in (10, 13):
                index += 1
            output.extend(b" ")
            continue
        if current == ord("("):
            # Literal strings can contain `/JavaScript`, `%`, and parentheses;
            # none are structural tokens. Respect escapes and nesting.
            depth = 1
            index += 1
            while index < len(value) and depth:
                if value[index] == ord("\\"):
                    index += 2
                    continue
                if value[index] == ord("("):
                    depth += 1
                elif value[index] == ord(")"):
                    depth -= 1
                index += 1
            output.extend(b" ")
            continue
        if current == ord("<") and index + 1 < len(value) and value[index + 1] == ord("<"):
            output.extend(b"<<")
            index += 2
            continue
        if current == ord("<"):
            # Hex strings are data, while `<<` above is a dictionary delimiter.
            index += 1
            while index < len(value) and value[index] != ord(">"):
                index += 1
            index += int(index < len(value))
            output.extend(b" ")
            continue
        if current == ord("/"):
            output.append(current)
            index += 1
            while index < len(value) and value[index] not in _DELIMITERS:
                if (
                    value[index] == ord("#")
                    and index + 2 < len(value)
                    and value[index + 1] in _HEX
                    and value[index + 2] in _HEX
                ):
                    output.append(int(value[index + 1 : index + 3], 16))
                    escapes += 1
                    index += 3
                else:
                    output.append(value[index])
                    index += 1
            continue
        output.append(current)
        index += 1
    return _WHITESPACE.sub(b" ", bytes(output)), escapes, comments


def canonicalize_pdf_syntax(
    raw_bytes: bytes,
    *,
    max_scan_bytes: int = 50 * 1024 * 1024,
) -> tuple[bytes, CanonicalizationReport]:
    """Normalize names/comments/whitespace outside opaque PDF streams.

    Stream bodies are replaced with one space. This prevents compressed or
    arbitrary payload bytes from masquerading as structural names while keeping
    the surrounding ``stream``/``endstream`` markers observable.
    """
    scanned = raw_bytes[:max_scan_bytes]
    truncated = len(raw_bytes) > len(scanned)
    output: list[bytes] = []
    position = decoded = comments = skipped = 0
    while position < len(scanned):
        start = _STREAM_START.search(scanned, position)
        if start is None:
            chunk, count, comment_count = _canonicalize_non_stream(scanned[position:])
            output.append(chunk)
            decoded += count
            comments += comment_count
            break
        prefix, count, comment_count = _canonicalize_non_stream(scanned[position:start.end()])
        output.append(prefix)
        decoded += count
        comments += comment_count
        end = _STREAM_END.search(scanned, start.end())
        if end is None:
            # Unterminated stream: do not inspect its body.
            output.append(b" ")
            skipped += 1
            break
        output.append(b" endstream ")
        skipped += 1
        position = end.end()
    canonical = _WHITESPACE.sub(b" ", b"".join(output)).strip()
    report = CanonicalizationReport(
        input_bytes=len(raw_bytes),
        scanned_bytes=len(scanned),
        output_bytes=len(canonical),
        decoded_name_escapes=decoded,
        comments_removed=comments,
        stream_bodies_skipped=skipped,
        truncated=truncated,
    )
    return canonical, report
