"""Allowlisted, resource-bounded PDF stream decoding for static inspection."""

from __future__ import annotations

import base64
import binascii
import re
import zlib
from dataclasses import asdict, dataclass
from typing import Sequence


class BoundedDecodeError(ValueError):
    """Raised when a stream/filter violates a static-analysis safety limit."""


@dataclass(frozen=True)
class DecodeLimits:
    maximum_compressed_bytes: int = 4 * 1024 * 1024
    maximum_decoded_bytes: int = 8 * 1024 * 1024
    maximum_expansion_ratio: float = 100.0
    maximum_filter_chain: int = 4


@dataclass(frozen=True)
class ObjectStreamInspection:
    object_streams_seen: int
    object_streams_validated: int
    decode_failures: int
    maximum_filter_chain: int
    limit_reached: bool

    def to_dict(self) -> dict:
        return asdict(self)


_ALIASES = {
    "/Fl": "/FlateDecode",
    "/AHx": "/ASCIIHexDecode",
    "/A85": "/ASCII85Decode",
    "/RL": "/RunLengthDecode",
}
_SUPPORTED = {"/FlateDecode", "/ASCIIHexDecode", "/ASCII85Decode", "/RunLengthDecode"}


def _run_length_decode(data: bytes, maximum: int) -> bytes:
    output = bytearray()
    index = 0
    while index < len(data):
        length = data[index]
        index += 1
        if length == 128:
            break
        if length <= 127:
            count = length + 1
            output.extend(data[index : index + count])
            index += count
        else:
            if index >= len(data):
                raise BoundedDecodeError("Truncated RunLengthDecode repeat.")
            output.extend(data[index : index + 1] * (257 - length))
            index += 1
        if len(output) > maximum:
            raise BoundedDecodeError("Decoded stream exceeds byte limit.")
    return bytes(output)


def _decode_once(data: bytes, filter_name: str, maximum: int) -> bytes:
    name = _ALIASES.get(filter_name, filter_name)
    if name not in _SUPPORTED:
        raise BoundedDecodeError(f"Unsupported or risky PDF filter: {filter_name}")
    if name == "/FlateDecode":
        decoder = zlib.decompressobj()
        output = decoder.decompress(data, maximum + 1)
        if decoder.unconsumed_tail or len(output) > maximum:
            raise BoundedDecodeError("Flate output exceeds byte limit.")
        output += decoder.flush(maximum + 1 - len(output))
        if len(output) > maximum:
            raise BoundedDecodeError("Flate output exceeds byte limit.")
        return output
    if name == "/ASCIIHexDecode":
        compact = re.sub(rb"\s+", b"", data).rstrip(b">")
        if len(compact) % 2:
            compact += b"0"
        if len(compact) // 2 > maximum:
            raise BoundedDecodeError("ASCIIHex output exceeds byte limit.")
        try:
            return binascii.unhexlify(compact)
        except binascii.Error as exc:
            raise BoundedDecodeError("Invalid ASCIIHex stream.") from exc
    if name == "/ASCII85Decode":
        try:
            output = base64.a85decode(data, adobe=True, ignorechars=b" \t\r\n\x0b")
        except (ValueError, binascii.Error) as exc:
            raise BoundedDecodeError("Invalid ASCII85 stream.") from exc
        if len(output) > maximum:
            raise BoundedDecodeError("ASCII85 output exceeds byte limit.")
        return output
    return _run_length_decode(data, maximum)


def bounded_decode_stream(
    data: bytes,
    filters: Sequence[str],
    *,
    limits: DecodeLimits | None = None,
) -> bytes:
    """Decode an allowlisted filter chain under strict cumulative limits."""
    active = limits or DecodeLimits()
    if len(data) > active.maximum_compressed_bytes:
        raise BoundedDecodeError("Compressed stream exceeds byte limit.")
    if len(filters) > active.maximum_filter_chain:
        raise BoundedDecodeError("Filter chain exceeds depth limit.")
    decoded = data
    original_size = max(len(data), 1)
    for filter_name in filters:
        decoded = _decode_once(decoded, str(filter_name), active.maximum_decoded_bytes)
        if len(decoded) / original_size > active.maximum_expansion_ratio:
            raise BoundedDecodeError("Stream exceeds expansion-ratio limit.")
    return decoded


_STREAM_OBJECT = re.compile(
    rb"\b\d+\s+\d+\s+obj\s*<<(.*?)>>\s*stream[\x20\x09]*(?:\r\n|\n|\r)"
    rb"(.*?)(?:\r\n|\n|\r)?endstream\b",
    re.DOTALL,
)
_NAME_TOKEN = re.compile(rb"/[A-Za-z0-9]+")


def _filter_names(dictionary: bytes) -> list[str]:
    array = re.search(rb"/Filter\s*\[(.*?)\]", dictionary, re.DOTALL)
    if array:
        return [name.decode("ascii", errors="strict") for name in _NAME_TOKEN.findall(array.group(1))]
    single = re.search(rb"/Filter\s*(/[A-Za-z0-9]+)", dictionary)
    return [single.group(1).decode("ascii")] if single else []


def inspect_bounded_object_streams(
    raw: bytes,
    *,
    limits: DecodeLimits | None = None,
    maximum_object_streams: int = 64,
) -> ObjectStreamInspection:
    """Decode only enough allowlisted ObjStm data to validate its object header."""
    active = limits or DecodeLimits()
    seen = validated = failures = max_chain = 0
    limit_reached = False
    for match in _STREAM_OBJECT.finditer(raw):
        dictionary, body = match.groups()
        if re.search(rb"/Type\s*/ObjStm\b", dictionary) is None:
            continue
        if seen >= maximum_object_streams:
            limit_reached = True
            break
        seen += 1
        filters = _filter_names(dictionary)
        max_chain = max(max_chain, len(filters))
        number = re.search(rb"/N\s+(\d+)", dictionary)
        first = re.search(rb"/First\s+(\d+)", dictionary)
        try:
            if number is None or first is None:
                raise BoundedDecodeError("ObjStm is missing /N or /First.")
            member_count = int(number.group(1))
            first_offset = int(first.group(1))
            if member_count < 0 or member_count > 100_000:
                raise BoundedDecodeError("ObjStm /N exceeds member limit.")
            decoded = bounded_decode_stream(body, filters, limits=active)
            if first_offset < 0 or first_offset > len(decoded):
                raise BoundedDecodeError("ObjStm /First is outside decoded data.")
            header_numbers = re.findall(rb"\d+", decoded[:first_offset])
            if len(header_numbers) < member_count * 2:
                raise BoundedDecodeError("ObjStm object-number/offset header is truncated.")
            validated += 1
        except (BoundedDecodeError, ValueError, UnicodeError):
            failures += 1
    return ObjectStreamInspection(seen, validated, failures, max_chain, limit_reached)
