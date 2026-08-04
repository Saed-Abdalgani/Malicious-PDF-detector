"""Bounded logical PDF object-graph inspection without stream decoding."""

from __future__ import annotations

import io
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.features.structural import MAX_PDF_BYTES
from src.features.filters import inspect_bounded_object_streams


@dataclass(frozen=True)
class SemanticFeatures:
    reachable_action_count: float = 0.0
    reachable_javascript_count: float = 0.0
    reachable_launch_count: float = 0.0
    reachable_uri_count: float = 0.0
    reachable_submitform_count: float = 0.0
    reachable_embedded_file_count: float = 0.0
    max_filter_chain_depth: float = 0.0
    max_graph_depth: float = 0.0
    visited_objects: float = 0.0
    parse_failure: float = 0.0
    extraction_timeout: float = 0.0
    extraction_limit_reached: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _walk(raw: bytes, max_nodes: int, max_depth: int) -> SemanticFeatures:
    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(raw), strict=False)
    root = reader.trailer.get("/Root")
    stack: list[tuple[Any, int]] = [(root, 0)]
    seen_indirect: set[tuple[int, int]] = set()
    counts = {
        "action": 0,
        "javascript": 0,
        "launch": 0,
        "uri": 0,
        "submitform": 0,
        "embedded": 0,
        "filter_depth": 0,
        "graph_depth": 0,
    }
    visited = 0
    limit_reached = False
    while stack:
        value, depth = stack.pop()
        if visited >= max_nodes or depth > max_depth:
            limit_reached = True
            continue
        try:
            if hasattr(value, "idnum") and hasattr(value, "generation"):
                key = (int(value.idnum), int(value.generation))
                if key in seen_indirect:
                    continue
                seen_indirect.add(key)
                value = value.get_object()
            visited += 1
            counts["graph_depth"] = max(counts["graph_depth"], depth)
            if hasattr(value, "items"):
                action_type = str(value.get("/S", "")) if hasattr(value, "get") else ""
                if action_type:
                    counts["action"] += 1
                if action_type == "/JavaScript" or "/JS" in value:
                    counts["javascript"] += 1
                if action_type == "/Launch":
                    counts["launch"] += 1
                if action_type == "/URI":
                    counts["uri"] += 1
                if action_type == "/SubmitForm":
                    counts["submitform"] += 1
                if str(value.get("/Type", "")) == "/EmbeddedFile":
                    counts["embedded"] += 1
                filters = value.get("/Filter")
                if filters is not None:
                    filter_depth = len(filters) if isinstance(filters, (list, tuple)) else 1
                    counts["filter_depth"] = max(counts["filter_depth"], filter_depth)
                for key, child in value.items():
                    # Never call get_data; stream dictionaries are traversed as metadata only.
                    if str(key) not in {"/Length"}:
                        stack.append((child, depth + 1))
            elif isinstance(value, (list, tuple)):
                stack.extend((child, depth + 1) for child in value)
        except Exception:
            continue
    return SemanticFeatures(
        reachable_action_count=float(counts["action"]),
        reachable_javascript_count=float(counts["javascript"]),
        reachable_launch_count=float(counts["launch"]),
        reachable_uri_count=float(counts["uri"]),
        reachable_submitform_count=float(counts["submitform"]),
        reachable_embedded_file_count=float(counts["embedded"]),
        max_filter_chain_depth=float(counts["filter_depth"]),
        max_graph_depth=float(counts["graph_depth"]),
        visited_objects=float(visited),
        extraction_limit_reached=float(limit_reached),
    )


def extract_semantic_features(
    pdf_path: str | Path,
    *,
    timeout_sec: int = 10,
    max_nodes: int = 25_000,
    max_depth: int = 64,
) -> SemanticFeatures:
    path = Path(pdf_path)
    if not path.is_file() or path.stat().st_size > MAX_PDF_BYTES:
        return SemanticFeatures(parse_failure=1.0, extraction_limit_reached=1.0)
    raw = path.read_bytes()
    object_streams = inspect_bounded_object_streams(raw)
    result: list[SemanticFeatures | None] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = _walk(raw, max_nodes, max_depth)
        except BaseException as exc:
            error[0] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        return SemanticFeatures(
            max_filter_chain_depth=float(object_streams.maximum_filter_chain),
            extraction_timeout=1.0,
            extraction_limit_reached=float(object_streams.limit_reached),
        )
    if error[0] is not None or result[0] is None:
        return SemanticFeatures(
            max_filter_chain_depth=float(object_streams.maximum_filter_chain),
            parse_failure=1.0,
            extraction_limit_reached=float(
                object_streams.limit_reached or object_streams.decode_failures > 0
            ),
        )
    walked = result[0]
    values = walked.to_dict()
    values["max_filter_chain_depth"] = max(
        values["max_filter_chain_depth"], float(object_streams.maximum_filter_chain)
    )
    values["extraction_limit_reached"] = float(
        bool(values["extraction_limit_reached"])
        or object_streams.limit_reached
        or object_streams.decode_failures > 0
    )
    return SemanticFeatures(**values)
