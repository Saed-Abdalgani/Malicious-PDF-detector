"""Safe, bounded PDF adversarial fixtures and robustness evaluation.

All fixtures are generated locally, contain no external links, files, launch
actions, or harmful script, and are parsed but never rendered.  Feature-space
stress tests are deliberately kept separate from valid-PDF mutations.
"""

from __future__ import annotations

import hashlib
import binascii
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.vectorizer import extract_features_record
from src.models.bundle import positive_class_probability
from src.models.metrics import metric_report
from src.security.file_validation import validate_pdf_envelope


MAX_SAFE_FIXTURE_BYTES = 2 * 1024 * 1024
FORBIDDEN_FIXTURE_MARKERS = (
    b"/URI",
    b"/Launch",
    b"/SubmitForm",
    b"/EmbeddedFile",
    b"http://",
    b"https://",
    b"file://",
    b"/GoToR",
)
HIGH_RISK_FEATURES = (
    "js_count",
    "javascript_count",
    "openaction_count",
    "action_count",
    "aa_count",
    "launch_count",
    "uri_count",
    "submitform_count",
)


class AdversarialSafetyError(RuntimeError):
    """Raised when a fixture or mutation violates the inert safety contract."""


class _PeakRSSMonitor:
    """Sample process RSS during one bounded extraction."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        import psutil

        self.interval_seconds = interval_seconds
        self.baseline = int(psutil.Process().memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSSMonitor":
        import psutil

        process = psutil.Process()

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak = max(self.peak, int(process.memory_info().rss))

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        import psutil

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        self.peak = max(self.peak, int(psutil.Process().memory_info().rss))


@dataclass(frozen=True)
class MutationValidation:
    valid: bool
    validator: str
    parseable: bool
    pages: int
    bytes: int
    sha256: str
    inert: bool
    reason: str
    qpdf_available: bool
    qpdf_passed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreatModel:
    black_box_score_queries: bool = True
    white_box_model_and_features: bool = True
    semantics_preserving_rewriting: bool = True
    feature_stuffing: bool = True
    parser_differential_exploitation: bool = True
    bounded_resource_exhaustion: bool = True
    poisoning_or_source_compromise: bool = True
    distribution_drift: bool = True
    live_malware_used: bool = False
    fixtures_rendered: bool = False
    feature_space_attacks_count_as_pdf_evasion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render_classic_pdf(
    objects: Mapping[int, bytes],
    *,
    root_object: int = 1,
    header: bytes = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n",
    spaced: bool = False,
) -> bytes:
    """Render a deterministic classic-xref PDF from inert object bodies."""
    chunks = [header]
    offsets: dict[int, int] = {}
    separator = b"\n% bounded-inert-fixture \n" if spaced else b"\n"
    for number in sorted(objects):
        offsets[number] = sum(map(len, chunks))
        chunks.append(
            f"{number} 0 obj\n".encode()
            + objects[number]
            + b"\nendobj"
            + separator
        )
    xref_offset = sum(map(len, chunks))
    maximum = max(objects, default=0)
    chunks.extend([b"xref\n", f"0 {maximum + 1}\n".encode(), b"0000000000 65535 f \n"])
    for number in range(1, maximum + 1):
        if number in offsets:
            chunks.append(f"{offsets[number]:010d} 00000 n \n".encode())
        else:
            chunks.append(b"0000000000 00000 f \n")
    chunks.extend(
        [
            b"trailer\n",
            f"<< /Size {maximum + 1} /Root {root_object} 0 R >>\n".encode(),
            b"startxref\n",
            str(xref_offset).encode() + b"\n%%EOF\n",
        ]
    )
    return b"".join(chunks)


def build_inert_pdf(*, security_marker: bool, serial: int = 0) -> bytes:
    """Create one valid local fixture with an empty, harmless action if requested."""
    content = f"BT /F1 10 Tf 20 50 Td (inert fixture {serial}) Tj ET".encode()
    catalog = b"<< /Type /Catalog /Pages 2 0 R"
    if security_marker:
        catalog += b" /OpenAction 5 0 R"
    catalog += b" >>"
    objects: dict[int, bytes] = {
        1: catalog,
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
            b"/Resources << >> /Contents 4 0 R >>"
        ),
        4: f"<< /Length {len(content)} >>\nstream\n".encode()
        + content
        + b"\nendstream",
        6: f"<< /Producer (local-inert-generator) /Fixture {serial} >>".encode(),
    }
    if security_marker:
        # Empty script: structural marker only, no executable behavior or I/O.
        objects[5] = b"<< /Type /Action /S /JavaScript /JS () >>"
    return _render_classic_pdf(objects)


_OBJECT = re.compile(rb"(?ms)^\s*(\d+)\s+(\d+)\s+obj\s*(.*?)\s*endobj\s*")


def _objects_from_pdf(data: bytes) -> dict[int, bytes]:
    prefix = data.split(b"xref", 1)[0]
    objects: dict[int, bytes] = {}
    for match in _OBJECT.finditer(prefix):
        number = int(match.group(1))
        generation = int(match.group(2))
        if generation == 0:
            objects[number] = match.group(3).strip()
    if 1 not in objects or 2 not in objects or 3 not in objects:
        raise AdversarialSafetyError("Mutation input is not a supported inert fixture.")
    return objects


def _rewrite(data: bytes, transform: Callable[[dict[int, bytes]], None], *, spaced: bool = False) -> bytes:
    objects = _objects_from_pdf(data)
    transform(objects)
    return _render_classic_pdf(objects, spaced=spaced)


def hex_escape_names(data: bytes) -> bytes:
    replacements = {
        b"/JavaScript": b"/J#61vaScript",
        b"/JS": b"/J#53",
        b"/OpenAction": b"/OpenActio#6E",
    }
    def transform(objects: dict[int, bytes]) -> None:
        for number, body in objects.items():
            for original, escaped in replacements.items():
                body = body.replace(original, escaped)
            objects[number] = body
    return _rewrite(data, transform)


def whitespace_comment_insertion(data: bytes) -> bytes:
    return _rewrite(data, lambda _objects: None, spaced=True)


def object_renumbering(data: bytes) -> bytes:
    objects = _objects_from_pdf(data)
    mapping = {number: (1 if number == 1 else number + 20) for number in objects}
    rewritten: dict[int, bytes] = {}
    for number, body in objects.items():
        for old in sorted(mapping, reverse=True):
            body = re.sub(
                rf"(?<!\d){old}\s+0\s+R\b".encode(),
                f"{mapping[old]} 0 R".encode(),
                body,
            )
        rewritten[mapping[number]] = body
    return _render_classic_pdf(rewritten, root_object=mapping[1])


def incremental_duplicate_revision(data: bytes) -> bytes:
    previous = int(re.search(rb"startxref\s+(\d+)", data).group(1))
    offset = len(data)
    update = b"7 0 obj\n<< /Note (inert incremental revision) >>\nendobj\n"
    xref = offset + len(update)
    return (
        data
        + update
        + b"xref\n7 1\n"
        + f"{offset:010d} 00000 n \n".encode()
        + b"trailer\n"
        + f"<< /Size 8 /Root 1 0 R /Prev {previous} >>\n".encode()
        + b"startxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )


def object_stream_filter_chain(data: bytes) -> bytes:
    decoded = b"50 0 << /Note (inert object stream member) >>"
    first = len(b"50 0 ")
    compressed = zlib.compress(decoded)
    encoded = binascii.hexlify(compressed) + b">"
    body = (
        f"<< /Type /ObjStm /N 1 /First {first} /Length {len(encoded)} "
        "/Filter [/ASCIIHexDecode /FlateDecode] >>\nstream\n"
    ).encode() + encoded + b"\nendstream"
    def transform(objects: dict[int, bytes]) -> None:
        objects[max(objects) + 1] = body
    return _rewrite(data, transform)


def relocate_action_marker(data: bytes) -> bytes:
    def transform(objects: dict[int, bytes]) -> None:
        if 5 not in objects:
            objects[1] = objects[1].replace(
                b"/Pages 2 0 R", b"/Pages 2 0 R /Names << /Dests << >> >>"
            )
            return
        objects[1] = objects[1].replace(b"/OpenAction 5 0 R", b"/AA << /O 5 0 R >>")
    return _rewrite(data, transform)


def benign_structure_inflation(data: bytes) -> bytes:
    def transform(objects: dict[int, bytes]) -> None:
        start = max(objects) + 1
        for index in range(24):
            objects[start + index] = (
                f"<< /Type /Metadata /FixturePadding {index} /Note (inert benign object) >>".encode()
            )
    return _rewrite(data, transform)


def xref_startxref_normalization_stress(data: bytes) -> bytes:
    match = re.search(rb"startxref\s+(\d+)\s+%%EOF", data)
    if match is None:
        raise AdversarialSafetyError("Fixture lacks startxref.")
    padded = f"{int(match.group(1)):020d}".encode()
    return data[: match.start(1)] + padded + data[match.end(1) :]


def inert_string_encoding(data: bytes) -> bytes:
    def transform(objects: dict[int, bytes]) -> None:
        if 5 in objects:
            # Octal string spells "void" but is never invoked outside the empty action fixture.
            objects[5] = objects[5].replace(b"/JS ()", rb"/JS (\166\157\151\144)")
    return _rewrite(data, transform)


def page_metadata_image_inflation(data: bytes) -> bytes:
    payload = b"\x00" * 128
    image = (
        f"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
        f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length {len(payload)} >>\nstream\n"
    ).encode() + payload + b"\nendstream"
    def transform(objects: dict[int, bytes]) -> None:
        objects[max(objects) + 1] = image
        objects[max(objects) + 1] = b"<< /Type /Metadata /Subtype /XML /Length 0 >>\nstream\n\nendstream"
    return _rewrite(data, transform)


def bounded_resource_nesting(data: bytes) -> bytes:
    def transform(objects: dict[int, bytes]) -> None:
        start = max(objects) + 1
        depth = 32
        for index in range(depth):
            target = start + index + 1
            child = f" /Next {target} 0 R" if index + 1 < depth else ""
            objects[start + index] = f"<< /Type /Metadata /Depth {index}{child} >>".encode()
    return _rewrite(data, transform)


def parser_disagreement_duplicate_key(data: bytes) -> bytes:
    def transform(objects: dict[int, bytes]) -> None:
        # Keep the duplicate key in an unreachable inert metadata dictionary so
        # strict page traversal remains valid while parser policy can disagree.
        objects[6] = objects.get(6, b"<< /Type /Metadata >>").replace(
            b">>", b" /DuplicateKey 0 /DuplicateKey 1 >>"
        )
    return _rewrite(data, transform)


def combined_adaptive_static(data: bytes) -> bytes:
    value = hex_escape_names(data)
    value = inert_string_encoding(value)
    value = benign_structure_inflation(value)
    return whitespace_comment_insertion(value)


MUTATIONS: dict[str, Callable[[bytes], bytes]] = {
    "name_hex_escaping": hex_escape_names,
    "whitespace_comment_insertion": whitespace_comment_insertion,
    "object_renumbering": object_renumbering,
    "incremental_duplicate_revision": incremental_duplicate_revision,
    "object_stream_filter_chain": object_stream_filter_chain,
    "action_marker_relocation": relocate_action_marker,
    "benign_structure_inflation": benign_structure_inflation,
    "xref_startxref_manipulation": xref_startxref_normalization_stress,
    "inert_string_encoding": inert_string_encoding,
    "page_metadata_image_inflation": page_metadata_image_inflation,
    "bounded_resource_nesting": bounded_resource_nesting,
    "parser_disagreement_duplicate_key": parser_disagreement_duplicate_key,
    "combined_adaptive_static": combined_adaptive_static,
}


def validate_inert_pdf(
    data: bytes,
    *,
    maximum_bytes: int = MAX_SAFE_FIXTURE_BYTES,
    require_qpdf: bool = False,
) -> MutationValidation:
    """Validate without rendering and reject network/file/launch behavior."""
    digest = hashlib.sha256(data).hexdigest()
    envelope = validate_pdf_envelope(data, maximum_bytes=maximum_bytes)
    forbidden = [marker.decode("latin-1") for marker in FORBIDDEN_FIXTURE_MARKERS if marker.lower() in data.lower()]
    inert = not forbidden
    parseable = False
    pages = 0
    reason = envelope.reason if not envelope.valid else "ok"
    if envelope.valid and inert:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(data), strict=True)
            pages = len(reader.pages)
            parseable = pages >= 1
            if not parseable:
                reason = "strict_parser_found_no_pages"
        except Exception as exc:
            reason = f"strict_parser_rejected:{type(exc).__name__}"
    elif forbidden:
        reason = "forbidden_active_marker:" + ",".join(forbidden)

    qpdf = shutil.which("qpdf")
    qpdf_passed: bool | None = None
    if qpdf:
        descriptor, name = tempfile.mkstemp(suffix=".pdf")
        os.close(descriptor)
        path = Path(name)
        try:
            path.write_bytes(data)
            result = subprocess.run(
                [qpdf, "--check", str(path)],
                check=False,
                capture_output=True,
                timeout=10,
            )
            qpdf_passed = result.returncode == 0
            if not qpdf_passed:
                reason = "qpdf_check_failed"
        finally:
            path.unlink(missing_ok=True)
    elif require_qpdf:
        reason = "qpdf_required_but_unavailable"

    valid = bool(
        envelope.valid
        and inert
        and parseable
        and (qpdf_passed is not False)
        and not (require_qpdf and qpdf is None)
    )
    return MutationValidation(
        valid=valid,
        validator="PyPDF2_strict+qpdf_check" if qpdf else "PyPDF2_strict_equivalent_non_rendering_validator",
        parseable=parseable,
        pages=pages,
        bytes=len(data),
        sha256=digest,
        inert=inert,
        reason=reason,
        qpdf_available=qpdf is not None,
        qpdf_passed=qpdf_passed,
    )


def _literal_signal(data: bytes) -> float:
    return float(
        sum(
            data.count(token)
            for token in (b"/JS", b"/JavaScript", b"/OpenAction", b"/AA")
        )
    )


def _safe_extract(data: bytes) -> tuple[dict[str, float], dict[str, Any], float, int]:
    descriptor, name = tempfile.mkstemp(suffix=".pdf")
    os.close(descriptor)
    path = Path(name)
    started = time.perf_counter()
    try:
        path.write_bytes(data)
        with _PeakRSSMonitor() as memory:
            features, diagnostics = extract_features_record(path)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        return (
            features,
            diagnostics,
            elapsed_ms,
            max(0, int(memory.peak - memory.baseline)),
        )
    finally:
        path.unlink(missing_ok=True)


def evaluate_fixture_corpus(
    pipeline: Any,
    model: Any,
    *,
    threshold: float,
    fixtures_per_class: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score clean and mutated inert fixtures without persisting any PDF bytes."""
    if fixtures_per_class < 2:
        raise ValueError("At least two fixtures per benchmark class are required.")
    rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    for expected_positive in (False, True):
        for serial in range(fixtures_per_class):
            clean = build_inert_pdf(security_marker=expected_positive, serial=serial)
            variants = {"none": clean, **{name: mutation(clean) for name, mutation in MUTATIONS.items()}}
            row_start = len(rows)
            validity_start = len(validity_rows)
            for mutation_name, data in variants.items():
                validation = validate_inert_pdf(data)
                validity_rows.append(
                    {
                        "expected_security_positive": expected_positive,
                        "fixture_serial": serial,
                        "mutation": mutation_name,
                        **validation.to_dict(),
                    }
                )
                if not validation.valid:
                    continue
                features, diagnostics, latency_ms, rss_delta = _safe_extract(data)
                transformed = pipeline.transform_record(features).to_numpy(dtype=np.float32)
                probability = float(positive_class_probability(model, transformed)[0])
                canonical_signal = float(sum(features.get(name, 0.0) for name in HIGH_RISK_FEATURES))
                rows.append(
                    {
                        "fixture_id_sha256": hashlib.sha256(
                            f"{expected_positive}:{serial}".encode()
                        ).hexdigest(),
                        "expected_security_positive": int(expected_positive),
                        "mutation": mutation_name,
                        "valid_pdf": True,
                        "literal_signal": _literal_signal(data),
                        "canonical_multiview_signal": canonical_signal,
                        "model_probability": probability,
                        "model_prediction": int(probability >= threshold),
                        "abstained": bool(diagnostics.get("abstain_recommended")),
                        "extraction_latency_ms": latency_ms,
                        "peak_rss_delta_bytes": rss_delta,
                        "fixture_bytes": len(data),
                        "raw_pdf_persisted": False,
                    }
                )
            fixture_rows = [row for row in rows[row_start:] if row["mutation"] != "none"]
            if fixture_rows:
                selected = (
                    min(fixture_rows, key=lambda row: row["model_probability"])
                    if expected_positive
                    else max(fixture_rows, key=lambda row: row["model_probability"])
                )
                adaptive = dict(selected)
                adaptive["adaptive_selected_from"] = selected["mutation"]
                adaptive["mutation"] = "adaptive_query_selected_worst_case"
                rows.append(adaptive)
                selected_validity = next(
                    row
                    for row in validity_rows[validity_start:]
                    if row["mutation"] == selected["mutation"]
                )
                adaptive_validity = dict(selected_validity)
                adaptive_validity["adaptive_selected_from"] = selected["mutation"]
                adaptive_validity["mutation"] = "adaptive_query_selected_worst_case"
                validity_rows.append(adaptive_validity)
    validity = pd.DataFrame(validity_rows)
    scored = pd.DataFrame(rows)
    valid_families = validity.loc[
        (validity["mutation"] != "none") & validity["valid"], "mutation"
    ].nunique()
    if valid_families < 8:
        raise AdversarialSafetyError(
            f"Only {valid_families} valid mutation families passed; at least eight are required."
        )
    metrics = robustness_metric_table(scored, threshold=threshold)
    return scored, validity, metrics


def robustness_metric_table(scored: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Compute clean/mutated benchmark metrics for each implemented defense view."""
    required = {
        "expected_security_positive", "mutation", "literal_signal",
        "canonical_multiview_signal", "model_probability", "abstained",
    }
    if scored.empty or not required.issubset(scored):
        raise ValueError("Robustness rows are incomplete.")
    labels = scored["expected_security_positive"].to_numpy(dtype=np.int8)
    views = {
        "literal_token_baseline": (scored["literal_signal"] > 0).astype(float).to_numpy(),
        "canonical_multiview_indicator": (scored["canonical_multiview_signal"] > 0).astype(float).to_numpy(),
        "champion_model": scored["model_probability"].to_numpy(dtype=float),
        "champion_plus_abstention": np.maximum(
            scored["model_probability"].to_numpy(dtype=float),
            scored["abstained"].astype(float).to_numpy(),
        ),
    }
    rows = []
    clean = scored["mutation"] == "none"
    for view, probability in views.items():
        active_threshold = threshold if view.startswith("champion") else 0.5
        for partition, mask in (("clean_inert_benchmark", clean), ("valid_mutated_inert_benchmark", ~clean)):
            report = metric_report(labels[mask], probability[mask], threshold=active_threshold)
            positives = labels[mask] == 1
            attack_success = float(
                np.mean((probability[mask][positives] < active_threshold))
            ) if positives.any() else 0.0
            rows.append(
                {
                    "defense_view": view,
                    "partition": partition,
                    "benchmark_labels_are_malware_ground_truth": False,
                    "attack_success_rate": attack_success,
                    "robust_recall_or_clean_recall": report["recall"],
                    "robust_f2_or_clean_f2": report["f2"],
                    "false_positive_rate": report["false_positive_rate"],
                    "threshold": active_threshold,
                    "rows": report["rows"],
                }
            )
    return pd.DataFrame(rows)


def defense_matrix() -> pd.DataFrame:
    """Separate demonstrated controls from conditional/future defenses."""
    rows = [
        ("name/string/whitespace/reference canonicalization", "demonstrated", "literal_token_baseline", "canonical_multiview_indicator", "literal-token evasion"),
        ("logical graph + bounded decoded object streams", "demonstrated", "literal_token_baseline", "canonical_multiview_indicator", "object/filter hiding"),
        ("multi-view consistency and semantic ratios", "demonstrated", "canonical_multiview_indicator", "champion_model", "stuffing/parser differential"),
        ("OOD/parser abstention and resource budgets", "demonstrated", "champion_model", "champion_plus_abstention", "novel/malformed/resource inputs"),
        ("feature-only source allowlist/checksums/confidence", "implemented_control", "uncontrolled_source", "phase1_fail_closed_source_gate", "training-data poisoning"),
        ("temporal drift monitoring", "implemented_control", "unmonitored", "phase5_drift_tables", "campaign/concept drift"),
        ("adversarial training", "future_requires_approved_labeled_pdf_corpus", "not_run", "not_run", "known rewrite patterns"),
        ("monotonic tree constraints", "future_requires_empirical_validation", "not_run", "not_run", "risk-feature stuffing"),
        ("sandbox escalation", "future_external_system", "not_run", "not_run", "high-risk uncertain cases"),
    ]
    return pd.DataFrame(rows, columns=("defense", "status", "pre", "post", "attack_addressed"))
