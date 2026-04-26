"""
structural.py
-------------
Extracts 25 structural features from raw PDF files using byte-level
regex analysis. These features capture the internal PDF specification
keywords, cross-reference structures, action triggers, and obfuscation
indicators that are highly discriminative for malicious PDF detection.

The extraction is performed via direct byte scanning — no PDF rendering
or JavaScript execution occurs — ensuring safe, static-only analysis
(SEC-05).

Features extracted:
    Structural keywords (13):  /JS, /JavaScript, /OpenAction, /Action,
        /AA, /Launch, /URI, /SubmitForm, /AcroForm, /XFA, /RichMedia,
        /JBig2Decode, /Colors
    PDF structure markers (7):  obj, endobj, stream, endstream, xref,
        trailer, startxref
    Object/stream analysis (3): /ObjStm, /Filter, indirect_obj_count
    Computed metrics (2):       avg_stream_size, obfuscation_count

Usage:
    from src.features.structural import extract_structural_features
    features = extract_structural_features("path/to/file.pdf")
"""

import re
import signal
import threading
from pathlib import Path
from typing import Dict, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeout mechanism (cross-platform via threading)
# ---------------------------------------------------------------------------
class _TimeoutError(Exception):
    """Raised when feature extraction exceeds the time limit."""
    pass


def _run_with_timeout(func, args=(), kwargs=None, timeout_sec: int = 30):
    """Execute *func* with a timeout. Returns (result, error)."""
    if kwargs is None:
        kwargs = {}
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as exc:
            error[0] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise _TimeoutError(
            f"Feature extraction timed out after {timeout_sec}s"
        )
    if error[0] is not None:
        raise error[0]
    return result[0]


# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once, reused across calls)
# ---------------------------------------------------------------------------

# Suspicious / action-related PDF keywords
_KEYWORD_PATTERNS: Dict[str, re.Pattern] = {
    "js_count":          re.compile(rb"/JS\b"),
    "javascript_count":  re.compile(rb"/JavaScript\b"),
    "openaction_count":  re.compile(rb"/OpenAction\b"),
    "action_count":      re.compile(rb"/Action\b"),
    "aa_count":          re.compile(rb"/AA\b"),
    "launch_count":      re.compile(rb"/Launch\b"),
    "uri_count":         re.compile(rb"/URI\b"),
    "submitform_count":  re.compile(rb"/SubmitForm\b"),
    "acroform_count":    re.compile(rb"/AcroForm\b"),
    "xfa_count":         re.compile(rb"/XFA\b"),
    "richmedia_count":   re.compile(rb"/RichMedia\b"),
    "jbig2decode_count": re.compile(rb"/JBig2Decode\b"),
    "colors_count":      re.compile(rb"/Colors\b"),
}

# PDF structural markers
_STRUCT_PATTERNS: Dict[str, re.Pattern] = {
    "obj_count":       re.compile(rb"\b\d+\s+\d+\s+obj\b"),
    "endobj_count":    re.compile(rb"\bendobj\b"),
    "stream_count":    re.compile(rb"\bstream\b"),
    "endstream_count": re.compile(rb"\bendstream\b"),
    "xref_count":      re.compile(rb"\bxref\b"),
    "trailer_count":   re.compile(rb"\btrailer\b"),
    "startxref_count": re.compile(rb"\bstartxref\b"),
}

# Object stream & filter keywords
_MISC_PATTERNS: Dict[str, re.Pattern] = {
    "objstm_count":  re.compile(rb"/ObjStm\b"),
    "filter_count":  re.compile(rb"/Filter\b"),
}

# Obfuscation: hex-encoded character references like #4A, #53
_OBFUSCATION_PATTERN = re.compile(rb"#[0-9A-Fa-f]{2}")

# Indirect object references: "12 0 R"
_INDIRECT_OBJ_PATTERN = re.compile(rb"\b\d+\s+\d+\s+R\b")

# Stream content boundaries (for computing average stream size)
_STREAM_CONTENT_PATTERN = re.compile(
    rb"\bstream\r?\n(.*?)\r?\nendstream\b", re.DOTALL
)


# ---------------------------------------------------------------------------
# Default feature dictionary (all zeros) — returned on error
# ---------------------------------------------------------------------------

def _zeroed_features() -> Dict[str, float]:
    """Return a feature dict with all 25 structural features set to 0.0."""
    keys = (
        list(_KEYWORD_PATTERNS.keys())
        + list(_STRUCT_PATTERNS.keys())
        + list(_MISC_PATTERNS.keys())
        + ["obfuscation_count", "avg_stream_size", "indirect_obj_count"]
    )
    return {k: 0.0 for k in keys}


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract(raw_bytes: bytes) -> Dict[str, float]:
    """Perform the actual regex-based feature extraction on raw bytes."""
    features: Dict[str, float] = {}

    # 1. Keyword pattern counts (13 features)
    for name, pattern in _KEYWORD_PATTERNS.items():
        features[name] = float(len(pattern.findall(raw_bytes)))

    # 2. Structural marker counts (7 features)
    for name, pattern in _STRUCT_PATTERNS.items():
        features[name] = float(len(pattern.findall(raw_bytes)))

    # 3. Object stream & filter counts (2 features)
    for name, pattern in _MISC_PATTERNS.items():
        features[name] = float(len(pattern.findall(raw_bytes)))

    # 4. Obfuscation count — hex-encoded sequences (#XX)
    #    Exclude common benign hex patterns (e.g. short isolated ones in names)
    obfusc_matches = _OBFUSCATION_PATTERN.findall(raw_bytes)
    features["obfuscation_count"] = float(len(obfusc_matches))

    # 5. Indirect object references
    indirect_matches = _INDIRECT_OBJ_PATTERN.findall(raw_bytes)
    features["indirect_obj_count"] = float(len(indirect_matches))

    # 6. Average stream size
    stream_bodies = _STREAM_CONTENT_PATTERN.findall(raw_bytes)
    if stream_bodies:
        total_stream_bytes = sum(len(s) for s in stream_bodies)
        features["avg_stream_size"] = float(
            total_stream_bytes / len(stream_bodies)
        )
    else:
        features["avg_stream_size"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_structural_features(
    pdf_path: Union[str, Path],
    timeout_sec: int = 30,
) -> Dict[str, float]:
    """Extract 25 structural features from a PDF file.

    Reads the raw bytes of the PDF and uses compiled regex patterns to
    count keyword occurrences, structural markers, obfuscation artifacts,
    and stream statistics.

    Args:
        pdf_path: Absolute or relative path to a PDF file.
        timeout_sec: Maximum seconds to allow for extraction before
            returning a zeroed feature dict. Defaults to 30.

    Returns:
        dict: A dictionary mapping 25 feature names to their numeric
        (float) values.  On any error (file not found, corrupted,
        timeout) a zeroed dict is returned so downstream code never
        crashes.

    Example:
        >>> feats = extract_structural_features("data/sample_pdfs/benign.pdf")
        >>> len(feats)
        25
        >>> feats["js_count"]
        0.0
    """
    pdf_path = Path(pdf_path)

    # Guard: file existence
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return _zeroed_features()

    try:
        # Read raw bytes
        raw_bytes = pdf_path.read_bytes()
        file_size_mb = len(raw_bytes) / (1024 * 1024)
        logger.info(
            f"Extracting structural features from {pdf_path.name} "
            f"({file_size_mb:.2f} MB)"
        )

        # Run extraction with timeout
        features = _run_with_timeout(
            _extract, args=(raw_bytes,), timeout_sec=timeout_sec
        )

        logger.info(
            f"Structural extraction complete — "
            f"{sum(1 for v in features.values() if v > 0)} / 25 "
            f"features non-zero"
        )
        return features

    except _TimeoutError:
        logger.warning(
            f"Structural extraction timed out after {timeout_sec}s "
            f"for {pdf_path.name}. Returning zeroed features."
        )
        return _zeroed_features()

    except Exception as exc:
        logger.error(
            f"Structural extraction failed for {pdf_path.name}: {exc}",
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
        print("Usage: python -m src.features.structural <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_structural_features(path)
    print(json.dumps(result, indent=2))
