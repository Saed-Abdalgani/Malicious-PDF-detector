"""
adversarial.py
--------------
Adversarial / evasion harness for the static-feature PDF detector.

The training set is *Evasive*-PDFMal2022, so the most interesting question for a
security tool is not "what is the accuracy?" but "**how easily can an attacker
evade it?**". This module applies realistic, byte-level PDF mutations that
preserve the malicious payload's intent while attempting to defeat the
structural feature extractor, then measures how much the detector's signal
drops.

Why this matters
~~~~~~~~~~~~~~~~~
The detector counts literal tokens such as ``/JavaScript``, ``/OpenAction`` and
``/Launch``. The PDF spec lets a name be written with hex escapes
(``/J#61vaScript`` == ``/JavaScript``), so a parser-faithful reader still
executes the action while a naive regex counter sees *nothing*. This is a
classic, real-world evasion — and an honest robustness section must show where
the static approach fails, not hide it.

Mutations implemented
~~~~~~~~~~~~~~~~~~~~~~~
1. ``hex_escape_names``     — rewrite high-risk PDF names using ``#XX`` escapes.
2. ``whitespace_padding``   — inject benign whitespace/comments around tokens.
3. ``junk_object_inflation``— append benign objects to dilute keyword ratios.
4. ``all``                  — apply every mutation in sequence (worst case).

Outputs
~~~~~~~
- ``reports/results/adversarial_robustness.csv`` — per-mutation feature drift.
- ``reports/results/adversarial_threat_model.md`` — narrative write-up.

Usage::

    python -m src.security.adversarial
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SAMPLE_PDFS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# High-risk structural features whose suppression indicates successful evasion.
_SIGNAL_FEATURES = [
    "js_count", "javascript_count", "openaction_count", "launch_count",
    "aa_count", "submitform_count", "uri_count", "richmedia_count",
    "jbig2decode_count", "action_count",
]

# Mapping of literal PDF names -> hex-escaped equivalents (parser-equivalent).
_HEX_ESCAPES = {
    b"/JavaScript": b"/J#61vaScript",
    b"/JS": b"/J#53",
    b"/OpenAction": b"/OpenActio#6E",
    b"/Launch": b"/Launc#68",
    b"/SubmitForm": b"/SubmitForm",  # left literal as a control
    b"/AA": b"/A#41",
    b"/RichMedia": b"/RichMedi#61",
    b"/URI": b"/UR#49",
}


# ---------------------------------------------------------------------------
# Mutations (operate on raw PDF bytes)
# ---------------------------------------------------------------------------

def hex_escape_names(data: bytes) -> bytes:
    """Rewrite high-risk PDF names using ``#XX`` hex escapes.

    A spec-compliant PDF reader treats ``/J#61vaScript`` identically to
    ``/JavaScript`` and still executes the action; a literal-token counter
    sees the keyword disappear.
    """
    for literal, escaped in _HEX_ESCAPES.items():
        if literal != escaped:
            data = data.replace(literal, escaped)
    return data


def whitespace_padding(data: bytes) -> bytes:
    """Inject benign comments/whitespace around object boundaries.

    Pure dilution: it does not change semantics but perturbs byte-level
    heuristics such as ``avg_stream_size`` and offsets.
    """
    return data.replace(b"endobj", b"%evasion-pad\nendobj")


def junk_object_inflation(data: bytes, n: int = 40) -> bytes:
    """Append benign indirect objects to dilute high-risk keyword ratios.

    Inserts ``n`` trivial objects before the final ``%%EOF`` so that
    ratio/percentage features shift toward "benign" territory.
    """
    junk = b"".join(
        f"\n{1000 + i} 0 obj\n<< /Type /Metadata /Note (benign) >>\nendobj".encode()
        for i in range(n)
    )
    if b"%%EOF" in data:
        return data.replace(b"%%EOF", junk + b"\n%%EOF", 1)
    return data + junk


MUTATIONS: Dict[str, Callable[[bytes], bytes]] = {
    "hex_escape_names": hex_escape_names,
    "whitespace_padding": whitespace_padding,
    "junk_object_inflation": junk_object_inflation,
    "all": lambda d: junk_object_inflation(whitespace_padding(hex_escape_names(d))),
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _features_for_bytes(data: bytes) -> Dict[str, float]:
    """Extract the raw feature dict for in-memory PDF bytes."""
    from src.features.vectorizer import extract_features_dict

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return extract_features_dict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _signal(features: Dict[str, float]) -> float:
    """Sum of high-risk structural features (the detector's 'suspicion signal')."""
    return float(sum(features.get(f, 0.0) for f in _SIGNAL_FEATURES))


def _model_prob(features: Dict[str, float], pipeline, model) -> float:
    """Run the (current deployment) model on a feature dict. Returns P(malicious)."""
    import torch

    vec = pipeline.transform_record(features).to_numpy(dtype=np.float32)
    x = torch.tensor(vec, dtype=torch.float32)
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).item())


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def run_harness(
    sample_dir: Path = SAMPLE_PDFS_DIR,
    save: bool = True,
    *,
    pipeline=None,
    verified_model=None,
) -> pd.DataFrame:
    """Apply every mutation to every sample PDF and measure detector drift.

    Args:
        sample_dir: Directory of source PDFs to mutate.
        save: If True, write the CSV + markdown write-up.

    Returns:
        pd.DataFrame: One row per (pdf, mutation) with feature/model drift.
    """
    from src.features.vectorizer import load_feature_pipeline

    active_pipeline = pipeline or load_feature_pipeline()
    model = verified_model

    rows: List[dict] = []
    pdfs = sorted(Path(sample_dir).glob("*.pdf"))

    for pdf in pdfs:
        original = pdf.read_bytes()
        base_feats = _features_for_bytes(original)
        base_signal = _signal(base_feats)
        base_prob = _model_prob(base_feats, active_pipeline, model) if model else float("nan")

        # baseline row
        rows.append({
            "pdf": pdf.name, "mutation": "none",
            "signal": round(base_signal, 2),
            "signal_drop_pct": 0.0,
            "model_prob_malicious": round(base_prob, 4),
        })

        for name, fn in MUTATIONS.items():
            mutated = fn(original)
            feats = _features_for_bytes(mutated)
            signal = _signal(feats)
            prob = _model_prob(feats, active_pipeline, model) if model else float("nan")
            drop = (1 - signal / base_signal) * 100 if base_signal else 0.0
            rows.append({
                "pdf": pdf.name, "mutation": name,
                "signal": round(signal, 2),
                "signal_drop_pct": round(drop, 1),
                "model_prob_malicious": round(prob, 4),
            })
            logger.info(
                f"{pdf.name} [{name}]: signal {base_signal:.0f} -> {signal:.0f} "
                f"({drop:+.0f}% high-risk-keyword signal)"
            )

    df = pd.DataFrame(rows)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "adversarial_robustness.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Adversarial robustness table saved -> {csv_path}")
        _write_threat_model(df)

    return df


def _write_threat_model(df: pd.DataFrame) -> None:
    """Write a concise, honest threat-model narrative alongside the CSV."""
    # Worst-case keyword suppression achieved by name obfuscation.
    hexrows = df[df["mutation"] == "hex_escape_names"]
    worst_drop = hexrows["signal_drop_pct"].max() if not hexrows.empty else 0.0

    md = f"""# Adversarial Robustness — Threat Model

*Author: Saed Abdalgani*

## Setup
The detector relies on **static structural features** — counts of literal PDF
tokens (`/JavaScript`, `/OpenAction`, `/Launch`, ...). We apply parser-faithful
byte-level mutations to the bundled sample PDFs and measure how much of the
detector's high-risk "suspicion signal" survives. Signal = sum of
{_SIGNAL_FEATURES}.

## Key result
**Hex-escaped PDF names defeat literal-token counting.** Rewriting
`/JavaScript` as `/J#61vaScript` (semantically identical to a compliant reader)
suppresses up to **{worst_drop:.0f}%** of the high-risk keyword signal, blinding
the structural detector while the payload still executes.

| Mutation | Technique | Static-feature impact |
|---|---|---|
| `hex_escape_names` | PDF name `#XX` hex escaping | High — collapses keyword counts toward 0 |
| `whitespace_padding` | comment/whitespace injection | Low — perturbs byte-level size features |
| `junk_object_inflation` | append benign objects | Medium — dilutes ratio/count features |
| `all` | combined worst case | Highest |

See `adversarial_robustness.csv` for the per-file, per-mutation numbers.

## Honest limitations
- The current deployment pipeline has a **normalization mismatch** (see
  `src/features/consistency.py`), so the reported `model_prob_malicious` is not
  yet meaningful end-to-end; the *feature-level* signal drop is the robust,
  model-independent finding.
- Static analysis alone cannot see semantics. Hex/octal name escaping, nested
  object streams, and filter chaining are well-known evasions.

## Recommended defenses (future work)
1. **Canonicalize names before counting** — decode `#XX` escapes so
   `/J#61vaScript` is counted as `/JavaScript`.
2. **Decode object streams / filters** before feature extraction.
3. **Combine static features with light dynamic triage** (e.g. detect the
   presence, not just the literal spelling, of auto-execute actions).
4. **Adversarial training** — include obfuscated variants in the training set.
"""
    out = RESULTS_DIR / "adversarial_threat_model.md"
    out.write_text(md, encoding="utf-8")
    logger.info(f"Threat-model write-up saved -> {out}")


if __name__ == "__main__":
    table = run_harness()
    print("\n" + table.to_string(index=False))
