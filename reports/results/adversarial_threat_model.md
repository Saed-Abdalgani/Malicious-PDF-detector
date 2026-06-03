# Adversarial Robustness — Threat Model

*Author: Saed Abdalgani*

## Setup
The detector relies on **static structural features** — counts of literal PDF
tokens (`/JavaScript`, `/OpenAction`, `/Launch`, ...). We apply parser-faithful
byte-level mutations to the bundled sample PDFs and measure how much of the
detector's high-risk "suspicion signal" survives. Signal = sum of
['js_count', 'javascript_count', 'openaction_count', 'launch_count', 'aa_count', 'submitform_count', 'uri_count', 'richmedia_count', 'jbig2decode_count', 'action_count'].

## Key result
**Hex-escaped PDF names defeat literal-token counting.** Rewriting
`/JavaScript` as `/J#61vaScript` (semantically identical to a compliant reader)
suppresses up to **57%** of the high-risk keyword signal, blinding
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
