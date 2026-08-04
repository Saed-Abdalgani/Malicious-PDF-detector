# Safe adversarial robustness contract

Phase 7 evaluates realizable PDF rewrites without creating or acquiring harmful
samples. The attacker may have black-box query access or white-box knowledge and
may try syntax rewriting, stuffing, parser differentials, bounded resource
exhaustion, source poisoning, or distribution drift. A feature-space edit that
is not a validator-accepted PDF is only an upper-bound stress test.

## Corpus safety

- Fixtures are generated locally and contain only inert page content and
  empty/harmless structural action markers.
- Network links, launch actions, embedded files, executables, and harmful script
  are rejected before scoring.
- PDFs are parsed in strict mode but never rendered or opened in a viewer.
- qpdf is added automatically when available; its absence is recorded.
- Temporary PDFs are deleted. Reports contain hashes, numeric evidence, timing,
  and memory—not PDF bytes.

## Metric interpretation

Attack success means a valid mutated positive benchmark fixture crossed from a
detected clean state to below the locked threshold without abstention. Robust
Recall/F2 use the expected-security-positive marker, not malware ground truth.
Probability drop measures rewrite sensitivity; benign rows measure false-positive
pressure. Clean deltas expose defense cost, while latency, peak RSS, and
abstention expose operational cost.

## Defense evidence levels

- Demonstrated: canonicalization; bounded logical/object-stream inspection;
  multi-view consistency/semantic features; and abstention/resource budgets.
  Each receives a pre/post table.
- Implemented elsewhere: source allowlists/checksums/label confidence and
  temporal drift monitoring.
- Future, not claimed: adversarial training needs an approved labeled corpus;
  monotonic constraints need validation; sandbox escalation is external.

Phase 7 fails closed until the checksummed Phase 6 manifest and champion exist.
No production robustness conclusion is currently claimed.
