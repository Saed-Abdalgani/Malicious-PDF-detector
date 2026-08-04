# Remediation Phase 7 — adversarial attacks and defenses

Status: implementation and safe-fixture contract tests complete; production
robustness execution awaits the approved dataset and completed Phases 4–6.

Phase 7 is implemented in `src/security/adversarial.py` and
`src/models/phase7.py`. It uses only locally generated inert PDFs, parses them
without rendering, deletes all temporary fixtures after extraction, and never
fetches or stores live malware. Fixtures contain no external links, embedded
files, network/launch actions, executables, or harmful JavaScript.

The suite covers name escaping, whitespace/comments, object renumbering,
incremental revisions, compressed object streams, action relocation, benign
stuffing, xref representation stress, inert string encoding, metadata/image
inflation, bounded nesting, parser-policy disagreement, combined mutations, and
a query-selected worst case. Thus the eight-family requirement is exceeded.

Every mutation is checked for its PDF envelope, size, forbidden active markers,
and parseability with `PyPDF2` strict mode. If `qpdf` is installed, `qpdf
--check` is also applied. qpdf is unavailable on this machine, so the strict
non-rendering equivalent is used and recorded transparently.

The runner compares literal-token detection, canonicalized multi-view signals,
the calibrated champion, and champion-plus-abstention. It reports attack success,
robust Recall/F2, probability drop, benign false positives, clean-performance
change, parser failure/abstention, extraction latency, and peak RSS overhead.
It separates demonstrated defenses from controls implemented in other phases
and future proposals such as adversarial training, monotonic constraints, and
external sandbox escalation.

Immutable outputs under `reports/adversarial/` include fixture scores, mutation
validity, robustness metrics by defense and mutation, defense pre/post tables,
resource overhead, threat-model JSON/Markdown, conclusions, and a checksummed
manifest. No PDF is included.

Benchmark-positive fixtures contain only empty/inert structural action markers;
they are not malware ground truth. After production Phase 6 succeeds, run:

```powershell
python -m src.models.phase7
```

The complete workflow also accepts `--through-phase 7` and retains the explicit
one-shot Phase 5 confirmation requirement.
