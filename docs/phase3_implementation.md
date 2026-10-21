# Remediation Phase 3 — Feature contract and intelligent engineering

Status: implementation and parity tests complete; production pipeline fit awaits
the sealed Phase 2 train set.

Key files:

- `src/features/schema_v2.py`: explicit 37-feature contract and compatibility
  policy.
- `src/features/canonicalize.py`: PDF-name escape decoding and safe lexical
  normalization outside opaque streams.
- `src/features/structural.py`: bounded static counts plus typed extraction status.
- `src/features/metadata.py`: bounded parser view with no embedded-stream decoding.
- `src/features/semantic.py`: limited logical object-graph traversal.
- `src/features/engineered.py`: scale, density, consistency, interaction,
  complexity, and parser-health families with complete lineage.
- `src/features/selection.py`: train-only constants/correlation candidates,
  held-out ablation sets, importance rank stability, and a train-direction /
  validation-evaluated file-size shortcut audit.
- `src/features/pipeline.py`: one checksummed serialized pipeline for batch and live
  feature transformation.

The code never treats parser failure as synonymous with maliciousness. Critical
timeouts, limits, or parse failures recommend abstention. Optional status fields
cannot be silently defaulted when a serialized pipeline requires them.

The golden parity tests require exact equality for 100 sanitized schema-v2 rows
and for 100 locally generated inert PDFs passed through live extraction versus
batch transformation. The generated catalog covers every final numeric, missing,
and clipping input with formula, unit, range basis, rationale, and lineage.

Creator/producer strings are deliberately absent from schema v2 because they are
high-cardinality provenance shortcuts and may contain sensitive free text.
Source/time shortcut detection is part of Phase 1; file-size separability is
measured on held-out validation in Phase 3 and requires ablation when suspicious.

The generated production dictionary will be written to
`reports/data/feature_dictionary_v2.json` after the real Phase 2 split passes.
