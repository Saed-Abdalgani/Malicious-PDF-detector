# Remediation Phase 1 — Safe, validated data layer

Status: implementation complete; production data gate pending.

The former fallback downloader, permissive numeric coercion, feature-vector
deduplication, global imputation, and pre-split IQR thresholds are retired.

Implemented components:

- `src/data/manifest.py`: typed source registry and feature-only policy.
- `src/data/downloader.py`: HTTPS/checksum/size verification with no browser or
  archive extraction.
- `src/data/schema.py`: schema-v2 names, types, ranges, unsafe-column rejection.
- `src/data/loader.py`: bounded CSV/JSONL batches and PyArrow Parquet scanning.
- `src/data/deduplicate.py`: two-pass SQLite sample-ID conflict handling and
  secondary feature fingerprints.
- `src/data/validate.py`: row-level quarantine, row flow, quality gates, immutable
  Parquet outputs.
- `src/data/audit.py`: source/time coverage, label contingency, and source-only
  diagnostic ROC AUC.

The gate is at least 2.50M unique approved rows. The current repository has no
approved primary source, so this report does not claim that gate has passed.

Detailed operating instructions are in `docs/data_source_approval.md` and
`docs/phase0_3_implementation.md`.
