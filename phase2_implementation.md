# Remediation Phase 2 — Leakage-resistant natural-prevalence splits

Status: implementation and split controls complete.

`src/data/splitter.py` now supports deterministic, streaming group-temporal
splitting. Complete groups are ordered by earliest observation time, with the
newest window sealed as test. Each partition must be at least 99.5% benign and
must satisfy the configured row minimums without oversampling.

Safeguards:

- random row stratification is disabled;
- SMOTE is prohibited;
- sample and group overlap must be zero;
- temporal windows must be strictly ordered;
- row/group IDs and every Parquet part are checksummed;
- an existing split version cannot be overwritten;
- medians, clipping, scaling, and feature selection happen only after splitting.

`src/data/preprocessing.py` enforces `partition_name="train"`. It supports
bounded-batch fitting, deterministic train-only quantile estimation, full-train
incremental scaling, missing indicators, and neural clipping flags.

The configured split minimums are 800,000 train, 100,000 validation, and
100,000 test rows, with zero sample/group overlap and strict temporal order.
