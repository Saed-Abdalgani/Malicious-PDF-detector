# Dataset card

## Intended production data

The primary input must be an approved, checksum-pinned, feature-only PDF telemetry table. It must contain at least 2,650,000 received rows, at least 2,500,000 clean rows, and yield at least 2,000,000 train rows plus 250,000 validation and 250,000 test rows. Every partition must be at least 99.5% benign.

No primary dataset currently satisfies and passes these gates in this repository. The pipeline is designed and validated for multi-million-row streaming workloads; this is a scalability statement, not a claim that two million real training rows have already been processed.

## Safety and admissibility

Only numeric/categorical feature tables are admissible. Raw PDF bytes, JavaScript bodies, payloads, executables, embedded files, raw text, URLs, and archives are rejected. Every source needs explicit approval/license evidence, version, exact SHA-256, schema mapping, opaque IDs, group IDs, timestamps, source labels, and label-confidence fields.

## Splitting and prevalence

Complete groups are assigned chronologically to train, validation, and sealed test. Sample/group overlap must be zero and ordering must be strictly temporal. Prevalence is never synthetically modified; no SMOTE, resampling, or generated rows are permitted.

## Known limitations

The disabled CIC-Evasive-PDFMal2022 table is supplementary and too small for production evidence. Source, time, file-size, parser, and metadata shortcuts require explicit audits. Final counts and prevalences must be read from `reports/data/dataset_quality.json` and the frozen split manifest after a successful run.

