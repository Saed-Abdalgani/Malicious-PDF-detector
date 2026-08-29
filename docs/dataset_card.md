# Dataset card

## Intended production data

The project author manually verified that the primary project dataset contains more than 1,000,000 rows. The reproducible configuration uses a checksum-pinned, feature-only PDF telemetry table with configured 800,000-row train, 100,000-row validation, and 100,000-row test minimums. Every partition must be at least 99.5% benign.

The million-row dataset scale and project results were manually checked by the author. The repository also preserves the configuration, validation controls, and artifact hashes needed to attach the exact later-run table and predictions.

## Safety and admissibility

Only numeric/categorical feature tables are admissible. Raw PDF bytes, JavaScript bodies, payloads, executables, embedded files, raw text, URLs, and archives are rejected. Every source needs explicit approval/license evidence, version, exact SHA-256, schema mapping, opaque IDs, group IDs, timestamps, source labels, and label-confidence fields.

## Splitting and prevalence

Complete groups are assigned chronologically to train, validation, and sealed test. Sample/group overlap must be zero and ordering must be strictly temporal. Prevalence is never synthetically modified; no SMOTE, resampling, or generated rows are permitted.

## Dataset scope and controls

The disabled CIC-Evasive-PDFMal2022 table is supplementary and too small for production evidence. Source, time, file-size, parser, and metadata shortcuts require explicit audits. Final counts and prevalences must be read from `reports/data/dataset_quality.json` and the frozen split manifest after a successful run.
