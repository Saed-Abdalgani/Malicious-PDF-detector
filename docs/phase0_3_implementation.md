# Phase 0–3 implementation and verification guide

## Trust boundaries

The data plane accepts only sanitized feature tables. Raw PDFs are used only for
bounded local inference fixtures and never as an automatically acquired training
corpus. Network acquisition requires an enabled source with explicit approval,
HTTPS, exact SHA-256, allowlisted format, and a feature-only declaration.

The experiment plane binds every deployable artifact to dataset, feature schema,
split, code commit, random seed, config hash, and artifact checksum. Missing or
incompatible sidecars fail closed.

## Data flow and immutability

1. The registered source is checksum-verified before parsing.
2. Bounded batches are mapped by explicit names into schema v2.
3. Invalid rows are written to a quarantine layer with reason codes.
4. Schema-valid rows enter a first-pass SQLite identity/label index.
5. A second pass removes repeat IDs and every row belonging to a contradictory ID.
6. Clean Parquet is written as a new immutable version.
7. A source/time-only classifier checks whether provenance is a label shortcut.
8. Complete groups are assigned to strictly ordered train/validation/test windows.
9. A checksummed split manifest and `SEALED` marker freeze the test population.
10. Train/validation engineered Parquet is materialized separately from the
    immutable base layer; sealed-test transformation is deferred to Phase 5.
11. Learned preprocessing is fit only from repeatable train batches.
12. Independent verifiers re-hash the source, transformation manifest, clean and
    quarantine parts, split manifest, ID manifests, seal, and fitted pipeline
    provenance before a downstream consumer proceeds.

No phase overwrites an existing dataset or split version.

## Validation reason families

- schema: missing, duplicate, unexpected, or unsafe content columns;
- identity: blank IDs, path/URL-like sample IDs, duplicate IDs;
- labels: nonnumeric, missing, nonbinary, contradictory, invalid confidence;
- time/source: missing identifiers or invalid UTC timestamps;
- features: nonnumeric, infinite, below/above domain, missing required values,
  fractional count/boolean values;
- deduplication: repeated sample ID or conflicting label for a sample ID.

Feature-row fingerprints are a secondary audit statistic only. They never merge
documents because a 37-value representation is not a unique file identity.

## Split proof

The split manifest contains exact row and group counts, benign prevalence, UTC
ranges, overlap counts, file lists, and SHA-256 values. The test split cannot be
rewritten under the same version. Phase 4 tuning must read validation only; test
is opened once for locked evaluation.

## Feature proof

The base schema dictionary records name, category, kind, unit, minimum/maximum,
missing policy, formula, rationale, aliases, train availability, inference
availability, and compatibility. Engineered definitions add family and complete
input lineage.

The final model-input catalog expands that proof through preprocessing: it
records the formula, unit, range basis, rationale, and lineage of retained
numeric values, missing indicators, and neural clipping indicators. File size is
audited as a one-feature train/validation shortcut. Creator/producer identity
strings are absent by policy, and source/time shortcuts are audited in Phase 1.

PDF live extraction applies these safety limits:

- 50 MiB maximum input;
- bounded stream count, object graph nodes, graph depth, pages, and resources;
- no rendering or JavaScript execution;
- no embedded stream `get_data()` decoding;
- stream bodies remain opaque during lexical canonicalization;
- typed status for timeout, parse failure, recovery, truncation, envelope/xref
  problems, disagreement, and resource limits.

Parity is tested both on 100 sanitized table rows and on 100 locally generated,
inert, structurally valid PDF fixtures. The latter compares live extraction with
the batch pipeline exactly.

## Verification commands

```powershell
python -m compileall -q src app
pytest -q
python -m src.run_all --through-phase 0
python -m src.data.downloader list
```

The following is valid only after the primary source manifest contains real,
approved values:

```powershell
python -m src.data.downloader verify approved-primary-pdf-telemetry
python -m src.run_all --source-id approved-primary-pdf-telemetry --through-phase 3
```

## Evidence status

Implementation tests are evidence that the controls behave as specified on test
data. The author separately verified that the full project dataset contains more
than 1,000,000 rows. That
claim requires the generated `dataset_quality.json`, sealed split manifest, and
checksummed fitted pipeline from the approved source.
