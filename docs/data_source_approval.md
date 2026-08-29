# Primary data-source approval checklist

This project must not solve the row-count requirement by downloading unknown PDF
collections, malware archives, executables, or unlabeled web content. The primary
input must be a controlled, sanitized feature table.

## Required evidence from the provider

- legal/academic-use approval and license;
- provider and dataset version;
- feature extraction/version documentation;
- label definition, provenance, confidence, and known error process;
- opaque sample ID and related-sample/group ID;
- first-seen UTC timestamp and source ID;
- class counts demonstrating realistic prevalence;
- exact file byte size and SHA-256;
- confirmation that the table contains no payload bytes, raw text, scripts, URLs,
  paths, or retrievable sample locations.

## Required manifest fields

Edit `configs/data_sources.yaml` only after the evidence above is available:

```yaml
approved-primary-pdf-telemetry:
  provider: REAL_PROVIDER
  version: REAL_IMMUTABLE_VERSION
  approval_status: approved
  enabled: true
  feature_only: true
  file_format: .parquet
  url: https://controlled.example/immutable-table.parquet
  local_path: null
  sha256: REAL_64_HEX_DIGEST
  expected_size_bytes: REAL_EXACT_SIZE
  expected_rows: null
  author_verified_minimum_rows: 1000000
  license: REAL_APPROVAL_REFERENCE
  role: primary
  column_mapping:
    obj_count: provider_object_count
  metadata_mapping:
    sample_id: provider_hash
    Class: malicious_label
    source_id: provider_source
    group_id: provider_family_or_cluster
    first_seen_at: provider_first_seen
    label_confidence: provider_confidence
```

Mappings are canonical-name to exact provider-column name. Do not map fields by
position or guess that similarly named xref/filter/object fields have identical
semantics.

## Rejection conditions

Reject the source if any of these are true:

- checksum, byte size, license, or provenance is missing;
- a link redirects away from HTTPS or returns HTML;
- the container is an archive rather than a direct allowlisted table;
- it includes raw documents or malware-capable content;
- labels were inferred from source identity alone;
- sample/group IDs or timestamps needed for leakage control are missing;
- the 37 base features cannot be mapped unambiguously;
- class prevalence was synthetically balanced before delivery;
- the provider cannot explain duplicates, contradictions, or label confidence.

## Supplementary datasets

CIC-Evasive-PDFMal2022 can be evaluated later as a cited external benchmark, but
its size cannot satisfy the primary gate. Unlabeled SafeDocs PDFs and arbitrary
malware feeds are not approved training substitutes: they introduce raw risky
files, uncertain labels, and unverifiable prevalence.
