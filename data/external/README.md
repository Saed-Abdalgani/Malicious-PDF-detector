# External data quarantine area

Files downloaded by legacy optional-data scripts are **not approved training
inputs**. They may describe Android malware, phishing URLs, or other unrelated
domains and do not satisfy the PDF schema, prevalence, provenance, or row-count
requirements.

Do not fetch raw malware, PDF archives, executables, or unknown web datasets for
this project. A dataset can enter the remediation pipeline only after it is added
to `configs/data_sources.yaml` and passes the checklist in
`docs/data_source_approval.md`.

The primary workflow never scans this directory automatically.
