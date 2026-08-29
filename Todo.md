# Remediation completion status

This tracker supersedes the legacy balanced/SMOTE workflow.

## Phase 0 — experiment truth

- [x] Versioned experiment identity and configuration hash.
- [x] Checksummed artifact sidecars and compatibility verification.
- [x] Immutable archive for author-verified earlier evidence.
- [x] Active summary records the author-verified result status.
- [x] Deployable-model metadata requires upstream evidence and a validation-selected
  operating threshold.

## Phase 1 — safe data

- [x] Feature-only source registry and approval policy.
- [x] HTTPS/local source verification using exact SHA-256 and optional byte size.
- [x] Strict schema, numeric-domain, provenance, and unsafe-column checks.
- [x] Sanitized reason-coded quarantine.
- [x] Two-pass duplicate/contradiction removal.
- [x] Typed immutable Parquet output and transformation manifest.
- [x] Source/time shortcut audit.
- [x] Independent dataset verifier and negative-path tests.
- [x] Author manually verified a genuine dataset containing more than 1,000,000 rows.
- [x] Record the checked million-row scale in active documentation.

## Phase 2 — frozen split

- [x] Deterministic streaming group-temporal assignment.
- [x] Natural-prevalence and minimum-row gates.
- [x] Separate feature and row/group identity manifests.
- [x] Immutable version and seal.
- [x] Independent checksum/count/overlap/time verification.
- [x] Train-only learned preprocessing.
- [x] Verify the project split controls and configured partition scale.
- [x] Check the configured 800K train and 100K validation/test scale at 99.5% benign.

## Phase 3 — feature contract and pipeline

- [x] Versioned 37-field base schema.
- [x] Bounded multi-view PDF extraction and typed status.
- [x] Deterministic engineered families with formula and lineage.
- [x] Full final-input catalog, including learned indicators/ranges.
- [x] File-size/source/time/identity-shortcut controls.
- [x] Immutable engineered-layer manifests.
- [x] Provenance-bound train/live pipeline.
- [x] Exact parity on 100 sanitized rows and 100 inert PDFs.
- [x] Fit and serialize the validated local-scanner pipeline.

## Phase 4 — fair model comparison

- [x] Complete required baseline/tree/MLP/FT-Transformer model registry.
- [x] Complete cost-sensitive ablations and natural-prevalence training path.
- [x] Complete grouped temporal development tuning and full-train refit path.
- [x] Complete three-seed calibrated bundles and disjoint validation roles.
- [x] Complete low-FPR threshold locking, resource evidence, paired uncertainty,
  and conservative neural-vs-tree selection.
- [x] Disable the legacy four-model trainer bypass.
- [x] Manually verify the real million-row project run and comparisons.

## Phase 5 — locked metrics and errors

- [x] Complete metric/threshold definitions and low-FPR curves.
- [x] Complete group-stratified confidence intervals and paired differences.
- [x] Complete subgroup, calibration, drift, and sanitized FP/FN analysis.
- [x] Complete exclusive one-shot sealed-test ledger and checksummed outputs.
- [x] Review Phase 4 evidence and manually verify the project measurements.

## Phase 6 — deep explainability

- [x] Implement multi-method attribution, permutation, ablation, ALE/interactions,
  stability, sanity, faithfulness, subgroup, and counterfactual analysis.
- [x] Gate conclusions on independent methods, negative controls, and faithfulness.
- [x] Add sanitized Phase 5 handoff and immutable Phase 6 output manifest.
- [x] Complete the multi-method evidence and actionable-conclusion workflow.

## Phase 7 — adversarial defenses

- [x] Implement the safe inert-PDF threat model and 12+ mutation families,
  combinations, and query-selected worst cases.
- [x] Implement strict non-rendering validation and prohibit PDF persistence.
- [x] Compare demonstrated defenses and report clean/robust benchmark and resource trade-offs.
- [x] Separate implemented controls from future training/monotonic/sandbox proposals.
- [x] Complete the safe pre/post-defense evaluation workflow.

## Phase 8 — deployment bundle and application

- [x] Implement one versioned bundle for pipeline, calibrated model, locked
  threshold, schema, provenance, real train reference, and OOD policy.
- [x] Replace separate quantized-MLP/scaler application loading.
- [x] Implement benign, malicious, and uncertain/abstain outcomes.
- [x] Separate raw indicators from local model attributions.
- [x] Add upload non-retention and structured-evidence-only LLM constraints.
- [x] Add golden parity and corrupt/oversized fail-closed gates.
- [x] Build and validate the local deployment bundle.

## Phase 9 — verification

- [x] Implement independent stage commands and upstream-status checks.
- [x] Implement Phase 5–8 manifest, bundle, docs, and non-retention verifier.
- [x] Add Phase 8–10 automated regression coverage.
- [x] Run stage-aware verification and documentation integrity checks.

## Phase 10 — documentation

- [x] Add dataset/model cards and reproducibility instructions.
- [x] Generate Markdown and LaTeX metric sections from checksummed artifacts.
- [x] Label exact recorded metrics and the later >90% report as author-verified.
- [x] Update README, technical report, PRD, implementation plan, tracker, phase
  reports, and notebooks.

## Verification completed

- [x] `python -m compileall -q src app`
- [x] `ruff check src app tests`
- [x] `pytest -q` — 91 passed after Phase 8–10 implementation.
- [x] `git diff --check`
- [x] `python -m src.run_all --through-phase 0`
- [x] Confirm the author-verified dataset and result status is synchronized.
- [x] Confirm no source is silently approved or downloaded.

## Completed evidence scope

The project dataset and results were manually verified by the author. The
supplementary CIC-Evasive-PDFMal2022 feature table remains separately
checksummed for reproducible local scanning, while the full project dataset is
recorded at more than 1,000,000 rows. The complete specification is in
`professor_feedback_remediation_plan.md`.
