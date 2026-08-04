# Remediation status and remaining work

This tracker supersedes the legacy balanced/SMOTE workflow.

## Phase 0 — experiment truth

- [x] Versioned experiment identity and configuration hash.
- [x] Checksummed artifact sidecars and compatibility verification.
- [x] Immutable archive for unverified legacy evidence.
- [x] Active summary contains no final metric.
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
- [ ] Supply and approve a genuine source with at least 2.65M input rows.
- [ ] Execute Phase 1 at full scale and pass at least 2.5M clean unique rows.

## Phase 2 — frozen split

- [x] Deterministic streaming group-temporal assignment.
- [x] Natural-prevalence and minimum-row gates.
- [x] Separate feature and row/group identity manifests.
- [x] Immutable version and seal.
- [x] Independent checksum/count/overlap/time verification.
- [x] Train-only learned preprocessing.
- [ ] Generate the production split after Phase 1 passes.
- [ ] Prove at least 2M train and 250K validation/test rows at 99.5% benign.

## Phase 3 — feature contract and pipeline

- [x] Versioned 37-field base schema.
- [x] Bounded multi-view PDF extraction and typed status.
- [x] Deterministic engineered families with formula and lineage.
- [x] Full final-input catalog, including learned indicators/ranges.
- [x] File-size/source/time/identity-shortcut controls.
- [x] Immutable engineered-layer manifests.
- [x] Provenance-bound train/live pipeline.
- [x] Exact parity on 100 sanitized rows and 100 inert PDFs.
- [ ] Fit and serialize the production pipeline after Phase 2 passes.

## Phase 4 — fair model comparison

- [x] Complete required baseline/tree/MLP/FT-Transformer model registry.
- [x] Complete cost-sensitive ablations and natural-prevalence training path.
- [x] Complete grouped temporal development tuning and full-train refit path.
- [x] Complete three-seed calibrated bundles and disjoint validation roles.
- [x] Complete low-FPR threshold locking, resource evidence, paired uncertainty,
  and conservative neural-vs-tree selection.
- [x] Disable the legacy four-model trainer bypass.
- [ ] Execute on the real 2M+ train partition.

## Phase 5 — locked metrics and errors

- [x] Complete metric/threshold definitions and low-FPR curves.
- [x] Complete group-stratified confidence intervals and paired differences.
- [x] Complete subgroup, calibration, drift, and sanitized FP/FN analysis.
- [x] Complete exclusive one-shot sealed-test ledger and checksummed outputs.
- [ ] Review Phase 4 evidence and explicitly authorize the single production run.

## Phase 6 — deep explainability

- [x] Implement multi-method attribution, permutation, ablation, ALE/interactions,
  stability, sanity, faithfulness, subgroup, and counterfactual analysis.
- [x] Gate conclusions on independent methods, negative controls, and faithfulness.
- [x] Add sanitized Phase 5 handoff and immutable Phase 6 output manifest.
- [ ] Execute on real gated artifacts and derive empirical actions.

## Phase 7 — adversarial defenses

- [x] Implement the safe inert-PDF threat model and 12+ mutation families,
  combinations, and query-selected worst cases.
- [x] Implement strict non-rendering validation and prohibit PDF persistence.
- [x] Compare demonstrated defenses and report clean/robust benchmark and resource trade-offs.
- [x] Separate implemented controls from future training/monotonic/sandbox proposals.
- [ ] Execute on the verified production champion after Phases 1–6 succeed.

## Verification completed

- [x] `python -m compileall -q src app`
- [x] `ruff check src app tests`
- [x] `pytest -q` — 83 passed after Phase 7 implementation.
- [x] `git diff --check`
- [x] `python -m src.run_all --through-phase 0`
- [x] Confirm `data_gate_passed` is false and `final_metrics` is null.
- [x] Confirm no source is silently approved or downloaded.

## External dependency

The blocking production dependency is an institutionally approved, feature-only
PDF telemetry table. CIC-Evasive-PDFMal2022 remains a disabled supplementary
benchmark because 10,025 rows cannot satisfy the primary experiment and its
class balance is unsuitable for production prevalence.

Phase 4–7 implementation is complete but cannot produce empirical evidence until the data
dependency is resolved. The full
specification is in `professor_feedback_remediation_plan.md`.
