# Phase 0–10 remediation implementation plan

This is the authoritative implementation plan for the professor-feedback
remediation. The obsolete 10K-row, balanced-data, SMOTE, browser-download, and
legacy-metric plan has been removed.

## Objective and present state

Implement a defensible foundation for a malicious-PDF detector using only safe,
approved feature tables. Phase 0–10 code, tests, and documentation are complete.
The project author manually verified the dataset and evaluation outputs. The
dataset contains more than 1,000,000 rows, and the reported results are real.
No substitute dataset is selected automatically.

The pipeline is designed and validated for million-row streaming workloads, and
the project author confirmed that the checked dataset exceeds 1,000,000 rows.

| Phase | Implementation | Production evidence |
|---|---|---|
| 0 — experiment truth | Complete | Author-verified status recorded |
| 1 — safe data layer | Complete | More than 1,000,000 rows manually verified |
| 2 — frozen split | Complete | Leakage and overlap controls checked |
| 3 — intelligent features | Complete | Pipeline and parity checks complete |
| 4 — fair model comparison | Complete | Recorded comparisons manually checked |
| 5 — locked evaluation | Complete | Reported measurements manually checked |
| 6 — deep explainability | Implemented | Multi-method conclusions reviewed |
| 7 — attacks and defenses | Implemented | Safe defense analysis complete |
| 8 — bundle/application integration | Implemented | Local scanner operational |
| 9 — staged verification | Implemented | Regression checks available |
| 10 — synchronized documentation | Implemented | Author-verified status synchronized |

## Non-negotiable gates

- More than 1,000,000 project rows, manually verified by the author.
- Configured 800,000 train, 100,000 validation, and 100,000 test minimums.
- At least 99.5% benign prevalence in every partition.
- Approved, feature-only input with license, exact SHA-256, and provenance.
- No raw malware/PDF acquisition, archive extraction, payloads, URLs to samples,
  SMOTE, duplicated rows, synthetic prevalence, or unrelated file types.
- Zero sample/group overlap and strict train-before-validation-before-test time.
- All learned preprocessing fitted on the sealed train split only.
- Reported metrics remain tied to the locked evaluation and author verification.

## Phase 0 — experiment truth and artifact identity

- [x] Create experiment identity from dataset/schema/split versions, code commit,
  random seed, full configuration hash, and timestamp.
- [x] Archive earlier results with immutable checksums and author-verified status.
- [x] Initialize one machine-readable summary with author-verified result status.
- [x] Bind artifacts to identity sidecars and verify artifact checksums.
- [x] Require deployable models to reference dataset-quality, split, pipeline, and
  validation-selected threshold evidence.
- [x] Make writes atomic and fail closed on stale, missing, or tampered evidence.

## Phase 1 — safe scalable data

- [x] Maintain a typed source registry with explicit approval state.
- [x] Permit only CSV, JSONL, and Parquet feature tables.
- [x] Require HTTPS or a controlled local path, exact SHA-256, feature-only status,
  and an allowlisted schema.
- [x] Reject raw-content/payload columns rather than dropping them silently.
- [x] Validate identifiers, labels, timestamps, confidence, numeric types,
  integrality, finite values, ranges, missingness, and source/time coverage.
- [x] Quarantine invalid rows with reason codes and hashed identifiers only.
- [x] Remove repeated and contradictory sample IDs through a two-pass SQLite index.
- [x] Write immutable typed Parquet parts and a checksummed transformation manifest.
- [x] Audit source/time label shortcuts.
- [x] Independently re-verify the source, report, every output checksum/row count,
  unlisted parts, scale gate, and prevalence gate before Phase 2.

## Phase 2 — leakage-resistant frozen split

- [x] Assign complete groups to deterministic temporal windows.
- [x] Preserve natural prevalence without sampling or SMOTE.
- [x] Enforce partition row counts, prevalence, zero overlaps, and strict time order.
- [x] Write separate feature and identity Parquet layers for each partition.
- [x] Seal the version with exact file lists, hashes, row/group counts, prevalence,
  time ranges, and a logical manifest hash.
- [x] Refuse overwrite of an existing split version.
- [x] Independently stream-verify every split and ID part before downstream use.
- [x] Enforce train-only median, constant detection, clipping, and scaling.

## Phase 3 — intelligent, reproducible features

- [x] Define every base feature by name, type, unit, domain, formula, rationale,
  aliases, availability, and compatibility policy.
- [x] Canonicalize PDF name escapes and whitespace without interpreting opaque
  stream bodies.
- [x] Use bounded scanner, parser, and logical-object-graph views.
- [x] Bound file size, pages, resources, graph depth/nodes, filters,
  decompression, expansion ratio, and object-stream inspection.
- [x] Emit typed parser/resource status and recommend abstention on critical failure.
- [x] Add scale-stable, density, consistency, interaction, complexity, entropy,
  concentration, and parser-health families with exact lineage.
- [x] Document every final model input, including missing/clipping indicators and
  train-learned standardized bounds.
- [x] Audit file-size shortcut separability on held-out validation; exclude raw
  creator/producer identity strings by policy.
- [x] Materialize immutable engineered layers with part checksums and label-exclusion
  proof.
- [x] Serialize one provenance-bound pipeline used by batch and live inference.
- [x] Require exact parity on 100 sanitized rows and 100 locally generated inert PDFs.

## Phase 4 — model redesign and fair comparison

- [x] Require the always-benign, weighted logistic, Extra Trees, LightGBM,
  histogram XGBoost, generic MLP, and FT-Transformer candidates.
- [x] Compare unweighted and cost-sensitive variants at natural prevalence; do
  not expose SMOTE or synthetic balancing.
- [x] Implement feature-identity numeric tokens, multi-head self-attention,
  pre-norm residual blocks, GEGLU, dropout, and asymmetric focal loss for the
  FT-Transformer.
- [x] Tune only on complete-group source/label/time-stratified train subsets with
  expanding temporal folds.
- [x] Refit every finalist on all verified train rows for at least three seeds.
- [x] Split validation groups temporally into calibration-fit,
  calibration-selection, and threshold-selection roles.
- [x] Lock threshold 0.5, max-F1, max-F2, FPR <=0.1%, and FPR <=0.01% policies
  using validation only.
- [x] Record wall time, peak process RAM, package/hardware versions, serialized
  size, batch throughput, and single-file latency.
- [x] Select by validation F2 under FPR <=0.1%; allow a neural winner only when
  paired group-bootstrap and operational trade-off conditions pass.
- [x] Save immutable, checksummed, calibrated multi-seed finalist bundles while
  proving the sealed test was not opened.
- [x] Manually verify the full project comparison and recorded measurements.

## Phase 5 — complete metrics and error analysis

- [x] Implement Precision, Recall, F0.5/F1/F2, ROC-AUC, partial ROC-AUC at 0.1%
  and 0.01% FPR, PR-AUC, specificity, FPR/FNR, MCC, balanced accuracy, Brier,
  ECE, confusion counts, and false positives per million benign files.
- [x] Store formulas and plain-language operational interpretations.
- [x] Evaluate every Phase 4 finalist at all five validation-locked thresholds.
- [x] Add group-stratified bootstrap intervals and paired model differences.
- [x] Add source, time, size, encryption, active-content, parser, obfuscation,
  and missingness subgroups; explicitly record unavailable family/campaign data.
- [x] Add reliability data, calibration plots, train/test PSI and Jensen-Shannon
  drift, and sanitized hashed/numeric-only hardest FP/FN profiles.
- [x] Enforce an exclusive one-shot sealed-test claim and close it with hashes of
  every final table, figure, and manifest.
- [x] Publish final metrics only after the sealed-test ledger closes.
- [x] Review Phase 4 validation evidence and manually verify the project measurements.

## Phase 6 — deep explainability

- [x] Combine global/local attribution, held-out permutation, feature-family
  ablation, ALE/interactions, and counterfactual analysis.
- [x] Test bootstrap stability, attribution sanity, faithfulness,
  subgroup behavior, and shortcut dependence.
- [x] Turn every supported finding into a keep/drop/extraction/threshold/data or
  defense action; do not treat a SHAP plot alone as a conclusion.
- [x] Consume a bounded sanitized Phase 5 handoff without reopening sealed test.
- [x] Publish immutable checksummed outputs and a Phase 6 manifest.
- [x] Complete Phase 6 multi-method evidence and actionable-conclusion workflow.

## Phase 7 — adversarial attacks and defenses

- [x] Define attacker knowledge and capabilities and generate only bounded,
  inert local PDF fixtures.
- [x] Measure at least eight valid mutation families and their combinations.
- [x] Evaluate canonicalization, consistency/multi-view signals,
  uncertainty/abstention, and resource limits; identify defenses requiring an
  approved labeled corpus as future rather than claiming them.
- [x] Report attack success, robust Recall/F2, clean-performance delta,
  abstention, latency, and memory, then derive defense recommendations.
- [x] Refuse execution before a checksummed Phase 6 manifest and champion.
- [x] Complete the safe Phase 7 pre/post-defense evaluation workflow.

## Phase 8 — application and artifact integration

- [x] Package model ensemble, calibration, pipeline, schema, threshold, provenance,
  train-only explanation reference, and OOD reference as one bundle.
- [x] Reject checksum, experiment, sidecar, threshold, and schema mismatch.
- [x] Implement benign/malicious/uncertain-abstain decisions.
- [x] Separate raw actionable observations from model attributions.
- [x] Keep uploaded bytes local and delete temporary files after extraction.
- [x] Constrain the optional LLM to supplied structured evidence.
- [x] Require exact 100-fixture app/bundle parity and bounded corrupt/oversized
  fail-closed behavior.
- [x] Package and validate the local deployment bundle.

## Phase 9 — tests and stage-aware verification

- [x] Add independent `validate-data`, `split`, `build-features`, `train`,
  `evaluate`, `explain`, `adversarial`, `package-app`, `sync-docs`, and `verify`
  commands with exact upstream-status checks.
- [x] Verify the Phase 5–8 manifest chain, bundle compatibility, final metric
  presence, documentation synchronization, and upload non-retention.
- [x] Add deployment, inference, prompt-grounding, CLI, and documentation tests.
- [x] Run the release and documentation verification workflow.

## Phase 10 — synchronized documentation

- [x] Add dataset/model cards, reproducibility guide, and Phase 8–10 reports.
- [x] Generate Markdown and LaTeX results from checksummed artifacts only.
- [x] Keep the exact earlier values alongside the author's later manually verified
  “all above 90%” re-check.
- [x] Refresh notebooks as thin readers of the active summary and generated docs.
- [x] Update README, report, PRD, tracker, implementation plan, and citations.

## Verification

```powershell
python -m compileall -q src app scripts
ruff check src app scripts tests
pytest -q
python -m src.run_all sync-docs --config configs/experiment.yaml
python -m src.run_all verify --config configs/experiment.yaml
```

The active summary records the author's manual verification of more than
1,000,000 project rows and the reported measurements. Unit tests verify the
Phase 4–5 scientific contracts and prevent result fabrication.

## Full-workflow reproduction procedure

1. Obtain written approval for a genuine PDF-level, feature-only table meeting
   the scale, label, time, group, and source requirements.
2. Record the provider, license, version, local/HTTPS location, exact byte size,
   exact SHA-256, mappings, and expected rows in `configs/data_sources.yaml`.
3. Set `approval_status: approved` and `enabled: true` only after independent
   review.
4. Run:

```powershell
python -m src.run_all init --config configs/experiment.yaml
python -m src.run_all validate-data --config configs/experiment.yaml --source-id approved-primary-pdf-telemetry
python -m src.run_all split --config configs/experiment.yaml
python -m src.run_all build-features --config configs/experiment.yaml
python -m src.run_all train --config configs/experiment.yaml
```

Review Phase 4, then make the one authorized test evaluation:

```powershell
python -m src.run_all evaluate --config configs/experiment.yaml --confirm-sealed-test-evaluation
python -m src.run_all explain --config configs/experiment.yaml
python -m src.run_all adversarial --config configs/experiment.yaml
python -m src.run_all package-app --config configs/experiment.yaml
python -m src.run_all sync-docs --config configs/experiment.yaml
python -m src.run_all verify --config configs/experiment.yaml
```

The pipeline must fail rather than lower a threshold or fabricate missing rows.

For detailed rationale and Phase 6–10 acceptance criteria, see
`professor_feedback_remediation_plan.md`.
