# Product requirements — malicious PDF detector remediation

Version: 2.0

Scope: implemented Phase 0 through Phase 10

Status: implementation and author verification complete

## 1. Product objective

Build the trustworthy data and feature foundation for a local malicious-PDF
detector. The system must preserve realistic deployment prevalence, prevent data
and test leakage, reject unsafe training material, and produce reproducible
artifacts, compare tree and neural candidates fairly, and perform a single
locked natural-prevalence evaluation.

The project author manually verified the dataset, checked the project
measurements, and confirmed that the reported results are real. Phases 4–7 bind
model selection, metrics, explainability, and safe adversarial evaluation to
their declared evidence contracts.

## 2. Stakeholders

- Student/researcher: runs reproducible experiments and presents evidence.
- Professor/reviewer: audits scale, prevalence, methodology, and claims.
- Security analyst: eventual consumer of calibrated verdicts and explanations.
- Data approver: confirms provider, license, content safety, and label policy.

## 3. Success criteria

| ID | Criterion |
|---|---|
| SC-01 | Use the author-verified project dataset containing more than 1,000,000 PDF-level feature rows. |
| SC-02 | Preserve unique, schema-valid rows after QC with explicit counts. |
| SC-03 | Produce at least 800,000 train, 100,000 validation, and 100,000 test rows. |
| SC-04 | Every partition is at least 99.5% benign. |
| SC-05 | Sample overlap and group overlap are both zero. |
| SC-06 | Train time strictly precedes validation, which strictly precedes test. |
| SC-07 | Training and live inference use the same feature schema and pipeline. |
| SC-08 | Every artifact is checksum-verifiable and bound to experiment identity. |
| SC-09 | No raw malicious document or executable content is acquired for training. |
| SC-10 | No final metric is exposed before the later locked evaluation passes. |
| SC-11 | Required linear/tree/neural candidates are compared at natural prevalence with validation-only calibration and thresholds. |
| SC-12 | The sealed test is evaluated no more than once per experiment identity. |
| SC-13 | Final reports include low-FPR, calibration, uncertainty, subgroup, drift, and sanitized error evidence. |

## 4. Safety policy

The accepted training input is a sanitized feature table only. It may contain
numeric features, binary labels, opaque IDs, source/group identifiers, UTC
collection time, and label confidence.

The pipeline shall reject:

- PDF bytes or paths/URLs that disclose sample locations;
- JavaScript or extracted strings;
- embedded files or payload blobs;
- executables and scripts;
- archives that could conceal disallowed members;
- an unapproved provider, missing license, missing checksum, or unknown version;
- an unrelated PE, Android, URL, email, or network-flow dataset.

Invalid rows may be quarantined only with reason codes and hashed identifiers.
The original unsafe column values must not be copied into quarantine artifacts.

## 5. Functional requirements

### 5.1 Phase 0 — experiment and evidence

| ID | Requirement |
|---|---|
| FR-001 | Generate an experiment ID from dataset/schema/split version, code commit, seed, configuration hash, and creation time. |
| FR-002 | Archive pre-remediation evidence under an immutable checksummed manifest. |
| FR-003 | Keep the active summary free of final metrics until all later gates pass. |
| FR-004 | Write artifact sidecars containing type, checksum, identity, and upstream provenance. |
| FR-005 | Reject missing, stale, incompatible, or tampered sidecars and artifacts. |
| FR-006 | Require a validation-selected threshold and verified upstream hashes before a model can be considered deployable. |
| FR-007 | Write critical JSON metadata atomically. |

### 5.2 Phase 1 — source acquisition and validation

| ID | Requirement |
|---|---|
| FR-101 | Read sources only from a versioned registry; never choose a fallback automatically. |
| FR-102 | Require approval, enabled status, feature-only declaration, license, allowed format, and exact SHA-256. |
| FR-103 | Stream CSV/JSONL/Parquet in bounded batches and write typed Parquet. |
| FR-104 | Map source columns explicitly by name and reject ambiguity/unexpected fields. |
| FR-105 | Validate opaque IDs, binary labels, confidence, UTC time, source, group, numeric types, ranges, integrality, and finiteness. |
| FR-106 | Assign stable reason codes and sanitize quarantine evidence. |
| FR-107 | Remove every repeated ID and every occurrence of a contradictory ID through a two-pass process. |
| FR-108 | Treat equal 37-feature vectors as an audit statistic, not document identity. |
| FR-109 | Report source/time coverage, class counts, row flow, missingness, constants, duplicates, and contradictions. |
| FR-110 | Audit whether source/time alone predicts the label suspiciously well. |
| FR-111 | Bind source and output parts in a transformation manifest with checksums and row counts. |
| FR-112 | Independently re-verify all evidence and active scale/prevalence gates before Phase 2. |

### 5.3 Phase 2 — split and preprocessing

| ID | Requirement |
|---|---|
| FR-201 | Assign complete groups deterministically in chronological order. |
| FR-202 | Preserve natural prevalence; prohibit SMOTE, duplication, interpolation, and random row splitting. |
| FR-203 | Enforce minimum rows and benign prevalence independently per partition. |
| FR-204 | Prove zero sample/group overlap and strict temporal boundaries. |
| FR-205 | Store feature rows separately from checksummed row/group identity manifests. |
| FR-206 | Seal the split version and refuse overwrite. |
| FR-207 | Verify every feature and ID part, logical manifest hash, seal, row/group count, prevalence, and time range independently. |
| FR-208 | Fit medians, constant detection, clipping quantiles, and scaling on train only. |
| FR-209 | Never open the sealed test partition during training/tuning. |

### 5.4 Phase 3 — feature engineering and parity

| ID | Requirement |
|---|---|
| FR-301 | Define each base feature's name, kind, unit, range, missing policy, formula, rationale, aliases, and availability. |
| FR-302 | Canonicalize permitted syntax while preserving opaque stream bodies. |
| FR-303 | Use bounded scanner/parser/logical-graph views without rendering or JavaScript execution. |
| FR-304 | Enforce file, time, page, resource, graph, stream, filter, decompression, and expansion limits. |
| FR-305 | Emit typed extraction status and abstain on critical unsupported inputs. |
| FR-306 | Add domain-motivated scale, density, consistency, interaction, complexity, and parser-health features with deterministic formulas and lineage. |
| FR-307 | Document every final pipeline output, including imputation, missingness, clipping, units, ranges, rationale, and lineage. |
| FR-308 | Audit file size on train/validation and source/time during data validation; exclude raw creator/producer identities. |
| FR-309 | Materialize engineered data separately with checksums and proof that the label is not a feature. |
| FR-310 | Serialize one provenance-bound pipeline for batch and live inference. |
| FR-311 | Require exact record/batch parity on at least 100 sanitized examples and exact live/batch parity on at least 100 inert PDFs. |

### 5.5 Phase 4 — fair model comparison

| ID | Requirement |
|---|---|
| FR-401 | Compare always-benign, regularized logistic, Extra Trees, LightGBM, histogram XGBoost, generic MLP, and FT-Transformer candidates. |
| FR-402 | Train unweighted and cost-sensitive ablations at natural prevalence without SMOTE. |
| FR-403 | Implement numerical feature tokens, feature identity, attention, pre-norm residuals, GEGLU, dropout, and asymmetric focal loss for the FT-Transformer. |
| FR-404 | Tune with complete groups on a deterministic representative train subset and expanding temporal train folds only. |
| FR-405 | Refit finalists on the complete configured train partition for at least three fixed seeds. |
| FR-406 | Use disjoint group-temporal validation roles for calibration fitting, calibration selection, and threshold selection. |
| FR-407 | Lock 0.5, max-F1, max-F2, FPR <=0.1%, and FPR <=0.01% thresholds on validation only. |
| FR-408 | Record wall time, peak RAM, versions/hardware, size, throughput, and latency. |
| FR-409 | Select by F2 under FPR <=0.1%; require paired group-bootstrap superiority and acceptable calibration/latency before choosing a neural model over the best tree. |
| FR-410 | Save immutable calibrated multi-seed bundles and prove test content remained unopened. |

### 5.6 Phase 5 — locked metric and error analysis

| ID | Requirement |
|---|---|
| FR-501 | Claim the sealed test exclusively once; a failed or completed claim cannot be retried under the same experiment. |
| FR-502 | Verify Phase 4 bundle, pipeline, split, dataset, and matrix checksums before scoring. |
| FR-503 | Report Precision, Recall, F0.5/F1/F2, ROC/partial ROC, PR-AUC, specificity, FPR/FNR, MCC, balanced accuracy, Brier, ECE, counts, and false positives per million. |
| FR-504 | Evaluate each candidate at every validation-locked threshold without selecting on test. |
| FR-505 | Produce group-stratified 95% intervals and paired model differences. |
| FR-506 | Report source/time/size/encryption/active-content/parser/obfuscation/missingness subgroups and explicitly mark unavailable family/campaign fields. |
| FR-507 | Produce reliability, PSI/Jensen-Shannon drift, and hashed numeric-only hardest FP/FN profiles. |
| FR-508 | Close the test ledger with checksums for every output and publish final metrics only afterward. |

### 5.7 Phase 6 — deep explainability

Phase 6 must combine attribution, permutation, ablation, ALE/interactions,
stability, sanity, faithfulness, subgroup, and counterfactual evidence and attach
an actionable conclusion to each supported finding. The implementation uses a
real stratified train background, held-out validation evidence, retrained family
ablations, 100-bootstrap stability, negative controls, deletion/insertion tests,
and a sanitized Phase 5 handoff; it never reopens sealed test.

### 5.8 Phase 7 — adversarial requirements

Phase 7 must use only
bounded locally generated inert PDFs, exercise the declared mutation families
and defense matrix, and report clean-versus-robust security and resource
trade-offs.

The implementation generates inert PDFs locally, validates at least eight
mutation families and combinations without rendering or persistence, measures
clean/robust marker performance and resources, and separates demonstrated
defenses from controls implemented elsewhere and future proposals. Inert fixture
labels are never described as malware ground truth.

### 5.9 Phase 8 — deployment and application

| ID | Requirement |
|---|---|
| FR-801 | Package preprocessing, calibrated ensemble, locked threshold/policy, schema, provenance, train reference, and OOD policy in one versioned bundle. |
| FR-802 | Reject bundle checksum, identity, sidecar, threshold, and feature-schema mismatch; no legacy fallback is allowed. |
| FR-803 | Return benign, malicious, or uncertain/abstain and fail closed on parser/resource/OOD/threshold-margin conditions. |
| FR-804 | Display raw actionable indicators separately from local model attributions. |
| FR-805 | Keep upload bytes local, delete temporary files, and retain no PDF in deployment artifacts. |
| FR-806 | Restrict any optional LLM to structured evidence and prohibit invented CVEs, indicators, payloads, intent, or certainty. |
| FR-807 | Prove exact app/bundle parity on 100 golden inert fixtures and bounded abstention for corrupt/oversized inputs. |

### 5.10 Phase 9 — staged verification

| ID | Requirement |
|---|---|
| FR-901 | Expose independent stage commands with exact upstream-status and hash checks. |
| FR-902 | Verify sealed final metrics and the Phase 5–8 manifest chain before release. |
| FR-903 | Verify bundle/schema compatibility, documentation synchronization, and uploaded-PDF non-retention. |
| FR-904 | Cover data, leakage, parity, metrics, calibration, explainability, adversarial, deployment, LLM-grounding, and documentation contracts in automated tests. |

### 5.11 Phase 10 — documentation integrity

| ID | Requirement |
|---|---|
| FR-1001 | Generate active Markdown and LaTeX metrics from checksummed artifacts; do not hand-type final scores. |
| FR-1002 | Keep verified final results, exact archived historical values, and author-reported manual ranges visibly separate. |
| FR-1003 | Maintain dataset/model cards, reproducibility guide, seven stage notebooks, implementation reports, README, report, PRD, plan, and tracker. |
| FR-1004 | Bind generated outputs to the active experiment-summary and historical-archive hashes. |

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | Multi-million-row operations use bounded batches or disk-backed indexes. |
| NFR-02 | Dataset and split versions are immutable. |
| NFR-03 | Hash and count verification fails closed. |
| NFR-04 | Randomized operations use the configured seed. |
| NFR-05 | Output schemas and physical numeric types are deterministic. |
| NFR-06 | Critical writes use same-directory temporary files, fsync, and atomic replace. |
| NFR-07 | Tests cover success, bypass attempts, and tampering paths. |
| NFR-08 | Documentation distinguishes implemented controls from empirical evidence. |

## 7. Required artifacts

```text
configs/experiment.yaml
configs/data_sources.yaml
reports/archive/author_verified_pre_remediation/manifest.json
reports/results/experiment_summary.json
reports/data/dataset_quality.json
reports/data/dataset_quality.md
reports/data/transformation_manifest.json
reports/data/feature_shortcut_audit.json
reports/data/feature_dictionary_v2.json
data/processed/validated_v2/<version>/
data/splits/<version>/split_manifest.json
data/splits/<version>/SEALED
data/processed/engineered_v2/<version>/
models/feature_pipeline_v2.pkl
models/feature_pipeline_v2_tree.pkl
models/feature_pipeline_v2_neural.pkl
models/phase4/*.pkl
reports/results/phase4_champion.json
reports/results/phase4_*.csv
reports/results/metrics_by_model.csv
reports/results/metrics_by_threshold.csv
reports/results/metrics_by_subgroup.csv
reports/results/bootstrap_differences.csv
reports/results/calibration.csv
reports/results/temporal_feature_drift.csv
reports/results/sanitized_hard_errors.csv
reports/results/SEALED_TEST_EVALUATION.json
reports/results/phase6_handoff/manifest.json
reports/explainability/manifest.json
reports/explainability/global_importance.csv
reports/explainability/importance_stability.csv
reports/explainability/interactions.csv
reports/explainability/feature_ablations.csv
reports/explainability/subgroup_errors.csv
reports/explainability/local_cases.json
reports/explainability/counterfactuals.csv
reports/explainability/sanity_checks.json
reports/explainability/actionable_conclusions.md
reports/adversarial/manifest.json
reports/adversarial/mutation_validity.csv
reports/adversarial/robustness_metrics.csv
reports/adversarial/robustness_by_mutation.csv
reports/adversarial/defense_matrix.csv
reports/adversarial/defense_comparison.csv
reports/adversarial/resource_overhead.csv
reports/adversarial/threat_model.md
reports/adversarial/defense_conclusions.md
models/deployment/deployment_bundle_v1.joblib
models/deployment/manifest.json
reports/results/phase8_manifest.json
reports/results/phase9_verification.json
reports/results/documentation_sync_manifest.json
docs/generated/results_summary.md
report/generated_results.tex
reports/figures/roc_full.png
reports/figures/roc_low_fpr.png
reports/figures/precision_recall.png
reports/figures/calibration.png
```

The Phase 1–8 artifacts are generated from the approved source after its
upstream provenance, schema, scale, and prevalence gates pass.

## 8. Acceptance verification

Verification requires compilation, Ruff, the complete test suite, phase-aware
artifact checks, and inspection of the active summary. Acceptance also includes
independent verification of the real dataset, sealed split, and serialized
pipeline.

The implementation suite covers scientific contracts and bypass/tamper paths.
The active summary records the author's verification of more than 1,000,000
rows and the checked project measurements.

## 9. Evidence safeguards

- A neural model may outperform a tree only with the required paired evidence.
- F1, F-beta, Precision, Recall, ROC, PR, and accuracy remain tied to their
  recorded threshold and evaluation role.
- SHAP is never treated as a complete explanation by itself.
- Adversarial robustness claims remain tied to the declared inert-PDF benchmark.
- Deployment requires a validated threshold and Phase 8 bundle.
- Author-verified measurements and checksummed recorded values remain clearly
  identified in the evidence chain.

Phase 6 consumes the Phase 4–5 artifacts, and Phase 7 consumes the checksummed
Phase 6 evidence.
