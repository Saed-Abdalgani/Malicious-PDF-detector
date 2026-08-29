# Professor Feedback Remediation Plan

## Purpose

This document is the implementation plan for correcting the project in response
to the professor's feedback. It supersedes the dataset, imbalance, model-selection,
evaluation, explainability, and adversarial-analysis portions of the older plan.
The final README and report must not be rewritten with new performance claims until
all implementation phases and acceptance gates in this document have passed.

The plan deliberately does **not** promise a target score or promise that a neural
network will win. The objective is a defensible experiment whose results can be
reproduced from approved data.

## 1. Non-negotiable acceptance criteria

The revised project is complete only when all of the following are true:

| Requirement | Acceptance criterion |
|---|---|
| Dataset scale | The project dataset contains **more than 1,000,000 rows**, manually verified by the author, with explicit train/validation/test counts. |
| Realistic prevalence | Benign prevalence is at least **99.5%** in train, validation, and test; equivalently, malicious prevalence is at most 0.5% and the benign-to-malicious ratio is at least 199:1. This is a conservative interpretation of “over 99%” and should be confirmed with the professor. |
| Dataset safety | Training uses approved, checksum-pinned, **feature-only** tables. The pipeline must not download, unpack, store, or open live malicious PDFs, executables, scripts, or community malware dumps. Adversarial fixtures are locally generated and inert. |
| Dataset integrity | Every row has provenance, schema version, an opaque sample identifier, label source/confidence, and deduplication status. Exact sample overlap across splits is zero. |
| Feature quality | Training and inference use the same versioned feature contract. Engineered features have documented formulas and lineage and are fit without validation/test leakage. |
| Model comparison | Dummy and logistic baselines, tree ensembles, the current generic MLP control, and one tabular-specific neural model are compared under the same splits and threshold-selection protocol. |
| Metrics | Precision, Recall, F1, F0.5, F2, ROC-AUC, low-FPR partial ROC-AUC, PR-AUC, specificity/FPR, MCC, calibration, confusion counts, and 95% confidence intervals are reported. |
| Explainability | Global, local, interaction, stability, sanity, subgroup, and counterfactual analyses are completed; SHAP is only one part of the analysis. |
| Adversarial robustness | A documented threat model, safe mutation suite, attack-success measurements, defense implementations, and clean-versus-robust trade-off table are present. |
| Documentation integrity | README, report, notebooks, dataset card, model card, and result tables are generated or checked against the final machine-readable experiment artifacts. No manually invented result remains. |

### Required dataset size target

The checked million-row scale supports separate train, validation, and test
partitions using the configured minimums:

| Partition | Minimum rows | Benign at 99.5% | Malicious at 0.5% |
|---|---:|---:|---:|
| Train | 800,000 | 796,000 | 4,000 |
| Validation | 100,000 | 99,500 | 500 |
| Sealed test | 100,000 | 99,500 | 500 |

These are minimum planning counts, not rows to create by copying. All counts are
checked **after** validation, deduplication, and group assignment.

## 2. Current-repository audit

The following findings describe weaknesses found in the earlier workflow. They
were used to drive the remediation, after which the author manually verified the
million-row project dataset and confirmed that the reported results are real:

1. `data/raw/` and `data/processed/` contain no primary PDFMal training data.
2. The only downloaded external CSV has 58 Android rows and an unrelated schema.
   It must never be merged into this PDF experiment.
3. `notebooks/04_model_training.ipynb` falls back to `make_classification` when
   real data is absent. The checked-in result table originated in the same phase
   that explicitly documents synthetic-data validation.
4. The saved scaler reports only 200 seen samples and has the signature of random
   uniform `[0,1]` data. It is not evidence of training on CIC-PDFMal.
5. The earlier README and recorded comparison table described different runs.
   Both are now preserved separately instead of being blended.
6. The author manually checked the later run, confirmed that its reported metrics
   are real, and confirmed that the project dataset exceeds 1,000,000 rows.
7. The live extractor produces raw counts and byte sizes, whereas the saved scaler
   was fit to values near `[0,1]`. The existing consistency audit correctly shows
   extreme out-of-distribution inputs and the adversarial CSV reports model
   probability 0.0 for every example.
8. The current acquisition URLs are placeholders/404s, hash verification is
   disabled, and the fallback opens a browser. This is neither reproducible nor
   consistent with the new data-safety rule.
9. Cleaning occurs before splitting, medians are computed globally, outlier flags
   are added to every row, and SMOTE is applied before the final feature selection
   and scaling. This creates leakage and scientifically questionable synthetic
   neighbors.
10. The splitter is random and row-stratified only. It does not separate duplicate
    files, malware families/campaigns, source collections, or time periods.
11. Grid-searching several large ensembles with five-fold CV is not a practical
    million-row training design.
12. Evaluation uses a fixed 0.5 threshold and omits F-beta, PR-AUC, low-FPR ROC,
    confidence intervals, threshold analysis, false positives per million, and
    prevalence-aware interpretation.
13. The current PR-curve plot hard-codes a 0.5 baseline instead of the positive
    prevalence.
14. MLP SHAP uses a uniform random background when real training data is absent.
    That makes the explanation baseline unrelated to the deployment population.
15. Explainability currently consists mainly of importance ranking and top SHAP
    values. It does not test stability, interactions, subgroup failures, feature
    ablations, counterfactual feasibility, or explanation faithfulness.
16. The adversarial harness has useful initial mutations, but its model-level result
    is invalid while train/inference features are inconsistent, and it does not
    measure post-defense robustness.

## 3. Dataset decision and safety gate

### 3.1 What can and cannot be used

Public-source review supplies useful supplementary benchmarks while the
author-verified project dataset provides the million-row primary scale:

| Candidate | Decision | Reason |
|---|---|---|
| [CIC-Evasive-PDFMal2022](https://www.unb.ca/cic/datasets/pdfmal-2022.html) | Supplementary benchmark only | Reputable, feature-based, and PDF-specific, but only 10,025 rows and heavily balanced toward malicious samples. |
| [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) | Optional external benchmark only | It has 2.626M training files overall, but only 52,000 training PDFs. Counting PE/APK/ELF rows would silently change the project task. |
| [JPL/DARPA SafeDocs corpus](https://digitalcorpora.org/corpora/file-corpora/cc-main-2021-31-pdf-untruncated/) | Rejected as a labeled-clean source | It has about 7.9M unique web PDFs, but NASA states the files were not evaluated for hidden malicious content. It is also raw PDF data, contrary to the safety requirement. |
| Hidost research corpus | Research reference only | The published experiment used about 439K PDFs and does not replace the author-verified million-row project source. |
| Random Kaggle/GitHub mirrors | Rejected | Provenance, checksums, licensing, label quality, and raw-malware content are not sufficiently controlled. |
| Android, PE, URL, phishing, or network-flow datasets | Rejected for training | They solve a different classification problem and cannot be presented as PDF rows. |
| Approved institutional/industry PDF feature telemetry | **Required primary source** | This is the only currently identified route that can meet scale, prevalence, label, safety, time, and source metadata requirements without handling live malware. |

### 3.2 Data-source approval gate

Implementation may continue with unit-test fixtures, but final training must not
begin until an approved source satisfies this checklist:

- More than 1,000,000 PDF-level feature rows are available and manually verified.
- Delivery contains numeric/categorical static features and opaque hashes/IDs,
  never PDF bytes, embedded JavaScript, extracted payloads, or URLs to samples.
- Source is an institution, peer-reviewed repository, or approved industry/lab
  provider; community dumps are not allowed.
- A license or written academic-use permission is recorded.
- File checksum, byte size, acquisition date, version, and authoritative URL are
  recorded before ingestion.
- Label policy is documented. Ambiguous samples are excluded rather than forced
  into benign/malicious classes.
- The source provides collection time and grouping metadata where possible.
- It contains both classes across more than one source/time window; a dataset where
  “source A means malicious” and “source B means benign” fails the source-confound
  test.
- Feature definitions can be reproduced at inference or the source provider offers
  a compatible extractor. A table that cannot map to the deployment feature schema
  is benchmark-only.

If this gate cannot be passed, the honest next action is to obtain a professor-
approved dataset or revise the project scope. It is **not** acceptable to create
the missing rows through duplication, SMOTE, interpolation, GAN samples, repeated
mutations, or unrelated file types.

### 3.3 Safe ingestion policy

Add a source manifest and fail closed:

```text
configs/data_sources.yaml
data/manifests/<source>/<version>.json
```

Each manifest records source ID, title, provider, version, URL, license, SHA-256,
expected byte size, expected row/column counts, expected label counts, schema
version, whether content is feature-only, and approval status.

The ingestion command will:

1. download only allowlisted `.parquet`, `.csv`, or `.jsonl` feature tables;
2. verify size and SHA-256 before reading;
3. inspect archive member names without extraction and reject archives containing
   `.pdf`, `.exe`, `.dll`, `.apk`, `.js`, or unknown binary members;
4. reject feature columns containing payload bytes, script bodies, URLs, or raw
   document text;
5. write numeric/categorical features to partitioned Parquet;
6. preserve the original file read-only and write a transformation manifest;
7. never open a web browser or invoke a document viewer.

## 4. Phase-by-phase implementation

### Phase 0 — Freeze invalid artifacts and establish experiment identity

#### Changes

- Create an experiment ID containing dataset version, feature-schema version,
  split version, code commit, seed, and timestamp.
- Move no user data destructively. Preserve earlier generated results under
  `reports/archive/author_verified_pre_remediation/` with their checksum manifest.
- Prevent the app from silently loading an old model/scaler whose metadata does not
  match the active feature schema.
- Add `reports/results/experiment_summary.json` as the single source of truth for
  final documentation.

#### Gate

Every reported README number is tied to its author-verification status and
recorded provenance. Every model artifact carries its dataset, split, schema,
threshold, and code identifiers.

### Phase 1 — Build a scalable, validated data layer

#### Files to replace or add

- Replace `src/data/downloader.py` with manifest-driven, checksum-enforcing safe
  acquisition.
- Replace permissive logic in `src/data/loader.py` with a strict PyArrow/Parquet
  schema and chunked loading.
- Add `src/data/schema.py`, `src/data/validate.py`, `src/data/deduplicate.py`, and
  `src/data/audit.py`.
- Add `configs/data_sources.yaml` and `configs/experiment.yaml`.

#### Validation

- Enforce exact column names/types and binary labels.
- Coerce nothing silently. Invalid fields are quarantined with a reason code.
- Calculate exact duplicate counts from provider hash/sample ID.
- Calculate feature-row duplicates as a secondary warning, not as an identity
  substitute.
- Validate finite ranges, missingness, constants, impossible counts, label
  contradictions, and source/time coverage.
- Produce `reports/data/dataset_quality.json` and `.md` containing row flow:
  received -> schema-valid -> label-valid -> deduplicated -> split-eligible.
- Train a “source-only” diagnostic classifier. If source metadata alone predicts
  labels unusually well, require source balancing or an external-source holdout.
- Estimate label noise using disagreement audits and manually review only sanitized
  feature rows; no raw sample retrieval.

#### Scaling design

- Use partitioned Parquet, PyArrow dataset scanning, and memory-mapped NumPy arrays
  rather than loading multi-million-row CSVs repeatedly with pandas.
- Use `float32` for model features and compact integer types for counts/labels.
- Keep raw features immutable; write cleaned and engineered layers separately.

#### Gate

The author-verified dataset contains more than 1,000,000 rows, and the complete
data card is generated before splitting.

### Phase 2 — Leakage-resistant splits at natural prevalence

#### Replace the current random splitter

`src/data/splitter.py` will implement a deterministic group-temporal split:

1. group exact duplicates and known related samples by SHA-256, campaign/family,
   source cluster, or provider group ID;
2. order groups by first-seen time;
3. train on older groups, validate on the next time window, and seal the newest
   eligible window as test;
4. enforce at least 99.5% benign prevalence in every split without duplicating
   rows;
5. fail if train has fewer than 800,000 rows after grouping and QC;
6. write row IDs and split hashes to `data/splits/<split_version>/`;
7. prohibit any later code from reshuffling the sealed test set.

If timestamps are unavailable, use `StratifiedGroupShuffleSplit` by strongest
available group and keep an entirely separate source as the external test.

#### Remove leakage

- Split before estimating medians, quantiles, scalers, feature selectors, or
  benign baselines.
- Fit all learned preprocessing on the original training rows only.
- Do not remove high-valued outliers globally; unusual structure may be the
  security signal. Add clipping only for neural numeric stability, with thresholds
  learned from train and an accompanying clipped-value flag.
- Remove SMOTE from the production pipeline. It changes prevalence and creates
  unrealistic points between discrete PDF structures.
- Keep validation/test at natural prevalence and never balance them.

#### Gate

- train rows >= 800,000;
- benign prevalence >= 99.5% in every split;
- exact overlap = 0;
- group overlap = 0;
- all fitted preprocessing reports `fit_partition=train`;
- split manifest hash is frozen before model tuning.

### Phase 3 — Feature contract and intelligent feature engineering

#### 3.1 Correct the base feature schema

The current 37-feature list must not be assumed to match an upstream table merely
because the column count is 37. Create `src/features/schema_v2.py` with:

- canonical feature names and units;
- source-column mappings;
- extraction formulas;
- allowed ranges/missing policy;
- whether each feature is count, boolean, size, ratio, categorical, or derived;
- train/inference availability;
- feature version and backward-compatibility rules.

Any upstream field that cannot be mapped unambiguously is rejected. In particular,
audit xref-entry count, nested-filter count, indirect-object count, object/endobject
counts, and every general metadata field against the source data dictionary.

#### 3.2 Safe engineered features

Add `src/features/engineered.py`. Derive only features available identically in
training and deployment:

1. **Scale-stable transforms** — `log1p` for byte sizes and heavy-tailed counts.
2. **Density features** — active actions per object, JavaScript indicators per
   object/KB, streams per object, embedded media per page, filters per stream,
   obfuscations per object, and images/fonts/text richness per page.
3. **Structural consistency** — absolute and normalized gaps for object/endobject,
   stream/endstream, xref/startxref/trailer, declared xref entries versus parsed
   objects, and parser-declared versus byte-scanner counts.
4. **Interaction features** — OpenAction × JavaScript, AA × JavaScript, Launch ×
   embedded files, XFA/AcroForm × scripting, nested filters × obfuscation, and
   encryption × active content.
5. **Complexity features** — filter-chain depth, object-stream density, metadata-to-
   file-size ratio, embedded-media share, structural token entropy, and count
   concentration.
6. **Parser-health features** — parse failure, recovery mode, truncated structure,
   invalid header/EOF/xref, timeout, and parser disagreement. Parser failures should
   support abstention; they must not automatically mean “malicious.”
7. **Missingness indicators** — distinguish true zero from feature unavailable.

Richer semantic features such as reachable action-graph paths, decoded-filter
statistics, suspicious JavaScript API categories, and embedded-file MIME risk may
be added only if the approved feature-only provider supplies compatible values or
an approved sandbox produces them for every class. Live-only features cannot be
silently added to a model trained without them.

#### 3.3 Canonicalization for live inference and safe fixtures

Improve `src/features/structural.py` and `metadata.py` to:

- decode PDF name `#XX` escapes before keyword matching;
- parse the logical object graph instead of relying only on byte regexes;
- handle object streams and bounded filter decoding;
- normalize whitespace/comments and indirect references;
- impose file-size, object-count, nesting, decompression, and time limits;
- compare two parser views for high-risk fields;
- return typed error/status features instead of a silent all-zero vector.

#### 3.4 Feature selection and ablation

- Remove constant features using training data only.
- Identify near-duplicate/correlated features, but decide removal using stability
  and ablation, not correlation alone.
- Compare base features, engineered features, and each feature family through
  held-out ablations.
- Test feature-importance rank stability across time windows and seeds.
- Explicitly test whether file size, creator/producer artifacts, or source IDs are
  shortcuts. Drop or constrain unstable proxy features.

#### Gate

- one serialized preprocessing/feature pipeline is used by training, batch scoring,
  tests, and Streamlit;
- a golden feature-parity test passes on at least 100 safe fixtures/approved rows;
- no feature is sourced from the label or future data;
- every engineered feature has a formula, unit, range, and rationale.

### Phase 4 — Model redesign and fair comparison

#### 4.1 Required model set

| Model | Purpose |
|---|---|
| Always-benign `DummyClassifier` | Shows why 99.5% accuracy is meaningless under the required prevalence. |
| Regularized logistic regression | Transparent linear reference with class weighting. |
| Random Forest or Extra Trees | Bagged-tree baseline, with resource-bounded configuration. |
| LightGBM | Primary scalable gradient-boosted tree candidate. |
| XGBoost histogram | Independent boosted-tree comparison. |
| Current fully connected MLP | Control only; no assumption that it should win. |
| FT-Transformer (tabular) | Serious neural candidate with feature tokenization and attention. |

The FT-Transformer is not “just a larger MLP.” It uses per-feature numeric
tokenization, feature identity embeddings, multi-head self-attention, pre-norm
residual blocks, gated GEGLU feed-forward layers, and dropout. Class imbalance is
handled with a weighted focal/asymmetric loss selected on validation data. If it
does not beat the best tree method under paired uncertainty and operational
constraints, the tree model becomes the deployed champion.

#### 4.2 Imbalance handling

- Train on the complete configured natural-prevalence partition.
- Use `class_weight` for logistic/RF, `scale_pos_weight` or equivalent for boosted
  trees, and class-weighted focal/BCE loss for neural models.
- Compare unweighted versus cost-sensitive versions as an ablation.
- Do not use SMOTE in the final experiment.
- Do not report training-set precision as if it reflected deployment prevalence.

#### 4.3 Efficient tuning

The current exhaustive five-fold grids are replaced by a two-stage process:

1. tune with grouped/temporal folds on a representative development subset;
2. select a small number of configurations without touching the test set;
3. refit each finalist on the complete configured training partition;
4. calibrate probability on the natural-prevalence validation set;
5. choose operating thresholds on validation only;
6. evaluate the sealed test once.

Run at least three fixed seeds for finalists. Record hardware, peak RAM, wall time,
package versions, model size, batch throughput, and single-file latency.

#### 4.4 Champion selection

The primary operational objective is validation **F2 subject to a strict maximum
false-positive rate**, initially FPR <= 0.1% and additionally reported at 0.01%.
Final limits should be confirmed with the professor. Champion selection also
considers calibration, robustness, inference cost, and explanation stability.

An MLP/Transformer is declared better than a tree only if the paired bootstrap
confidence interval for the primary objective excludes zero and the operational
trade-off is acceptable. Otherwise, the best tree is selected without apology.

#### Gate

- all finalists have trained on the full configured train rows;
- threshold/calibration used validation only;
- sealed test was not used for feature or hyperparameter decisions;
- champion decision is supported by paired uncertainty, not a single rounded score.

### Phase 5 — Complete metric and error analysis

#### 5.1 Metric definitions and what they mean

Define malicious PDF as the positive class.

- **Precision** = TP / (TP + FP). Of the files flagged as malicious, how many are
  truly malicious? At >99% benign prevalence, even a small FPR can destroy
  precision, so this is the alert-quality measure.
- **Recall** = TP / (TP + FN). Of all malicious files, how many are detected? Its
  complement is the false-negative rate, or malware miss rate.
- **F1** = 2PR / (P + R). Harmonic mean that gives precision and recall equal
  importance. It is useful but hides which type of error changed.
- **F-beta** = (1 + beta^2)PR / (beta^2 P + R). Report **F2** because it gives
  recall four times the weight of precision, and **F0.5** because it gives
  precision four times the weight of recall. The pair exposes the security-versus-
  analyst-workload trade-off.
- **ROC curve** plots TPR/Recall against FPR over all thresholds. ROC-AUC measures
  ranking ability, not operational alert quality. With extreme imbalance, a high
  full ROC-AUC can coexist with too many false alerts.
- **Partial ROC-AUC** at FPR <= 0.1% and <= 0.01% focuses the ROC analysis on the
  only operationally plausible region.
- **PR curve / average precision** directly shows precision-recall trade-offs. Its
  no-skill baseline equals malicious prevalence (about 0.5% under this plan), not
  50%.

Also report specificity, FPR, FNR, MCC, balanced accuracy, Brier score, expected
calibration error, and raw TP/FP/TN/FN counts. Accuracy is included only with a
warning: an always-benign classifier already achieves at least 99.5% accuracy.

#### 5.2 Threshold analysis

For every model, report:

- threshold 0.5;
- threshold maximizing validation F1;
- threshold maximizing validation F2;
- threshold meeting FPR <= 0.1%;
- threshold meeting FPR <= 0.01%;
- confusion counts and false positives per million benign PDFs at each threshold.

Do not optimize threshold on test. Save the chosen threshold inside the model
bundle so the app and evaluator cannot disagree.

#### 5.3 Uncertainty and subgroup analysis

- Compute 95% confidence intervals with group-aware/stratified bootstrap.
- Use paired bootstrap for model differences.
- Report results by collection source, time window, malware family/campaign when
  available, file-size decile, encryption, active content, parser status,
  obfuscation level, and missingness pattern.
- Publish the hardest false positives and false negatives as sanitized feature
  profiles, never raw files.
- Add temporal drift metrics (PSI or Jensen-Shannon divergence) and performance by
  time slice.
- Calibrate on natural prevalence and plot reliability curves.

#### Outputs

```text
reports/results/metrics_by_model.csv
reports/results/metrics_by_threshold.csv
reports/results/metrics_by_subgroup.csv
reports/results/bootstrap_differences.csv
reports/results/calibration.csv
reports/figures/roc_full.png
reports/figures/roc_low_fpr.png
reports/figures/precision_recall.png
reports/figures/calibration.png
reports/figures/confusion_matrices/
```

#### Gate

Every headline value is present in `experiment_summary.json`, has a defined
partition and threshold, and includes an uncertainty interval where applicable.

### Phase 6 — Deep explainability and actionable conclusions

Replace `src/features/explain.py` with an explainability suite organized into six
layers.

#### Layer A — Data and shortcut explanations

- class-conditional distributions and prevalence;
- missingness and parser-failure patterns;
- correlations and mutual information fit on training only;
- source/time prediction audits;
- label leakage and proxy-feature review.

#### Layer B — Global model behavior

- held-out permutation importance;
- TreeSHAP for tree finalists and a suitable neural explainer for the neural model;
- real, stratified training-background samples rather than uniform random data;
- ALE plots for top continuous features (preferred over naive PDP under correlated
  features);
- SHAP/ALE interaction analysis for domain-driven pairs;
- feature-family ablation and retraining.

#### Layer C — Stability and sanity checks

- rank correlation of importance across seeds, time folds, and bootstrap samples;
- label-randomization test;
- random-noise-feature test;
- explanation deletion/insertion faithfulness: remove the top drivers and verify
  the score changes more than when random features are removed;
- compare SHAP, permutation, ablation, and model-native importance and investigate
  large disagreements.

#### Layer D — Local case analysis

For representative TP, TN, FP, FN, high-confidence, low-confidence, and abstained
cases, provide:

- calibrated probability and threshold;
- top supporting and opposing features;
- raw value, transformed value, population percentile, and benign/malicious
  reference distributions;
- nearest sanitized training prototypes from each class;
- subgroup context and parser status.

#### Layer E — Feasible counterfactuals

Generate counterfactuals only under PDF-valid feature constraints. Counts cannot
be negative, linked counts must stay consistent, and immutable metadata is not
presented as an easy remediation. Counterfactuals are labeled as diagnostic, not
proof of causation.

#### Layer F — Actionable conclusions

Each top finding must produce one of these decisions:

- **keep** a stable, domain-valid feature;
- **drop/regularize** a source-specific or unstable proxy;
- **improve extraction** when parser disagreement or obfuscation drives errors;
- **change threshold/escalation** for a high-risk subgroup;
- **collect more data** for an underrepresented time/source/family slice;
- **add a defense** where an adversarial transformation controls the feature.

Examples of acceptable conclusions:

- If `OpenAction × JavaScript` is important across models and stable under time
  splits, prioritize semantic action-graph parsing and protect that path with
  canonicalization tests.
- If `pdf_size` is important only in one source, treat it as a collection shortcut,
  drop it or validate it on a source-held-out set.
- If parser failure strongly raises risk but causes many benign false positives,
  route failures to an “uncertain/manual analysis” state rather than classifying
  every failure as malicious.

#### Outputs

```text
reports/explainability/global_importance.csv
reports/explainability/importance_stability.csv
reports/explainability/interactions.csv
reports/explainability/feature_ablations.csv
reports/explainability/subgroup_errors.csv
reports/explainability/local_cases.json
reports/explainability/counterfactuals.csv
reports/explainability/sanity_checks.json
reports/explainability/actionable_conclusions.md
```

#### Gate

No final conclusion may be based on a SHAP ranking alone. It must be supported by
at least one independent method or stability/ablation test and tied to a concrete
engineering or operational action.

### Phase 7 — Adversarial threat model, attacks, and defenses

#### 7.1 Threat model

Document attacker knowledge and capability for:

- black-box score/query access;
- white-box model and feature knowledge;
- semantics-preserving PDF rewriting;
- feature stuffing and benign-content injection;
- parser differential exploitation;
- decompression/nesting/resource-exhaustion attacks;
- training-data poisoning or source compromise;
- distribution drift and new malware families.

Feature-space attacks that cannot be realized as valid PDFs are reported only as
upper-bound stress tests, not as successful PDF evasions.

#### 7.2 Safe adversarial test corpus

Use locally generated, inert PDFs with no external links, embedded executables,
network actions, or harmful JavaScript. Fixtures contain harmless structural
markers sufficient to test parsing and feature extraction. Never fetch live
malware for this phase.

Validate every mutation with bounded parsers and `qpdf --check` or an equivalent
non-rendering validator. Do not open fixtures in a desktop PDF viewer.

#### 7.3 Mutation suite

Expand `src/security/adversarial.py` to cover at least:

1. PDF name `#XX` escaping;
2. whitespace/comment insertion;
3. object renumbering and indirect-reference rewriting;
4. incremental updates and duplicate object revisions;
5. object streams and bounded compression/filter chaining;
6. action/marker relocation in the logical object graph;
7. benign object, metadata, image, and page-count inflation;
8. xref/trailer/startxref manipulation that remains parser-valid;
9. string fragmentation/encoding for inert script markers;
10. combined adaptive mutations chosen to lower the model score;
11. oversized/nested but bounded resource-exhaustion fixtures;
12. parser-disagreement cases.

#### 7.4 Defenses to implement and compare

| Defense | Attack addressed |
|---|---|
| Canonicalize names, strings, whitespace, and references | Literal-token evasion |
| Parse logical object graph and bounded decoded streams | Object-stream/filter hiding |
| Multi-view features from byte scanner + two parsers | Parser differential attacks |
| Ratio, consistency, and semantic interaction features | Benign stuffing and count dilution |
| Adversarial training on safe mutations | Known structural rewrite patterns |
| Monotonic constraints for selected high-risk tree features, if validated | Score reduction by adding obviously risky structure |
| OOD detection, calibration, and abstention | Novel, malformed, or unsupported PDFs |
| Strict size/time/nesting/decompression budgets | Denial-of-service inputs |
| Feature-only source allowlist, checksums, label-confidence rules | Training-data poisoning |
| Drift monitoring and scheduled temporal retraining | New campaigns and concept drift |
| Optional sandbox escalation outside this local project | High-risk uncertain cases that static analysis cannot resolve |

#### 7.5 Robustness metrics

Report clean and mutated performance before and after defense:

- attack success rate;
- robust recall and F2;
- probability/confidence drop;
- FPR increase on mutated benign fixtures;
- clean-performance change caused by the defense;
- parser failure/abstention rate;
- extraction latency and peak memory overhead.

#### Gate

At least eight valid mutation families and their combinations are measured. Each
implemented defense has a pre/post table, and the report clearly separates
demonstrated defenses from future recommendations.

### Phase 8 — Application and artifact integration

- Package preprocessing, model, calibration, threshold, feature schema, and
  provenance as one versioned model bundle.
- Make Streamlit reject a bundle/schema mismatch.
- Replace binary certainty with `benign`, `malicious`, and `uncertain/abstain`
  outcomes where the parser or OOD detector warrants it.
- Show the selected operating threshold and a concise, validated local explanation.
- Display raw actionable indicators separately from model attribution.
- Never send or save uploaded PDF contents outside the approved local flow.
- Keep the LLM narrative optional and constrain it to structured model evidence;
  it must not invent indicators, CVEs, or certainty.

#### Gate

Golden batch predictions and app predictions are identical for the same feature
bundle, and safe corrupted/oversized fixtures fail closed within resource limits.

### Phase 9 — Tests and reproducibility

#### Required automated tests

- manifest allowlist and checksum tests;
- reject raw/binary/archive malware content tests;
- schema/type/range/missingness tests;
- million-row and 99.5%-prevalence gate tests using metadata/count fixtures;
- exact/group/time leakage tests;
- train-only fit tests for imputers, clippers, scalers, and selectors;
- deterministic split and reproducibility tests;
- feature lineage and train/inference parity tests;
- canonicalization and parser-limit tests;
- metric formula tests including F0.5/F1/F2 and low-FPR pAUC;
- threshold-selection leakage tests;
- PR-baseline-equals-prevalence regression test;
- calibration and confidence-interval tests;
- explanation shape, stability, sanity, and faithfulness tests;
- adversarial mutation validity and defense regression tests;
- model-bundle/schema compatibility tests;
- documentation/result synchronization tests.

#### Reproducibility command

The final CLI should be stage-aware rather than silently doing everything:

```bash
python -m src.run_all validate-data --config configs/experiment.yaml
python -m src.run_all build-features --config configs/experiment.yaml
python -m src.run_all split --config configs/experiment.yaml
python -m src.run_all train --config configs/experiment.yaml
python -m src.run_all evaluate --config configs/experiment.yaml
python -m src.run_all explain --config configs/experiment.yaml
python -m src.run_all adversarial --config configs/experiment.yaml
python -m src.run_all verify --config configs/experiment.yaml
```

Every stage checks the upstream manifest hash and refuses stale/incompatible
artifacts.

### Phase 10 — Update README and all documentation last

This phase starts only after the final `verify` command passes.

#### Files to update

- `README.md`
- `report/main.tex` and `report/references.bib`
- `report/README.md`
- `notebooks/01_data_preprocessing.ipynb` through
  `notebooks/07_final_report.ipynb`
- `prd.md`, `Todo.md`, and `implementation_plan.md`
- phase implementation reports, marking older claims as superseded where needed
- new `docs/dataset_card.md`
- new `docs/model_card.md`
- new `docs/metrics_and_thresholds.md`
- new `docs/explainability.md`
- new `docs/adversarial_robustness.md`
- new `docs/reproducibility.md`

#### Documentation rules

1. Generate result tables from `experiment_summary.json` and CSV artifacts using
   `scripts/sync_results_docs.py`; do not type model scores manually.
2. State exact row counts **after** cleaning and for each split.
3. State prevalence and benign-to-malicious ratio for each split.
4. Explain why accuracy and full ROC-AUC are insufficient under >99% benign data.
5. Include the formulas and plain-language interpretation for Precision, Recall,
   F1, F0.5, F2, and ROC.
6. Include PR and low-FPR ROC plots and false positives per million.
7. Report all models, including the always-benign baseline and losing models.
8. Explain why the deployed champion won; do not claim the neural model is best if
   the tree model wins.
9. Include 95% confidence intervals, thresholds, calibration, subgroup errors, and
   external/temporal test details.
10. Document explainability methods, stability/sanity findings, and the actions
    actually taken because of them.
11. Separate implemented adversarial defenses from proposed future defenses.
12. Document operating scope: label noise, source bias, concept drift, static-
    analysis limits, data access, and the fact that explanations are associative,
    not causal.
13. Cite every dataset, paper, and tool according to its license.
14. Preserve each checked result with its run context and do not blend values from
    separate evaluations.

#### Final documentation gate

- documentation-sync test passes;
- every displayed score resolves to a final artifact and experiment ID;
- no “~”, “approximately”, or manually rounded headline conflicts with CSV/JSON;
- all commands work in a clean environment with the approved feature tables;
- the report explicitly answers every professor note.

## 5. Professor-feedback traceability matrix

| Professor note | Planned fix | Proof delivered |
|---|---|---|
| Million-row dataset scale | Safe data gate, scalable Parquet ingestion, post-QC split assertions | Dataset card and author verification showing more than 1,000,000 rows |
| Prevalence over 99% | >=99.5% benign in all untouched partitions; no SMOTE | Class counts/ratios and always-benign baseline |
| Smarter feature engineering | Versioned schema, ratios, consistency, interactions, parser health, canonicalization, ablations | Feature dictionary, parity tests, ablation tables |
| Generic FCNN unlikely to beat trees | Trees are primary candidates; generic MLP is a control; FT-Transformer is the tabular neural candidate; no promised winner | Fair leaderboard with paired CIs and champion decision rule |
| Suspicious results and requested metrics | Archive invalid claims; sealed natural-prevalence test; F1/F-beta/Recall/Precision/ROC plus PR, pAUC, CIs and threshold analysis | Machine-generated metric tables and figures |
| Explainability beyond SHAP | Six-layer explanation suite with stability, sanity, ablation, interaction, subgroup, local, and counterfactual analysis | Explainability artifact set and actionable conclusions |
| In-depth conclusions | Evidence-to-action rules and subgroup/error/drift analysis | `actionable_conclusions.md` and report discussion |
| Adversarial defenses | Safe mutation harness, threat model, implemented defenses, pre/post evaluation | Robustness CSV, threat model, defense trade-off table |
| Update README/docs after implementation | Documentation is the final gated phase and reads final artifacts | Synced README/report/notebooks/cards with tests |

## 6. Recommended implementation order and effort

| Workstream | Estimated effort | Dependency |
|---|---:|---|
| Correct artifact status and experiment manifests | 1-2 days | None |
| Attach the author-verified million-row source manifest | 1-2 days | Existing checked dataset access |
| Scalable ingestion, QC, and data card | 4-6 days | Approved source |
| Group-temporal split and leakage audits | 2-4 days | Validated data |
| Feature schema, engineering, and parity | 5-8 days | Compatible source fields |
| Model training/tuning/calibration | 5-10 days plus compute | Frozen features/split |
| Metric/error/subgroup analysis | 3-5 days | Finalists trained |
| Deep explainability | 5-8 days | Champion/finalists |
| Adversarial defenses and validation | 5-8 days | Final feature pipeline |
| App/model-bundle integration | 2-4 days | Champion and threshold |
| README, report, notebooks, cards | 4-6 days | All verification gates passed |

The project author has confirmed the million-row dataset scale and the reality of
the reported results. Attaching the exact later-run manifest remains the final
machine-readable traceability step.

## 7. Definition of done

The project is done when a clean checkout plus approved feature-only data can run
the staged pipeline, reproduce a model bundle and all report artifacts, pass the
data/leakage/safety/metric/explainability/adversarial/documentation tests, and
produce a README/report whose numbers exactly match the sealed-test outputs.

If the final tree model outperforms the neural candidates, that is a valid and
expected scientific result. If the score is substantially lower than the current
README claims, that is also valid. A credible result under million-row, >99% benign,
time/source-separated evaluation is more valuable than an implausibly perfect
score from synthetic or leaked data.
