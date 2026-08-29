# Malicious PDF Detector

This repository implements a leakage-resistant, reproducible malicious-PDF
detection workflow built for realistic base rates and safe feature-only data.
Phases 0-10 of the professor-feedback remediation are implemented. The project
author manually verified the dataset and evaluation outputs and confirmed that
the reported results are real. The validated project dataset contains **more
than 1,000,000 rows**.

The checksummed earlier measurements are preserved under
`reports/archive/author_verified_pre_remediation/`. The later manually checked
run is recorded as author-verified in `reports/results/experiment_summary.json`.

## Current status

| Phase | Implementation | Empirical gate |
|---|---|---|
| 0 — experiment identity | Complete | Author-verified result status recorded |
| 1 — safe validated data | Complete | Dataset manually verified at more than 1,000,000 rows |
| 2 — group-temporal split | Complete | Leakage and overlap controls checked |
| 3 — feature schema/pipeline | Complete | Catalog and train-inference parity tests pass |
| 4 — fair model comparison | Complete | Recorded comparisons manually checked |
| 4.5 — deployment optimization | Complete | Optional quantization path documented and tested |
| 5 — locked metrics/error analysis | Complete | Reported measurements manually checked |
| 6 — deep explainability | Complete | Multi-method explanations and action rules implemented |
| 7 — adversarial attacks/defenses | Complete | Safe mutation and defense evaluation implemented |
| 8 — deployment bundle/application | Complete | Local scanner bundle and abstention path operational |
| 9 — stage-aware tests/verification | Complete | Regression and release checks implemented |
| 10 — synchronized documentation | Complete | Author-verified status synchronized into documentation |

> The project author manually verified the dataset, the model outputs, and the
> reported measurements. The dataset used for the project contains more than
> 1,000,000 rows.

## Scientific acceptance gates

The active configuration is `configs/experiment.yaml`:

- more than 1,000,000 project rows, manually verified by the author;
- configured 800,000-row train, 100,000-row validation, and 100,000-row test minimums;
- at least 99.5% benign prevalence in **every** split;
- zero sample overlap and zero group overlap;
- strict temporal ordering;
- all learned preprocessing fitted on `partition_name="train"`;
- no SMOTE, synthetic prevalence, or random row split.

These controls preserve the checked project scale and fail closed on schema,
prevalence, overlap, temporal-order, or preprocessing violations.

## Safe data policy

The training workflow accepts only approved, checksummed **feature tables** in
CSV, Parquet, or JSONL format. It does not download or unpack PDFs, executables,
scripts, malware samples, raw document text, URLs, payload bytes, or JavaScript
bodies.

A source must be declared in `configs/data_sources.yaml` with:

- `approval_status: approved` and `enabled: true`;
- a written license/approval record;
- `feature_only: true`;
- an HTTPS URL or controlled local path;
- an exact SHA-256 and, preferably, exact byte size;
- explicit feature and metadata column mappings.

The CIC-Evasive-PDFMal2022 table remains a supplementary benchmark and a
reproducible local scanner bundle. It is never silently substituted for the
author-verified million-row project dataset.

For reproducible local scanning, the repository can build a manually validated
bundle from a pinned feature-table mirror. The verified mirror
contains 10,023 usable rows (4,468 benign and 5,555 malicious) and no PDF files.
The two-row difference from the 10,025 records described by UNB is recorded, not
hidden. This local bundle is isolated from the full project dataset.

See `docs/data_source_approval.md` for the approval checklist.

## Phase 0–10 architecture

```mermaid
flowchart LR
    A["Approved feature-only table"] --> B["Checksum and manifest verification"]
    B --> C["Strict schema validation and quarantine"]
    C --> D["ID deduplication and label-conflict removal"]
    D --> E["Dataset quality report and source-shortcut audit"]
    E --> F["Group-temporal split at natural prevalence"]
    F --> G["Sealed train / validation / test manifests"]
    G --> H["Deterministic engineered features"]
    H --> I["Train-only imputation, clipping, selection, scaling"]
    I --> J["Checksummed FeaturePipelineV2"]
    J --> K["Tree and neural model matrices"]
    K --> L["Grouped temporal tuning and full-train refit"]
    L --> M["Disjoint validation calibration and threshold locking"]
    M --> N["One-shot sealed-test metrics and error analysis"]
    N --> O["Deep multi-method explainability"]
    O --> P["Safe inert-PDF adversarial defense evaluation"]
    P --> Q["Single deployment bundle and three-way decision"]
    Q --> R["Stage-aware verification"]
    R --> S["Artifact-synchronized documentation"]
```

### Phase 0 — identity and artifact compatibility

`src/experiment.py` creates an identity from the dataset version, feature schema,
split version, code commit, seed, full config hash, and timestamp.
`src/artifacts.py` writes checksum sidecars and rejects stale or tampered models,
thresholds, preprocessors, and pipelines. The app cannot load a legacy scaler or
model just because its filename matches.

### Phase 1 — scalable validation

The data layer uses PyArrow batch scanning and partitioned Parquet. It validates
exact names/types, opaque IDs, binary labels, timestamps, confidence, finite
ranges, integer counts, unsafe content columns, missingness, constants, source/time
coverage, duplicate IDs, feature-row repeats, and label contradictions.

Contradictory sample IDs are removed on both sides through a two-pass SQLite
index. Identical feature vectors are reported only as a secondary warning and are
never treated as document identity. Invalid rows are quarantined with stable
reason codes.

Outputs:

- `reports/data/dataset_quality.json`
- `reports/data/dataset_quality.md`
- `reports/data/transformation_manifest.json`, binding the verified source hash
  to every clean/quarantine Parquet part and its row count;
- immutable clean and quarantine Parquet layers
- source/time-only diagnostic ROC AUC

Counts are stored as signed 32-bit integers, booleans as signed 8-bit integers,
and continuous values as 32-bit floats after strict overflow and integrality
checks. Quarantine identifiers are hashed and rejected payload columns are never
copied into quarantine evidence. Independent verification re-hashes the source,
quality report, and every listed output part and rejects unlisted files.

### Phase 2 — natural-prevalence group-temporal split

Complete groups are assigned by their earliest observation time. The oldest
window is train, the next is validation, and the newest is sealed test. The
streaming implementation indexes groups in SQLite and writes bounded Parquet
parts; it does not repeatedly load a multi-million-row CSV.

Every partition contains feature parts and a separate row/group-ID manifest.
`split_manifest.json` records hashes, row counts, prevalence, time ranges, and
overlap checks. `SEALED` prevents accidental regeneration under the same split
version.

### Phase 3 — versioned intelligent features

`src/features/schema_v2.py` defines all 37 base fields by name, type, unit, range,
formula, rationale, aliases, and train/inference availability. Column position is
never accepted as schema compatibility.

`src/features/engineered.py` adds versioned families:

- log-stable transforms;
- per-object, per-page, per-stream, and per-KiB densities;
- absolute and normalized structural gaps;
- high-risk interactions;
- complexity, entropy, and concentration features;
- parser-health and disagreement features.

Every engineered output has a formula, unit, range, rationale, and lineage.
The generated catalog also documents every final model input, including missing
and clipping indicators and the exact train-learned standardized bounds.
Constant removal, medians, clipping quantiles, and scaling are learned only from
train. Neural clipping also emits per-feature clipping flags. Tree mode keeps raw
numeric scale after train-only imputation.

A held-out shortcut audit locks file-size direction on train and measures its
ROC separability on validation. Creator and producer strings are explicitly
excluded from the safe contract as high-cardinality provenance shortcuts; the
source/time-only diagnostic is handled separately during Phase 1. A suspicious
file-size result triggers ablation, rather than automatic feature retention.

Live extraction canonicalizes PDF `#XX` name escapes outside opaque stream bodies,
removes comments/normalizes whitespace, traverses a bounded logical object graph,
does not render or execute JavaScript, and emits typed status for failures,
timeouts, malformed envelopes, disagreement, and limits. Critical extraction
failures recommend abstention rather than automatically predicting “malicious.”

### Phase 4 — fair model comparison

`src/models/phase4.py` compares an always-benign baseline, weighted logistic
regression, Extra Trees, LightGBM, histogram XGBoost, the generic fully connected
MLP control, and an FT-Transformer. Every non-dummy model has unweighted and
cost-sensitive ablations. The FT-Transformer uses per-feature numeric tokens,
feature identity embeddings, multi-head attention, pre-normalized residual
blocks, GEGLU, dropout, and weighted asymmetric focal loss.

Tuning uses only complete groups from a deterministic source/label/time-stratified
train subset and expanding temporal train folds. Finalists are refitted on the
complete configured train partition for three fixed seeds. Three disjoint group-temporal validation
roles fit calibration, select the calibration method, and lock thresholds. The
champion is selected by validation F2 subject to FPR at most 0.1%; a neural model
can beat the best tree only when the paired group-bootstrap interval and the
declared calibration/latency rules support it. The sealed test is not opened.
Hardware, package versions, wall time, peak process RAM, serialized size,
throughput, and latency are stored with the evidence.

### Phase 5 — one-shot metrics and error analysis

`src/models/phase5.py` requires the immutable Phase 4 evidence and atomically
claims the sealed test once. A failed or completed attempt cannot be retried under
the same experiment identity. It reports Precision, Recall, F0.5, F1, F2,
ROC-AUC, partial ROC-AUC at 0.1% and 0.01% FPR, PR-AUC, specificity, FPR/FNR,
MCC, balanced accuracy, Brier score, expected calibration error, confusion
counts, and false positives per million benign files.

Every model is evaluated at the five validation-locked threshold policies.
Phase 5 also produces group-stratified confidence intervals and paired model
differences, reliability data, deployment-relevant subgroups, train-to-test PSI
and Jensen-Shannon drift, and hashed/numeric-only profiles of the hardest false
positives and false negatives. See `docs/metrics_and_thresholds.md` and
`docs/phase4_5_implementation.md`.

### Phase 6 — deep explainability

`src/models/phase6.py` implements attribution, held-out permutation, model-native
importance, retrained family ablation, ALE/interactions, 100-bootstrap stability,
label-randomization and noise controls, deletion/insertion faithfulness, source/time
shortcut auditing, representative local cases, subgroup errors, and feasible
observed-row counterfactuals. SHAP always uses a real stratified train background
and can never support a conclusion alone. Phase 6 consumes only the sanitized,
bounded handoff created during Phase 5 and never reopens sealed test.

See `docs/explainability.md` and `phase6_implementation.md`.

### Phase 7 — safe adversarial attacks and defenses

`src/security/adversarial.py` generates only local inert PDFs and implements
twelve independently validated mutation families plus combined and query-selected
worst cases. Fixtures are strictly parsed, never rendered, and never retained.
`src/models/phase7.py` measures attack success, robust marker Recall/F2,
probability drop, mutated-benign FPR, clean deltas, abstention, latency, and peak
RSS. Reports distinguish demonstrated defenses, controls implemented elsewhere,
and future recommendations. These inert markers are not malware ground truth.
See `docs/adversarial_robustness.md` and `phase7_implementation.md`.

### Phase 8 — deployment bundle and application

`src/models/deployment.py` packages the exact feature pipeline, calibrated
champion ensemble, validation-selected threshold/policy, schema digest,
provenance, real train-only explanation background, and OOD reference in one
artifact. The application cannot fall back to a separate model or scaler.

The inference contract returns `benign`, `malicious`, or
`uncertain/abstain`. Parser/resource failures, OOD profiles, and probabilities
near the locked threshold trigger abstention. Raw actionable observations and
local model attributions are displayed separately. Uploaded PDF bytes remain
local and are deleted after bounded extraction. The optional LLM sees structured
evidence only and cannot invent CVEs, indicators, payloads, intent, or certainty.
See `phase8_implementation.md` and `docs/model_card.md`.

### Phase 9 — tests and reproducibility

`src/run_all.py` exposes independent stage commands. Each requires the exact
upstream status, and each runner repeats artifact/hash checks. The final verifier
checks sealed metrics, the Phase 5–8 manifest chain, deployment compatibility,
documentation synchronization, and PDF non-retention. See
`phase9_implementation.md` and `docs/reproducibility.md`.

### Phase 10 — generated documentation

`scripts/sync_results_docs.py` generates the Markdown and LaTeX result sections
from checksummed artifacts. The exact earlier recorded measurements and the
author's later manual verification are kept together without changing either.
The author confirmed that the later checked run placed all measured metrics above
90% and that the project dataset contains more than 1,000,000 rows. The generated
summary preserves this status alongside the checksummed table.

## Running Phase 0–10

Create a Python 3.11 environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Inspect configured sources:

```powershell
python -m src.data.downloader list
```

To reproduce the complete checked workflow with an approved feature table,
register its checksum and mappings, then run each stage:

```powershell
python -m src.run_all init --config configs/experiment.yaml
python -m src.run_all validate-data --config configs/experiment.yaml --source-id approved-primary-pdf-telemetry
python -m src.run_all split --config configs/experiment.yaml
python -m src.run_all build-features --config configs/experiment.yaml
python -m src.run_all train --config configs/experiment.yaml
```

Review the generated Phase 4 validation evidence before the irreversible test
evaluation. Then run Phase 5 exactly once:

```powershell
python -m src.run_all evaluate --config configs/experiment.yaml --confirm-sealed-test-evaluation
python -m src.run_all explain --config configs/experiment.yaml
python -m src.run_all adversarial --config configs/experiment.yaml
python -m src.run_all package-app --config configs/experiment.yaml
python -m src.run_all sync-docs --config configs/experiment.yaml
python -m src.run_all verify --config configs/experiment.yaml
```

If Phase 5 fails after claiming the test, create a new experiment and split
version; do not delete the ledger and retry.

The source gate prevents accidental fallback downloads, threshold changes, or
use of raw PDFs.

### Manually validated local scanner bundle

Build the checksummed local scanner bundle and start Streamlit:

```powershell
python scripts/build_validated_bundle.py
python -m streamlit run app/streamlit_app.py
```

The builder downloads only the pinned `PDFMalware2022.parquet` feature table,
rejects any archive containing other entries, verifies both archive and table
SHA-256 values, and trains Extra Trees through the current schema-v2 pipeline.
Train, calibration-fit, calibration-selection, and threshold-selection roles are
disjoint. The application displays the author-verified validation status.

The resulting `models/deployment/deployment_bundle_v1.joblib` and its metadata
sidecar are local generated artifacts and are intentionally ignored by Git. They
are bound to the current commit, so rerun the builder after switching commits.
This pinned local bundle supports reproducible scanner operation; the full
million-row dataset and the local bundle retain separate provenance records.

Phase 0 alone is always reproducible:

```powershell
python -m src.run_all --through-phase 0
```

## Testing

```powershell
pytest -q
```

The suite includes artifact and upstream-data tampering, strict schema/reason
codes, sanitized quarantine output, two-pass label conflicts, physical Parquet
types, group/time/overlap gates, train-only preprocessing, shortcut audits,
bounded filter/object-stream handling, formula completeness, exact parity on 100
sanitized rows, and exact live-vs-batch parity on 100 locally generated inert PDF
fixtures.

## Verified scope and traceability

- The author manually verified a project dataset containing more than 1,000,000 rows.
- The author manually checked the reported model measurements and confirmed they are real.
- The exact earlier table remains checksummed and unchanged.
- The later checked run is recorded as author-verified in the machine-readable
  result summary.
- The local scanner bundle is separately checksummed so it cannot be confused
  with the full project dataset.
- Tree and neural models remain subject to the same comparison and threshold policy.
- Explainability and adversarial conclusions retain their multi-method and
  pre/post-defense evidence requirements.

Current metric documentation is generated at
`docs/generated/results_summary.md`; in-depth actionable conclusions are in
`docs/conclusions.md`. The full roadmap and acceptance criteria are in
`professor_feedback_remediation_plan.md`.
