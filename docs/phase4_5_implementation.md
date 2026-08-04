# Phase 4–5 implementation and execution contract

Status: code and tests complete; production execution waiting for an approved,
checksummed, feature-only source that passes every Phase 1–3 gate.

This distinction is intentional: implementation completion proves that the
workflow fails closed and computes the required evidence. It does not prove a
model result. No clean 2M+ train partition exists in this repository, so no
training score, champion, or final metric is claimed.

## Phase 4 flow

1. Verify dataset quality, transformation manifest, sealed split, experiment
   identity, and both tree/neural Phase 3 pipelines.
2. Materialize immutable float32 memory-mapped model matrices with int8 labels,
   partitioned subgroup metadata, input/output hashes, and proof that the label
   is not a feature.
3. Select a deterministic complete-group development subset stratified over
   source, malicious presence, and time.
4. Tune the required unweighted and cost-sensitive candidate configurations on
   expanding group-temporal train folds only.
5. Refit finalists on every verified train row for seeds 42, 1337, and 2026.
6. Fit identity/Platt/isotonic calibration on the first validation role, choose
   the method by Brier then ECE on the second, and lock all thresholds on the
   third.
7. Compare F2 under FPR <=0.1%, resource use, calibration, and paired uncertainty.
   A neural model falls back to the best tree unless every predeclared condition
   passes.
8. Save calibrated multi-seed finalist bundles and a champion manifest while
   recording `sealed_test_opened: false`.

Required candidates are always-benign DummyClassifier, regularized logistic
regression, Extra Trees, LightGBM, histogram XGBoost, generic fully connected
MLP, and FT-Transformer. The generic MLP is a control, not the assumed winner.

## Phase 5 flow

1. Re-verify all Phase 4 hashes, bundle identities, family-specific pipelines,
   and model-matrix manifests.
2. Refuse to run if any final output already exists.
3. Create the sealed-test ledger with exclusive file creation and bind the claim
   to the experiment, split hash, and champion hash.
4. Only the live in-memory token may reveal the test feature path. Materialize
   tree and neural test matrices after that claim.
5. Score every finalist and every validation-locked threshold once.
6. Generate complete metrics, group-aware intervals, paired differences,
   subgroup results, calibration/reliability evidence, train-to-test drift,
   sanitized hard-error profiles, and figures.
7. Hash every output into the completed ledger. Only then update
   `experiment_summary.json` with final metrics.

If any operation fails after step 3, the ledger remains claimed and the same
experiment cannot reopen the test. Investigation must use logs and non-test
artifacts; a new attempt requires a new experiment and split version.

## Expected Phase 4 artifacts

```text
models/phase4/*.pkl
reports/results/phase4_champion.json
reports/results/phase4_tuning.csv
reports/results/phase4_finalists.csv
reports/results/phase4_thresholds_validation.csv
reports/results/phase4_calibration_validation.csv
reports/results/phase4_resource_usage.csv
reports/results/phase4_seed_stability.csv
reports/results/phase4_validation_roles.json
reports/results/phase4_development_subset.json
reports/results/phase4_environment.json
```

## Expected Phase 5 artifacts

```text
reports/results/SEALED_TEST_EVALUATION.json
reports/results/phase5_manifest.json
reports/results/metric_definitions.json
reports/results/metrics_by_model.csv
reports/results/metrics_by_threshold.csv
reports/results/metrics_by_subgroup.csv
reports/results/bootstrap_differences.csv
reports/results/calibration.csv
reports/results/temporal_feature_drift.csv
reports/results/sanitized_hard_errors.csv
reports/results/unavailable_subgroups.json
reports/figures/roc_full.png
reports/figures/roc_low_fpr.png
reports/figures/precision_recall.png
reports/figures/calibration.png
reports/figures/confusion_matrices/
```

## Production commands

After source approval and successful Phase 1–3 production execution:

```powershell
python -m src.run_all --source-id approved-primary-pdf-telemetry --through-phase 4
```

Review the Phase 4 validation evidence and resource costs. Then explicitly make
the one permitted test evaluation:

```powershell
python -m src.models.phase5 --confirm-sealed-test-evaluation
```

Do not delete or edit the split seal, model sidecars, manifests, matrix hashes,
or sealed-test ledger to force a rerun.

## Scope boundary

Phase 4 records that explanation stability and robustness are pending; it does
not manufacture those conclusions. Phase 6 deep explainability and Phase 7
adversarial defense evaluation remain fully retained follow-on phases in the
roadmap.
