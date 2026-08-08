# Model card

## Model selection

The locked comparison includes an always-benign reference, logistic regression, Extra Trees, LightGBM, histogram XGBoost, a fully connected MLP control, and an FT-Transformer. A neural model is not presumed superior. Selection uses validation F2 under the declared false-positive constraint, calibration, paired uncertainty, stability, and resource evidence.

## Deployment contract

Phase 8 packages the champion ensemble, calibrators, feature pipeline, schema digest, validation-selected threshold, provenance hashes, train-only explanation background, and OOD reference into one artifact. The application rejects missing, tampered, stale, or schema-incompatible bundles and never falls back to a legacy model/scaler pair.

Outputs are `benign`, `malicious`, or `uncertain/abstain`. Parser failures, bounded-resource exits, OOD profiles, and probabilities near the locked threshold trigger abstention. Raw actionable observations are displayed separately from local model attributions.

## Evaluation

Final metrics are permitted only after the single sealed-test evaluation. See [artifact-synchronized results](generated/results_summary.md) for the current state and `docs/metrics_and_thresholds.md` for interpretations. Accuracy alone is not a success criterion at 99.5% benign prevalence.

## Explainability and limitations

Operational local explanations use real train-only references. Phase 6 requires multi-method global evidence, stability, sanity, faithfulness, ablations, interactions, representative errors, and feasible observed-row comparisons; SHAP alone cannot support an actionable conclusion.

Static evidence does not prove malicious intent, exploitability, a malware family, or a CVE. Distribution drift, parser disagreement, and adversarial rewriting remain material risks.

