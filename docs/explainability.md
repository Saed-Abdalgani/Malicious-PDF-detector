# Deep explainability contract

Phase 6 separates explanation methods from conclusions. A visually convincing
SHAP plot is not evidence by itself. A finding must have at least two independent
global methods and must survive the applicable stability, retraining-ablation,
negative-control, and faithfulness checks before it becomes actionable.

## Partition rules

- Train supplies class-conditional data audits, mutual information, the real
  SHAP background, prototypes, and all retraining.
- Validation supplies permutation importance, ALE/interactions, faithfulness,
  and sanity-check scoring.
- Sealed test is never reopened. Local cases use only the bounded sanitized
  handoff created and checksummed during the one authorized Phase 5 run.
- Test labels or explanations never change a model, calibration, threshold, or
  feature set.

## Interpretation rules

- Permutation importance measures held-out average-precision loss when one
  feature is disrupted; correlated features can share or hide importance.
- SHAP assigns model-specific local contributions relative to observed train
  data; it is neither causal proof nor an instruction to edit a PDF.
- ALE shows average local response over empirical feature ranges and is safer
  than unconstrained partial dependence for correlated inputs.
- Family ablation retrains the champion configuration without a complete feature
  family. This is stronger evidence than masking a column after training.
- Stability reports rank agreement across model seeds, validation-time slices,
  and 100 bootstrap samples. Low agreement
  means the ranking is not operationally trustworthy.
- Deletion/insertion compares claimed top drivers with random feature sets.
- Label randomization and a fitted noise column are negative controls. Failure
  blocks conclusions.
- Counterfactuals replace a case with a complete observed benign train row. They
  are feasible feature vectors but remain diagnostic and non-causal.

## Operational actions

Supported findings map to explicit actions: retain a stable security signal;
drop or regularize a collection proxy; improve parsing/canonicalization; change
an already validation-governed threshold policy in a new experiment; collect
missing subgroup/campaign data; or add a defense for Phase 7 evaluation.

The manifest records every output checksum and states that sealed test was not
reopened. Production results remain absent until the approved source passes the
2.65M input, 2.5M clean, 2M train, and 99.5% benign gates.
