# Remediation Phase 6 — deep explainability

Status: implementation and synthetic contract tests complete; production
explainability run pending the approved dataset and successful Phase 4–5 run.

Phase 6 is implemented in `src/models/phase6.py` and
`src/features/explain.py`. It fails closed unless Phase 5 has completed, the
sealed-test ledger is closed, all upstream hashes match, and final metrics are
non-null. It never reopens the sealed test. Phase 5 creates a bounded,
feature-only, hashed-ID handoff specifically for representative local analysis.

The suite implements six evidence layers:

1. train-only class distributions, prevalence, correlations, mutual
   information, missing/parser patterns, and source/time shortcut auditing;
2. held-out permutation importance, model-native importance, TreeSHAP or neural
   gradient explanation with a real stratified train background, ALE,
   domain-driven interactions, and retrained feature-family ablations;
3. rank stability across model seeds, validation-time slices, and 100 bootstraps;
   label randomization, a fitted random-noise
   negative control, deletion/insertion faithfulness, and method-disagreement
   flags;
4. representative TP/TN/FP/FN, high/low-confidence, and parser-abstained local
   cases with probabilities, locked threshold, supporting/opposing drivers,
   transformed/raw-when-available values, train percentiles, reference ranges,
   subgroup fields, and hashed identifiers;
5. feasible diagnostic comparisons against complete observed benign train rows,
   explicitly marked non-causal; and
6. actionable keep, drop/regularize, extraction, threshold, data-collection, or
   defense conclusions only when independent evidence passes the gates.

Required immutable outputs are written under `reports/explainability/`:

- `global_importance.csv`
- `importance_stability.csv`
- `interactions.csv`
- `feature_ablations.csv`
- `subgroup_errors.csv`
- `local_cases.json`
- `counterfactuals.csv`
- `sanity_checks.json`
- `actionable_conclusions.md`
- `manifest.json`, plus the supporting Layer-A and ALE tables

SHAP alone can never create an actionable conclusion. Failed sanity or
faithfulness checks produce a no-conclusion report instead of a deployment
claim. No empirical importance, local explanation, or action is claimed in the
repository until the approved multi-million-row production workflow reaches
Phase 6.

Run Phase 6 after the one-shot Phase 5 evaluation succeeds:

```powershell
python -m src.models.phase6
```

Or execute it in the full workflow with `--through-phase 6`; this still requires
the explicit Phase 5 sealed-test confirmation flag.
