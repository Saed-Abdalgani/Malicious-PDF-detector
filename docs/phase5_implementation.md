# Remediation Phase 5 — metrics and error analysis

Status: implementation complete; production execution has not occurred and no
final metric is currently claimed.

The locked evaluation must report Precision, Recall, F1, F0.5, F2, ROC-AUC,
low-FPR partial ROC-AUC, PR-AUC, specificity/FPR, MCC, calibration, confusion
counts, false positives per million, thresholds, subgroup results, and 95%
confidence intervals on the sealed natural-prevalence test. Threshold selection
uses validation only. Full definitions and acceptance gates are in
`professor_feedback_remediation_plan.md`.

`src/models/phase5.py` implements an exclusive one-shot ledger, verifies every
Phase 4 finalist and its family-specific pipeline, creates test matrices only
after the claim, evaluates all locked thresholds, produces group-aware
uncertainty and paired differences, performs subgroup/calibration/drift/error
analysis, and publishes checksummed CSV/JSON/figure artifacts. It writes final
metrics to `experiment_summary.json` only after the ledger closes successfully.

After reviewing Phase 4 evidence, run once with explicit confirmation:

```powershell
python -m src.models.phase5 --confirm-sealed-test-evaluation
```

See `docs/metrics_and_thresholds.md` for formulas and interpretations and
`docs/phase4_5_implementation.md` for the artifact contract.
