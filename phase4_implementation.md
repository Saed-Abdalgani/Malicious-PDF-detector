# Remediation Phase 4 — model comparison

Status: implementation complete; production execution blocked by the missing
approved 2.65M-row feature-only source.

Phase 4 may begin only after the production Phase 3 pipeline is fitted from the
verified 2M+ sealed training partition. No legacy model, leaderboard, metric, or
champion is accepted as Phase 4 evidence.

The implementation is in `src/models/phase4.py`, with supporting modules for
immutable matrices, calibration, low-FPR metrics, group bootstrap uncertainty,
the FT-Transformer, and calibrated model bundles. It includes the complete
required candidate set, unweighted/cost-sensitive ablations, grouped temporal
tuning, full-train refitting for three seeds, disjoint validation roles,
validation-only thresholds, resource measurements, and conservative neural-vs-
tree selection. The sealed test remains unopened throughout this phase.

Run it only after Phases 1–3 have passed at production scale:

```powershell
python -m src.models.phase4
```

Code completion is not empirical completion. No champion or validation score is
currently claimed.
