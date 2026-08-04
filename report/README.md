# Technical report status

`main.tex` is now the truthful Phase 0–7 remediation report. The former legacy
metric tables, balanced-data workflow, SMOTE discussion, and unsupported
conclusions have been removed.

The document reports verified implementation controls and explicitly states that
the production data gate has not passed. It contains no model-performance claim.
Phase 4–7 methodology is documented without invented results. A later final
report may add empirical metrics, explainability, and adversarial conclusions
only after the approved 2.5M-row clean-data gate, sealed 2M-row train split, and
Phase 4–7 production evidence exists.

Current authoritative sources:

- `../professor_feedback_remediation_plan.md`
- `../docs/professor_feedback_traceability.md`
- `../docs/metrics_and_thresholds.md`
- `../docs/phase4_5_implementation.md`
- `../docs/explainability.md`
- `../docs/adversarial_robustness.md`
- `../reports/results/experiment_summary.json`

The last file intentionally contains no final metrics.
