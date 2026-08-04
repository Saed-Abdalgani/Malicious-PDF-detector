# Metrics and threshold policy

Status: implemented contract; no production value is claimed.

Malicious PDF is always class 1. All reported values must identify the model,
partition, threshold policy, numeric threshold, and confidence interval where
defined. The active summary remains `final_metrics: null` until the single sealed
test evaluation completes.

## Metrics and what they indicate

| Metric | Formula | Operational indication |
|---|---|---|
| Precision | TP / (TP + FP) | Alert quality. At 99.5% benign prevalence, even a small FPR may generate many false alerts and depress precision. |
| Recall / TPR | TP / (TP + FN) | Malicious-PDF detection coverage. Its complement is the miss rate. |
| F1 | 2PR / (P + R) | Equal-weight harmonic balance; it must be read with the separate Precision and Recall values. |
| F0.5 | 1.25PR / (0.25P + R) | Precision-weighted view emphasizing analyst workload. |
| F2 | 5PR / (4P + R) | Recall-weighted security view and Phase 4 primary objective under the FPR constraint. |
| ROC-AUC | Area under TPR versus FPR | Threshold-free ranking, not alert quality. A high value can still be unusable at extreme imbalance. |
| Partial ROC-AUC | Standardized ROC area up to FPR 0.1% or 0.01% | Ranking within the operational low-false-positive region. |
| PR-AUC / average precision | Area summary of Precision versus Recall | Imbalance-sensitive ranking. The no-skill baseline equals malicious prevalence, about 0.5%, not 50%. |
| Specificity | TN / (TN + FP) | Fraction of benign PDFs left unflagged. |
| FPR | FP / (FP + TN) | Benign false-alert rate; multiplied by one million for false positives per million benign PDFs. |
| FNR | FN / (FN + TP) | Malicious-PDF miss rate. |
| MCC | (TP×TN − FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Correlation-style summary using all four confusion cells. |
| Balanced accuracy | (Recall + Specificity) / 2 | Equal class-level weighting without changing prevalence. |
| Brier score | mean((probability − label)^2) | Probability accuracy; lower is better. |
| ECE | sum(bin weight × absolute calibration gap) | Probability-to-observed-rate mismatch; lower is better. |
| Accuracy | (TP + TN) / N | Diagnostic only: an always-benign classifier already achieves at least 99.5%. |

Raw TP, FP, TN, and FN counts are mandatory. A threshold-free metric never
replaces the locked operating-point confusion matrix.

## Validation-only thresholds

Every finalist stores these policies in its calibrated bundle:

| Policy | Selection rule |
|---|---|
| `fixed_0_5` | Fixed threshold 0.5. |
| `max_f1` | Maximize F1 on `validation_threshold_selection`. |
| `max_f2` | Maximize F2 on `validation_threshold_selection`. |
| `fpr_lte_0_001` | Maximize F2 subject to validation FPR <=0.1%. |
| `fpr_lte_0_0001` | Maximize F2 subject to validation FPR <=0.01%. |

The calibration-fit and calibration-method-selection validation roles are
different complete-group temporal slices. Test labels never fit calibration,
select a method, choose a threshold, select a feature, tune a model, or choose a
champion. If the low-FPR validation budget is smaller than one false positive,
the threshold search includes a valid zero-alert operating point rather than
silently violating the constraint.

## Uncertainty and interpretation rules

- Confidence intervals resample complete groups and stratify malicious-containing
  versus benign-only groups when possible.
- Candidate differences use the same sampled groups for both models.
- The Phase 4 champion maximizes validation F2 subject to FPR <=0.1%.
- A neural point estimate is insufficient: its paired F2 interval versus the best
  tree must exclude zero, and its calibration and latency must satisfy the
  declared trade-off rule.
- Phase 5 evaluates every locked threshold on test but never changes one.
- Subgroups below the declared minimum size are omitted rather than overread.
- Malware-family/campaign results are explicitly marked unavailable when the safe
  feature schema does not supply those fields.

The machine-readable definitions are generated as
`reports/results/metric_definitions.json` during Phase 5.
