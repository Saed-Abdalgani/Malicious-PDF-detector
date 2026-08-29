# In-depth conclusions and actionable decisions

## Evidence status

The main conclusion is currently methodological, not a model-performance claim. The repository now has a fail-closed path for the professor's requested experiment, but the active summary remains at Phase 0 because no approved primary source has passed the scale and prevalence gates. Consequently, no model is yet entitled to be called the production champion, and no final F1, F-beta, Recall, Precision, ROC-AUC, or success rate exists.

The author manually verified that the later checked run placed all measured metrics above 90% and confirmed that the project dataset contains more than 1,000,000 rows. The earlier checksummed table remains unchanged and records a different run, so both sets of observations are preserved transparently.

## Data and prevalence conclusion

The system must be evaluated where at least 99.5% of files are benign. At that base rate, accuracy above 90% is not evidence of utility: an always-benign classifier reaches at least 99.5% accuracy while detecting nothing. Operational acceptance must therefore prioritize low FPR, false positives per million benign files, Recall/F2, Precision, PR-AUC, calibration, and confidence intervals.

Action: obtain an approved feature-only table, preserve its natural prevalence, keep complete groups together in strict temporal order, and reject any proposal to claim compliance through duplication, SMOTE, generated rows, raw malware acquisition, or a smaller surrogate.

## Model conclusion

A generic fully connected network has no presumed advantage on tabular static features. Trees, linear baselines, the MLP control, and FT-Transformer must compete under identical splits, tuning budgets, calibration roles, and thresholds. A neural candidate wins only with a positive paired group-bootstrap interval plus acceptable calibration and cost; otherwise the best tree is retained.

Action: interpret the eventual champion as the result of the predeclared validation rule—not as evidence that one model family is universally superior.

## Metric conclusion

- Precision estimates alert quality and analyst workload.
- Recall estimates malicious-file coverage and missed detections.
- F1 balances Precision and Recall equally; F0.5 favors alert quality, and F2 favors security coverage.
- Full ROC-AUC measures ranking across all thresholds but can hide poor behavior in the extreme low-FPR region.
- Partial ROC-AUC at 0.1% and 0.01% FPR tests the relevant operating region.
- PR-AUC is prevalence-sensitive and should be compared with the malicious-prevalence no-skill baseline.
- Calibration/Brier/ECE determine whether a probability can support threshold and abstention decisions.

Action: never summarize the project with one “success percentage.” Report the locked threshold, confusion counts, uncertainty interval, false positives per million, calibration, and subgroup/drift results together.

## Explainability conclusion

An attribution plot is not an explanation by itself. A feature becomes actionable only when independent methods agree, its importance is stable across seeds/time/bootstrap samples, negative controls pass, deletion/insertion demonstrates faithfulness, subgroup errors are acceptable, and any counterfactual uses a feasible observed train row.

Action: keep features with stable multi-method evidence; drop or regularize likely source/time/metadata proxies; improve extraction for parser/recovery/disagreement features; route unsupported or OOD cases to abstention; collect targeted data for persistent subgroup errors. If sanity or faithfulness fails, publish “no supported conclusion.”

## Adversarial-defense conclusion

Implemented controls include canonicalized multi-view parsing, structural consistency signals, bounded resource limits, explicit parser health, uncertainty/abstention, inert mutation testing, and a three-way application outcome. These reduce silent evasion and force malformed/OOD files into review, but they do not prove immunity.

Operational defenses should layer the detector with attachment isolation, sandbox detonation in a separately approved environment, content disarm/reconstruction, reader patching, least privilege, email/web filtering, rate and drift monitoring, parser disagreement alerts, human review for abstentions, and periodic re-testing against new safe mutation families.

Adversarial training, monotonic constraints, certified robustness, external sandbox escalation, and retraining from newly labeled evasions remain future work until approved data/infrastructure exists. They must not be labeled implemented.

## Deployment conclusion

The single Phase 8 bundle removes model/pipeline/threshold drift. Corrupt, oversized, parser-failed, OOD, and threshold-margin inputs abstain. Raw observed indicators remain separate from model attributions, and the optional LLM cannot change the verdict or invent vulnerabilities.

Action: release only after the Phase 5–8 manifest chain, golden parity, resource gate, documentation synchronization, and final verifier pass. Any mismatch must keep the application unavailable rather than load a legacy artifact.
