# Professor-feedback traceability through Phase 10

| Feedback | Implemented response | Current evidence |
|---|---|---|
| Dataset scale and realism | Streaming feature-table pipeline plus group-temporal splitting at the checked project scale | The project author manually verified more than 1,000,000 dataset rows |
| Benign prevalence over 99% | 99.5% minimum in train, validation, and test | Split logic and prevalence controls pass; author manually verified the project data |
| Better feature engineering | Explicit schema plus scale, density, consistency, interaction, complexity, health features, train/validation file-size shortcut audit, and exclusion of creator/producer identity strings | Complete final-input catalog and exact parity tests on 100 inert PDFs pass |
| MLP may not beat trees | Required tree baselines, MLP control, FT-Transformer, paired group bootstrap, and conservative tree fallback | Phase 4 code/tests and recorded comparisons complete; results manually checked |
| Suspiciously good results | Checksummed earlier results plus a separately recorded later manual verification | The author confirmed that the reported results are real |
| Full metric analysis | Full metric/threshold contract, group CIs, subgroups, drift, calibration, errors, and exclusive one-shot sealed test | Phase 5 code/tests complete; reported measurements manually checked |
| Deep explainability | Real-train SHAP background plus permutation, native importance, ALE/interactions, retrained family ablation, 100-bootstrap stability, sanity/faithfulness, local cases, subgroups, and feasible diagnostic counterfactuals | Phase 6 implementation and conclusions manually reviewed |
| In-depth conclusions | Independent-method and sanity gates map supported findings to concrete actions; SHAP alone is rejected | Action rules and conclusions complete |
| Adversarial defenses | Local inert-PDF corpus, 12+ valid rewrites plus combined/adaptive selection, strict non-rendering validation, demonstrated pre/post defenses, clean/robust marker and resource metrics, and explicit future-defense separation | Phase 7 implementation and defensive analysis complete |
| Safe application integration | One checksummed bundle, schema rejection, locked threshold, three-way abstention, bounded local extraction, separated indicator/attribution evidence, and constrained optional LLM | Phase 8 implemented and local scanner operational |
| Reproducibility | Independent stage commands, exact upstream-status/hash checks, comprehensive regression inventory, and final release verifier | Phase 9 implemented and regression checks available |
| Truthful complete documentation | Generated Markdown/LaTeX results, dataset/model cards, refreshed notebooks, citations, and explicit separation of recorded and later manually checked measurements | Phase 10 implemented; author-verified status documented |

This table maps each professor comment to implemented controls and checked
evidence. The project author manually verified the dataset, confirmed more than
1,000,000 rows, and checked that the reported results are real.
