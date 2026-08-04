# Professor-feedback traceability through Phase 7

| Feedback | Implemented response | Current evidence |
|---|---|---|
| Train on at least 2M rows | 2M train / 2.5M clean / 2.65M input hard gates | Code tested; real source pending |
| Benign prevalence over 99% | 99.5% minimum in train, validation, and test | Split tests pass; real source pending |
| Better feature engineering | Explicit schema plus scale, density, consistency, interaction, complexity, health features, train/validation file-size shortcut audit, and exclusion of creator/producer identity strings | Complete final-input catalog and exact parity tests on 100 inert PDFs pass |
| MLP may not beat trees | Required tree baselines, MLP control, FT-Transformer, paired group bootstrap, and conservative tree fallback | Phase 4 code/tests complete; production comparison pending |
| Suspiciously good results | Legacy results archived; final summary metrics are null until gates pass | Phase 0 active |
| Full metric analysis | Full metric/threshold contract, group CIs, subgroups, drift, calibration, errors, and exclusive one-shot sealed test | Phase 5 code/tests complete; no metric claimed yet |
| Deep explainability | Real-train SHAP background plus permutation, native importance, ALE/interactions, retrained family ablation, 100-bootstrap stability, sanity/faithfulness, local cases, subgroups, and feasible diagnostic counterfactuals | Phase 6 code/tests complete; production evidence pending |
| In-depth conclusions | Independent-method and sanity gates map supported findings to concrete actions; SHAP alone is rejected | Mechanism complete; empirical conclusions pending |
| Adversarial defenses | Local inert-PDF corpus, 12+ valid rewrites plus combined/adaptive selection, strict non-rendering validation, demonstrated pre/post defenses, clean/robust marker and resource metrics, and explicit future-defense separation | Phase 7 code/tests complete; production champion evidence pending |

This table distinguishes implemented controls from empirical evidence. A green
unit test is not substituted for a real 2M-row training run.
