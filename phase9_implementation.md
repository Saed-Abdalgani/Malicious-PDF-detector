# Phase 9 implementation — verification

Phase 9 adds independent stage commands and the fail-closed verifier in `src/verification.py`. Each stage requires the exact prior status; runners then re-check upstream manifests and hashes.

The test inventory covers data/schema gates, leakage and train-only behavior, deterministic splits, batch/live parity, canonicalization and resource limits, calibration/threshold/metric formulas, group uncertainty, explainability gates, adversarial fixtures, deployment mismatch, corrupt/oversized abstention, LLM grounding, and documentation synchronization.

`python -m src.run_all verify --config configs/experiment.yaml` refuses release unless the sealed final metrics, Phase 5–8 manifest chain, deployment bundle/sidecar, synchronized docs, and PDF non-retention checks all pass.

