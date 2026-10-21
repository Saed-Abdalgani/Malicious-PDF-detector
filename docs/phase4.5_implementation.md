# Remediation Phase 4.5 — deployment optimization

Status: optional optimization path documented and verified.

Quantization or another deployment optimization is accepted only when it binds
the exact selected model bundle, preprocessing pipeline, calibration, threshold,
schema, and upstream dataset/split provenance. The comparison reports the clean
performance delta so deployment efficiency cannot hide a loss in detection
quality.
