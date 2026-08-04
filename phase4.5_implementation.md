# Remediation Phase 4.5 — deployment optimization

Status: not executed.

Quantization or other deployment optimization may be evaluated only after Phase
4 selects a verified model and threshold. Historical quantized files are not
compatible production artifacts. Any future comparison must bind the exact model
bundle, preprocessing pipeline, calibration, threshold, schema, and upstream
dataset/split provenance and must report the clean performance delta.
