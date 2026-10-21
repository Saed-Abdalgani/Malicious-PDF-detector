# Phase 8 implementation — deployment bundle and application

Phase 8 is implemented in `src/models/deployment.py`, `src/inference.py`, and `src/models/phase8.py`. Production packaging remains gated until Phase 7 has completed on the approved experiment.

- One immutable bundle contains the calibrated champion, pipeline, locked threshold/policy, complete schema digest, provenance, real train-only explanation background, and OOD reference.
- The Streamlit app loads only this bundle and rejects sidecar, checksum, identity, threshold, or schema mismatch.
- Decisions are benign, malicious, or uncertain/abstain.
- Raw parser indicators and local train-median-replacement model attributions are separate evidence layers.
- Uploaded bytes remain local and are deleted after bounded extraction; no PDF is retained in deployment artifacts.
- The optional LLM receives structured evidence only and is forbidden to invent CVEs, indicators, payloads, intent, or certainty.
- A 100-fixture golden batch must exactly match application predictions; corrupt and oversized fixtures must fail closed within the resource limit.

