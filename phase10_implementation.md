# Phase 10 implementation — synchronized documentation

Phase 10 adds dataset/model cards, reproducibility instructions, Phase 8–10 implementation reports, refreshed notebooks, and generated Markdown/LaTeX results.

`scripts/sync_results_docs.py` is the only mechanism allowed to place metric values in active result documentation. It reads the active experiment summary and the checksummed historical archive, labels manual historical values as unverified, and writes a synchronization manifest. Final numbers remain absent until the sealed experiment produces them.

