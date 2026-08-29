# Reproducibility and stage execution

Install Python 3.11 dependencies, then run one stage at a time. Every stage checks the exact upstream status and checksums before doing work.

```powershell
python -m src.run_all init --config configs/experiment.yaml
python -m src.run_all validate-data --config configs/experiment.yaml --source-id approved-primary-pdf-telemetry
python -m src.run_all split --config configs/experiment.yaml
python -m src.run_all build-features --config configs/experiment.yaml
python -m src.run_all train --config configs/experiment.yaml
python -m src.run_all evaluate --config configs/experiment.yaml --confirm-sealed-test-evaluation
python -m src.run_all explain --config configs/experiment.yaml
python -m src.run_all adversarial --config configs/experiment.yaml
python -m src.run_all package-app --config configs/experiment.yaml
python -m src.run_all sync-docs --config configs/experiment.yaml
python -m src.run_all verify --config configs/experiment.yaml
```

The scientific dependency order is split before feature fitting, because preprocessing and engineered matrices must be learned from the frozen train partition. The sealed test can be opened exactly once. A failed or completed Phase 5 claim requires a new experiment/split version rather than deletion and retry.

Use `pytest -q`, `python -m compileall -q src app scripts`, and `git diff --check` for code verification. `verify` also checks artifact hashes, deployment compatibility, final-metric presence, documentation synchronization, and non-retention of uploaded PDF bytes.

## Reproducible manually validated local scanner

The production stages above remain fail-closed without approved data. A separate historical path exists only for local interface testing:

```powershell
python scripts/build_validated_bundle.py
python -m streamlit run app/streamlit_app.py
```

The builder pins the download and extracted feature-table SHA-256 values, accepts exactly one Parquet member and no PDFs, validates the expected 10,023 rows and class counts, and records all source mappings and unavailable schema-v2 fields. Negative sentinels and unparseable PDFiD-like values become missing values; unavailable fields are never fabricated. The local pipeline masks those same unavailable fields at live inference so they cannot create a train/serve mismatch or influence a verdict. Learned preprocessing is fitted only on the local train partition, while calibration and threshold roles remain disjoint.

The generated bundle is ignored by Git and cryptographically bound to the current commit. Rebuild it after any commit change. Its `manually_validated_local` provenance keeps it separate from the author-verified project dataset, which contains more than 1,000,000 rows.
