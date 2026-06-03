# Optional external datasets

**Author:** Saed Abdalgani

This folder holds **curated pointers** and **simple downloads** from public hubs
(Zenodo, Hugging Face, Google Dataset Search links, etc.). It does **not** replace
`data/raw/pdfmal2022.csv` for the main CIC pipeline unless you adapt the schema
yourself.

## Manifest

See `manifest.json` for each entry’s `provider`, `simple` flag, and exact command.

## One-command examples

**List manifest (what is “simple” vs manual):**

```bash
python scripts/fetch_optional_datasets.py list-manifest
```

**Download a small CSV from Zenodo (no API key):**

```bash
python scripts/fetch_optional_datasets.py zenodo-get 18627925 android_malware_dataset.csv
```

**Search Zenodo for deposits (pick a `record_id` from the output):**

```bash
python scripts/fetch_optional_datasets.py zenodo-search "malware csv"
```

**Export a slice of a Hugging Face dataset to CSV:**

```bash
pip install -r requirements-optional-datasets.txt
python scripts/fetch_optional_datasets.py hf-export pirocheto/phishing-url "train[:2000]" data/external/hf_cache/phishing_url_sample.csv
```

Files land under `data/external/downloads/` and `data/external/hf_cache/` (both
are gitignored except `.gitkeep`).

## Safety

- **Malware Bazaar / raw VirusTotal binaries** are intentionally **not**
  auto-fetched here — only **documented** in `manifest.json` with `simple: false`.
- Use only datasets your **course / employer** permits, and cite Zenodo DOIs /
  dataset cards in your report.
