#!/usr/bin/env python3
"""
fetch_optional_datasets.py
--------------------------
Pull **simple** public datasets from Zenodo (no API key) and optionally from
Hugging Face (requires ``pip install -r requirements-optional-datasets.txt``).

This is a thin companion to ``data/external/manifest.json`` — it does **not**
replace the main CIC PDFMal pipeline.

Examples::

    python scripts/fetch_optional_datasets.py list-manifest
    python scripts/fetch_optional_datasets.py zenodo-search "malware csv"
    python scripts/fetch_optional_datasets.py zenodo-files 18627925
    python scripts/fetch_optional_datasets.py zenodo-get 18627925 android_malware_dataset.csv
    python scripts/fetch_optional_datasets.py hf-export pirocheto/phishing-url "train[:500]" data/external/hf_cache/phish.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "manifest.json"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "external" / "downloads"
HF_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "hf_cache"


def _zenodo_record(record_id: int) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def cmd_zenodo_files(record_id: int) -> None:
    data = _zenodo_record(record_id)
    print(f"Title: {data['metadata'].get('title', '')}\n")
    for f in data.get("files") or []:
        print(f"  {f['key']}\t{f['size']} bytes")


def cmd_zenodo_search(query: str, size: int = 8) -> None:
    q = urllib.parse.quote(query)
    url = f"https://zenodo.org/api/records/?size={size}&q={q}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    for h in payload.get("hits", {}).get("hits", []):
        rid = h["id"]
        title = h["metadata"].get("title", "")[:80]
        print(f"{rid}\t{title}")


def cmd_zenodo_get(record_id: int, file_key: str, out: Path | None) -> Path:
    data = _zenodo_record(record_id)
    files = {f["key"]: f for f in (data.get("files") or [])}
    if file_key not in files:
        raise SystemExit(f"File '{file_key}' not found on record {record_id}. Run zenodo-files first.")
    link = files[file_key]["links"]["self"]
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = out or (DOWNLOAD_DIR / file_key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading -> {dest}")
    with urllib.request.urlopen(link, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print("Done.")
    return dest


def cmd_hf_export(dataset_id: str, split: str, out_path: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install -r requirements-optional-datasets.txt"
        ) from exc

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {dataset_id!r} split={split!r} ...")
    ds = load_dataset(dataset_id, split=split)
    df = ds.to_pandas()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")


def cmd_list_manifest() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for e in entries:
        flag = "SIMPLE" if e.get("simple") else "manual"
        print(f"[{flag}] {e.get('id')} — {e.get('provider')} — {e.get('title')}")
        if e.get("how"):
            print(f"       {e['how']}")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch simple optional public datasets.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-manifest", help="Print data/external/manifest.json entries")

    sp = sub.add_parser("zenodo-search", help="Search Zenodo public API")
    sp.add_argument("query", help='e.g. "malware csv"')
    sp.add_argument("--size", type=int, default=8)

    sp = sub.add_parser("zenodo-files", help="List files for a Zenodo record id")
    sp.add_argument("record_id", type=int)

    sp = sub.add_parser("zenodo-get", help="Download one file from a Zenodo record")
    sp.add_argument("record_id", type=int)
    sp.add_argument("file_key", help="Exact filename key from zenodo-files")
    sp.add_argument("-o", "--out", type=Path, default=None, help="Output path")

    sp = sub.add_parser("hf-export", help="Export a Hugging Face split slice to CSV")
    sp.add_argument("dataset_id", help="e.g. pirocheto/phishing-url")
    sp.add_argument("split", help='e.g. train[:500]')
    sp.add_argument("out_csv", type=Path)

    args = p.parse_args(argv)

    try:
        if args.cmd == "list-manifest":
            cmd_list_manifest()
        elif args.cmd == "zenodo-search":
            cmd_zenodo_search(args.query, size=args.size)
        elif args.cmd == "zenodo-files":
            cmd_zenodo_files(args.record_id)
        elif args.cmd == "zenodo-get":
            cmd_zenodo_get(args.record_id, args.file_key, args.out)
        elif args.cmd == "hf-export":
            cmd_hf_export(args.dataset_id, args.split, args.out_csv)
        else:
            p.error("unknown command")
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
