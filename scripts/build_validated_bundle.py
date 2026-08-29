"""Create the manually validated local bundle used by the PDF scanner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validated_bundle import build_validated_bundle, format_build_result  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Pinned validated feature-table path.")
    parser.add_argument("--output", type=Path, help="Deployment bundle destination.")
    args = parser.parse_args()
    result = build_validated_bundle(dataset_path=args.dataset, output_path=args.output)
    print(format_build_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
