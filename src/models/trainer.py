"""Compatibility entry point for the fail-closed Phase 4 training workflow.

The original trainer loaded the validation and test partitions into memory and
trained only four uncalibrated candidates. That route could bypass the
professor-feedback remediation gates. ``TrainingPipeline`` is retained for
callers that imported it, but every invocation now delegates to
``Phase4Runner``. There is no legacy per-model training entry point.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from src.models.phase4 import Phase4Runner


class TrainingPipeline:
    """Backward-compatible facade over :class:`Phase4Runner`.

    ``models_dir`` is intentionally rejected because Phase 4 artifacts have a
    fixed, checksummed location. Allowing arbitrary legacy output directories
    would weaken the immutable evidence chain.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        *,
        batch_size: int = 100_000,
    ) -> None:
        if models_dir is not None:
            raise ValueError(
                "Custom legacy models_dir is disabled; Phase 4 controls its "
                "checksummed artifact locations."
            )
        warnings.warn(
            "TrainingPipeline is a compatibility name; use Phase4Runner directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.runner = Phase4Runner(
            split_root=Path(data_dir) if data_dir is not None else None,
            batch_size=batch_size,
        )

    def train_all_models(self) -> dict[str, Any]:
        """Run the complete required Phase 4 comparison and return its manifest."""
        manifest = self.runner.run()
        return {
            "phase": 4,
            "status": "complete_test_still_sealed",
            "champion_manifest": str(manifest.resolve()),
        }


def main() -> int:
    """Run Phase 4 through the historical module name."""
    result = TrainingPipeline().train_all_models()
    print(result["champion_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
