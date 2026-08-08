"""Replace legacy notebooks with deterministic artifact-reading stage notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT


NOTEBOOKS = {
    "01_data_preprocessing.ipynb": (
        "Phase 1 — data validation",
        "Inspect the approved feature-only source, row-flow, safety, prevalence, and checksum gates. No raw PDF or payload content is loaded.",
        "python -m src.run_all validate-data --config configs/experiment.yaml --source-id approved-primary-pdf-telemetry",
        "reports/data/dataset_quality.json",
    ),
    "02_eda.ipynb": (
        "Phase 2 — leakage-resistant split",
        "Inspect exact train/validation/test counts, natural prevalence, time ordering, and zero sample/group overlap from the frozen manifest.",
        "python -m src.run_all split --config configs/experiment.yaml",
        "data/splits/2.0.0/split_manifest.json",
    ),
    "03_feature_engineering.ipynb": (
        "Phase 3 — intelligent features",
        "Inspect schema definitions, formulas, lineage, train-only preprocessing, parser health, and final model-input specifications.",
        "python -m src.run_all build-features --config configs/experiment.yaml",
        "reports/data/feature_dictionary_v2.json",
    ),
    "04_model_training.ipynb": (
        "Phase 4 — fair model comparison",
        "Inspect validation-only tuning, calibration, threshold locking, three-seed stability, resources, and conservative tree-versus-neural selection.",
        "python -m src.run_all train --config configs/experiment.yaml",
        "reports/results/phase4_champion.json",
    ),
    "05_quantization.ipynb": (
        "Phase 5 — sealed metrics and errors",
        "Legacy MLP quantization is superseded. Inspect one-shot sealed metrics, low-FPR thresholds, calibration, uncertainty, subgroups, drift, and errors.",
        "python -m src.run_all evaluate --config configs/experiment.yaml --confirm-sealed-test-evaluation",
        "reports/results/phase5_manifest.json",
    ),
    "06_llm_integration.ipynb": (
        "Phases 6–8 — explanation, adversarial defense, and deployment",
        "Inspect multi-method explanation, inert adversarial evidence, the single deployment bundle, abstention, and the evidence-only optional LLM boundary.",
        "python -m src.run_all explain --config configs/experiment.yaml\npython -m src.run_all adversarial --config configs/experiment.yaml\npython -m src.run_all package-app --config configs/experiment.yaml",
        "reports/results/phase8_manifest.json",
    ),
    "07_final_report.ipynb": (
        "Phases 9–10 — verification and final report",
        "Read artifact-synchronized results, verify provenance, and never type a model score into this notebook.",
        "python -m src.run_all sync-docs --config configs/experiment.yaml\npython -m src.run_all verify --config configs/experiment.yaml",
        "reports/results/documentation_sync_manifest.json",
    ),
}


def _cell(cell_type: str, source: str) -> dict:
    value = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def sync_notebooks() -> list[Path]:
    root = PROJECT_ROOT / "notebooks"
    outputs: list[Path] = []
    for filename, (title, description, command, artifact) in NOTEBOOKS.items():
        notebook = {
            "cells": [
                _cell(
                    "markdown",
                    f"# {title}\n\n{description}\n\n"
                    "The active experiment summary is authoritative. Historical/manual values are not final evidence. "
                    "See `../docs/generated/results_summary.md` for the generated result statement.\n",
                ),
                _cell(
                    "code",
                    "from pathlib import Path\nimport json\n\n"
                    "project_root = Path.cwd().resolve()\n"
                    "if project_root.name == 'notebooks':\n    project_root = project_root.parent\n"
                    "summary_path = project_root / 'reports/results/experiment_summary.json'\n"
                    "summary = json.loads(summary_path.read_text(encoding='utf-8'))\n"
                    "{'status': summary['status'], 'data_gate_passed': summary['data_gate_passed'], "
                    "'final_metrics_present': summary['final_metrics'] is not None}\n",
                ),
                _cell("markdown", f"## Stage command\n\n```powershell\n{command}\n```\n"),
                _cell(
                    "code",
                    f"artifact_path = project_root / {artifact!r}\n"
                    "if artifact_path.is_file():\n"
                    "    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))\n"
                    "    artifact\n"
                    "else:\n"
                    "    print(f'Pending gated artifact: {artifact_path}')\n",
                ),
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
                "phase10_generated": True,
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = root / filename
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs.append(path)

    html = root / "07_final_report.html"
    html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Artifact-synchronized report</title></head>"
        "<body><h1>Artifact-synchronized report</h1><p>This legacy rendered notebook has been superseded. "
        "Open <a href='../docs/generated/results_summary.md'>the generated result summary</a> and "
        "<a href='../report/main.tex'>the Phase 0–10 technical report</a>. No metric is embedded in this HTML.</p></body></html>\n",
        encoding="utf-8",
    )
    outputs.append(html)
    return outputs


if __name__ == "__main__":
    for output in sync_notebooks():
        print(output)
