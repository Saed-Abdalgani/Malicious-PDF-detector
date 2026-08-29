"""Regression checks that active documentation cannot resurrect legacy claims."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _active_document_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "implementation_plan.md",
        PROJECT_ROOT / "Todo.md",
        PROJECT_ROOT / "prd.md",
        PROJECT_ROOT / "report" / "README.md",
        PROJECT_ROOT / "report" / "main.tex",
    ]
    paths.extend(PROJECT_ROOT.glob("phase*_implementation.md"))
    paths.extend((PROJECT_ROOT / "docs").glob("*.md"))
    # Generated historical metrics are checked by a dedicated provenance test;
    # exact archived values are intentionally permitted there and nowhere else.
    return sorted(set(paths))


def test_active_documentation_contains_no_legacy_metric_or_workflow_claims():
    forbidden = {
        "99.84": "fabricated legacy performance",
        "99.83": "fabricated legacy performance",
        "0.8636": "earlier recorded performance outside the synchronized summary",
        "reports/results/model_comparison.csv": "generic result artifact outside the synchronized summary",
        "manual_fallback()": "unsafe browser fallback",
        "apply_smote(": "prohibited synthetic balancing workflow",
    }
    violations: list[str] = []
    for path in _active_document_paths():
        if path == PROJECT_ROOT / "docs" / "generated" / "results_summary.md":
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase, reason in forbidden.items():
            if phrase.lower() in text:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}: {reason} ({phrase!r})"
                )
    assert not violations, "\n".join(violations)


def test_documented_status_matches_fail_closed_experiment_and_source_registry():
    summary = json.loads(
        (PROJECT_ROOT / "reports" / "results" / "experiment_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["data_gate_passed"] is False
    assert summary["final_metrics"] is None

    registry = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "data_sources.yaml").read_text(encoding="utf-8")
    )
    sources = registry["sources"]
    assert not any(
        source["enabled"] and source["approval_status"] == "approved"
        for source in sources.values()
    )
    primary = sources["approved-primary-pdf-telemetry"]
    assert primary["enabled"] is False
    assert primary["approval_status"] == "requires_approval"


def test_phases4_10_are_documented_as_implemented_without_production_results():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    implementation = (PROJECT_ROOT / "implementation_plan.md").read_text(
        encoding="utf-8"
    )
    metrics = (PROJECT_ROOT / "docs" / "metrics_and_thresholds.md").read_text(
        encoding="utf-8"
    )
    details = (PROJECT_ROOT / "docs" / "phase4_5_implementation.md").read_text(
        encoding="utf-8"
    )
    for phase in ("4", "5", "6", "7", "8", "9", "10"):
        assert f"Phase {phase}" in readme
        assert f"Phase {phase}" in implementation
    assert "| 4 — fair model comparison | Complete" in readme
    assert "| 5 — locked metrics/error analysis | Complete" in readme
    assert "| 6 — deep explainability | Complete" in readme
    assert "| 7 — adversarial attacks/defenses | Complete" in readme
    assert "| 8 — deployment bundle/application | Complete" in readme
    assert "| 9 — stage-aware tests/verification | Complete" in readme
    assert "| 10 — synchronized documentation | Complete" in readme
    assert "author manually verified" in readme.lower()
    assert "manually verified by the project author" in metrics.lower()
    assert "exclusive file creation" in details.lower()
    assert "cannot reopen the test" in details.lower()
    for formula in ("TP / (TP + FP)", "TP / (TP + FN)", "5PR / (4P + R)"):
        assert formula in metrics
