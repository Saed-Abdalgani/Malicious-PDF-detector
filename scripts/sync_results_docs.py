"""Generate metric documentation exclusively from checksummed result artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.experiment import sha256_file
from src.utils.atomic import atomic_write_json, atomic_write_text
from scripts.sync_notebooks import sync_notebooks


SUMMARY_PATH = RESULTS_DIR / "experiment_summary.json"
ARCHIVE_ROOT = PROJECT_ROOT / "reports" / "archive" / "unverified_pre_remediation"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "manifest.json"
ARCHIVE_METRICS = ARCHIVE_ROOT / "model_comparison.csv"
MARKDOWN_OUTPUT = PROJECT_ROOT / "docs" / "generated" / "results_summary.md"
TEX_OUTPUT = PROJECT_ROOT / "report" / "generated_results.tex"
SYNC_MANIFEST = RESULTS_DIR / "documentation_sync_manifest.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _verified_historical_rows() -> list[dict[str, str]]:
    manifest = _json(ARCHIVE_MANIFEST)
    entry = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("archived_copy", "").endswith("model_comparison.csv")
        ),
        None,
    )
    if not entry or sha256_file(ARCHIVE_METRICS) != entry.get("sha256"):
        raise RuntimeError("Historical metric CSV does not match its archive manifest.")
    with ARCHIVE_METRICS.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _tex_percent(value: Any) -> str:
    return _percent(value).replace("%", "\\%")


def _verified_metric_markdown(summary: dict[str, Any]) -> list[str]:
    final = summary.get("final_metrics")
    lines = ["## Verified sealed-test results", ""]
    if final is None:
        lines.extend(
            [
                "No verified final model metrics exist. The active experiment has not passed the data and sealed-test gates.",
                "",
                "Accordingly, there is no verified final success metric above 90% to report.",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            f"Model: `{final['model']}` / `{final['variant']}`; partition: `{final['partition']}`.",
            f"Locked threshold: `{float(final['threshold']):.8f}` (`{final['threshold_policy']}`).",
            "",
            "| Metric | Value | 95% interval | Above 90%? |",
            "|---|---:|---:|:---:|",
        ]
    )
    high_is_good = {
        "precision", "recall", "f0_5", "f1", "f2", "roc_auc",
        "partial_roc_auc_standardized_fpr_0_001",
        "partial_roc_auc_standardized_fpr_0_0001",
        "pr_auc_average_precision", "specificity", "matthews_correlation_coefficient",
    }
    for name, record in final.get("metrics", {}).items():
        value = float(record["value"])
        lower, upper = record.get("lower_95"), record.get("upper_95")
        interval = "not available" if lower is None or upper is None else f"{_percent(lower)}–{_percent(upper)}"
        above = "yes" if name in high_is_good and value > 0.90 else "no/not a higher-is-better metric"
        lines.append(f"| `{name}` | {_percent(value)} | {interval} | {above} |")
    lines.append("")
    return lines


def _historical_markdown(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Manual and historical measurements (not final evidence)",
        "",
        "The author reports that a later manual re-check placed all measured metrics above 90%. Exact values, predictions, labels, split identifiers, and a checksummed evaluation artifact were not supplied, so that statement is preserved as author-reported and unverified—not as a final result.",
        "",
        "The older checksummed pre-remediation CSV contains the exact values below. It is detached from the approved 2M-row, 99.5%-benign experiment and is shown only for transparent historical comparison.",
        "",
        "| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | Metrics actually above 90% |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    fields = ("Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC")
    for row in rows:
        above = [name for name in fields if float(row[name]) > 0.90]
        lines.append(
            "| {model} | {accuracy} | {f1} | {precision} | {recall} | {auc} | {above} |".format(
                model=row["Model"],
                accuracy=_percent(row["Accuracy"]),
                f1=_percent(row["F1-Score"]),
                precision=_percent(row["Precision"]),
                recall=_percent(row["Recall"]),
                auc=_percent(row["AUC-ROC"]),
                above=", ".join(above) or "none",
            )
        )
    lines.extend(
        [
            "",
            f"Historical CSV SHA-256: `{sha256_file(ARCHIVE_METRICS)}`.",
            "",
            "These measurements must not be used to claim compliance with the professor's final data, prevalence, split, or evaluation requirements.",
            "",
        ]
    )
    return lines


def _tex(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    final = summary.get("final_metrics")
    if final is None:
        verified = (
            "No verified final score is available, and therefore no verified final "
            "metric above 90\\% is claimed."
        )
    else:
        metric_parts = []
        for name, record in final.get("metrics", {}).items():
            safe_name = name.replace("_", "\\_")
            metric_parts.append(f"\\texttt{{{safe_name}}}={_tex_percent(record['value'])}")
        verified = "Verified sealed-test metrics: " + ", ".join(metric_parts) + "."
    historical_rows = []
    for row in rows:
        model = row["Model"].replace("_", "\\_")
        historical_rows.append(
            f"{model} & {_tex_percent(row['Accuracy'])} & "
            f"{_tex_percent(row['F1-Score'])} & "
            f"{_tex_percent(row['Precision'])} & "
            f"{_tex_percent(row['Recall'])} & "
            f"{_tex_percent(row['AUC-ROC'])} \\\\"
        )
    return "\n".join(
        [
            "% Auto-generated by scripts/sync_results_docs.py; do not hand-edit.",
            "\\section{Artifact-synchronized results}",
            verified,
            "",
            "The author reports a later manual re-check in which all measured metrics were above 90\\%. Because exact values and a checksummed evaluation artifact were not supplied, this is an unverified author report, not final evidence.",
            "",
            "\\begin{table}[h]",
            "\\centering\\small",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Historical model & Accuracy & F1 & Precision & Recall & ROC--AUC \\\\ ",
            "\\midrule",
            *historical_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Checksummed pre-remediation manual measurements, retained only as unverified historical evidence.}",
            "\\end{table}",
            "",
        ]
    )


def sync_results_docs() -> Path:
    summary = _json(SUMMARY_PATH)
    historical = _verified_historical_rows()
    markdown = [
        "# Artifact-synchronized results",
        "",
        "<!-- Auto-generated by scripts/sync_results_docs.py; do not hand-edit. -->",
        "",
        f"Active experiment status: `{summary.get('status')}`.",
        "",
        *_verified_metric_markdown(summary),
        *_historical_markdown(historical),
    ]
    MARKDOWN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(MARKDOWN_OUTPUT, "\n".join(markdown).rstrip() + "\n")
    atomic_write_text(TEX_OUTPUT, _tex(summary, historical))
    notebook_outputs = sync_notebooks()
    manifest = {
        "version": "1.0.0",
        "experiment_summary_sha256": sha256_file(SUMMARY_PATH),
        "historical_archive_manifest_sha256": sha256_file(ARCHIVE_MANIFEST),
        "historical_metrics_sha256": sha256_file(ARCHIVE_METRICS),
        "outputs": {
            "markdown_results": {
                "path": MARKDOWN_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(MARKDOWN_OUTPUT),
            },
            "latex_results": {
                "path": TEX_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(TEX_OUTPUT),
            },
            **{
                f"notebook_{path.name.replace('.', '_')}": {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in notebook_outputs
            },
        },
        "final_metrics_present": summary.get("final_metrics") is not None,
        "manual_metrics_labeled_unverified": True,
    }
    atomic_write_json(SYNC_MANIFEST, manifest)
    return SYNC_MANIFEST


def main() -> int:
    print(sync_results_docs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
