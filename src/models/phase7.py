"""Phase 7 safe adversarial robustness and defense evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_REPORTS_DIR, REPORTS_DIR, RESULTS_DIR, SPLITS_DIR
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.pipeline import FeaturePipelineV2
from src.models.bundle import Phase4ModelBundle
from src.security.adversarial import (
    MUTATIONS,
    ThreatModel,
    defense_matrix,
    evaluate_fixture_corpus,
)
from src.utils.atomic import atomic_write_json, atomic_write_text


PHASE7_VERSION = "1.0.0"


class Phase7GateError(RuntimeError):
    """Raised when Phase 7 cannot prove safe and complete upstream evidence."""


def _json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase7GateError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase7GateError(f"{description} is unreadable.") from exc
    if not isinstance(value, dict):
        raise Phase7GateError(f"{description} must be a JSON object.")
    return value


def _checked(path: Path, digest: str, description: str) -> Path:
    if not path.is_file() or sha256_file(path) != digest:
        raise Phase7GateError(f"{description} checksum mismatch: {path}")
    return path


def _robustness_by_mutation(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    clean = scored[scored["mutation"] == "none"].set_index(
        ["fixture_id_sha256", "expected_security_positive"]
    )
    mutated = scored[scored["mutation"] != "none"].copy()
    rows = []
    for mutation, frame in mutated.groupby("mutation", sort=True):
        joined = frame.join(
            clean[["model_probability"]].rename(
                columns={"model_probability": "clean_probability"}
            ),
            on=["fixture_id_sha256", "expected_security_positive"],
        )
        positive = joined["expected_security_positive"] == 1
        benign = ~positive
        probability_drop = (
            joined.loc[positive, "clean_probability"]
            - joined.loc[positive, "model_probability"]
        )
        attack_success = (
            (joined.loc[positive, "clean_probability"] >= threshold)
            & (joined.loc[positive, "model_probability"] < threshold)
            & ~joined.loc[positive, "abstained"]
        )
        clean_benign_fpr = float(
            (joined.loc[benign, "clean_probability"] >= threshold).mean()
        ) if benign.any() else 0.0
        mutated_benign_fpr = float(
            (joined.loc[benign, "model_probability"] >= threshold).mean()
        ) if benign.any() else 0.0
        rows.append(
            {
                "mutation": mutation,
                "valid_scored_fixtures": len(joined),
                "attack_success_rate": float(attack_success.mean()) if len(attack_success) else 0.0,
                "mean_probability_drop_on_security_positive": float(probability_drop.mean()) if len(probability_drop) else 0.0,
                "maximum_probability_drop_on_security_positive": float(probability_drop.max()) if len(probability_drop) else 0.0,
                "clean_benign_false_positive_rate": clean_benign_fpr,
                "mutated_benign_false_positive_rate": mutated_benign_fpr,
                "benign_false_positive_rate_increase": (
                    mutated_benign_fpr - clean_benign_fpr
                ),
                "parser_failure_or_abstention_rate": float(joined["abstained"].mean()),
                "mean_extraction_latency_ms": float(joined["extraction_latency_ms"].mean()),
                "p95_extraction_latency_ms": float(joined["extraction_latency_ms"].quantile(0.95)),
                "maximum_peak_rss_delta_bytes": int(joined["peak_rss_delta_bytes"].max()),
            }
        )
    return pd.DataFrame(rows)


def _defense_comparison(metrics: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    lookup = metrics.set_index(["defense_view", "partition"])
    rows = []
    for record in matrix.itertuples(index=False):
        if record.status != "demonstrated":
            continue
        for partition in ("clean_inert_benchmark", "valid_mutated_inert_benchmark"):
            pre = lookup.loc[(record.pre, partition)]
            post = lookup.loc[(record.post, partition)]
            rows.append(
                {
                    "defense": record.defense,
                    "attack_addressed": record.attack_addressed,
                    "partition": partition,
                    "pre_view": record.pre,
                    "post_view": record.post,
                    "pre_recall": pre["robust_recall_or_clean_recall"],
                    "post_recall": post["robust_recall_or_clean_recall"],
                    "recall_delta": post["robust_recall_or_clean_recall"] - pre["robust_recall_or_clean_recall"],
                    "pre_f2": pre["robust_f2_or_clean_f2"],
                    "post_f2": post["robust_f2_or_clean_f2"],
                    "f2_delta": post["robust_f2_or_clean_f2"] - pre["robust_f2_or_clean_f2"],
                    "pre_false_positive_rate": pre["false_positive_rate"],
                    "post_false_positive_rate": post["false_positive_rate"],
                    "false_positive_rate_delta": post["false_positive_rate"] - pre["false_positive_rate"],
                    "benchmark_labels_are_malware_ground_truth": False,
                }
            )
    return pd.DataFrame(rows)


def _threat_model_markdown(valid_families: int, qpdf_available: bool) -> str:
    qpdf_note = (
        "qpdf --check was available and applied in addition to strict parsing."
        if qpdf_available
        else "qpdf was unavailable; PyPDF2 strict mode was used as the equivalent non-rendering validator."
    )
    return f"""# Phase 7 adversarial threat model

The attacker may have black-box score access or full model/feature knowledge and
may attempt semantics-preserving rewriting, benign stuffing, parser differential
exploitation, bounded resource exhaustion, source poisoning, or campaign drift.
Feature-space-only changes never count as successful PDF evasions.

The corpus is generated locally and is inert. It contains no external links,
embedded files, launch/network actions, executables, or harmful JavaScript. PDFs
are parsed but never rendered or opened in a desktop viewer. No fixture PDF is
persisted in the report. {qpdf_note}

This run validated {valid_families} mutation families/combinations, including
query-selected worst cases. Benchmark-positive fixtures contain only an empty or
inert structural action marker; they are not malware. Consequently robust Recall,
F2, and attack success in these tables measure preservation of a security-marker
benchmark, not real-malware ground truth or a production robustness guarantee.
"""


def _recommendations_markdown(matrix: pd.DataFrame) -> str:
    demonstrated = matrix[matrix["status"] == "demonstrated"]
    implemented = matrix[matrix["status"] == "implemented_control"]
    future = matrix[matrix["status"].str.startswith("future")]
    lines = [
        "# Phase 7 defense conclusions",
        "",
        "## Demonstrated in the inert-PDF benchmark",
        "",
    ]
    lines.extend(
        f"- {row.defense}: addresses {row.attack_addressed}; see the pre/post table."
        for row in demonstrated.itertuples(index=False)
    )
    lines.extend(["", "## Implemented controls verified outside fixture scoring", ""])
    lines.extend(
        f"- {row.defense}: addresses {row.attack_addressed}."
        for row in implemented.itertuples(index=False)
    )
    lines.extend(["", "## Future recommendations — not demonstrated", ""])
    lines.extend(
        f"- {row.defense}: {row.status}; no robustness claim is made."
        for row in future.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "Production conclusions require the approved natural-prevalence dataset, a verified champion, and real Phase 4–6 artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


class Phase7Runner:
    def __init__(self, *, split_root: Path | None = None) -> None:
        self.config = load_experiment_config()
        self.phase7 = self.config["phase7"]
        self.identity = create_experiment_identity(self.config)
        self.split_root = Path(
            split_root or SPLITS_DIR / str(self.config["split_version"])
        )
        self.output_root = REPORTS_DIR / "adversarial"

    def _verify_upstream(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.output_root.exists():
            raise Phase7GateError(
                f"Phase 7 output is immutable and already exists: {self.output_root}"
            )
        summary = _json(RESULTS_DIR / "experiment_summary.json", "Experiment summary")
        if summary.get("status") != "phase_6_complete_deep_explainability":
            raise Phase7GateError("Phase 7 requires completed Phase 6 evidence.")
        phase6_entry = summary.get("phase_6", {})
        phase6_path = Path(str(phase6_entry.get("manifest", "")))
        _checked(
            phase6_path,
            str(phase6_entry.get("manifest_sha256", "")),
            "Phase 6 manifest",
        )
        phase6 = _json(phase6_path, "Phase 6 manifest")
        if phase6.get("sealed_test_reopened"):
            raise Phase7GateError("Phase 6 reports that sealed test was reopened.")
        for name, entry in phase6.get("outputs", {}).items():
            _checked(phase6_path.parent / name, str(entry.get("sha256", "")), f"Phase 6 output {name}")
        phase4_path = RESULTS_DIR / "phase4_champion.json"
        phase4 = _json(phase4_path, "Phase 4 champion manifest")
        if sha256_file(phase4_path) != phase6.get("phase4_champion_manifest_sha256"):
            raise Phase7GateError("Phase 4/6 champion provenance mismatch.")
        return summary, phase4

    def run(self) -> Path:
        summary, phase4 = self._verify_upstream()
        pipeline_path = _checked(
            Path(phase4["champion_feature_pipeline"]),
            phase4["champion_feature_pipeline_sha256"],
            "Champion feature pipeline",
        )
        champion_path = _checked(
            Path(phase4["champion_bundle"]),
            phase4["champion_bundle_sha256"],
            "Champion model bundle",
        )
        pipeline = FeaturePipelineV2.load(
            pipeline_path,
            identity=self.identity,
            split_manifest_path=self.split_root / "split_manifest.json",
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
        )
        champion = Phase4ModelBundle.load_champion(
            champion_path,
            identity=self.identity,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            split_manifest_path=self.split_root / "split_manifest.json",
            feature_pipeline_path=pipeline_path,
        )
        temporary = Path(tempfile.mkdtemp(prefix=".phase7-", dir=REPORTS_DIR))
        try:
            scored, validity, metrics = evaluate_fixture_corpus(
                pipeline,
                champion,
                threshold=champion.threshold,
                fixtures_per_class=int(self.phase7["fixtures_per_benchmark_class"]),
            )
            valid_families = int(
                validity.loc[
                    (validity["mutation"] != "none") & validity["valid"], "mutation"
                ].nunique()
            )
            if valid_families < int(self.phase7["minimum_valid_mutation_families"]):
                raise Phase7GateError("The valid mutation-family gate did not pass.")
            matrix = defense_matrix()
            comparison = _defense_comparison(metrics, matrix)
            by_mutation = _robustness_by_mutation(scored, champion.threshold)
            resources = (
                scored.groupby("mutation", as_index=False)
                .agg(
                    mean_latency_ms=("extraction_latency_ms", "mean"),
                    p95_latency_ms=("extraction_latency_ms", lambda values: float(np.quantile(values, 0.95))),
                    maximum_peak_rss_delta_bytes=("peak_rss_delta_bytes", "max"),
                    mean_fixture_bytes=("fixture_bytes", "mean"),
                    abstention_rate=("abstained", "mean"),
                )
            )
            scored.to_csv(temporary / "fixture_scores.csv", index=False)
            validity.to_csv(temporary / "mutation_validity.csv", index=False)
            metrics.to_csv(temporary / "robustness_metrics.csv", index=False)
            by_mutation.to_csv(temporary / "robustness_by_mutation.csv", index=False)
            matrix.to_csv(temporary / "defense_matrix.csv", index=False)
            comparison.to_csv(temporary / "defense_comparison.csv", index=False)
            resources.to_csv(temporary / "resource_overhead.csv", index=False)
            atomic_write_json(
                temporary / "threat_model.json",
                {
                    **ThreatModel().to_dict(),
                    "mutation_registry": sorted(MUTATIONS),
                    "valid_families_and_combinations": valid_families,
                    "raw_fixture_pdfs_persisted": False,
                    "benchmark_labels_are_malware_ground_truth": False,
                },
            )
            qpdf_available = bool(validity["qpdf_available"].any())
            atomic_write_text(
                temporary / "threat_model.md",
                _threat_model_markdown(valid_families, qpdf_available),
            )
            atomic_write_text(
                temporary / "defense_conclusions.md",
                _recommendations_markdown(matrix),
            )
            required = {
                "fixture_scores.csv", "mutation_validity.csv", "robustness_metrics.csv",
                "robustness_by_mutation.csv", "defense_matrix.csv", "defense_comparison.csv",
                "resource_overhead.csv", "threat_model.json", "threat_model.md",
                "defense_conclusions.md",
            }
            actual = {path.name for path in temporary.iterdir() if path.is_file()}
            if not required.issubset(actual):
                raise Phase7GateError("Phase 7 output inventory is incomplete.")
            manifest = {
                "phase7_version": PHASE7_VERSION,
                "experiment": self.identity.to_dict(),
                "phase6_manifest_sha256": sha256_file(
                    Path(summary["phase_6"]["manifest"])
                ),
                "valid_mutation_families_and_combinations": valid_families,
                "minimum_required_valid_mutation_families": int(self.phase7["minimum_valid_mutation_families"]),
                "gate_passed": True,
                "raw_fixture_pdfs_persisted": False,
                "live_malware_used": False,
                "benchmark_labels_are_malware_ground_truth": False,
                "outputs": {
                    path.name: {"sha256": sha256_file(path)}
                    for path in sorted(temporary.iterdir())
                    if path.is_file()
                },
            }
            atomic_write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, self.output_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        manifest_path = self.output_root / "manifest.json"
        summary.update(
            {
                "status": "phase_7_complete_safe_adversarial_evaluation",
                "phase_7": {
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha256_file(manifest_path),
                    "live_malware_used": False,
                    "raw_fixture_pdfs_persisted": False,
                },
            }
        )
        atomic_write_json(RESULTS_DIR / "experiment_summary.json", summary)
        return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(Phase7Runner().run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
