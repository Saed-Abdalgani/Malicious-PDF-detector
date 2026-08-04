"""Phase 6 deep explainability with immutable, multi-method evidence.

The runner never opens the sealed test split.  It consumes the bounded,
sanitized Phase 5 handoff and verified train/validation model matrices only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import DATA_REPORTS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RESULTS_DIR, SPLITS_DIR
from src.experiment import create_experiment_identity, load_experiment_config, sha256_file
from src.features.explain import (
    actionable_conclusions_markdown,
    ale_table,
    data_explanation_tables,
    deletion_insertion_faithfulness,
    feature_family_ablation,
    importance_stability_table,
    interaction_table,
    method_agreement_table,
    native_importance_table,
    observed_prototype_counterfactuals,
    permutation_importance_table,
    predict_probability,
    select_local_cases,
    shap_attributions,
    shap_importance_table,
    source_time_shortcut_audit,
    stratified_reference_indices,
    top_attributions,
)
from src.features.pipeline import FeaturePipelineV2
from src.models.bundle import Phase4ModelBundle
from src.models.matrix import load_model_matrix
from src.models.phase4 import CandidateSpec, fit_candidate
from src.utils.atomic import atomic_write_json, atomic_write_text


PHASE6_VERSION = "1.0.0"


class Phase6GateError(RuntimeError):
    """Raised before publication when Phase 6 provenance or evidence is incomplete."""


def _csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _verified_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise Phase6GateError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase6GateError(f"{description} is unreadable.") from exc
    if not isinstance(value, dict):
        raise Phase6GateError(f"{description} must be a JSON object.")
    return value


def _verify_file_entry(entry: dict[str, Any], *, description: str) -> Path:
    path = Path(str(entry.get("path", "")))
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise Phase6GateError(f"{description} checksum/provenance mismatch: {path}")
    return path


def _bounded_indices(labels: np.ndarray, maximum_rows: int, seed: int) -> np.ndarray:
    metadata = pd.DataFrame({"group_id": np.arange(len(labels)).astype(str)})
    return stratified_reference_indices(
        labels, metadata, maximum_rows=min(maximum_rows, len(labels)), random_seed=seed
    )


def _feature_families(feature_names: Sequence[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = defaultdict(list)
    for name in feature_names:
        lowered = name.lower()
        if "__missing" in lowered:
            family = "missingness"
        elif any(token in lowered for token in ("parse", "recovery", "timeout", "invalid_eof")):
            family = "parser_status"
        elif any(token in lowered for token in ("javascript", "js_", "openaction", "launch", "uri", "submitform", "action")):
            family = "active_content"
        elif any(token in lowered for token in ("obfuscat", "entropy", "hex", "encoding")):
            family = "obfuscation"
        elif any(token in lowered for token in ("pdf_size", "page", "object", "stream", "xref")):
            family = "document_structure"
        elif any(token in lowered for token in ("ratio", "density", "consistency", "disagreement")):
            family = "engineered_consistency"
        else:
            family = "other_numeric"
        families[family].append(name)
    return dict(families)


def _domain_pairs(feature_names: Sequence[str], maximum_pairs: int = 20) -> list[tuple[str, str]]:
    families = _feature_families(feature_names)
    pairs: list[tuple[str, str]] = []
    for members in families.values():
        for index in range(0, len(members) - 1, 2):
            pairs.append((members[index], members[index + 1]))
            if len(pairs) >= maximum_pairs:
                return pairs
    return pairs


def _bootstrap_stability(
    attributions: np.ndarray,
    feature_names: Sequence[str],
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rankings: dict[str, dict[str, float]] = {}
    for replicate in range(replicates):
        indices = rng.integers(0, len(attributions), size=len(attributions))
        values = np.mean(np.abs(attributions[indices]), axis=0)
        rankings[f"bootstrap_{replicate:03d}"] = dict(zip(feature_names, values))
    # Pairing all 100 bootstraps creates 4,950 valid stability comparisons.
    result = importance_stability_table(rankings)
    result.insert(0, "scope", "bootstrap")
    return result


def _local_records(
    model: Phase4ModelBundle,
    cases: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    attributions: np.ndarray,
    feature_names: Sequence[str],
    train_reference: np.ndarray,
    train_reference_labels: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, Any]]:
    percentiles = {
        name: np.sort(train_reference[:, column])
        for column, name in enumerate(feature_names)
    }
    output = []
    scale = np.std(train_reference, axis=0)
    scale[scale < 1e-6] = 1.0
    for position in selected:
        row = cases[position]
        drivers = top_attributions(attributions[position], feature_names, k=10)
        driver_rows = []
        for name, attribution, direction in drivers:
            column = feature_names.index(name)
            raw_value = metadata.iloc[position].get(name)
            driver_rows.append(
                {
                    "feature": name,
                    "direction": direction,
                    "attribution": attribution,
                    "transformed_value": float(row[column]),
                    "raw_value": None if pd.isna(raw_value) else raw_value,
                    "train_percentile": float(
                        np.searchsorted(percentiles[name], row[column], side="right")
                        / len(train_reference)
                    ),
                    "train_reference_q05": float(np.quantile(percentiles[name], 0.05)),
                    "train_reference_q50": float(np.quantile(percentiles[name], 0.50)),
                    "train_reference_q95": float(np.quantile(percentiles[name], 0.95)),
                }
            )
        meta = metadata.iloc[position]
        prototypes = []
        for prototype_label, prototype_name in ((0, "benign"), (1, "malicious")):
            candidates = train_reference[train_reference_labels == prototype_label]
            if not len(candidates):
                continue
            distances = np.mean(((candidates - row) / scale) ** 2, axis=1)
            nearest = candidates[int(np.argmin(distances))]
            prototypes.append(
                {
                    "class": prototype_name,
                    "standardized_mean_squared_distance": float(np.min(distances)),
                    "model_probability": float(
                        model.predict_proba(nearest.reshape(1, -1))[0]
                    ),
                    "source": "sanitized_real_train_background_row",
                    "raw_identifier_present": False,
                }
            )
        output.append(
            {
                "case_index_in_sanitized_handoff": int(position),
                "sample_id_sha256": meta.get("sample_id_sha256"),
                "label": int(labels[position]),
                "probability": float(probabilities[position]),
                "threshold": model.threshold,
                "predicted_label": int(probabilities[position] >= model.threshold),
                "outcome": meta.get("outcome"),
                "subgroup": {
                    name: (None if pd.isna(meta.get(name)) else meta.get(name))
                    for name in ("source_id", "is_encrypted", "parse_failure", "recovery_mode")
                    if name in metadata
                },
                "nearest_sanitized_prototypes": prototypes,
                "supporting_and_opposing_drivers": driver_rows,
            }
        )
    return output


class Phase6Runner:
    def __init__(self, *, split_root: Path | None = None) -> None:
        self.config = load_experiment_config()
        self.phase6 = self.config["phase6"]
        self.identity = create_experiment_identity(self.config)
        self.split_root = Path(
            split_root or SPLITS_DIR / str(self.config["split_version"])
        )
        self.output_root = REPORTS_DIR / "explainability"

    def _verify_upstream(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self.output_root.exists():
            raise Phase6GateError(f"Phase 6 output is immutable and already exists: {self.output_root}")
        summary_path = RESULTS_DIR / "experiment_summary.json"
        summary = _verified_json(summary_path, description="Experiment summary")
        if summary.get("status") != "phase_5_complete_sealed_test_closed" or summary.get("final_metrics") is None:
            raise Phase6GateError("Phase 6 requires completed Phase 5 and non-null final metrics.")
        phase5_entry = summary.get("phase_5", {})
        phase5_path = Path(str(phase5_entry.get("manifest", "")))
        if not phase5_path.is_file() or sha256_file(phase5_path) != phase5_entry.get("manifest_sha256"):
            raise Phase6GateError("Phase 5 manifest does not match the experiment summary.")
        phase5 = _verified_json(phase5_path, description="Phase 5 manifest")
        ledger_path = Path(str(phase5_entry.get("sealed_test_ledger", "")))
        if (
            not ledger_path.is_file()
            or sha256_file(ledger_path) != phase5_entry.get("sealed_test_ledger_sha256")
        ):
            raise Phase6GateError("Sealed-test ledger does not match the experiment summary.")
        ledger = _verified_json(ledger_path, description="Sealed-test ledger")
        if ledger.get("status") != "completed_test_closed":
            raise Phase6GateError("The sealed test ledger is not closed.")
        phase4_path = RESULTS_DIR / "phase4_champion.json"
        phase4 = _verified_json(phase4_path, description="Phase 4 champion manifest")
        if sha256_file(phase4_path) != phase5.get("phase4_champion_manifest_sha256"):
            raise Phase6GateError("Phase 4/5 provenance mismatch.")
        handoff = _verified_json(
            RESULTS_DIR / "phase6_handoff" / "manifest.json",
            description="Sanitized Phase 6 handoff",
        )
        if handoff.get("raw_pdf_content_present") or handoff.get("raw_sample_identifiers_present"):
            raise Phase6GateError("The Phase 6 handoff contains prohibited raw content or IDs.")
        if handoff.get("raw_group_identifiers_present"):
            raise Phase6GateError("The Phase 6 handoff contains prohibited raw group IDs.")
        expected_handoff = phase5.get("outputs", {}).get("phase6_handoff_manifest")
        if not isinstance(expected_handoff, dict):
            raise Phase6GateError("Phase 5 does not bind the Phase 6 handoff manifest.")
        bound_handoff_path = _verify_file_entry(
            expected_handoff, description="Phase 5-bound handoff manifest"
        )
        if bound_handoff_path.resolve() != (RESULTS_DIR / "phase6_handoff" / "manifest.json").resolve():
            raise Phase6GateError("Phase 5 binds a different handoff manifest path.")
        _verify_file_entry(handoff["matrix"], description="Phase 6 handoff matrix")
        _verify_file_entry(handoff["metadata"], description="Phase 6 handoff metadata")
        return summary, phase4, phase5

    def run(self) -> Path:
        summary, phase4, phase5 = self._verify_upstream()
        pipeline_path = Path(phase4["champion_feature_pipeline"])
        champion_path = Path(phase4["champion_bundle"])
        for path, digest, name in (
            (pipeline_path, phase4["champion_feature_pipeline_sha256"], "pipeline"),
            (champion_path, phase4["champion_bundle_sha256"], "champion"),
        ):
            if not path.is_file() or sha256_file(path) != digest:
                raise Phase6GateError(f"Phase 4 {name} checksum mismatch.")
        champion = Phase4ModelBundle.load_champion(
            champion_path,
            identity=self.identity,
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            split_manifest_path=self.split_root / "split_manifest.json",
            feature_pipeline_path=pipeline_path,
        )
        FeaturePipelineV2.load(
            pipeline_path,
            identity=self.identity,
            split_manifest_path=self.split_root / "split_manifest.json",
            dataset_quality_path=DATA_REPORTS_DIR / "dataset_quality.json",
            transformation_manifest_path=DATA_REPORTS_DIR / "transformation_manifest.json",
        )
        family = "neural" if champion.model_family == "neural" else "tree"
        matrix_root = PROCESSED_DATA_DIR / "model_matrices_v2" / str(self.config["split_version"]) / family
        X_train, y_train, train_metadata, train_manifest = load_model_matrix(
            matrix_root / "train", expected_partition="train", feature_pipeline_path=pipeline_path,
            split_manifest_path=self.split_root / "split_manifest.json"
        )
        X_validation, y_validation, validation_metadata, validation_manifest = load_model_matrix(
            matrix_root / "validation", expected_partition="validation", feature_pipeline_path=pipeline_path,
            split_manifest_path=self.split_root / "split_manifest.json"
        )
        if tuple(champion.feature_names) != tuple(train_manifest.feature_names) or tuple(train_manifest.feature_names) != tuple(validation_manifest.feature_names):
            raise Phase6GateError("Champion/train/validation feature schemas differ.")
        names = tuple(champion.feature_names)
        seed = int(self.config["random_seed"])
        background_index = stratified_reference_indices(
            y_train, train_metadata, maximum_rows=int(self.phase6["background_rows"]), random_seed=seed
        )
        global_train_index = stratified_reference_indices(
            y_train, train_metadata, maximum_rows=min(int(self.phase6["development_sanity_rows"]), len(y_train)), random_seed=seed + 1
        )
        validation_index = _bounded_indices(
            np.asarray(y_validation), min(int(self.phase6["global_explanation_rows"]), len(y_validation)), seed + 2
        )
        background = np.asarray(X_train[background_index])
        train_sample = np.asarray(X_train[global_train_index])
        train_y_sample = np.asarray(y_train[global_train_index])
        validation_sample = np.asarray(X_validation[validation_index])
        validation_y_sample = np.asarray(y_validation[validation_index])

        handoff_manifest = _verified_json(RESULTS_DIR / "phase6_handoff" / "manifest.json", description="Phase 6 handoff")
        handoff_path = _verify_file_entry(handoff_manifest["matrix"], description="Phase 6 handoff matrix")
        with np.load(handoff_path) as handoff:
            test_cases = np.asarray(handoff["X"], dtype=np.float32)
            test_y = np.asarray(handoff["y"], dtype=np.int8)
            test_probability = np.asarray(handoff["probability"], dtype=float)
        test_metadata = pd.read_parquet(_verify_file_entry(handoff_manifest["metadata"], description="Phase 6 handoff metadata"))
        if tuple(handoff_manifest["feature_names"]) != names or not (len(test_cases) == len(test_y) == len(test_metadata)):
            raise Phase6GateError("Phase 6 handoff schema/row alignment is invalid.")

        temporary = Path(tempfile.mkdtemp(prefix=".phase6-", dir=REPORTS_DIR))
        try:
            data_tables = data_explanation_tables(
                train_sample, train_y_sample, train_metadata.iloc[global_train_index].reset_index(drop=True), names,
                partition_name="train", random_seed=seed
            )
            for table_name, frame in data_tables.items():
                _csv(temporary / f"data_{table_name}.csv", frame)
            permutation = permutation_importance_table(
                champion, validation_sample, validation_y_sample, names,
                partition_name="validation", repeats=int(self.phase6["permutation_repeats"]), random_seed=seed
            )
            native = native_importance_table(champion, names)
            validation_shap, shap_method = shap_attributions(champion, validation_sample, background)
            shap_importance = shap_importance_table(validation_shap, names, method=shap_method)
            global_importance = pd.concat([permutation, native, shap_importance], ignore_index=True, sort=False)
            _csv(temporary / "global_importance.csv", global_importance)
            top = global_importance.groupby("feature")["importance"].mean().nlargest(int(self.phase6["top_features"])).index.tolist()
            _csv(temporary / "ale.csv", ale_table(champion, validation_sample, names, top, bins=int(self.phase6["ale_bins"])))
            _csv(temporary / "interactions.csv", interaction_table(champion, validation_sample, names, _domain_pairs(names)))
            stability = _bootstrap_stability(
                validation_shap, names, replicates=int(self.phase6["bootstrap_replicates"]), random_seed=seed
            )
            seed_rankings: dict[str, dict[str, float]] = {}
            for member in champion.members:
                member_attribution, _ = shap_attributions(
                    member.estimator, validation_sample, background
                )
                seed_rankings[f"seed_{member.seed}"] = dict(
                    zip(names, np.mean(np.abs(member_attribution), axis=0))
                )
            seed_stability = importance_stability_table(seed_rankings)
            if not seed_stability.empty:
                seed_stability.insert(0, "scope", "model_seed")
                stability = pd.concat([stability, seed_stability], ignore_index=True)
            if "first_seen_at" in validation_metadata:
                explanation_time = pd.to_datetime(
                    validation_metadata.iloc[validation_index]["first_seen_at"],
                    utc=True,
                    errors="coerce",
                )
                time_bins = pd.qcut(
                    explanation_time.rank(method="first"),
                    q=min(4, len(explanation_time)),
                    labels=False,
                    duplicates="drop",
                )
                time_rankings = {
                    f"time_bin_{int(value)}": dict(
                        zip(names, np.mean(np.abs(validation_shap[time_bins.to_numpy() == value]), axis=0))
                    )
                    for value in sorted(time_bins.dropna().unique())
                }
                time_stability = importance_stability_table(time_rankings)
                if not time_stability.empty:
                    time_stability.insert(0, "scope", "validation_time")
                    stability = pd.concat([stability, time_stability], ignore_index=True)
            _csv(temporary / "importance_stability.csv", stability)

            spec = CandidateSpec(**champion.members[0].training_configuration)
            def train_and_score(columns: tuple[int, ...]) -> float:
                estimator, _ = fit_candidate(spec, X_train[:, columns], y_train, seed=seed, tuning=False)
                return average_precision_score(y_validation, predict_probability(estimator, X_validation[:, columns]))
            ablations = feature_family_ablation(train_and_score, names, _feature_families(names))
            _csv(temporary / "feature_ablations.csv", ablations)

            sanity_rows = min(int(self.phase6["development_sanity_rows"]), len(global_train_index))
            sanity_X = train_sample[:sanity_rows]
            sanity_y = train_y_sample[:sanity_rows]
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(sanity_y)
            randomized_estimator, _ = fit_candidate(spec, sanity_X, shuffled, seed=seed + 91, tuning=True)
            randomized_ap = average_precision_score(validation_y_sample, predict_probability(randomized_estimator, validation_sample))
            randomization_passed = randomized_ap <= max(
                0.05, 5 * float(np.mean(validation_y_sample))
            )
            noise_train = rng.normal(size=(len(sanity_X), 1)).astype(np.float32)
            noise_validation = rng.normal(size=(len(validation_sample), 1)).astype(np.float32)
            noise_estimator, _ = fit_candidate(
                spec,
                np.column_stack((sanity_X, noise_train)),
                sanity_y,
                seed=seed + 92,
                tuning=True,
            )
            noise_names = (*names, "__random_noise_negative_control")
            noise_importance = permutation_importance_table(
                noise_estimator,
                np.column_stack((validation_sample, noise_validation)),
                validation_y_sample,
                noise_names,
                partition_name="validation",
                repeats=max(3, int(self.phase6["permutation_repeats"])),
                random_seed=seed + 92,
            ).reset_index(drop=True)
            noise_rank = int(
                noise_importance.index[
                    noise_importance["feature"] == "__random_noise_negative_control"
                ][0]
                + 1
            )
            noise_passed = noise_rank > max(1, int(np.ceil(0.2 * len(noise_names))))
            faithfulness = deletion_insertion_faithfulness(
                champion, validation_sample, background,
                dict(zip(shap_importance["feature"], shap_importance["importance"])), names,
                top_k=min(10, len(names)), random_seed=seed
            )
            shortcut = source_time_shortcut_audit(
                train_metadata.iloc[global_train_index].reset_index(drop=True), train_y_sample,
                validation_metadata.iloc[validation_index].reset_index(drop=True), validation_y_sample,
                random_seed=seed
            )
            agreement = method_agreement_table(global_importance)
            sanity_checks = {
                "label_randomization": {
                    "partition": "bounded_development_train_to_validation",
                    "rows": sanity_rows,
                    "average_precision": float(randomized_ap),
                    "validation_prevalence": float(np.mean(validation_y_sample)),
                    "passed": bool(randomization_passed),
                },
                "random_noise_feature": {
                    "implemented_as_negative_control": True,
                    "heldout_permutation_rank": noise_rank,
                    "features_including_noise": len(noise_names),
                    "passed": noise_passed,
                },
                "deletion_insertion_faithfulness": faithfulness,
                "source_time_shortcut_audit": shortcut,
                "method_agreement": agreement.to_dict(orient="records"),
                "all_required_checks_passed": bool(
                    faithfulness["passed"] and noise_passed and randomization_passed
                ),
            }
            atomic_write_json(temporary / "sanity_checks.json", sanity_checks)

            parser_columns = [name for name in ("parse_failure", "extraction_timeout", "file_too_large", "extraction_limit_reached") if name in test_metadata]
            abstained = test_metadata[parser_columns].fillna(0).astype(bool).any(axis=1).to_numpy() if parser_columns else np.zeros(len(test_metadata), dtype=bool)
            selected = select_local_cases(
                test_y, test_probability, threshold=champion.threshold,
                per_type=int(self.phase6["local_cases_per_type"]), abstained=abstained
            )
            test_shap, _ = shap_attributions(champion, test_cases, background)
            local = _local_records(
                champion, test_cases, test_y, test_probability, test_metadata, test_shap,
                names, background, np.asarray(y_train[background_index]), selected
            )
            atomic_write_json(temporary / "local_cases.json", {"cases": local, "raw_pdf_content_present": False})
            counterfactuals = observed_prototype_counterfactuals(
                champion, test_cases[selected], train_sample, train_y_sample, names, threshold=champion.threshold
            )
            counterfactual_frame = pd.DataFrame([asdict(record) for record in counterfactuals])
            counterfactual_frame["changed_features"] = counterfactual_frame["changed_features"].map(lambda values: json.dumps(list(values)))
            counterfactual_frame["diagnostic_not_causal"] = True
            _csv(temporary / "counterfactuals.csv", counterfactual_frame)

            subgroup_source = _verify_file_entry(phase5["outputs"]["metrics_by_subgroup"], description="Phase 5 subgroup metrics")
            subgroup = pd.read_csv(subgroup_source)
            if "bundle_key" in subgroup:
                subgroup = subgroup[subgroup["bundle_key"] == phase4["selected_bundle_key"]]
            _csv(temporary / "subgroup_errors.csv", subgroup)
            conclusions = (
                actionable_conclusions_markdown(
                    global_importance,
                    stability,
                    ablations,
                    minimum_methods=int(
                        self.phase6["minimum_independent_support_methods"]
                    ),
                )
                if sanity_checks["all_required_checks_passed"]
                else (
                    "# Actionable conclusions\n\n"
                    "No operational conclusion passed the Phase 6 sanity and "
                    "faithfulness gates. Investigate the failed checks in "
                    "`sanity_checks.json` before deployment.\n"
                )
            )
            atomic_write_text(temporary / "actionable_conclusions.md", conclusions)

            output_files = sorted(path for path in temporary.iterdir() if path.is_file())
            required = {
                "global_importance.csv", "importance_stability.csv", "interactions.csv",
                "feature_ablations.csv", "subgroup_errors.csv", "local_cases.json",
                "counterfactuals.csv", "sanity_checks.json", "actionable_conclusions.md",
            }
            if not required.issubset({path.name for path in output_files}):
                raise Phase6GateError("Phase 6 required output inventory is incomplete.")
            manifest_path = temporary / "manifest.json"
            manifest = {
                "phase6_version": PHASE6_VERSION,
                "experiment": self.identity.to_dict(),
                "phase4_champion_manifest_sha256": sha256_file(RESULTS_DIR / "phase4_champion.json"),
                "phase5_manifest_sha256": sha256_file(RESULTS_DIR / "phase5_manifest.json"),
                "sealed_test_reopened": False,
                "test_input": "bounded_sanitized_phase5_handoff_only",
                "train_reference_rows": int(len(background)),
                "validation_explanation_rows": int(len(validation_sample)),
                "outputs": {path.name: {"sha256": sha256_file(path)} for path in output_files},
            }
            atomic_write_json(manifest_path, manifest)
            os.replace(temporary, self.output_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        manifest_path = self.output_root / "manifest.json"
        summary.update(
            {
                "status": "phase_6_complete_deep_explainability",
                "phase_6": {
                    "manifest": str(manifest_path.resolve()),
                    "manifest_sha256": sha256_file(manifest_path),
                    "sealed_test_reopened": False,
                },
            }
        )
        atomic_write_json(RESULTS_DIR / "experiment_summary.json", summary)
        return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(Phase6Runner().run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
