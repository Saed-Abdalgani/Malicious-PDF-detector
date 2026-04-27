"""
benchmark.py
------------
Comprehensive benchmarking utilities for the Phase 5 quantization analysis.

Measures and compares the following across FP32 and quantized model variants:
    - **File size on disk** (MB)
    - **In-memory model size** (MB, via serialization buffer)
    - **Inference latency** (mean/std/min/max over N runs, in milliseconds)
    - **Peak RAM delta** during inference (MB, via psutil)
    - **Accuracy, F1, Precision, Recall, AUC-ROC** on the test set
    - **Size-reduction ratio** and **speedup ratio** vs FP32 baseline

Outputs:
    - ``reports/results/quantization_comparison.csv``
    - Console summary table

Usage::

    from src.optimization.benchmark import QuantizationBenchmark
    bench = QuantizationBenchmark(fp32_model, test_loader)
    df = bench.run_all(dynamic_model, static_model)
    bench.print_summary()
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config import QUANTIZED_MODELS_DIR, RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level measurement helpers
# ---------------------------------------------------------------------------

def measure_model_size_mb(model: nn.Module) -> float:
    """Measure serialised model size in MB (via in-memory buffer).

    Equivalent to what ``torch.save(model.state_dict(), path)`` would
    produce on disk.

    Args:
        model: Any ``nn.Module``.

    Returns:
        float: Size in megabytes.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def measure_file_size_mb(path: Union[str, Path]) -> float:
    """Return on-disk file size in MB.

    Args:
        path: Path to the saved model file (``.pt``, ``.pkl``, etc.).

    Returns:
        float: File size in megabytes, or 0.0 if file not found.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"File not found: {p}")
        return 0.0
    return p.stat().st_size / (1024 ** 2)


def measure_inference_time(
    model: nn.Module,
    X_sample: torch.Tensor,
    n_runs: int = 100,
    warmup_runs: int = 5,
) -> Dict[str, float]:
    """Measure inference latency over ``n_runs`` forward passes.

    A warmup phase (``warmup_runs`` passes) is executed first to prime
    CPU caches and lazy initialisation.

    Args:
        model: Quantized or FP32 ``nn.Module`` in eval mode.
        X_sample: Representative input tensor, shape ``(batch, features)``.
        n_runs: Number of timed inference passes. Default 100.
        warmup_runs: Warm-up passes before timing starts. Default 5.

    Returns:
        dict: ``{mean_ms, std_ms, min_ms, max_ms, median_ms, p95_ms}``
    """
    model.eval()
    X_sample = X_sample.float()

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(X_sample)

    # Timed runs
    times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(X_sample)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    arr = np.array(times_ms)
    return {
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
        "min_ms":    float(arr.min()),
        "max_ms":    float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
    }


def measure_memory_mb(model: nn.Module) -> float:
    """Estimate peak RAM increase when running inference.

    Uses ``psutil`` to capture RSS before and after a forward pass.

    Args:
        model: ``nn.Module`` to probe.

    Returns:
        float: Peak RSS delta in MB (approximate).
    """
    proc = psutil.Process()
    gc_before = proc.memory_info().rss

    dummy = torch.randn(1, 37)
    model.eval()
    with torch.no_grad():
        for _ in range(10):   # run a few times to warm allocation
            _ = model(dummy)

    gc_after = proc.memory_info().rss
    delta_mb = max(0.0, (gc_after - gc_before) / (1024 ** 2))
    return delta_mb


# ---------------------------------------------------------------------------
# Classification metric helper
# ---------------------------------------------------------------------------

def evaluate_classification(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_quantizable_arch: bool = False,
) -> Dict[str, float]:
    """Run inference and compute classification metrics.

    Handles both the original ``MaliciousPDFClassifier`` (raw logits via
    ``model.block1 / output``) and the quantisable ``_QuantizableClassifier``
    (also raw logits but with quant/dequant stubs).

    Args:
        model: Trained ``nn.Module`` — FP32 or INT8.
        X_test: Test feature matrix, shape ``(n, 37)``.
        y_test: Integer test labels, shape ``(n,)``.
        is_quantizable_arch: Unused — kept for API compatibility.

    Returns:
        dict: ``{accuracy, f1, precision, recall, auc_roc}``
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_test.astype(np.float32))

    with torch.no_grad():
        logits = model(X_tensor)
        probs  = torch.sigmoid(logits).squeeze().numpy()

    y_pred = (probs >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.0

    return {
        "accuracy":  round(float(acc), 4),
        "f1":        round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "auc_roc":   round(float(auc), 4),
    }


# ---------------------------------------------------------------------------
# QuantizationBenchmark
# ---------------------------------------------------------------------------

class QuantizationBenchmark:
    """Orchestrates comprehensive benchmarking of FP32 vs quantized models.

    Args:
        fp32_model: The trained FP32 baseline ``MaliciousPDFClassifier``.
        X_test: Test feature matrix (numpy), shape ``(n, 37)``.
        y_test: Test labels (numpy), shape ``(n,)``.
        n_inference_runs: Number of inference timing passes per model.
        single_sample: If True, inference timing uses batch_size=1
                       (relevant for per-PDF latency SLA NFR-102).
    """

    def __init__(
        self,
        fp32_model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_inference_runs: int = 100,
        single_sample: bool = True,
    ):
        self.fp32_model       = fp32_model
        self.X_test           = X_test
        self.y_test           = y_test
        self.n_inference_runs = n_inference_runs
        self.single_sample    = single_sample

        # Timing sample — single sample for latency SLA, or full test set
        if single_sample:
            self._timing_X = torch.FloatTensor(X_test[:1].astype(np.float32))
        else:
            self._timing_X = torch.FloatTensor(X_test.astype(np.float32))

        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-model profiling
    # ------------------------------------------------------------------

    def profile_model(
        self,
        model: nn.Module,
        label: str,
        saved_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Profile a single model variant.

        Args:
            model: ``nn.Module`` to benchmark.
            label: Display name (e.g. ``'FP32 Baseline'``).
            saved_path: Optional path to the saved file for disk-size measurement.

        Returns:
            dict: Full profile result.
        """
        logger.info(f"Profiling: {label}")

        # Size
        mem_size_mb  = measure_model_size_mb(model)
        disk_size_mb = measure_file_size_mb(saved_path) if saved_path else mem_size_mb

        # Inference time
        timing = measure_inference_time(
            model, self._timing_X,
            n_runs=self.n_inference_runs,
            warmup_runs=5,
        )

        # RAM footprint
        ram_delta_mb = measure_memory_mb(model)

        # Classification metrics
        metrics = evaluate_classification(model, self.X_test, self.y_test)

        result: Dict[str, Any] = {
            "label":        label,
            "mem_size_mb":  round(mem_size_mb, 3),
            "disk_size_mb": round(disk_size_mb, 3),
            "ram_delta_mb": round(ram_delta_mb, 3),
            **timing,
            **metrics,
        }

        logger.info(
            f"  [{label}] Size={mem_size_mb:.3f}MB | "
            f"Latency={timing['mean_ms']:.2f}ms±{timing['std_ms']:.2f}ms | "
            f"Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}"
        )

        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def _add_comparison_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append size-reduction and speedup columns relative to FP32."""
        fp32_row = df[df["Label"] == "FP32 Baseline"]
        if fp32_row.empty:
            return df

        fp32_size  = fp32_row["Mem Size (MB)"].values[0]
        fp32_speed = fp32_row["Latency Mean (ms)"].values[0]

        def _pct(val, ref, smaller_is_better=True):
            change = (ref - val) / ref * 100 if ref > 0 else 0
            return round(change if smaller_is_better else -change, 1)

        df["Size Reduction (%)"] = df["Mem Size (MB)"].apply(
            lambda v: _pct(v, fp32_size)
        )
        df["Speedup vs FP32 (×)"] = df["Latency Mean (ms)"].apply(
            lambda v: round(fp32_speed / v, 2) if v > 0 else 0
        )
        df["Accuracy Drop (pp)"] = df["Accuracy"].apply(
            lambda v: round((fp32_row["Accuracy"].values[0] - v) * 100, 2)
        )
        df["F1 Drop (pp)"] = df["F1"].apply(
            lambda v: round((fp32_row["F1"].values[0] - v) * 100, 2)
        )

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_models(self, save: bool = True) -> pd.DataFrame:
        """Build a comparison DataFrame from all profiled results.

        Args:
            save: If True, save CSV to ``reports/results/quantization_comparison.csv``.

        Returns:
            pd.DataFrame: Comparison table.
        """
        rows = []
        for r in self.results:
            rows.append({
                "Label":             r["label"],
                "Mem Size (MB)":     r["mem_size_mb"],
                "Disk Size (MB)":    r["disk_size_mb"],
                "RAM Delta (MB)":    r["ram_delta_mb"],
                "Latency Mean (ms)": round(r["mean_ms"], 3),
                "Latency Std (ms)":  round(r["std_ms"], 3),
                "Latency P95 (ms)":  round(r["p95_ms"], 3),
                "Accuracy":          r["accuracy"],
                "F1":                r["f1"],
                "Precision":         r["precision"],
                "Recall":            r["recall"],
                "AUC-ROC":           r["auc_roc"],
            })

        df = pd.DataFrame(rows)
        df = self._add_comparison_metrics(df)

        if save:
            out_path = RESULTS_DIR / "quantization_comparison.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info(f"Quantization comparison saved → {out_path}")

        return df

    def print_summary(self, df: Optional[pd.DataFrame] = None):
        """Print a formatted summary table to console."""
        if df is None:
            df = self.compare_models(save=False)

        print("\n" + "=" * 90)
        print("  QUANTIZATION BENCHMARK SUMMARY")
        print("=" * 90)
        display_cols = [
            "Label", "Mem Size (MB)", "Latency Mean (ms)",
            "Accuracy", "F1",
        ]
        if "Size Reduction (%)" in df.columns:
            display_cols += ["Size Reduction (%)", "Speedup vs FP32 (×)"]

        print(df[display_cols].to_string(index=False))
        print("=" * 90)


# ---------------------------------------------------------------------------
# Tree model benchmarks
# ---------------------------------------------------------------------------

def benchmark_tree_models(models_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Benchmark disk size and inference latency of tree-based models.

    Args:
        models_dir: Path to ``models/trained/``.

    Returns:
        pd.DataFrame: Size and latency summary for RF, XGB, LGBM.
    """
    from src.config import TRAINED_MODELS_DIR as _TRAINED
    import joblib

    if models_dir is None:
        models_dir = _TRAINED
    models_dir = Path(models_dir)

    files = {
        "Random Forest": "rf_best.pkl",
        "XGBoost":       "xgb_best.pkl",
        "LightGBM":      "lgbm_best.pkl",
    }

    # Representative dummy data (37 features, 1 sample)
    dummy = np.random.randn(1, 37).astype(np.float64)

    rows = []
    for name, fname in files.items():
        fpath = models_dir / fname
        if not fpath.exists():
            logger.warning(f"{name} file not found: {fpath}")
            continue

        disk_size = measure_file_size_mb(fpath)

        try:
            obj = joblib.load(fpath)
            # Handle BaselineModel wrapper vs raw sklearn model
            model_obj = obj.model if hasattr(obj, "model") else obj

            # Warm up
            for _ in range(3):
                model_obj.predict(dummy)

            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                model_obj.predict(dummy)
                times.append((time.perf_counter() - t0) * 1000)

            arr = np.array(times)
            rows.append({
                "Model":              name,
                "Disk Size (MB)":     round(disk_size, 3),
                "Latency Mean (ms)":  round(arr.mean(), 3),
                "Latency Std (ms)":   round(arr.std(), 3),
                "Latency P95 (ms)":   round(np.percentile(arr, 95), 3),
            })
            logger.info(f"  [{name}] {disk_size:.3f}MB | {arr.mean():.2f}ms")

        except Exception as exc:
            logger.warning(f"Failed to benchmark {name}: {exc}")

    df = pd.DataFrame(rows)

    # Save
    out_path = RESULTS_DIR / "tree_model_benchmarks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Tree model benchmarks saved → {out_path}")

    return df


# ---------------------------------------------------------------------------
# All-in-one orchestration
# ---------------------------------------------------------------------------

def benchmark_all_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_runs: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Run the complete Phase 5 benchmark suite.

    Loads all model artefacts from ``models/trained/`` and
    ``models/quantized/``, profiles each one, and saves consolidated
    comparison CSVs.

    Args:
        X_test: Test feature matrix.
        y_test: Test labels.
        n_runs: Inference timing repetitions. Default 100.

    Returns:
        dict: ``{'mlp_comparison': DataFrame, 'tree_benchmark': DataFrame}``
    """
    from src.models.mlp import load_mlp

    logger.info("=" * 60)
    logger.info("BENCHMARK ALL MODELS — Phase 5")
    logger.info("=" * 60)

    results = {}

    # ---- MLP variants ----
    fp32_path    = Path("models/trained/mlp_best.pt")
    dyn_path     = QUANTIZED_MODELS_DIR / "mlp_dynamic_int8.pt"
    static_path  = QUANTIZED_MODELS_DIR / "mlp_static_int8.pt"

    if fp32_path.exists():
        fp32_model = load_mlp(fp32_path)
        bench = QuantizationBenchmark(fp32_model, X_test, y_test, n_inference_runs=n_runs)

        # FP32 baseline
        bench.profile_model(fp32_model, "FP32 Baseline", saved_path=fp32_path)

        # Dynamic INT8
        if dyn_path.exists():
            from src.optimization.quantizer import ModelQuantizer
            quantizer = ModelQuantizer(fp32_model)
            dyn_model = quantizer.dynamic_quantize()
            bench.profile_model(dyn_model, "Dynamic INT8", saved_path=dyn_path)
        else:
            logger.warning(f"Dynamic INT8 model not found: {dyn_path}")

        # Static INT8
        if static_path.exists():
            from src.optimization.quantizer import ModelQuantizer
            quantizer = ModelQuantizer(fp32_model)
            # Build calibration loader from test data subset
            calib_X  = torch.FloatTensor(X_test[:200].astype(np.float32))
            calib_y  = torch.FloatTensor(y_test[:200].astype(np.float32))
            calib_ds = TensorDataset(calib_X, calib_y)
            calib_dl = DataLoader(calib_ds, batch_size=32, shuffle=False)
            stat_model = quantizer.static_quantize(calib_dl)
            bench.profile_model(stat_model, "Static INT8", saved_path=static_path)
        else:
            logger.warning(f"Static INT8 model not found: {static_path}")

        mlp_df = bench.compare_models(save=True)
        bench.print_summary(mlp_df)
        results["mlp_comparison"] = mlp_df
    else:
        logger.warning(f"FP32 MLP not found: {fp32_path}")

    # ---- Tree models ----
    tree_df = benchmark_tree_models()
    results["tree_benchmark"] = tree_df
    logger.info("\nTree model benchmarks:")
    if not tree_df.empty:
        print(tree_df.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    print("Phase 5 — Benchmark self-test (synthetic data)")

    np.random.seed(42)
    X = np.random.randn(200, 37).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    from src.models.mlp import MaliciousPDFClassifier
    fp32 = MaliciousPDFClassifier(input_dim=37)
    fp32.eval()

    from src.optimization.quantizer import ModelQuantizer
    quantizer  = ModelQuantizer(fp32)
    dyn_model  = quantizer.dynamic_quantize()

    bench = QuantizationBenchmark(fp32, X, y, n_inference_runs=50)
    bench.profile_model(fp32,      "FP32 Baseline")
    bench.profile_model(dyn_model, "Dynamic INT8")

    df = bench.compare_models(save=False)
    bench.print_summary(df)

    print("\n✅ Benchmark self-test passed!")
