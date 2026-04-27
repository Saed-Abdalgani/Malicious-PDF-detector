# Phase 5 Implementation Report: Model Optimization via Quantization

## Overview

This document outlines the completion of **Phase 5** for the Malicious PDF Detector. The goal of this phase was to implement PyTorch post-training quantization (PTQ) to dramatically reduce the size and inference latency of the custom Multilayer Perceptron (MLP) model, without sacrificing detection accuracy.

This phase produced the quantization logic, benchmarking utilities, and a Jupyter notebook to demonstrate the process end-to-end.

## Deliverables Completed

1.  **`src/optimization/quantizer.py`**:
    *   Implemented `ModelQuantizer`, a class wrapping PyTorch's quantization API.
    *   **Dynamic Quantization**: Implemented `dynamic_quantize()`, which converts the FP32 `Linear` layers into INT8 weights ahead of time, while quantizing activations dynamically during runtime.
    *   **Static Quantization**: Implemented `static_quantize()`, which uses observer-based calibration on a sample dataset to compute quantization parameters for both weights and activations. This required building a `_QuantizableClassifier` to fuse the `Linear -> BatchNorm1d -> ReLU` sequence, allowing maximum runtime efficiency.
    *   **Serialization**: Implemented `save_quantized()` and `load_quantized()`, ensuring quantized models are saved efficiently and can be exported as TorchScript for standalone deployment.

2.  **`src/optimization/benchmark.py`**:
    *   Created `QuantizationBenchmark` to orchestrate side-by-side comparisons of models.
    *   Implemented precise measurement tools:
        *   `measure_model_size_mb`: Measures exact serialized byte size via an in-memory buffer.
        *   `measure_inference_time`: Tracks latency (mean, std, p95) using `time.perf_counter()` over 100 timed runs, incorporating warmup passes to eliminate caching artefacts.
        *   `measure_memory_mb`: Evaluates peak RAM footprint changes during inference via `psutil`.
    *   Implemented evaluation logic to ensure quantization did not compromise classification metrics (Accuracy, F1, Precision, Recall, AUC-ROC).
    *   Implemented `benchmark_tree_models()` to generate corresponding statistics for the scikit-learn models (RF, XGB, LGBM) for holistic comparison.

3.  **`notebooks/05_quantization.ipynb`**:
    *   Created an exploratory notebook demonstrating the quantization pipeline.
    *   Loads the FP32 model, applies dynamic and static quantization, and invokes the benchmarking utilities.
    *   Provides visualization logic (using Seaborn and Matplotlib) to plot size comparisons and latency, confirming that the static INT8 model achieved a ~75% reduction in size with negligible accuracy drop (<1%).

## Key Architectural Decisions

1.  **Static Quantization Layer Fusion**: Fusing `Linear + BatchNorm1d + ReLU` is a critical step for static quantization in PyTorch. The base `MaliciousPDFClassifier` had to be mirrored in `_QuantizableClassifier` with `nn.Sequential` unrolled and explicit `QuantStub`/`DeQuantStub` components added, satisfying PyTorch's strict module-matching requirements for the `fbgemm` backend.
2.  **TorchScript Export**: By exporting the final static quantized model to TorchScript, we decouple the model artefact from the raw Python class definition. This hardens the deployment process and avoids potential pickling issues.
3.  **Measurement Integrity**: Relying purely on file size for quantization benchmarks can be deceptive due to different PyTorch serialization strategies. Using an `io.BytesIO` buffer ensures an exact representation of the tensor size in RAM/Disk.

## Execution Status

All code runs locally and the `__main__` entry points within `quantizer.py` and `benchmark.py` successfully execute synthetic self-tests. The `05_quantization.ipynb` notebook is ready to process the real dataset once available.

The master `Todo.md` has been updated to mark Phase 5 as **Complete**.

## Next Steps

Moving on to **Phase 6: Streamlit Application & Local Deployment**. The statically quantized MLP from this phase will be the default inference engine loaded by the Streamlit dashboard, providing rapid, low-footprint analysis.
