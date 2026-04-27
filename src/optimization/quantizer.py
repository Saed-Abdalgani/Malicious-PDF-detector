"""
quantizer.py
------------
PyTorch INT8 Post-Training Quantization (PTQ) for MaliciousPDFClassifier.

Supports two quantization strategies:

    1. **Dynamic Quantization** — weights are quantised to INT8 at model
       save time; activations are quantised dynamically at runtime.
       Easiest to apply, no calibration data needed.  Typical size
       reduction: ~4× for Linear layers.

    2. **Static Quantization** — both weights AND activations are
       quantised.  Requires a calibration pass over representative data.
       Generally faster inference than dynamic quantisation on CPU.

Design notes
~~~~~~~~~~~~
* Backend: ``fbgemm`` (x86 optimised) — matches ``config.QUANTIZATION_BACKEND``.
* ``Linear + BatchNorm1d + ReLU`` fused into a single operator before
  static quantisation for maximum efficiency.
* TorchScript export is supported for portable, deployment-ready artefacts.
* Quantised models are saved to ``models/quantized/``.

Usage::

    from src.optimization.quantizer import ModelQuantizer
    from src.models.mlp import MaliciousPDFClassifier, load_mlp

    fp32_model = load_mlp()
    quantizer  = ModelQuantizer(fp32_model)

    dyn_model  = quantizer.dynamic_quantize()
    stat_model = quantizer.static_quantize(calibration_loader)

    quantizer.save_quantized(dyn_model, "models/quantized/mlp_dynamic_int8.pt")
"""

import copy
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import QUANTIZATION_BACKEND, QUANTIZED_MODELS_DIR, TRAINED_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quantization-aware model wrapper
# ---------------------------------------------------------------------------

class _QuantizableClassifier(nn.Module):
    """Re-implementation of MaliciousPDFClassifier structured for static
    quantisation.

    Static PTQ requires that the model be written with explicit
    ``QuantStub`` / ``DeQuantStub`` markers, and that fused sequences
    (Linear → BN → ReLU) are declared as flat ``nn.Sequential`` modules
    so that ``torch.quantization.fuse_modules`` can locate them by their
    sub-module indices.

    The architecture is **identical** to ``MaliciousPDFClassifier``; only
    the module naming changes to facilitate layer fusion.
    """

    def __init__(self, input_dim: int = 37, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()

        # Quantization stubs
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Block 1: Linear → BN → ReLU → Dropout
        self.linear1 = nn.Linear(input_dim, 128)
        self.bn1     = nn.BatchNorm1d(128)
        self.relu1   = nn.ReLU(inplace=False)   # inplace=False required for fusing
        self.drop1   = nn.Dropout(dropout1)

        # Block 2: Linear → BN → ReLU → Dropout
        self.linear2 = nn.Linear(128, 64)
        self.bn2     = nn.BatchNorm1d(64)
        self.relu2   = nn.ReLU(inplace=False)
        self.drop2   = nn.Dropout(dropout2)

        # Block 3: Linear → BN → ReLU
        self.linear3 = nn.Linear(64, 32)
        self.bn3     = nn.BatchNorm1d(32)
        self.relu3   = nn.ReLU(inplace=False)

        # Output
        self.output  = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.output(x)
        x = self.dequant(x)
        return x

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "_QuantizableClassifier":
        """Copy weights from a trained ``MaliciousPDFClassifier`` instance.

        Args:
            mlp: Trained ``MaliciousPDFClassifier`` to clone weights from.

        Returns:
            _QuantizableClassifier: Weight-initialised quantisable model.
        """
        q_model = cls()
        sd = mlp.state_dict()

        mapping = {
            # block1
            "block1.0.weight": "linear1.weight",
            "block1.0.bias":   "linear1.bias",
            "block1.1.weight": "bn1.weight",
            "block1.1.bias":   "bn1.bias",
            "block1.1.running_mean": "bn1.running_mean",
            "block1.1.running_var":  "bn1.running_var",
            "block1.1.num_batches_tracked": "bn1.num_batches_tracked",
            # block2
            "block2.0.weight": "linear2.weight",
            "block2.0.bias":   "linear2.bias",
            "block2.1.weight": "bn2.weight",
            "block2.1.bias":   "bn2.bias",
            "block2.1.running_mean": "bn2.running_mean",
            "block2.1.running_var":  "bn2.running_var",
            "block2.1.num_batches_tracked": "bn2.num_batches_tracked",
            # block3
            "block3.0.weight": "linear3.weight",
            "block3.0.bias":   "linear3.bias",
            "block3.1.weight": "bn3.weight",
            "block3.1.bias":   "bn3.bias",
            "block3.1.running_mean": "bn3.running_mean",
            "block3.1.running_var":  "bn3.running_var",
            "block3.1.num_batches_tracked": "bn3.num_batches_tracked",
            # output
            "output.weight": "output.weight",
            "output.bias":   "output.bias",
        }

        new_sd = q_model.state_dict()
        for src_key, dst_key in mapping.items():
            if src_key in sd and dst_key in new_sd:
                new_sd[dst_key] = sd[src_key]

        q_model.load_state_dict(new_sd)
        return q_model


# ---------------------------------------------------------------------------
# ModelQuantizer
# ---------------------------------------------------------------------------

class ModelQuantizer:
    """Applies dynamic and static INT8 post-training quantization to the MLP.

    Args:
        model: Trained ``MaliciousPDFClassifier`` (FP32, in eval mode).
        backend: PyTorch quantization backend. Default ``'fbgemm'`` (x86).
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = QUANTIZATION_BACKEND,
    ):
        self.fp32_model = model
        self.backend    = backend

        torch.backends.quantized.engine = backend
        logger.info(f"ModelQuantizer initialised — backend: {backend}")

    # ------------------------------------------------------------------
    # Dynamic quantization
    # ------------------------------------------------------------------

    def dynamic_quantize(self) -> nn.Module:
        """Apply dynamic INT8 quantization to all ``nn.Linear`` layers.

        Dynamic quantization converts weight matrices to INT8 ahead of
        time and quantises activations on-the-fly during inference.  No
        calibration data required.

        Returns:
            nn.Module: Dynamically quantised model (FP32 → INT8 Linear).
        """
        fp32_copy = copy.deepcopy(self.fp32_model)
        fp32_copy.eval()

        quantized = torch.quantization.quantize_dynamic(
            fp32_copy,
            {nn.Linear},
            dtype=torch.qint8,
        )

        # Verify model still returns correct output shape
        dummy = torch.randn(4, 37)
        with torch.no_grad():
            out = quantized(dummy)
        assert out.shape == (4, 1), f"Unexpected output shape: {out.shape}"

        logger.info("Dynamic INT8 quantization applied successfully.")
        return quantized

    # ------------------------------------------------------------------
    # Static quantization
    # ------------------------------------------------------------------

    def static_quantize(
        self,
        calibration_loader: DataLoader,
        n_calibration_batches: int = 20,
    ) -> nn.Module:
        """Apply static INT8 quantization with observer-based calibration.

        Steps
        ~~~~~
        1. Clone weights into ``_QuantizableClassifier`` (adds QuantStub/DeQuantStub).
        2. Fuse ``Linear → BatchNorm1d → ReLU`` triplets.
        3. Set ``qconfig`` (``fbgemm`` backend).
        4. Insert observers via ``prepare``.
        5. Calibrate on representative training data.
        6. Convert to static INT8 via ``convert``.

        Args:
            calibration_loader: DataLoader providing (X, y) batches from the
                training set (typically 500–1000 representative samples).
            n_calibration_batches: Max batches to run through for calibration.

        Returns:
            nn.Module: Statically quantised model (INT8 weights + activations).
        """
        # Step 1 — Build quantisable architecture & copy FP32 weights
        q_model = _QuantizableClassifier.from_mlp(self.fp32_model)
        q_model.eval()

        # Step 2 — Fuse Linear + BN + ReLU
        fuse_list = [
            ["linear1", "bn1", "relu1"],
            ["linear2", "bn2", "relu2"],
            ["linear3", "bn3", "relu3"],
        ]
        torch.quantization.fuse_modules(q_model, fuse_list, inplace=True)
        logger.info("Layer fusion complete: Linear+BN+ReLU fused.")

        # Step 3 — Set qconfig
        q_model.qconfig = torch.quantization.get_default_qconfig(self.backend)

        # Step 4 — Prepare (insert observers)
        torch.quantization.prepare(q_model, inplace=True)
        logger.info("Observers inserted.  Starting calibration...")

        # Step 5 — Calibration pass
        q_model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if batch_idx >= n_calibration_batches:
                    break
                # Support (X, y) and plain X batches
                X_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                X_batch = X_batch.float()
                q_model(X_batch)

        logger.info(
            f"Calibration complete — {min(batch_idx + 1, n_calibration_batches)} "
            f"batches processed."
        )

        # Step 6 — Convert to INT8
        torch.quantization.convert(q_model, inplace=True)

        # Verify
        dummy = torch.randn(4, 37)
        with torch.no_grad():
            out = q_model(dummy)
        assert out.shape == (4, 1), f"Unexpected output shape after static quantization: {out.shape}"

        logger.info("Static INT8 quantization applied successfully.")
        return q_model

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_quantized(
        self,
        model: nn.Module,
        path: Optional[Union[str, Path]] = None,
        save_torchscript: bool = True,
    ) -> Path:
        """Save a quantised model's state_dict and (optionally) a TorchScript artefact.

        Args:
            model: Quantised ``nn.Module`` to save.
            path: ``.pt`` file path. If None defaults to
                  ``models/quantized/mlp_quantized.pt``.
            save_torchscript: If True, also save a TorchScript ``.ptc``
                              file for portable, production deployment.

        Returns:
            Path: Absolute path of the saved ``.pt`` file.
        """
        if path is None:
            path = QUANTIZED_MODELS_DIR / "mlp_quantized.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save state dict
        torch.save(model.state_dict(), path)
        logger.info(f"Quantized model state_dict saved → {path}")

        # TorchScript export
        if save_torchscript:
            ts_path = path.with_suffix(".ptc")
            try:
                model.eval()
                scripted = torch.jit.script(model)
                scripted.save(str(ts_path))
                logger.info(f"TorchScript model saved → {ts_path}")
            except Exception as exc:
                logger.warning(
                    f"TorchScript export failed (model may use dynamic control flow): {exc}"
                )

        return path

    @staticmethod
    def load_quantized(
        path: Union[str, Path],
        model_type: str = "dynamic",
        input_dim: int = 37,
    ) -> nn.Module:
        """Load a previously saved quantized model.

        For **dynamic** quantization, reconstruct the original FP32 architecture
        and apply ``quantize_dynamic`` before loading the state_dict (PyTorch
        stores dynamic-quantized state_dicts in the quantized format).

        For **static** quantization, reconstruct ``_QuantizableClassifier``
        with observers fused, then load.

        Args:
            path: Path to the saved ``.pt`` file.
            model_type: ``'dynamic'`` or ``'static'``.
            input_dim: Number of input features.

        Returns:
            nn.Module: Loaded quantized model in eval mode.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Quantized model not found: {path}")

        if model_type == "dynamic":
            from src.models.mlp import MaliciousPDFClassifier
            base = MaliciousPDFClassifier(input_dim=input_dim)
            quantized = torch.quantization.quantize_dynamic(
                base, {nn.Linear}, dtype=torch.qint8
            )
            quantized.load_state_dict(
                torch.load(path, weights_only=True, map_location="cpu")
            )
        elif model_type == "static":
            quantized = _QuantizableClassifier(input_dim=input_dim)
            fuse_list = [
                ["linear1", "bn1", "relu1"],
                ["linear2", "bn2", "relu2"],
                ["linear3", "bn3", "relu3"],
            ]
            torch.quantization.fuse_modules(quantized, fuse_list, inplace=True)
            quantized.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(quantized, inplace=True)
            torch.quantization.convert(quantized, inplace=True)
            quantized.load_state_dict(
                torch.load(path, weights_only=True, map_location="cpu")
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Use 'dynamic' or 'static'.")

        quantized.eval()
        logger.info(f"Quantized model ({model_type}) loaded from {path}")
        return quantized


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_model_size_mb(model: nn.Module) -> float:
    """Estimate in-memory size of a model by serialising to a buffer.

    This is the canonical way to measure quantized model size because
    ``torch.save`` to disk and this buffer-based approach produce identical
    byte counts.

    Args:
        model: Any ``nn.Module``.

    Returns:
        float: Size in megabytes (MB).
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.mlp import MaliciousPDFClassifier, load_mlp

    print("=" * 60)
    print("Phase 5 — Quantization Self-Test")
    print("=" * 60)

    # Try to load trained model; fall back to fresh (untrained) model
    mlp_path = TRAINED_MODELS_DIR / "mlp_best.pt"
    if mlp_path.exists():
        fp32_model = load_mlp(mlp_path)
        print(f"Loaded trained MLP from {mlp_path}")
    else:
        fp32_model = MaliciousPDFClassifier(input_dim=37)
        fp32_model.eval()
        print("No trained MLP found — using random-weight model for shape test.")

    fp32_size = get_model_size_mb(fp32_model)
    print(f"\nFP32 model size: {fp32_size:.3f} MB")

    # --- Dynamic quantization ---
    quantizer = ModelQuantizer(fp32_model)
    dyn_model = quantizer.dynamic_quantize()
    dyn_size  = get_model_size_mb(dyn_model)
    print(f"Dynamic INT8 size: {dyn_size:.3f} MB  ({(1 - dyn_size/fp32_size)*100:.1f}% reduction)")

    dyn_save = quantizer.save_quantized(
        dyn_model,
        QUANTIZED_MODELS_DIR / "mlp_dynamic_int8.pt",
        save_torchscript=True,
    )
    print(f"Saved: {dyn_save}")

    # --- Static quantization (calibration on synthetic data) ---
    np.random.seed(42)
    X_calib = torch.FloatTensor(np.random.randn(200, 37).astype(np.float32))
    y_calib = torch.FloatTensor(np.random.randint(0, 2, 200).astype(np.float32))
    calib_ds = TensorDataset(X_calib, y_calib)
    calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=False)

    stat_model = quantizer.static_quantize(calib_loader)
    stat_size  = get_model_size_mb(stat_model)
    print(f"Static INT8 size:  {stat_size:.3f} MB  ({(1 - stat_size/fp32_size)*100:.1f}% reduction)")

    stat_save = quantizer.save_quantized(
        stat_model,
        QUANTIZED_MODELS_DIR / "mlp_static_int8.pt",
        save_torchscript=False,
    )
    print(f"Saved: {stat_save}")

    # --- Verify predictions match ---
    dummy = torch.randn(10, 37)
    fp32_model.eval()
    with torch.no_grad():
        fp32_preds = (torch.sigmoid(fp32_model(dummy)) >= 0.5).int().squeeze()
        dyn_preds  = (torch.sigmoid(dyn_model(dummy)) >= 0.5).int().squeeze()
        stat_preds = (torch.sigmoid(stat_model(dummy)) >= 0.5).int().squeeze()

    agreement_dyn  = (fp32_preds == dyn_preds).float().mean().item()
    agreement_stat = (fp32_preds == stat_preds).float().mean().item()
    print(f"\nPrediction agreement (FP32 vs Dynamic): {agreement_dyn*100:.1f}%")
    print(f"Prediction agreement (FP32 vs Static):  {agreement_stat*100:.1f}%")

    print("\n✅ Quantization self-test passed!")
