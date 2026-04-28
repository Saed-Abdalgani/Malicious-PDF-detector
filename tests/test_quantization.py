"""
test_quantization.py
--------------------
Tests for dynamic and static post-training quantization.
"""

import time
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.mlp import MaliciousPDFClassifier
from src.optimization.quantizer import ModelQuantizer, get_model_size_mb

@pytest.fixture
def fp32_model():
    model = MaliciousPDFClassifier(input_dim=37)
    model.eval()
    return model

@pytest.fixture
def calibration_loader():
    X = torch.randn(100, 37)
    y = torch.randint(0, 2, (100,)).float()
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=32)

def skip_if_quantization_unsupported(exc):
    if "is not supported" in str(exc) or "FBGEMM" in str(exc) or "QNNPACK" in str(exc):
        pytest.skip("Quantization engine is not supported on this platform/PyTorch build.")
    raise exc

def test_dynamic_quantization_size(fp32_model):
    try:
        quantizer = ModelQuantizer(fp32_model, backend="qnnpack")
        dyn_model = quantizer.dynamic_quantize()
    except RuntimeError as e:
        skip_if_quantization_unsupported(e)
    
    fp32_size = get_model_size_mb(fp32_model)
    dyn_size = get_model_size_mb(dyn_model)
    
    assert dyn_size < fp32_size * 0.6  # Expected ~50% reduction
    assert dyn_size < 1.0  # G2 limit

def test_static_quantization_size(fp32_model, calibration_loader):
    try:
        quantizer = ModelQuantizer(fp32_model, backend="qnnpack")
        stat_model = quantizer.static_quantize(calibration_loader, n_calibration_batches=5)
    except RuntimeError as e:
        skip_if_quantization_unsupported(e)
    
    fp32_size = get_model_size_mb(fp32_model)
    stat_size = get_model_size_mb(stat_model)
    
    assert stat_size < fp32_size * 0.6
    assert stat_size < 1.0

def test_quantized_predictions_match_fp32(fp32_model):
    try:
        quantizer = ModelQuantizer(fp32_model, backend="qnnpack")
        dyn_model = quantizer.dynamic_quantize()
    except RuntimeError as e:
        skip_if_quantization_unsupported(e)
    
    dummy = torch.randn(100, 37)
    with torch.no_grad():
        fp32_preds = (torch.sigmoid(fp32_model(dummy)) >= 0.5).int()
        dyn_preds = (torch.sigmoid(dyn_model(dummy)) >= 0.5).int()
        
    agreement = (fp32_preds == dyn_preds).float().mean().item()
    assert agreement > 0.95

def test_quantized_inference_speed_nfr_102(fp32_model):
    try:
        quantizer = ModelQuantizer(fp32_model, backend="qnnpack")
        dyn_model = quantizer.dynamic_quantize()
    except RuntimeError as e:
        skip_if_quantization_unsupported(e)
    
    dummy = torch.randn(1, 37)
    
    # Warmup
    with torch.no_grad():
        dyn_model(dummy)
        
    start = time.perf_counter()
    with torch.no_grad():
        dyn_model(dummy)
    duration = time.perf_counter() - start
    
    assert duration < 0.500  # < 500ms single-sample inference

def test_quantizer_save_load(fp32_model, tmp_path):
    try:
        quantizer = ModelQuantizer(fp32_model, backend="qnnpack")
        dyn_model = quantizer.dynamic_quantize()
    except RuntimeError as e:
        skip_if_quantization_unsupported(e)
    
    save_path = tmp_path / "dyn.pt"
    quantizer.save_quantized(dyn_model, save_path, save_torchscript=False)
    
    loaded_model = ModelQuantizer.load_quantized(save_path, model_type="dynamic", input_dim=37)
    
    dummy = torch.randn(5, 37)
    with torch.no_grad():
        out1 = dyn_model(dummy)
        out2 = loaded_model(dummy)
    assert torch.allclose(out1, out2, atol=1e-5)
