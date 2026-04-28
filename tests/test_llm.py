"""
test_llm.py
-----------
Tests for the local LLM integration (GemmaClient, ThreatAnalyzer, ThreatReport).
"""

import json
import httpx
from src.llm.client import GemmaClient
from src.llm.report_generator import ThreatReport
from src.llm.analyzer import ThreatAnalyzer

def test_client_check_health_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "gemma4:e4b"}]}
    
    mocker.patch("httpx.Client.get", return_value=mock_response)
    
    client = GemmaClient()
    assert client.check_health() is True
    assert client.is_available is True

def test_client_check_health_offline(mocker):
    mocker.patch("httpx.Client.get", side_effect=httpx.ConnectError("Offline"))
    
    client = GemmaClient()
    assert client.check_health() is False
    assert client.is_available is False

def test_client_check_ram_e4b(mocker):
    mock_mem = mocker.Mock()
    mock_mem.available = 6 * 1024 * 1024 * 1024  # 6GB
    mock_mem.percent = 50.0
    mock_mem.total = 16 * 1024 * 1024 * 1024
    mocker.patch("psutil.virtual_memory", return_value=mock_mem)
    
    client = GemmaClient(model="gemma4:e4b")
    model = client.check_ram()
    assert model == "gemma4:e4b"

def test_client_check_ram_disabled(mocker):
    mock_mem = mocker.Mock()
    mock_mem.available = 2 * 1024 * 1024 * 1024  # 2GB
    mock_mem.percent = 90.0
    mock_mem.total = 16 * 1024 * 1024 * 1024
    mocker.patch("psutil.virtual_memory", return_value=mock_mem)
    
    client = GemmaClient()
    model = client.check_ram()
    assert model is None

def test_threat_report_formats():
    report = ThreatReport(
        timestamp="2026-04-28 10:00:00",
        file_hash="dummy_hash",
        ml_prediction="Malicious",
        ml_confidence=0.99,
        risk_severity="High",
        threat_explanation="Explains the threat",
        attack_vector="Embedded JS",
        suspicious_features=[("js_count", 5, 3.0, "Too much JS")],
        remediation="Delete it",
        processing_time_ms=100
    )
    
    md = report.to_markdown()
    assert "dummy_hash" in md
    assert "Explains the threat" in md
    
    js = report.to_json()
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["file_hash"] == "dummy_hash"

def test_analyzer_identify_suspicious_features(mocker, dummy_features_dict):
    client = GemmaClient()
    baseline = {col: {"mean": 0.0, "std": 1.0} for col in dummy_features_dict.keys()}
    
    # Mock exists to true and load to return baseline
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("joblib.load", return_value=baseline)
    
    analyzer = ThreatAnalyzer(client)
    
    dummy_features_dict["js_count"] = 5.0 # Deviation of 5 std
    
    suspicious = analyzer.identify_suspicious_features(dummy_features_dict)
    assert len(suspicious) == 1
    assert suspicious[0][0] == "js_count"
