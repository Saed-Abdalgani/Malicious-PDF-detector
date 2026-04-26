"""
llm — LLM Integration Package for Malicious PDF Detector
==========================================================

Provides local LLM-powered threat intelligence via Ollama and
Gemma 4 E4B / E2B models. The LLM serves as an explainability
and deep analysis layer — it does NOT replace the ML classifier.

Modules:
    client:           Ollama API client wrapper with health/RAM checks
    prompts:          Cybersecurity-tuned prompt templates (SOC analyst persona)
    analyzer:         Threat analysis pipeline (features → LLM → report)
    report_generator: Structured ThreatReport dataclass with export formats

Quick Start:
    >>> from src.llm import GemmaClient, ThreatAnalyzer
    >>> client = GemmaClient()
    >>> client.auto_initialize()
    >>> analyzer = ThreatAnalyzer(client)
    >>> report = analyzer.analyze(features, "Malicious", 0.97)
"""

from src.llm.client import GemmaClient
from src.llm.prompts import (
    FEATURE_DESCRIPTIONS,
    FOLLOW_UP_TEMPLATE,
    JAVASCRIPT_ANALYSIS_TEMPLATE,
    QUICK_SUMMARY_TEMPLATE,
    SYSTEM_PROMPT,
    THREAT_ANALYSIS_TEMPLATE,
    format_feature_summary,
    format_suspicious_features,
)
from src.llm.report_generator import (
    ThreatReport,
    generate_file_hash,
    save_report,
)
from src.llm.analyzer import ThreatAnalyzer

__all__ = [
    # Client
    "GemmaClient",
    # Prompts
    "SYSTEM_PROMPT",
    "THREAT_ANALYSIS_TEMPLATE",
    "JAVASCRIPT_ANALYSIS_TEMPLATE",
    "QUICK_SUMMARY_TEMPLATE",
    "FOLLOW_UP_TEMPLATE",
    "FEATURE_DESCRIPTIONS",
    "format_feature_summary",
    "format_suspicious_features",
    # Analyzer
    "ThreatAnalyzer",
    # Report
    "ThreatReport",
    "generate_file_hash",
    "save_report",
]
