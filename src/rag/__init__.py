"""Garmin Biometric Knowledge Base & Generative RAG Engine.

Provides token chunking, ledger state tracking, dense embeddings, vector storage,
multi-layer guardrails, and hybrid RRF reranking for Garmin telemetry and Firstbeat
Analytics physiology papers.
"""

from __future__ import annotations

from src.rag.guardrails import (
    LLMBudgetTracker,
    check_domain,
    detect_query_language,
    normalize_text,
    rerank_candidates,
    scan_prompt_injection,
)
from src.rag.reranker import rerank_candidates as rerank

__all__ = [
    "LLMBudgetTracker",
    "check_domain",
    "detect_query_language",
    "normalize_text",
    "rerank",
    "rerank_candidates",
    "scan_prompt_injection",
]
