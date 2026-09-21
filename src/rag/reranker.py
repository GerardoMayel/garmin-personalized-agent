"""Hybrid Reranker Engine for Biometric Knowledge Base Retrieval.

Combines dense vector search (ChromaDB cosine distance) with lexical BM25/keyword
coverage and metadata tag matching using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from src.rag.guardrails import rerank_candidates

__all__ = ["rerank_candidates"]
