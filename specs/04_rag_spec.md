# Specification 04: Clinical RAG Engine Specification

## 1. Scope
Defines the document loading, chunking strategy, embeddings model, metadata schema, and re-ranking for medical consensus documents.

## 2. Ingestion & Chunking
- **Sources**: PDFs/Markdown guidelines from SEC (Sociedad Española de Cardiología), SEPAR (Neumología/Sueño), GuíaSalud.
- **Chunk Size**: 512 tokens with 64 token overlap.
- **Embeddings**: `BAAI/bge-m3` or `text-embedding-3-small`.
- **Hybrid Search**: Vector dense cosine distance combined with BM25 sparse lexical search via Reciprocal Rank Fusion (RRF).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` or BGE-reranker-large.
