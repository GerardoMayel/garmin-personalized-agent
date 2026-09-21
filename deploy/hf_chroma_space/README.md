---
title: Garmin Personal Data
emoji: 🏃
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Garmin Biometric RAG Backend

Full RAG Backend hosting ChromaDB vector storage, multi-layer guardrails (Rate Limiting, Semantic Domain Classifier, Prompt Injection Guard), and Gemini generative synthesis for Garmin physiological and biomechanical telemetry.

## Endpoints

- `GET /`: Service metadata and active guardrail list.
- `GET /health`: Healthcheck, uptime, and collection count.
- `GET /stats`: Breakdown of chunks per source in `biometric_knowledge_base`.
- `POST /ask`: End-to-end RAG question answering with 3-layer guardrail pipeline, ChromaDB retrieval, and Gemini synthesis with citations.
- `POST /upsert`: Batch upsert of chunk IDs, embeddings, text documents, and metadata.
- `POST /query`: Vector search query with embedding vector, `top_k`, and metadata filtering.
- `POST /delete`: Delete chunks by IDs.
- `POST /refresh-from-r2`: Synchronize and decompress the latest ChromaDB index directly from Cloudflare R2.
- `GET /docs`: Interactive Swagger UI documentation.

## Guardrails Architecture

1. **Rate Limiting (`slowapi`)**: 15 requests/minute and 150 requests/day per client IP.
2. **Prompt Injection Guard**: Regex detection for jailbreak patterns, instruction override, and persona spoofing.
3. **Semantic Domain Intent Classifier**: Cosine similarity against centroid reference text of Garmin/Firstbeat topics (threshold `0.55`). Out-of-domain queries are rejected instantly with zero LLM cost.

## Security

When configured as a **Private** Space on Hugging Face, all incoming requests are authenticated via:
`Authorization: Bearer <HF_TOKEN>`
