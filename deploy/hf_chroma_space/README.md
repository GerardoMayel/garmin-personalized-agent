---
title: Garmin Chroma Backend
emoji: 🏃
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Garmin Biometric ChromaDB Vector Backend

FastAPI service exposing ChromaDB vector storage for the Garmin Personal Insight Agent.

## Endpoints

- `GET /`: Service metadata.
- `GET /health`: Healthcheck, uptime, and collection count.
- `GET /stats`: Breakdown of chunks per source in `biometric_knowledge_base`.
- `POST /upsert`: Batch upsert of chunk IDs, embeddings, text documents, and metadata.
- `POST /query`: Vector search query with embedding vector, `top_k`, and metadata filtering.
- `GET /docs`: Interactive Swagger UI documentation.

## Security

When configured as a **Private** Space on Hugging Face, all incoming requests are authenticated via:
`Authorization: Bearer <HF_TOKEN>`
