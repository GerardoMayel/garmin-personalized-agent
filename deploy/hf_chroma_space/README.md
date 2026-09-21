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

Full RAG Backend hosting ChromaDB vector storage, multi-layer guardrails (Rate Limiting, Semantic Domain Classifier, Anti-Obfuscation Injection Guard, Strict Language Filtering), Hybrid Reranking (Reciprocal Rank Fusion), and Gemini generative synthesis for Garmin physiological and biomechanical telemetry.

## Endpoints

- `GET /`: Service metadata, active guardrails, and LLM budget usage.
- `GET /health`: Healthcheck, uptime, Chroma collection count, and budget status.
- `GET /stats`: Breakdown of indexed chunks per source in `biometric_knowledge_base`.
- `POST /ask`: End-to-end RAG question answering with 4-layer guardrails, strict source filtering (`variables_fisiologia_humana` & `dispositivos_garmin_sensores`), hybrid RRF reranking, and Gemini generative synthesis.
- `POST /upsert`: Batch upsert of chunk IDs, embeddings (768 dims), clean text documents, and structured metadata.
- `POST /query`: Vector search query with embedding vector, `top_k`, and metadata filtering.
- `POST /delete`: Delete chunks by IDs.
- `POST /refresh-from-r2`: Synchronize and decompress the latest ChromaDB index directly from Cloudflare R2.
- `GET /docs`: Interactive Swagger UI documentation.

## Knowledge Base Sources & Filtering Scope

The vector database hosts 3 distinct sources (702 total chunks):

1. **`variables_fisiologia_humana` (423 chunks - Active in `/ask`)**:
   Human physiology and biometrics: Autonomic Nervous System (ANS), heart rate variability (rMSSD, SDNN, LF/HF), sleep architecture (NREM, REM, slow-wave deep sleep), VO2 Max, lactate kinetics, EPOC, and systemic recovery.
2. **`dispositivos_garmin_sensores` (268 chunks - Active in `/ask`)**:
   Garmin hardware and algorithms: Multi-channel PPG Elevate v4/v5 optical sensors, PulseOx SpO2, triaxial accelerometry, barometric altimetry, multi-band GPS, and Firstbeat Analytics algorithms (Body Battery, Sleep Score, Training Readiness, Training Status, Training Effect, Stamina, nocturnal HRV Status).
3. **`descripciones_metricas_garmin` (11 chunks - Metadata & Lookup only)**:
   Technical glossary and JSON schemas of Garmin Connect telemetry (`sleep.json`, `stress.json`, `rMSSD`, data types and units). **Excluded from `/ask` generation** to ensure answers are grounded in deep scientific and sensor engineering papers.

## Hybrid Reranker Engine

Retrieval expands to candidate pool `min(max(top_k * 3, 10), count)` and applies Reciprocal Rank Fusion (RRF $k=60$):
- **Dense Rank (0.55 weight)**: ChromaDB cosine distance.
- **Lexical Rank (0.45 weight)**: BM25 keyword density, acronym matches (`rMSSD`, `Elevate`, `VO2Max`), and metadata tag overlap.
- Runs entirely in-memory on CPU (< 1 ms), adding zero external API calls or latency.

## Multi-Layer Guardrails Architecture

1. **Strict Language Filter**: Only **Spanish** (`es`) and **English** (`en`) are accepted. Queries in French, German, Italian, Portuguese, or other languages are immediately intercepted and rejected with `status="unsupported_language"` with 0 LLM calls.
   - English queries receive complete answers in English.
   - Spanish queries receive complete answers in Spanish.
2. **Anti-Obfuscation Prompt Injection Guard**: Text desanitization (Unicode NFKD, leetspeak translation, repetition collapse, spaced bypass collapse) and regex rules across 5 attack families (`INSTRUCTION_OVERRIDE`, `ROLEPLAY_JAILBREAK`, `PROMPT_LEAK`, `SYNTAX_INJECTION`, `HYPOTHETICAL_BYPASS`).
3. **Semantic Domain Intent Classifier**: Reuses the query embedding vector against a Garmin/Firstbeat centroid reference text (threshold `0.52`) with O(1) lexical fast-path bypass.
4. **LLM Budget & Rate Protection**:
   - `slowapi` IP rate limiting (15/min, 60/hr, 100/day).
   - In-memory budget tracker strictly enforcing max 60 LLM generation calls/hour and max 100 LLM calls/day.

## Security

When configured as a **Private** Space on Hugging Face, all incoming requests are authenticated via:
`Authorization: Bearer <HF_TOKEN>`
