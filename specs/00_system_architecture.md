# Specification 00: System Architecture Macro Specification

## 1. System Overview
The **Garmin Personal Insight Agent** is an end-to-end analytical framework and autonomous agent architecture designed to ingest, process, model, and interpret continuous biomechanical and physiological telemetry from Garmin wearables.

---

## 2. Architectural Flow

```text
       ┌────────────────────────────────────────────────────────┐
       │                 Garmin Connect API                    │
       └──────────────────────────┬─────────────────────────────┘
                                  │ (Sync / Ingestion)
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       Data Engine & Ingestion (Spec 02)                 │
 │  - Raw Partitioning (data/raw/YYYY-MM-DD/)                              │
 │  - Biometric JSONs: Sleep, HRV, Stress, VO2 Max, Daily Summaries        │
 │  - Binary .FIT Telemetry Parsing (fitparse -> Parquet)                  │
 └────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                  Data Schemas & Type Contracts (Spec 01)                │
 │  - Pydantic models (SleepRecord, HRVRecord, ActivityTelemetry)          │
 │  - Validation, unit normalizations (timestamps, heart rates, watts)     │
 └────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │               Time Series & Classical Machine Learning (Spec 03)        │
 │  - Rolling Baselines (7d, 28d windowing for rMSSD and Resting HR)       │
 │  - Anomaly Detection (Isolation Forest, Z-score robust filtering)       │
 │  - Granger Causality & Delayed Effects (Training Load vs. Deep Sleep)   │
 └────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │              Clinical RAG & Knowledge Base Engine (Spec 04)             │
 │  - Domain knowledge (SEC, SEPAR, GuíaSalud sports physiology consensuses)│
 │  - Hybrid Retrieval (Dense Vector via Qdrant/Chroma + BM25 Sparse)      │
 │  - Cross-Encoder Re-ranking                                             │
 └────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                  Agentic Reasoning Layer - LangGraph (Spec 05)          │
 │  - Stateful Diagnostic Graph (Router -> Analytics -> RAG -> Synthesizer)│
 │  - Fine-Tuned Domain LLM (PEFT/LoRA adapters for physiological terms)   │
 └────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       User Interfaces & Delivery                        │
 │  - Streamlit Dashboard (Biometrics, Hypnograms, Interactive Map, Chat)  │
 │  - FastAPI REST API (/health, /sync, /agent/invoke, /metrics)           │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack & Boundaries
- **Runtime**: Python 3.11+ managed via `uv`.
- **Packaging**: Hatchling (`pyproject.toml`).
- **Ingestion**: `garminconnect` (v0.3+), `fitparse` (v1.2+).
- **Processing**: `pandas`, `pyarrow` (Parquet series), `scipy`, `statsmodels`.
- **Orchestration**: `langgraph`, `langchain-core`.
- **Storage**:
  - Raw: `data/raw/` (ignored in Git, tracked via DVC).
  - Processed: `data/processed/` (Parquet).
  - Vector: Qdrant / ChromaDB (`data/chroma_db`).
- **Frontend**: Streamlit multi-page UI.
