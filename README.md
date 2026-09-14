# Garmin Personal Insight Agent 🏃‍♂️💤🧠

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Framework](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://www.langchain.com/langgraph)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

Sistema analítico end-to-end y arquitectura de agentes personalizada para la ingestión, modelado e interpretación biomecánica y fisiológica continua a partir de telemetría de Garmin Connect.

El proyecto integra:
1. **Modelado estadístico y Machine Learning clásico** sobre series temporales fisiológicas densas (descomposición, causalidad Granger, correlación no lineal).
2. **Fine-Tuning de un modelo fundacional de lenguaje (LLM)** adaptado a terminología biomédica y estilos de razonamiento fisiológico deportivo.
3. **Agentes orquestados por grafos de estados (LangGraph)** con acceso a herramientas analíticas y base vectorial (RAG) de consensos clínicos en español.
4. **Dashboard analítico interactivo en Streamlit** para observabilidad de inferencias, hipnogramas y telemetría de actividades.

---

## 🏗️ Arquitectura del Sistema

```text
       ┌────────────────────────────────────────────────────────┐
       │                 Garmin Connect API                    │
       └──────────────────────────┬─────────────────────────────┘
                                  │ (Sync / Ingestion)
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       Data Engine & Pipelines                           │
 │  ┌───────────────────────────────┐     ┌─────────────────────────────┐  │
 │  │      Telemetry Processing     │     │      Sleep & Recovery       │  │
 │  │ (.fit files: GPS, HR, Cadence)│     │ (Hypnogram, HRV, SpO2, Resp)│  │
 │  └──────────────┬────────────────┘     └──────────────┬──────────────┘  │
 └─────────────────┼─────────────────────────────────────┼─────────────────┘
                   │                                     │
                   ▼                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │               Time Series & Classical Machine Learning Layer            │
 │  - Feature Store & Rolling Windows (Baselines personales de 7-28 días) │
 │  - Modelos de Series de Tiempo (ARIMA/Prophet, Detección de Anomalías) │
 │  - Inferencia Causal & Correlación (Granger Causality, Shapley Values)  │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │ (Extracted Insights & Signals)
                                   ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                Agentic Reasoning Layer (LangGraph)                      │
 │                                                                         │
 │   ┌────────────────────────┐         ┌──────────────────────────────┐   │
 │   │ Fine-Tuned Foundation  │         │ Medical Knowledge Base (RAG) │   │
 │   │ Model (Domain-Adapted) │◄───────►│ (SEC, GuíaSalud, SEPAR)      │   │
 │   └───────────┬────────────┘         └──────────────┬───────────────┘   │
 │               │                                     │                   │
 │               └───────────────┬─────────────────────┘                   │
 │                               ▼                                         │
 │         Stateful Orchestration Workflow (Diagnostic Graph)              │
 └───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                 Interactive Dashboard (Streamlit)                       │
 │  - Multi-page UI: Biometric Monitoring, Time Series Lab, Agent Chat     │
 │  - Visualizaciones interactivas de hipnogramas y rutas con Plotly/Pydeck│
 └─────────────────────────────────────────────────────────────────────────┘

## Estructura del Proyecto

garmin-personal-insight-agent/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Linting (ruff), type checking (mypy), tests
│       ├── cd.yml                    # Build y push de imágenes Docker (GHCR / Docker Hub)
│       └── data_sync_cron.yml        # Scheduled job para disparar pipeline de Garmin
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.api            # Imagen para FastAPI / LangGraph runtime
│   │   └── Dockerfile.ui             # Imagen liviana para Streamlit
│   ├── docker-compose.yml            # Orquestación local (API + UI + Qdrant/Chroma)
│   └── k8s/                          # Opcional: Manifiestos de despliegue cloud
│
├── configs/
│   ├── base_config.yaml              # Configuración general del sistema
│   ├── fine_tuning_lora.yaml         # Hyperparámetros (PEFT, rank, alpha, learning rate)
│   ├── rag_settings.yaml             # Embeddings model, reranker, chunk size/overlap
│   └── agent_graph_config.yaml       # Parámetros de estados y timeouts de LangGraph
│
├── data/                             # Ignorado en git (gestión vía DVC)
│   ├── raw/                          # Archivos .fit binarios y JSONs crudos de Garmin
│   ├── processed/                    # Series temporales limpias en Parquet
│   ├── knowledge_base/               # Literatura médica, consensos y guías (PDFs/MD)
│   ├── multimodal/                   # Capturas, hipnogramas renderizados e imágenes para VLM
│   └── training/                     # Splits de train/val/test para fine-tuning (JSONL)
│
├── notebooks/                        # Laboratorios de experimentación (EDA y prototipado)
│   ├── 01_garmin_data_exploration.ipynb
│   ├── 02_time_series_causality_lab.ipynb
│   ├── 03_rag_chunking_and_eval.ipynb
│   └── 04_vlm_hypnogram_analysis.ipynb
│
├── src/
│   ├── common/                       # Utilidades transversales y contratos de datos
│   │   ├── __init__.py
│   │   ├── config.py                 # Carga tipada de configuraciones (Pydantic Settings)
│   │   ├── logger.py                 # Logging estructurado JSON
│   │   └── schemas.py                # Schemas Pydantic (SleepRecord, FitTelemetry, etc.)
│   │
│   ├── ingestion/                    # Extracción y parsing de datos
│   │   ├── __init__.py
│   │   ├── garmin_client.py          # Conector a Garmin Connect API
│   │   ├── fit_decoder.py            # Decodificador de telemetría densa .fit (fitparse)
│   │   └── sync_pipeline.py          # Orquestador de ingestión y guardado en Parquet
│   │
│   ├── analytics/                    # Machine Learning clásico y series temporales
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py      # Outlier detection en rMSSD / HR basal
│   │   ├── causality_engine.py       # Granger causality y retardos entrenamiento vs. sueño
│   │   └── time_series_models.py     # Descomposición STL, rolling baselines y tendencias
│   │
│   ├── rag/                          # RAG clínico avanzado
│   │   ├── __init__.py
│   │   ├── loader_and_chunker.py     # Ingestión con metadatos estructurados
│   │   ├── vector_store.py           # Conector a Qdrant / ChromaDB
│   │   ├── reranker.py               # Cross-encoder para reordenar chunks relevantes
│   │   └── retriever.py              # Recuperador híbrido (Vector + BM25)
│   │
│   ├── vision/                       # Módulo Multimodal (VLM)
│   │   ├── __init__.py
│   │   ├── chart_renderer.py         # Convierte telemetría en artefactos visuales
│   │   └── vlm_analyzer.py           # Inferencia VLM para lectura de patrones en gráficas
│   │
│   ├── training/                     # Fine-tuning con GPU externa
│   │   ├── __init__.py
│   │   ├── dataset_builder.py        # Generación de pares sintéticos con CoT para SFT
│   │   ├── train_lora.py             # Script de entrenamiento (HF TRL / PEFT / Unsloth)
│   │   └── evaluate_model.py         # Evaluación de benchmarks de fisiología y guardrails
│   │
│   ├── agents/                       # Arquitectura Agéntica (LangGraph)
│   │   ├── __init__.py
│   │   ├── state.py                  # Definición del TypedDict / AgentState
│   │   ├── nodes.py                  # Nodos de decisión, análisis estadístico y RAG
│   │   ├── edges.py                  # Lógica de routing condicional
│   │   ├── tools.py                  # Tools invocables (ejecución de scripts ML y queries)
│   │   └── graph.py                  # Compilación del grafo orquestador de LangGraph
│   │
│   ├── api/                          # Backend desacoplado (FastAPI)
│   │   ├── __init__.py
│   │   ├── routes.py                 # Endpoints (/health, /sync, /agent/invoke, /metrics)
│   │   └── main.py                   # Servidor ASGI
│   │
│   └── ui/                           # Frontend (Streamlit)
│       ├── app.py                    # Entry point de la aplicación Streamlit
│       ├── components/               # Elementos visuales reutilizables
│       │   ├── charts.py             # Renderizado de hipnogramas y métricas con Plotly
│       │   ├── maps.py               # Visualizaciones de rutas GPS con Pydeck
│       │   └── chat_interface.py     # Interfaz conversacional con streaming
│       └── pages/                    # Vistas multipágina de Streamlit
│           ├── 1_📊_Recovery_&_Sleep.py
│           ├── 2_📈_Time_Series_&_ML.py
│           ├── 3_🖼️_VLM_Inspection.py
│           └── 4_🤖_Agent_Chat.py
│
├── tests/
│   ├── unit/                         # Pruebas unitarias de parsers, RAG y herramientas
│   ├── integration/                  # Pruebas de integración de la API y LangGraph
│   └── eval/                         # Evaluaciones de Ragas o TruLens para calidad de RAG
│
├── .dvc/                             # Configuración de DVC (Data Version Control)
├── .env.example                      # Variables de entorno requeridas
├── .gitignore
├── .pre-commit-config.yaml           # Hooks de pre-commit para calidad de código
├── pyproject.toml                    # Configuración central de dependencias (uv / pip)
├── uv.lock                           # Lockfile de versiones reproducibles
└── README.md