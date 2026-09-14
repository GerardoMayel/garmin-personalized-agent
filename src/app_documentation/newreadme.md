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
│       ├── cd.yml
│       ├── ci.yml
│       └── data_sync_cron.yml
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.ui
│   ├── k8s/
│   │   ├── deployment-api.yaml
│   │   ├── deployment-ui.yaml
│   │   └── service.yaml
│   └── docker-compose.yml
│
├── configs/
│   ├── agent_graph_config.yaml
│   ├── base_config.yaml
│   ├── fine_tuning_lora.yaml
│   └── rag_settings.yaml
│
├── data/
│   ├── knowledge_base/
│   ├── multimodal/
│   ├── processed/
│   ├── raw/
│   └── training/
│
├── notebooks/
│   ├── 01_garmin_data_exploration.ipynb
│   ├── 02_time_series_causality_lab.ipynb
│   ├── 03_rag_chunking_and_eval.ipynb
│   └── 04_vlm_hypnogram_analysis.ipynb
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── edges.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py
│   │   ├── causality_engine.py
│   │   └── time_series_models.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   ├── app_documentation/
│   │   ├── __init__.py
│   │   ├── newreadme.md
│   │   ├── project_structure.txt
│   │   ├── run_map.sh
│   │   └── update_readme.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fit_decoder.py
│   │   ├── garmin_client.py
│   │   └── sync_pipeline.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader_and_chunker.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py
│   │   ├── evaluate_model.py
│   │   └── train_lora.py
│   ├── ui/
│   │   ├── components/
│   │   │   ├── charts.py
│   │   │   ├── chat_interface.py
│   │   │   └── maps.py
│   │   ├── pages/
│   │   │   ├── 1_📊_Recovery_&_Sleep.py
│   │   │   ├── 2_📈_Time_Series_&_ML.py
│   │   │   ├── 3_🖼️_VLM_Inspection.py
│   │   │   └── 4_🤖_Agent_Chat.py
│   │   └── app.py
│   └── vision/
│       ├── __init__.py
│       ├── chart_renderer.py
│       └── vlm_analyzer.py
│
├── tests/
│   ├── eval/
│   ├── integration/
│   └── unit/
│
├── .dvc/
│   ├── .gitignore
│   └── config
│
├── .dvcignore
├── .env.example
├── .gitignore
├── LICENSE
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
