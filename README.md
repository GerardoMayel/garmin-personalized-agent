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
