# Garmin Personal Insight Agent 🏃‍♂️💤🧠

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Framework](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://www.langchain.com/langgraph)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

Sistema analítico end-to-end y arquitectura de agentes personalizada para la ingestión, modelado e interpretación biomecánica y fisiológica continua a partir de telemetría de Garmin Connect.

El proyecto integra:
1. **Modelado estadístico y Machine Learning clásico** sobre series temporales fisiológicas densas (descomposición, causalidad Granger, correlación no lineal).
2. **Base de Datos Histórica SQLite** para almacenamiento relacional optimizado, consultas analíticas y particionamiento en `data/processed/garmin_history.db`.
3. **Fine-Tuning de un modelo fundacional de lenguaje (LLM)** adaptado a terminología biomédica y estilos de razonamiento fisiológico deportivo.
4. **Agentes orquestados por grafos de estados (LangGraph)** con acceso a herramientas analíticas y base vectorial (RAG) de consensos clínicos en español.
5. **Dashboard analítico interactivo en Streamlit** para observabilidad de inferencias, hipnogramas y telemetría de actividades.

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
 │              Historical Storage & Relational Database Layer             │
 │  - SQLite (`data/processed/garmin_history.db`): Daily, HRV, Sleep, FIT  │
 │  - Raw Partitioned Snapshots (`data/raw/YYYY-MM-DD/` via DVC)           │
 └─────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
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
 │  - Structured JSON Logging & Rotational Auditing (`logs/`)              │
 └───────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                 Interactive Dashboard (Streamlit)                       │
 │  - Multi-page UI: Biometric Monitoring, Time Series Lab, Agent Chat     │
 │  - Visualizaciones interactivas de hipnogramas y rutas con Plotly/Pydeck│
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Base de Datos Histórica (SQLite)

Para superar las limitaciones de ventanas móviles y permitir modelado longitudinal a largo plazo, el sistema cuenta con un motor relacional en **`data/processed/garmin_history.db`** ([`src/common/database.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/common/database.py)):

| Tabla | Clave Primaria | Métricas Principales Almacenadas |
| :--- | :--- | :--- |
| **`daily_summaries`** | `calendar_date` | Pasos, distancia, FC reposo, calorías activas, estrés promedio, minutos vigorosos/moderados. |
| **`sleep_records`** | `calendar_date` | Puntuación de sueño, fases (profundo, ligero, REM, vigilia), SpO2, respiración, inicio/fin. |
| **`hrv_records`** | `calendar_date` | rMSSD nocturno, media semanal, estado (`BALANCED`, `UNBALANCED`), baselines personalizadas. |
| **`stress_records`** | `calendar_date` | Nivel promedio y máximo de estrés, duraciones en reposo, actividad y niveles bajo/medio/alto. |
| **`max_metrics`** | `calendar_date` | VO2 Max de carrera/ciclismo, edad de condición física (Fitness Age). |
| **`activities`** | `activity_id` | Nombre, tipo, distancia, duración, desnivel, velocidad, FC media/máx, ruta local al `.fit.zip`. |

### Operaciones de Base de Datos
```bash
# Cargar/migrar datos históricos existentes desde data/raw/ hacia SQLite:
uv run python -m src.common.database --backfill

# Consultar el recuento y estado de registros en todas las tablas:
uv run python -m src.common.database --stats
```

---

## 📝 Sistema de Logs Centralizado

El sistema implementa un servicio de logging estructurado y asíncrono con **Loguru** ([`src/common/logger.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/common/logger.py)):
- **Consola:** Trazas coloreadas con marcas de tiempo, módulo emisor y nivel de gravedad.
- **Archivo rotativo:** Registro persistente en `logs/garmin_agent.log` (rotación cada 10 MB, retención de 14 días y compresión `.zip`).
- **Nivel configurable:** Controlado mediante `LOG_LEVEL` en tu archivo `.env` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- **Intercepción:** Captura automática de logs emitidos por librerías estándar (`urllib3`, `requests`, `garminconnect`).

---

## 📁 Estructura del Proyecto

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
│   │   ├── database.py
│   │   ├── logger.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fit_decoder.py
│   │   ├── garmin_client.py
│   │   ├── garmin_sync.py
│   │   ├── sample_sync.py
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
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── chart_renderer.py
│   │   └── vlm_analyzer.py
│   └── __init__.py
│
├── tests/
│   ├── eval/
│   ├── integration/
│   └── unit/
│       ├── test_database.py
│       ├── test_garmin_client.py
│       ├── test_logger.py
│       ├── test_sample_sync.py
│       └── test_sync_pipeline.py
│
├── specs/
│   ├── 00_system_architecture.md
│   ├── 01_data_schemas_spec.md
│   ├── 02_ingestion_spec.md
│   ├── 03_time_series_spec.md
│   ├── 04_rag_spec.md
│   └── 05_agent_graph_spec.md
│
├── .dvc/
│   ├── .gitignore
│   └── config
│
├── .agents/
│   └── rules/
│       └── git-commits.md
│
├── .dvcignore
├── .env.example
├── .gitignore
├── LICENSE
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
├── AGENTS.md
└── README.md


## ⚡ Guía de Inicio Rápido

### 1. Configurar Entorno
```bash
# Sincronizar dependencias con uv
uv sync --extra dev

# Copiar variables de entorno y configurar credenciales
cp .env.example .env
# Modifica .env con tu GARMIN_EMAIL y GARMIN_PASSWORD
```

### 2. Probar Conexión con Muestra Ligera (Recomendado)
Descarga en 3 segundos los datos de ayer y tu última actividad para verificar credenciales y tokens de sesión:
```bash
uv run python -m src.ingestion.sample_sync
```

### 3. Sincronización Completa y Persistencia en SQLite
Descarga la ventana móvil de los últimos 7 días y persiste automáticamente en `data/raw/` y `data/processed/garmin_history.db`:
```bash
uv run python -m src.ingestion.garmin_sync
```

### 4. Ejecución de Tests Automatizados
```bash
uv run pytest tests/unit/ -v
```
