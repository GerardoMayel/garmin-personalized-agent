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
| **`fitness_age_records`** | `calendar_date` | Edad cronológica, edad física, brecha biológica, grasa corporal, FC reposo y potenciales. |
| **`activities`** | `activity_id` | Nombre, tipo, distancia, duración, desnivel, velocidad, FC media/máx, ruta local al `.fit.zip`. |
| **`consolidated_daily_actuals`** | `calendar_date` | **Tabla Única de Reales**: Consolidación aplanada (48 columnas) de telemetría diaria cerrada. |
| **`consolidated_biometric_forecasts`** | `(target_date, metric)` | **Tabla Única de Pronósticos**: Predicciones futuras con intervalos de confianza y linaje de modelos. |
| **`unified_biometrics_timeline`** | *(Vista SQL)* | **Línea Temporal Continua**: Empalme automático de reales pasados (`ACTUAL`) y proyecciones futuras (`FORECAST`). |

### 📊 Modelo de Datos Unificado y Consolidado

El sistema proporciona un modelo relacional unificado para simplificar el análisis longitudinal y la visualización sin necesidad de múltiples JOINs manuales:

#### 1. Tabla Única de Reales: `consolidated_daily_actuals`
- **Ubicación**: `data/processed/garmin_history.db` (respaldada en Cloudflare R2 en `processed/garmin_history.db`).
- **Gatillado**: Se actualiza y recalcula automáticamente al concluir la sincronización diaria de telemetría a día vencido (`ingest_raw_directory()` y `run_pipeline()`).
- **Diccionario de Campos Principal**:
  - `calendar_date` (*TEXT*, PK): Fecha calendario en formato ISO `YYYY-MM-DD`.
  - `total_steps` (*INTEGER*): Total de pasos diarios registrados.
  - `total_distance_meters` (*REAL*): Distancia total recorrida en metros.
  - `floors_ascended` (*REAL*): Pisos o tramos de escaleras subidos.
  - `active_kilocalories` / `resting_kilocalories` / `total_kilocalories` (*REAL*): Desglose energético en kcal.
  - `resting_heart_rate` (*INTEGER*): Frecuencia cardíaca en reposo en latidos por minuto (ppm).
  - `min_heart_rate` / `max_heart_rate` (*INTEGER*): Frecuencia cardíaca mínima y máxima del día.
  - `daily_avg_stress` / `daily_max_stress` (*INTEGER*): Puntuación de estrés diario (escala 0-100).
  - `rest_stress_duration_sec` / `low_stress_duration_sec` / `medium_stress_duration_sec` / `high_stress_duration_sec` (*INTEGER*): Tiempo acumulado en segundos por zona de estrés.
  - `sleep_score` (*INTEGER*): Puntuación global de calidad de sueño (0-100).
  - `total_sleep_seconds` (*INTEGER*): Duración total del periodo de sueño en segundos.
  - `deep_sleep_seconds` / `light_sleep_seconds` / `rem_sleep_seconds` / `awake_sleep_seconds` (*INTEGER*): Desglose de fases de sueño en segundos.
  - `avg_spo2` / `lowest_spo2` (*REAL*): Saturación de oxígeno en sangre media y mínima nocturna (%).
  - `avg_respiration` (*REAL*): Tasa respiratoria promedio nocturna (respiraciones por minuto).
  - `avg_sleep_stress` (*REAL*): Nivel de estrés vegetativo promedio durante el sueño.
  - `hrv_rmssd` (*REAL*): Variabilidad de la frecuencia cardíaca nocturna en milisegundos (rMSSD).
  - `hrv_weekly_avg` (*REAL*): Media móvil semanal de HRV (ms).
  - `hrv_status` (*TEXT*): Clasificación autonómica de Garmin (`BALANCED`, `UNBALANCED`, `LOW`).
  - `hrv_baseline_low` / `hrv_baseline_balanced_low` / `hrv_baseline_balanced_upper` (*REAL*): Límites de normalidad personal.
  - `vo2_max_running` / `vo2_max_precise` (*REAL*): Consumo máximo de oxígeno estimado en ml/kg/min.
  - `fitness_age` (*REAL*): Edad de condición física o biológica calculada por Garmin.
  - `chronological_age` (*REAL*): Edad cronológica real del usuario en años.
  - `achievable_fitness_age` (*REAL*): Edad biológica mínima alcanzable estimada.
  - `fitness_age_gap` (*REAL*): Ventaja biológica (`chronological_age - fitness_age`, positivo = menor edad física).
  - `body_fat_pct` (*REAL*): Porcentaje de grasa corporal registrado en báscula o manual.
  - `vigorous_minutes_avg` (*REAL*): Promedio de minutos semanales de intensidad vigorosa.
  - `activity_count` (*INTEGER*): Número de actividades deportivas estructuradas registradas en el día.
  - `total_activity_duration_sec` / `total_activity_distance_m` / `total_activity_calories` (*REAL*): Totales acumulados en sesiones deportivas.
  - `avg_activity_hr` / `max_activity_hr` (*REAL*): Frecuencia cardíaca media y máxima en actividades del día.

#### 2. Tabla Única de Pronósticos: `consolidated_biometric_forecasts`
- **Ubicación**: `data/processed/garmin_history.db`.
- **Gatillado**: Se actualiza y persiste de forma determinista cada vez que se ejecutan los pipelines de Machine Learning o el administrador de inferencias (`BiometricPredictionsManager.generate_and_update_forecasts()`).
- **Diccionario de Campos**:
  - `forecast_generated_date` (*TEXT*): Fecha ISO en la que se calculó la proyección.
  - `target_date` (*TEXT*, PK compuesta): Fecha futura proyectada `YYYY-MM-DD`.
  - `metric` (*TEXT*, PK compuesta): Indicador biométrico proyectado (`resting_heart_rate`, `total_steps`, `daily_avg_stress`, `sleep_score`, `hrv_rmssd`, `fitness_age`, `fitness_age_gap`, etc.).
  - `predicted_value` (*REAL*): Valor numérico proyectado puntual.
  - `ci_lower` / `ci_upper` (*REAL*): Intervalos de confianza inferior y superior al 95%.
  - `model_name` (*TEXT*): Algoritmo de modelado (`SARIMAX`, `Prophet`, `EnsembleBiometricForecaster`).
  - `is_locked` (*INTEGER*): `1` si la proyección está fijada e inmutable para evaluación post-hoc de error.
  - `updated_at` (*TEXT*): Timestamp UTC de inserción.

#### 3. Vista Continua Unificada: `unified_biometrics_timeline`
Proporciona una única interfaz SQL que une el pasado real y el futuro proyectado:
```sql
SELECT calendar_date, record_type, resting_heart_rate, hrv_rmssd, daily_avg_stress, sleep_score, fitness_age
FROM unified_biometrics_timeline
WHERE calendar_date BETWEEN '2026-09-01' AND '2026-09-28';
```
- Para fechas históricas cerradas: devuelve `record_type = 'ACTUAL'` con la telemetría real consolidada.
- Para fechas futuras: devuelve `record_type = 'FORECAST'` con las proyecciones pivoteadas por métrica.

### Operaciones de Base de Datos
```bash
# Construir o refrescar la tabla de reales consolidados:
uv run python -m src.common.database --build-consolidated

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

### 3. Sincronización a Día Vencido y Reconciliación Autorreparable
El pipeline de ingestión opera bajo la estricta **regla de día vencido ($T-1$)**: Garmin Connect mantiene campos incompletos o nulos (`averageStressLevel: null`, `totalSteps: null`) durante el día en curso ($T$). Para garantizar la máxima fidelidad fisiológica:
- **Día Vencido:** Nunca ingiere el día en curso; procesa únicamente jornadas cerradas y consolidadas ($< \text{hoy}$).
- **Reconciliador de 15 Días:** Audita una ventana móvil de 15 días comprobando integridad de archivos raw y registros en SQLite. Si detecta ficheros ausentes o nulos debidos a ejecuciones previas parciales, los repara automáticamente descargando la telemetría oficial de Garmin.
- **Purga de Días Incompletos:** Detecta y elimina cualquier partición o fila local o en R2 de fechas $\ge \text{hoy}$.

```bash
# Ejecutar reconciliación autorreparable de 15 días y respaldar en Cloudflare R2:
uv run python -m src.ingestion.sync_pipeline --days-back 15 --reconcile --r2-sync

# Forzar re-sincronización completa de toda la ventana:
uv run python -m src.ingestion.sync_pipeline --days-back 15 --force --r2-sync

# Sincronizar una fecha específica cerrada (día vencido):
uv run python -m src.ingestion.sync_pipeline --date 2026-09-20 --r2-sync
```

### 4. Almacenamiento en la Nube con Cloudflare R2
Persistencia y recuperación rápida de particiones raw y base de datos histórica mediante `src.common.r2_storage`:
```bash
# Respaldar SQLite y particiones raw hacia Cloudflare R2:
uv run python -m src.common.r2_storage --backup-db --sync-raw

# Restaurar SQLite y particiones raw completas desde Cloudflare R2:
uv run python -m src.common.r2_storage --restore-db --restore-raw
```

### 5. Ejecución de Tests Automatizados
```bash
uv run pytest tests/unit/ -v
```
