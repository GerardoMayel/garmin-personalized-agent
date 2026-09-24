# Garmin Personal Insight Agent 🏃‍♂️💤🧠

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Storage: Cloudflare R2](https://img.shields.io/badge/Storage-Cloudflare_R2-F38020.svg)](https://www.cloudflare.com/developer-platform/r2/)
[![Database: SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging_Face-FFD21E.svg)](https://huggingface.co/datasets/GerardoMayel/garmin-mexican-fitness-coach-sft)
[![CI](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/ci.yml)
[![Science Sync](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/sync_science_papers.yml/badge.svg)](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/sync_science_papers.yml)

Sistema analítico end-to-end y arquitectura de agentes personalizados para la ingestión, persistencia relacional, modelado biométrico longitudinal y procesamiento de base de conocimiento científica (RAG) a partir de telemetría de Garmin Connect y literatura de Firstbeat Analytics.

---

## 📌 Estado Actual del Proyecto (Fase 1: Ingestión, Fase 2: RAG Backend & Fase 3/4: Dataset SFT)

El proyecto cuenta con sus componentes fundamentales activos, probados y desplegados con CI/CD automatizado:

1. **Ingestión de Telemetría Garmin Connect**: Cliente con autenticación MFA, persistencia de tokens de sesión (`~/.garminconnect`) y extracción de resúmenes diarios, sueño, estrés, HRV, VO2Max y actividades FIT.
2. **Capa Relacional Histórica (SQLite)**: Base de datos estructurada en `data/garmin_personal.db` para análisis longitudinales con 6 esquemas analíticos (`daily_summaries`, `sleep_records`, `hrv_records`, `stress_records`, `max_metrics`, `activities`).
3. **Machine Learning Clásico & Series de Tiempo**: Modelos predictivos (ARIMA, Prophet, Holt-Winters), descomposición temporal (tendencia, estacionalidad, residuos), motores de detección de anomalías (Z-Score, IQR, Isolation Forest) y análisis de causalidad de Granger.
4. **Almacenamiento en Cloudflare R2**: Bucket S3-compatible `garmin-personal-data` para resguardo automatizado de snapshots crudos, base SQLite histórica, documentos científicos, datasets particionados Parquet y el tarball vectorial `chroma_db.tar.gz`.
5. **RAG Knowledge Base & Chunking Pipeline (Fase 1 & Fase 2)**:
   - **3 Fuentes Científicas y Técnicas**: `variables_fisiologia_humana` (423 chunks), `dispositivos_garmin_sensores` (268 chunks) y `descripciones_metricas_garmin` (11 chunks).
   - **Embeddings Densos**: Modelo `gemini-embedding-2` de Google (768 dimensiones) con indexación incremental batch.
   - **ChromaDB en Hugging Face Spaces**: Desplegado en el Space privado `GerardoMayel/garmin-personal-data` sobre hardware gratuito `cpu-basic` ($0.00/mes) con sincronización Direct-to-Storage desde Cloudflare R2.
   - **Restricción Estricta de Fuentes en `/ask`**: Generación basada exclusivamente en `variables_fisiologia_humana` y `dispositivos_garmin_sensores`, preservando `descripciones_metricas_garmin` para lookup de esquemas.
   - **Motor de Reranking Híbrido**: Reciprocal Rank Fusion (RRF $k=60$) combinando distancia vectorial de coseno con densidad léxica BM25 y coincidencia de tags sin llamadas extra a APIs.
   - **Guardrails Multicapa de Producción**: Rate limiting (`slowapi`), escáner anti-ofuscación de inyecciones de prompt, clasificador semántico de dominio y limitador de presupuesto en memoria (máx. 60 llamadas/hora y 100/día).
   - **Control Estricto de Idioma**: Español base (respuestas siempre en español), inglés permitido (respuestas en inglés) y rechazo inmediato de cualquier otro idioma sin coste LLM.
6. **Automatización en GitHub Actions**: Flujos programados para sincronización diaria de telemetría, actualización de literatura científica y keep-alive de Hugging Face Spaces.
7. **Suite de Pruebas Unitaria**: **102 tests automatizados pasando** con `pytest`, formateo con `ruff` y tipado estricto al 100% con `mypy`.
8. **Pipeline de Datos Sintéticos & Dataset SFT en Hugging Face**: Pipeline multi-modelo de generación sintética basado en Google Gemini (`src/training/generate_synthetic_dataset.py`) con sondeo dinámico de latencias (`models_registry.json`), balanceo en 3 niveles de esfuerzo (alta intensidad, base aeróbica, recuperación/alerta) y control estricto de tokens (200-350 tokens). Publicado el dataset bilingüe (196 ES / 204 EN) de 400 pares ChatML en [Hugging Face Datasets: GerardoMayel/garmin-mexican-fitness-coach-sft](https://huggingface.co/datasets/GerardoMayel/garmin-mexican-fitness-coach-sft) para Supervised Fine-Tuning (SFT / LoRA).

---

## 🏗️ Arquitectura del Sistema

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FUENTES DE DATOS                                     │
├──────────────────────────────────────┬─────────────────────────────────────────────────┤
│          Garmin Connect API          │            Firstbeat Science & Tech             │
│   (Telemetría personal, sueño, HRV)  │      (30 White Papers + Glosario Oficial)       │
└──────────────────┬───────────────────┴────────────────────────┬────────────────────────┘
                   │                                            │
                   ▼ (Sync Pipeline)                            ▼ (Download & Clean)
┌──────────────────────────────────────┐       ┌─────────────────────────────────────────┐
│     Ingestión & SQLite Histórico     │       │     Knowledge Base (3 Fuentes RAG)      │
│  - data/garmin_personal.db           │       │  1. dispositivos_garmin_sensores (PDF)  │
│  - Snapshots raw en JSON             │       │  2. variables_fisiologia_humana (PDF)   │
│  - DVC tracking                      │       │  3. descripciones_metricas_garmin (MD)  │
└──────────────────┬───────────────────┘       └────────────────────┬────────────────────┘
                   │                                                │
                   │                                                ▼ (Token Chunker)
                   │                               ┌─────────────────────────────────────┐
                   │                               │        BPE Token Splitter           │
                   │                               │  - 400 tokens / 40% overlap         │
                   │                               │  - Hybrid NLP (langdetect / Gemini) │
                   │                               │  - DocumentLedger (SHA-256 hashes)  │
                   │                               └────────────────┬────────────────────┘
                   │                                                │
                   │                                                ▼ (PyArrow Schema)
                   │                               ┌─────────────────────────────────────┐
                   │                               │    Columnar Apache Parquet Output   │
                   │                               │   dataset_v1/part-00001.parquet     │
                   │                               └────────────────┬────────────────────┘
                   │                                                │
                   ▼                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  Cloudflare R2 Remote Storage (garmin-personal-data)                   │
│   - raw/snapshots/               - knowledge_base/firstbeat/                           │
│   - processed/garmin_personal.db - knowledge_base/processed_chunks/                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
                   │                                                │
                   ▼                                                ▼ (Fase 2 Próxima)
┌──────────────────────────────────────┐       ┌─────────────────────────────────────────┐
│ Analytics & Classical ML Engine      │       │     Vector Database & Embeddings        │
│ - ARIMA / Prophet / Holt-Winters     │       │ - Embeddings densos                     │
│ - Detección de Anomalías (Isolation) │       │ - Búsqueda híbrida (Dense + BM25)       │
│ - Causalidad de Granger              │       │ - Inferencia personalizada para agentes │
└──────────────────────────────────────┘       └─────────────────────────────────────────┘
```

---

## 📚 Base de Conocimiento RAG & Motor de Inferencia

El sistema de recuperación aumentada por generación (RAG) organiza el conocimiento biomédico y técnico en **3 fuentes estructuradas e independientes**, asegurando una clara separación semántica entre biología humana, ingeniería de dispositivos y esquemas de datos:

| Fuente | Chunks | Alcance Temático & Evidencia Científica | Rol en el Sistema RAG |
| :--- | :---: | :--- | :--- |
| **`variables_fisiologia_humana`** | **423** | **Fisiología y Biometría Humana**: Dinámica del Sistema Nervioso Autónomo (SNA), equilibrio simpático/parasimpático, variabilidad de frecuencia cardíaca (VFC / HRV: métricas rMSSD, SDNN, LF/HF), arquitectura y fases del sueño (NREM, REM, sueño profundo de ondas lentas), consumo máximo de oxígeno (VO2 Max), cinéticas de lactato, exceso de consumo de oxígeno post-ejercicio (EPOC) y mecanismos biológicos de recuperación y sobreentrenamiento. | **Fuente Activa en `/ask`** (Base científica para responder preguntas fisiológicas del usuario). |
| **`dispositivos_garmin_sensores`** | **268** | **Dispositivos, Sensores y Algoritmos Garmin**: Especificaciones de hardware y sensórica óptica PPG (Garmin Elevate v4 y v5 de múltiples canales LED), pulsioximetría PulseOx (SpO2), acelerometría triaxial, altimetría barométrica, GPS multibanda y algoritmos Firstbeat Analytics licenciados por Garmin (cálculo de Body Battery, Sleep Score, Training Readiness, Training Status, Training Effect aeróbico/anaeróbico, Stamina en tiempo real y HRV Status nocturno). | **Fuente Activa en `/ask`** (Base técnica para explicar cómo los sensores y algoritmos calculan las métricas). |
| **`descripciones_metricas_garmin`** | **11** | **Glosario y Esquemas de Telemetría Garmin Connect**: Catálogo técnico con definiciones formales, tipos de datos, unidades de medida y semántica exacta de las variables extraídas de las APIs y esquemas JSON (`sleep.json`, `stress.json`, `daily_summary.json`, etc.). | **Fuente de Metadatos & Lookup** (Reservada para validación y enriquecimiento de esquemas; **excluida deliberadamente** de `/ask` para priorizar la profundidad fisiológica y de sensores). |

### 🎯 Restricción de Alcance en el RAG (`/ask`)
El endpoint generativo `/ask` aplica un filtro estricto a nivel de ChromaDB (`where={"source": {"$in": ["variables_fisiologia_humana", "dispositivos_garmin_sensores"]}}`). Esto garantiza que la generación de respuestas por parte del LLM esté fundamentada **únicamente en evidencia científica de fisiología y especificaciones técnicas de sensores**, evitando respuestas triviales o superficiales basadas en simples glosarios.

### ⚡ Motor de Reranking Híbrido (Reciprocal Rank Fusion)
Para seleccionar los fragmentos más relevantes y pedagógicos para el LLM:
1. **Recuperación Expandida**: Se recupera un grupo inicial de candidatos (`top_k * 3`, mínimo 10) desde ChromaDB mediante búsqueda vectorial densa (cosine distance).
2. **Scoring Léxico BM25 & Cobertura**: Se evalúa la densidad de palabras clave, la presencia de acrónimos técnicos (`rMSSD`, `SpO2`, `VO2Max`, `Elevate`) y la coincidencia con las etiquetas (`tags`) de los metadatos.
3. **Fusión Reciprocal Rank (RRF $k=60$)**: Se combinan los rangos denso y léxico con bonificación por cobertura:
   $$\text{Score}(d) = \frac{0.55}{60 + \text{Rank}_{\text{denso}}(d)} + \frac{0.45}{60 + \text{Rank}_{\text{léxico}}(d)} + \text{Bonus}_{\text{cobertura}}$$
4. **Cero Latencia y Cero Coste**: Ejecutado 100% en memoria en CPU (< 1 ms), sin consumir llamadas a APIs externas.

### 🌐 Política Estricta de Idiomas
- **Español (`es`)**: Idioma predeterminado del asistente. Las consultas en español siempre se responden en español con rigor técnico y pedagógico.
- **Inglés (`en`)**: Si el usuario formula su consulta en inglés, el asistente detecta el idioma e instruye al LLM a responder íntegramente en inglés.
- **Otros Idiomas NO Permitidos**: Consultas en francés, alemán, italiano, portugués u otros idiomas son interceptadas inmediatamente por el guardrail y rechazadas con estado `unsupported_language` **sin realizar llamadas al LLM**, protegiendo el presupuesto de API.

### Pipeline de Chunking y Metadatos
- **Segmentación**: `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=400` y `chunk_overlap=160` tokens.
- **Esquema Parquet PyArrow**: Cada fragmento almacena `chunk_id`, `source_id`, `document_id`, `chunk_index`, `content`, `token_count`, `char_count`, `language`, `source_category`, `tags` (JSON serializado) y `created_at`.
- **Control de Estado (Ledger)**: `DocumentLedger` registra en `_ledger.json` la firma SHA-256 de cada documento y los IDs de chunks generados. Si un PDF se modifica o elimina, los chunks huérfanos se purgan automáticamente de los Parquets y de Cloudflare R2.

---

## 🗄️ Base de Datos Histórica (SQLite)

Motor relacional en **`data/garmin_personal.db`** ([`src/common/database.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/common/database.py)):

| Tabla | Clave Primaria | Métricas Principales Almacenadas |
| :--- | :--- | :--- |
| **`daily_summaries`** | `calendar_date` | Pasos, distancia, FC reposo, calorías activas, estrés promedio, minutos vigorosos/moderados. |
| **`sleep_records`** | `calendar_date` | Sleep score, fases (profundo, ligero, REM, vigilia), SpO2 promedio, frecuencia respiratoria. |
| **`hrv_records`** | `calendar_date` | rMSSD nocturno, media de 7 días, estado (`BALANCED`, `UNBALANCED`), línea base personal. |
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

## 🛠️ Herramientas Text-to-SQL (Function Calling Determinista)

Para consultar métricas biométricas personales con máxima precisión sin depender de que un LLM genere sentencias SQL arbitrarias y propensas a errores de sintaxis o alucinación de columnas, el sistema implementa una arquitectura de **Function Calling Determinista** ([`src/tools/garmin_sql_tools.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/tools/garmin_sql_tools.py) y [`src/agents/tools.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/agents/tools.py)):

```text
  [ Usuario / Agente ] 
           │ (Pregunta: "¿Cómo anduvo mi sueño y VFC los últimos 3 días?")
           ▼
  [ LLM Router / Classifier ] ──► Extrae parámetros: {"days": 3}
           │
           ▼
  [ Function Calling Tool ] ────► get_garmin_actuals(days=3)
           │
           ▼ (Consulta SQL parametrizada y optimizada en <1ms)
  [ garmin_history.db (SQLite) ]
           │
           ▼
  [ JSON Estructurado ] ────────► Inyectado en Contexto al Coach Mexicano / RAG
```

### 1. Grupo 1: Datos Reales (`get_garmin_actuals`)
Recupera la telemetría real consolidada de la tabla `consolidated_daily_actuals` para los últimos `N` días cerrados a día vencido:
- **Métricas Equivalentes a Pronósticos**: Frecuencia cardíaca en reposo (`resting_heart_rate`), variabilidad (`hrv_rmssd`, `hrv_weekly_avg`, `hrv_status`), estrés diario (`daily_avg_stress`, `daily_max_stress`), puntuación de sueño (`sleep_score`), horas de sueño (`total_sleep_hours`), pasos (`total_steps`) y desglose calórico (`active_kilocalories`, `resting_kilocalories`, `total_kilocalories`).
- **4 Métricas Clínicas Extras (Sin Pronóstico)**:
  1. *Arquitectura de Fases de Sueño*: Horas y segundos de sueño profundo (`deep_sleep_hours`), REM (`rem_sleep_hours`), ligero y tiempo despierto, junto al estrés vegetativo durante el sueño (`avg_sleep_stress`).
  2. *Oximetría y Respiración Nocturna*: Saturación de oxígeno media (`avg_spo2`), mínima (`lowest_spo2`) y frecuencia respiratoria (`avg_respiration` en rpm).
  3. *Distribución Temporal de Estrés*: Horas de reposo restaurativo parasimpático (`rest_stress_hours`), horas en estrés alto (`high_stress_hours`) y duraciones intermedias.
  4. *Carga Cardiovascular Deportiva*: Número de actividades del día (`activity_count`), duración total en minutos (`total_activity_minutes`), distancia en km (`total_activity_distance_km`), calorías quemadas y FC media/máx del entrenamiento.
- **Regla Estricta**: Excluye permanentemente cualquier campo de edad biológica (`fitness_age`, `chronological_age`, `fitness_age_gap`).

```python
from src.tools.garmin_sql_tools import get_garmin_actuals

# Consultar los últimos 3 días cerrados
res = get_garmin_actuals(days=3)
print(res["period"])   # {'start_date': '2026-09-18', 'end_date': '2026-09-20', 'total_days_retrieved': 3}
print(res["summary"])  # Promedios de RHR, HRV, estrés, sueño, pasos y actividades del periodo
```

### 2. Grupo 2: Pronósticos Biométricos (`get_garmin_forecasts`)
Consulta las inferencias estadísticas de Machine Learning almacenadas en `consolidated_biometric_forecasts`:
- **Parámetros**: `metric` (opcional, para filtrar una métrica específica como `resting_heart_rate` o consultar todas) y `horizon_days` (default: 14 días).
- **Cálculo de Días Hacia Adelante**: Determina automáticamente `days_ahead` (ej. $+1, +2, +3 \dots +14$ días a futuro) respecto a la última fecha real cerrada.
- **Frases Resumen en Lenguaje Natural**: Genera declaraciones formateadas para inyectar directo al prompt:
  > *"El pronóstico para dentro de 3 día(s) (2026-09-23) de resting_heart_rate es de 58.03 ppm (IC 95%: [55.56 a 60.51] ppm), proyectado por Ensemble_Prophet_HoltWinters."*
- **Regla Estricta**: Bloquea y excluye consultas sobre edad biológica o fitness age.

```python
from src.tools.garmin_sql_tools import get_garmin_forecasts

# Consultar el pronóstico de pulso en reposo para los próximos 7 días
fc = get_garmin_forecasts(metric="resting_heart_rate", horizon_days=7)
for statement in fc["summary_statements"][:3]:
    print(statement)
```

### 3. Integración con Agentes LangGraph
Ambas herramientas están decoradas con `@tool` de `langchain_core` en [`src/agents/tools.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/agents/tools.py) y listas para ser enlazadas al grafo de estados:
```python
from src.agents.tools import get_garmin_actuals_tool, get_garmin_forecasts_tool

tools = [get_garmin_actuals_tool, get_garmin_forecasts_tool]
```

---

## 🤖 Pipeline de Datos Sintéticos & Dataset SFT (Hugging Face)

Para dotar a modelos de lenguaje pequeños (e.g., *Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, SmolLM2*) del tono, modismos, energía y cadencia de un **Coach de Alto Rendimiento Mexicano** fundamentado rigurosamente en la ciencia biomédica de Firstbeat Analytics y en la telemetría de Garmin Connect, se diseñó un pipeline de generación y curaduría de datos sintéticos:

- **Dataset Oficial en Hugging Face**: [🤗 GerardoMayel/garmin-mexican-fitness-coach-sft](https://huggingface.co/datasets/GerardoMayel/garmin-mexican-fitness-coach-sft)
- **Formato**: ChatML nativo (`messages`: `system`, `user`, `assistant`).
- **Volumen Total**: 400 ejemplos de alta fidelidad curados (`data/synthetic/synthetic_dataset_400.jsonl`).
- **Distribución Bilingüe**:
  - **Español (ES - 196 ejemplos):** Vocabulario de entrenamiento mexicano (*carnal, mi rey, machín, al tiro, chamba, paliza, a reventar la barra*), calibrado entre 240 y 360 tokens (150-220 palabras).
  - **Inglés (EN - 204 ejemplos):** Energía de coach hispanoamericano bilingüe (*my friend, engine, get after it, dialed in, beast mode*), calibrado entre 190 y 280 tokens (150-220 palabras).
- **Equilibrio de Esfuerzo Fisiológico (3 Tiers de Intensidad):**
  1. **Alta Intensidad & Rendimiento Pico (`high_intensity`, 134 ejemplos):** Series de VO2max en pista (6x800m), récords personales (PRs) de sentadilla pesada, tempo en umbral de lactato, trail running con desnivel positivo (+1,000m) y circuitos funcionales de alta demanda metabólica.
  2. **Base Aeróbica & Consistencia (`moderate_base`, 128 ejemplos):** Rodajes suaves en Zona 2, hipertrofia submáxima controlada, tempo aeróbico y acumulación de volumen mitocondrial.
  3. **Descarga, Fatiga & Alertas Médicas (`recovery_fatigue`, 138 ejemplos):** Desvelo agudo, marcadores de sobreentrenamiento (HRV deprimida, RHR elevado), estrés alostático elevado y detección temprana de procesos infecciosos.

### 🏛️ Arquitectura de Respuesta en 3 Secciones Obligatorias
Cada turno del asistente sigue una estructura didáctica y clínica inmutable:
1. `### 1. El Diagnóstico Rápido` / `### 1. Quick Diagnosis`: Evaluación empática y directa del estado del atleta a partir de su telemetría.
2. `### 2. La Explicación Fisiológica (Lo que dice la ciencia)` / `### 2. Physiological Breakdown (What Science Says)`: Argumentación con base en literatura médica de Firstbeat (tono vagal rMSSD, aclaramiento de lactato, síntesis miofibrilar en ondas lentas, EPOC).
3. `### 3. La Chamba de Hoy (Plan de Acción)` / `### 3. Today's Work (Action Plan)`: Prescripción operativa de 4 a 5 puntos concretos (ritmos, descansos, técnica, nutrición peri-entrenamiento y sueño).

### 🛠️ Componentes y Scripts de Entrenamiento (`src/training/`)
- **`src/training/generate_synthetic_dataset.py`**: Generador por bloques con selección dinámica de modelos Gemini (`gemini-flash-lite-latest`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`), sondeo y registro de latencias en `data/synthetic/models_registry.json`, auto-reparación de JSON truncado, checkpointing incremental y rate-limiting configurable.
- **`src/training/upload_to_huggingface.py`**: Publicador automatizado que sincroniza el dataset `train.jsonl` hacia Hugging Face Hub y genera el Dataset Card profesional con metadatos YAML y plantilla de entrenamiento con TRL (`SFTTrainer`).
- **`src/training/preview_few_shots.py`**: Inspector y validador de sintaxis, reglas de longitud y modismos de las semillas de few-shots (`data/synthetic/seed_few_shots.jsonl`).

```python
# Carga directa del dataset desde Hugging Face para Fine-Tuning:
from datasets import load_dataset

dataset = load_dataset("GerardoMayel/garmin-mexican-fitness-coach-sft", split="train")
print(f"Total ejemplos: {len(dataset)}")  # 400 ejemplos
print(dataset[0]["messages"])
```

---

## ☁️ Almacenamiento en Cloudflare R2

El cliente `src/common/r2_storage.py` gestiona la sincronización remota contra Cloudflare R2:
- Sincronización bidireccional de datos crudos (`--sync-raw`, `--restore-raw`).
- Resguardo y restauración de la base de datos SQLite procesada (`--sync-processed`, `--restore-processed`).
- Sincronización de los datasets particionados Parquet de la base de conocimiento (`--sync-chunks`, `--restore-chunks`).

---

## 📁 Estructura del Proyecto

garmin-personal-insight-agent/
├── .github/
│   └── workflows/
│       ├── cd.yml
│       ├── ci.yml
│       ├── data_sync_cron.yml
│       └── sync_science_papers.yml
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.ui
│   ├── hf_chroma_space/
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
│   ├── artifacts/
│   │   ├── models/
│   │   │   ├── active_kilocalories_ensemble.joblib
│   │   │   ├── daily_avg_stress_ensemble.joblib
│   │   │   ├── fitness_age_ensemble.joblib
│   │   │   ├── fitness_age_gap_ensemble.joblib
│   │   │   ├── gym_avg_hr_ensemble.joblib
│   │   │   ├── hrv_rmssd_ensemble.joblib
│   │   │   ├── models_registry.json
│   │   │   ├── resting_heart_rate_ensemble.joblib
│   │   │   ├── resting_kilocalories_ensemble.joblib
│   │   │   ├── running_avg_hr_ensemble.joblib
│   │   │   ├── sleep_score_ensemble.joblib
│   │   │   ├── total_kilocalories_ensemble.joblib
│   │   │   ├── total_sleep_hours_ensemble.joblib
│   │   │   ├── total_steps_ensemble.joblib
│   │   │   └── walking_avg_hr_ensemble.joblib
│   │   ├── imputation_metadata.json
│   │   └── imputation_pipeline.joblib
│   ├── dvc/
│   │   ├── garmin_clean_features.csv
│   │   └── garmin_clean_features.parquet
│   ├── knowledge_base/
│   ├── multimodal/
│   ├── processed/
│   ├── raw/
│   ├── synthetic/
│   │   ├── models_registry.json
│   │   ├── seed_few_shots.jsonl
│   │   └── synthetic_dataset_400.jsonl
│   ├── training/
│   └── garmin_personal.db
│
├── notebooks/
│   ├── 01_garmin_data_exploration.ipynb
│   ├── 02.1_forecast_heart_rates.ipynb
│   ├── 02.2_forecast_sleep.ipynb
│   ├── 02.3_forecast_calories.ipynb
│   ├── 02.4_forecast_steps_activity.ipynb
│   ├── 02.5_forecast_autonomic_stress_hrv.ipynb
│   ├── 02_time_series_causality_lab.ipynb
│   ├── 03_correlation_analysis.ipynb
│   └── 04_causality_analysis.ipynb
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
│   │   ├── dvc_manager.py
│   │   ├── predictions_manager.py
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
│   │   ├── r2_storage.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── fit_decoder.py
│   │   ├── garmin_client.py
│   │   ├── garmin_sync.py
│   │   ├── sample_sync.py
│   │   ├── sync_garmin_device_papers.py
│   │   ├── sync_garmin_metric_descriptions.py
│   │   ├── sync_human_physiology_papers.py
│   │   └── sync_pipeline.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── language_detector.py
│   │   ├── ledger.py
│   │   ├── loader_and_chunker.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   ├── schemas.py
│   │   └── vector_store.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── garmin_sql_tools.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py
│   │   ├── evaluate_model.py
│   │   ├── generate_synthetic_dataset.py
│   │   ├── preview_few_shots.py
│   │   ├── README.md
│   │   ├── train_lora.py
│   │   └── upload_to_huggingface.py
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
│       ├── test_anomaly_detection.py
│       ├── test_causality_engine.py
│       ├── test_database.py
│       ├── test_dvc_manager.py
│       ├── test_fine_tuning.py
│       ├── test_garmin_client.py
│       ├── test_logger.py
│       ├── test_predictions_manager.py
│       ├── test_r2_storage.py
│       ├── test_rag_chunking.py
│       ├── test_sample_sync.py
│       ├── test_sql_tools.py
│       ├── test_sync_pipeline.py
│       ├── test_synthetic_data_seeds.py
│       └── test_time_series_models.py
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

### 1. Clonar e Instalar Dependencias

Utilizando [uv](https://github.com/astral-sh/uv) (recomendado) o `pip`:

```bash
# Con uv:
uv sync --extra dev

# O alternativamente con pip:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Copia la plantilla y configura únicamente las credenciales activas del proyecto:

```bash
cp .env.example .env
```

Consulta y completa las variables requeridas en `.env` (guíate con la plantilla [`.env.example`](file:///.env.example)):
- **Garmin Connect**: `GARMIN_EMAIL`, `GARMIN_PASSWORD`, `GARMIN_TOKEN_STORE`
- **Base de datos SQLite**: `GARMIN_DB_PATH` (opcional, default: `data/garmin_personal.db`)
- **Logging**: `LOG_LEVEL` (default: `INFO`)
- **RAG NLP**: `GEMINI_API_KEY` (opcional, fallback para detección de idioma)
- **Hugging Face**: `HUGGINGFACE_TOKEN` / `HF_TOKEN` (opcional, para modelos, embeddings y rerankers)
- **Cloudflare R2**: `R2_ACCOUNT_ID`, `R2_BUCKET_NAME`, `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`


### 3. Verificar Conexión con Garmin (Smoke Test)
Descarga en 3 segundos los datos de ayer y tu última actividad para validar credenciales y tokens de sesión:
```bash
uv run python -m src.ingestion.sample_sync
```

### 4. Sincronización a Día Vencido y Reconciliación Autorreparable
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

### 5. Almacenamiento en la Nube con Cloudflare R2
Persistencia y recuperación rápida de particiones raw y base de datos histórica mediante `src.common.r2_storage`:
```bash
# Respaldar SQLite y particiones raw hacia Cloudflare R2:
uv run python -m src.common.r2_storage --backup-db --sync-raw

# Restaurar SQLite y particiones raw completas desde Cloudflare R2:
uv run python -m src.common.r2_storage --restore-db --restore-raw
```

### 6. Ingestión de Literatura y Chunking RAG
Descarga los 30 White Papers científicos de Firstbeat, genera el glosario de métricas y procesa los chunks Parquet:
```bash
# Descargar White Papers de sensores y algoritmos (15 documentos)
uv run python -m src.ingestion.sync_garmin_device_papers

# Descargar White Papers de fisiología humana y biología (15 documentos)
uv run python -m src.ingestion.sync_human_physiology_papers

# Generar glosario estructurado de métricas Garmin Connect
uv run python -m src.ingestion.sync_garmin_metric_descriptions

# Ejecutar el chunker BPE (400 tokens / 40% overlap) y sincronizar a Cloudflare R2
uv run python -m src.rag.loader_and_chunker --sync-r2
```

### 7. Indexación Vectorial en ChromaDB (Google Gemini 768-dim)
Genera embeddings densos para los 702 fragmentos e indexa en ChromaDB (local o Space remoto en Hugging Face):
```bash
# Indexación en base local embebida (data/chroma_db)
uv run python -m src.rag.index_to_chroma

# O indexación hacia el Space remoto de Hugging Face
uv run python -m src.rag.index_to_chroma --remote
```

### 8. Despliegue del Backend ChromaDB a Hugging Face Space (Docker)
```bash
# Despliegue automatizado del contenedor FastAPI a tu Space
uv run python deploy/hf_chroma_space/deploy_space.py --space tu_usuario_hf/garmin-chroma-backend
```

### 9. Ejecución de Tests y Verificación de Código
```bash
# Ejecutar suite de tests unitarios
uv run pytest

# Verificación de linter y formateo
uv run ruff check .
uv run ruff format --check .

# Verificación de tipos estáticos
uv run mypy src tests
```

### 10. Validación y Previsualización de Semillas Few-Shot
Valida el esquema ChatML, las tres secciones obligatorias y las restricciones de longitud (200-350 tokens) de las semillas bilingües:
```bash
uv run python -m src.training.preview_few_shots
```

### 11. Generación de Dataset Sintético Multi-Modelo
Ejecuta la generación por bloques balanceados (alta intensidad, base aeróbica, recuperación) rotando dinámicamente entre modelos Gemini:
```bash
# Generación por bloques de 25 ejemplos hasta alcanzar la meta de 400:
uv run python -m src.training.generate_synthetic_dataset --target 400 --block-size 25

# O especificar un modelo específico de la API de Gemini:
uv run python -m src.training.generate_synthetic_dataset --model gemini-flash-lite-latest --block-size 20
```

### 12. Publicación del Dataset a Hugging Face Hub
Sube automáticamente el archivo `train.jsonl` junto a la ficha técnica (Dataset Card) a tu cuenta de Hugging Face:
```bash
# Simulación previa (Dry-Run):
uv run python -m src.training.upload_to_huggingface --dry-run

# Publicación oficial en Hugging Face Datasets:
uv run python -m src.training.upload_to_huggingface
```

---

## 🗺️ Hoja de Ruta (Siguientes Fases)

- [x] **Fase 0**: Ingestión de telemetría Garmin y persistencia relacional en SQLite.
- [x] **Fase 1**: Ingestión de literatura Firstbeat (3 fuentes), glosario de métricas, chunker BPE de 400 tokens, esquemas Parquet, reconciliación con DocumentLedger y CI/CD mensual en GitHub Actions.
- [x] **Fase 2**: Generación de Embeddings Densos con Google Gemini (768-dim), Backend Vectorial ChromaDB en Hugging Face Spaces (Docker), pipeline de indexación batch y Keep-Alive automatizado.
- [x] **Fase 3**: Recuperador Híbrido (Dense Semantic + BM25 Lexical con Reranker RRF), Guardrails Multicapa NLP, Filtro Estricto de Idioma (Español/Inglés) y Control de Presupuesto LLM.
- [x] **Fase 4**: Pipeline de Datos Sintéticos Multi-Modelo y Dataset SFT Bilingüe publicado en Hugging Face ([GerardoMayel/garmin-mexican-fitness-coach-sft](https://huggingface.co/datasets/GerardoMayel/garmin-mexican-fitness-coach-sft)).
- [ ] **Fase 5**: Fine-Tuning LoRA / QLoRA de modelos SLM (Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B) y evaluación cuantitativa de perplejidad y retención de formato.
- [ ] **Fase 6**: Agente de Razonamiento Fisiológico con LangGraph y Dashboard interactivo en Streamlit.
