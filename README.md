# Garmin Personal Insight Agent 🏃‍♂️💤🧠

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Storage: Cloudflare R2](https://img.shields.io/badge/Storage-Cloudflare_R2-F38020.svg)](https://www.cloudflare.com/developer-platform/r2/)
[![Database: SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![CI](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/ci.yml)
[![Science Sync](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/sync_science_papers.yml/badge.svg)](https://github.com/GerardoMayel/garmin-personalized-agent/actions/workflows/sync_science_papers.yml)

Sistema analítico end-to-end y arquitectura de agentes personalizados para la ingestión, persistencia relacional, modelado biométrico longitudinal y procesamiento de base de conocimiento científica (RAG) a partir de telemetría de Garmin Connect y literatura de Firstbeat Analytics.

---

## 📌 Estado Actual del Proyecto (Fase 1: Ingestión & Fase 2: RAG Backend en Vivo)

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
| **`activities`** | `activity_id` | Nombre, tipo, distancia, duración, desnivel, velocidad, FC media/máx, ruta local al `.fit.zip`. |

---

## ☁️ Almacenamiento en Cloudflare R2

El cliente `src/common/r2_storage.py` gestiona la sincronización remota contra Cloudflare R2:
- Sincronización bidireccional de datos crudos (`--sync-raw`, `--restore-raw`).
- Resguardo y restauración de la base de datos SQLite procesada (`--sync-processed`, `--restore-processed`).
- Sincronización de los datasets particionados Parquet de la base de conocimiento (`--sync-chunks`, `--restore-chunks`).

---

## 📁 Estructura del Proyecto

```text
garmin-personalized-agent/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── data_sync_cron.yml
│       ├── ping_hf_space.yml
│       └── sync_science_papers.yml
├── configs/
├── data/
│   ├── garmin_personal.db
│   ├── knowledge_base/
│   │   ├── descripciones_metricas_garmin/
│   │   ├── firstbeat/
│   │   │   ├── dispositivos_garmin_sensores/
│   │   │   └── variables_fisiologia_humana/
│   │   └── processed_chunks/
│   └── raw/
├── deploy/
│   └── hf_chroma_space/
│       ├── Dockerfile
│       ├── README.md
│       ├── app.py
│       ├── deploy_space.py
│       ├── guardrails.py
│       └── requirements.txt
├── src/
│   ├── analytics/
│   │   ├── anomaly_detection.py
│   │   ├── causality_engine.py
│   │   ├── dvc_manager.py
│   │   ├── predictions_manager.py
│   │   └── time_series_models.py
│   ├── common/
│   │   ├── database.py
│   │   ├── logger.py
│   │   └── r2_storage.py
│   ├── ingestion/
│   │   ├── garmin_client.py
│   │   ├── garmin_sync.py
│   │   ├── sample_sync.py
│   │   ├── sync_garmin_device_papers.py
│   │   ├── sync_human_physiology_papers.py
│   │   └── sync_garmin_metric_descriptions.py
│   └── rag/
│       ├── README.md
│       ├── embeddings.py
│       ├── guardrails.py
│       ├── index_to_chroma.py
│       ├── language_detector.py
│       ├── ledger.py
│       ├── loader_and_chunker.py
│       ├── reranker.py
│       ├── schemas.py
│       └── vector_store.py
├── tests/
│   └── unit/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Descripción de Componentes Principales

| Directorio / Módulo | Responsabilidad Principal |
| :--- | :--- |
| **`.github/workflows/`** | Automatizaciones CI/CD: verificación (`ci.yml`), sincronización diaria (`data_sync_cron.yml`), pipeline mensual RAG (`sync_science_papers.yml`) y keep-alive de HF Space (`ping_hf_space.yml`). |
| **`data/`** | Capa de persistencia: base SQLite relacional (`garmin_personal.db`), snapshots JSON (`raw/`) y base de conocimiento científica (`knowledge_base/`). |
| **`deploy/hf_chroma_space/`** | Backend FastAPI containerizado para Hugging Face Spaces (Docker SDK, Private) con colección unificada `biometric_knowledge_base`, guardrails multicapa y presupuesto LLM. |
| **`src/analytics/`** | Machine Learning y series de tiempo: descomposición temporal, modelos ARIMA/Prophet/Holt-Winters, detección de anomalías (Isolation Forest) y causalidad de Granger. |
| **`src/common/`** | Infraestructura base: gestor SQLite (`database.py`), cliente Cloudflare R2 (`r2_storage.py`) y logging con Loguru (`logger.py`). |
| **`src/ingestion/`** | Clientes de ingestión: API de Garmin Connect y descargadores de literatura Firstbeat y glosario de métricas. |
| **`src/rag/`** | Pipeline de RAG: chunking BPE, embeddings Google Gemini (768 dims), cliente ChromaDB (Local/Remoto), guardrails NLP, filtro estricto de fuentes y motor de reranking híbrido RRF. |
| **`tests/unit/`** | Suite completa de 102 pruebas unitarias automatizadas con `pytest`. |



---

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

### 4. Sincronizar Telemetría Reciente (7 Días)
```bash
uv run python -m src.ingestion.garmin_sync
```

### 5. Ingestión de Literatura y Chunking RAG
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

### 6. Indexación Vectorial en ChromaDB (Google Gemini 768-dim)
Genera embeddings densos para los 702 fragmentos e indexa en ChromaDB (local o Space remoto en Hugging Face):
```bash
# Indexación en base local embebida (data/chroma_db)
uv run python -m src.rag.index_to_chroma

# O indexación hacia el Space remoto de Hugging Face
uv run python -m src.rag.index_to_chroma --remote
```

### 7. Despliegue del Backend ChromaDB a Hugging Face Space (Docker)
```bash
# Despliegue automatizado del contenedor FastAPI a tu Space
uv run python deploy/hf_chroma_space/deploy_space.py --space tu_usuario_hf/garmin-chroma-backend
```

### 8. Ejecución de Tests y Verificación de Código
```bash
# Ejecutar suite de 69 tests unitarios
uv run pytest

# Verificación de linter y formateo
uv run ruff check .
uv run ruff format --check .

# Verificación de tipos estáticos
uv run mypy src tests
```

---

## 🗺️ Hoja de Ruta (Siguientes Fases)

- [x] **Fase 0**: Ingestión de telemetría Garmin y persistencia relacional en SQLite.
- [x] **Fase 1**: Ingestión de literatura Firstbeat (3 fuentes), glosario de métricas, chunker BPE de 400 tokens, esquemas Parquet, reconciliación con DocumentLedger y CI/CD mensual en GitHub Actions.
- [x] **Fase 2**: Generación de Embeddings Densos con Google Gemini (768-dim), Backend Vectorial ChromaDB en Hugging Face Spaces (Docker), pipeline de indexación batch y Keep-Alive automatizado.
- [ ] **Fase 3**: Recuperador Híbrido (Dense Semantic + BM25 Lexical con Reranker cross-encoder).
- [ ] **Fase 4**: Agente de Razonamiento Fisiológico con LangGraph y Dashboard interactivo en Streamlit.

