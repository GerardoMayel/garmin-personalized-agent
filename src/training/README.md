# 🇲🇽 Módulo de Entrenamiento & Datos Sintéticos (SFT / LoRA)

Este módulo gestiona la generación, validación, curaduría y publicación de datasets sintéticos de alta fidelidad para el ajuste fino instruccional (**Supervised Fine-Tuning / LoRA**) de Modelos de Lenguaje Pequeños (*Small Language Models* como *Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, SmolLM2*).

El objetivo es enseñar al modelo a comportarse como un **Coach de Alto Rendimiento Mexicano** con fundamento fisiológico estricto derivado de **Firstbeat Technologies** y datos de telemetría de **Garmin Connect**.

---

## 🔗 Dataset Oficial en Hugging Face Hub

El dataset generado se encuentra publicado y accesible en la comunidad de Hugging Face:
- **Repositorio Hub:** [🤗 GerardoMayel/garmin-mexican-fitness-coach-sft](https://huggingface.co/datasets/GerardoMayel/garmin-mexican-fitness-coach-sft)
- **Licencia:** Apache 2.0
- **Formato:** ChatML (`system`, `user`, `assistant`) en `train.jsonl`

### Carga Directa con Python (`datasets`)

```python
from datasets import load_dataset

# Cargar split de entrenamiento directamente desde Hugging Face Hub
dataset = load_dataset("GerardoMayel/garmin-mexican-fitness-coach-sft", split="train")

print(f"Total de ejemplos disponibles: {len(dataset)}")  # 400
ejemplo = dataset[0]
print(f"ID: {ejemplo['id']} | Categoría: {ejemplo['metric_category']}")
for msg in ejemplo["messages"]:
    print(f"\n[{msg['role'].upper()}]:\n{msg['content']}")
```

---

## 📊 Especificaciones del Dataset

- **Volumen Total:** 400 ejemplos balanceados.
- **Distribución Bilingüe (50% / 50%):**
  - **Español (ES - 196 ejemplos):** Modismos mexicanos auténticos de entrenamiento (*carnal, mi rey, machín, al tiro, chamba, a reventar la barra, quemar llanta, paliza*). Rango calibrado: **240 a 360 tokens** (150-220 palabras).
  - **Inglés (EN - 204 ejemplos):** Cadencia y energía de coach hispanoamericano bilingüe (*my friend, engine, get after it, dialed in, beast mode*). Rango calibrado: **190 a 280 tokens** (150-220 palabras).
- **Métricas Biométricas Cubiertas:**
  1. `hrv_rmssd`: Variabilidad cardíaca nocturna, balance autonómico simpático/parasimpático.
  2. `sleep_score`: Arquitectura del sueño (fases N3 profunda, REM, ligero), saturación SpO2 y estrés nocturno.
  3. `daily_avg_stress`: Estrés alostático diario, horas en reposo restaurativo y sobrecarga laboral.
  4. `resting_heart_rate`: Frecuencia cardíaca en reposo basal y volumen sistólico miocárdico.
  5. `total_steps`: Volumen de actividad, zonas de frecuencia cardíaca y deuda metabólica (EPOC).
- **Equilibrio de Esfuerzo (3 Tiers de Intensidad):**
  - **`high_intensity` (134 ejemplos):** Series de VO2max en pista, récords personales (PRs), ritmos tempo en umbral de lactato, trail running con gran desnivel y circuitos metabólicos.
  - **`moderate_base` (128 ejemplos):** Rodajes aeróbicos en Zona 2, hipertrofia submáxima controlada y consistencia de volumen.
  - **`recovery_fatigue` (138 ejemplos):** Privación de sueño, fatiga autonómica simpática, estrés alto sostenido y detección temprana de alertas infecciosas.

---

## 🏛️ Arquitectura de Respuesta en 3 Secciones Obligatorias

Cada respuesta del asistente sigue rigurosamente esta estructura didáctica:

1. `### 1. El Diagnóstico Rápido` / `### 1. Quick Diagnosis`: Evaluación empática, directa y con energía deportiva del estado actual a partir de la telemetría.
2. `### 2. La Explicación Fisiológica (Lo que dice la ciencia)` / `### 2. Physiological Breakdown (What Science Says)`: Explicación médica sustentada en literatura de Firstbeat (tono vagal, aclaramiento de lactato, síntesis de hormona de crecimiento, fosfatos ATP-PCr).
3. `### 3. La Chamba de Hoy (Plan de Acción)` / `### 3. Today's Work (Action Plan)`: Prescripción operativa de 4 a 5 puntos concretos (intensidad, pausas, técnica, hidratación/nutrición peri-entreno y descanso).

---

## 🛠️ Herramientas y Scripts del Módulo

### 1. `preview_few_shots.py`
Herramienta de inspección y validación de las semillas base (`data/synthetic/seed_few_shots.jsonl`). Valida la sintaxis ChatML, la presencia de las tres secciones obligatorias, los modismos de personalidad y los rangos de longitud de tokens.

```bash
uv run python -m src.training.preview_few_shots
```

### 2. `generate_synthetic_dataset.py`
Generador por bloques que utiliza rotación dinámica entre modelos Gemini disponibles en la API (`gemini-flash-lite-latest`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`). Incluye:
- Sondeo automático de modelos activos y latencias guardadas en `data/synthetic/models_registry.json`.
- Sistema de auto-reparación sintáctica para strings JSON truncados.
- Checkpointing incremental (continúa donde se quedó si el archivo ya tiene ejemplos).
- Control de pausas y rate-limiting configurable.

```bash
# Generar hasta 400 ejemplos en bloques de 25:
uv run python -m src.training.generate_synthetic_dataset --target 400 --block-size 25

# Usar un modelo específico:
uv run python -m src.training.generate_synthetic_dataset --model gemini-flash-lite-latest --block-size 20
```

### 3. `upload_to_huggingface.py`
Sube el dataset `train.jsonl` a Hugging Face Hub y genera automáticamente una ficha técnica profesional (Dataset Card en `README.md`) con metadatos YAML, estadísticas y código de entrenamiento con TRL (`SFTTrainer`).

```bash
# Simular subida (Dry-Run):
uv run python -m src.training.upload_to_huggingface --dry-run

# Subir al repositorio oficial en Hugging Face:
uv run python -m src.training.upload_to_huggingface
```

### 4. `dataset_builder.py`
Carga, valida y divide el dataset en splits deterministas de entrenamiento (90%) y validación (10%), formateándolo para su consumo directo por `TRL` o `Transformers`.

```bash
# Ver reporte de distribución y estadísticas:
uv run python -m src.training.dataset_builder

# Descargar y validar directamente desde Hugging Face Hub:
uv run python -m src.training.dataset_builder --hub
```

### 5. `train_lora.py`
Entrenador de ajuste fino instruccional con adaptadores LoRA / PEFT y `SFTTrainer` de `trl`. Detecta automáticamente el hardware (NVIDIA CUDA con cuantización 4-bit, Apple Silicon MPS en FP16, o CPU):

```bash
# Simular pipeline de preparación de datos y modelo (Dry-Run):
uv run python -m src.training.train_lora --dry-run

# Entrenar modelo SLM (ejemplo: Qwen 2.5 1.5B o Llama 3.2 1B):
uv run python -m src.training.train_lora --config configs/fine_tuning_lora.yaml
```

### 6. `evaluate_model.py`
Suite de evaluación cualitativa y cuantitativa que mide:
1. **Cumplimiento de Estructura:** Presencia de las tres secciones obligatorias.
2. **Marcadores de Jerga/Tono:** Presencia de modismos auténticos de coaching deportivo.
3. **Calibración de Longitud:** Respuestas estrictamente en el rango de 200 a 350 tokens.

```bash
# Evaluar respuestas del dataset sintético:
uv run python -m src.training.evaluate_model --file data/synthetic/synthetic_dataset_400.jsonl
```

