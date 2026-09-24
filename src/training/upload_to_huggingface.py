"""Hugging Face Hub Dataset Uploader for Garmin Mexican Fitness Coach SFT Dataset.

Uploads:
1. The synthetic ChatML dataset (.jsonl) as `train.jsonl`.
2. Automatically generates and uploads a comprehensive Dataset Card (`README.md`)
   with YAML metadata, dataset description, schema specifications, and sample SFTTrainer code.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

DEFAULT_DATASET_PATH = Path("data/synthetic/synthetic_dataset_400.jsonl")
DEFAULT_REPO_NAME = "garmin-mexican-fitness-coach-sft"


def generate_dataset_card_content(repo_id: str, total_examples: int) -> str:
    """Generates a professional Hugging Face Dataset Card with YAML frontmatter."""
    return f"""---
language:
- es
- en
license: apache-2.0
task_categories:
- text-generation
task_ids:
- dialogue-modeling
tags:
- garmin
- firstbeat
- fitness
- coaching
- mexican-spanish
- bilingual
- sft
- lora
- biometrics
- synthetic-data
pretty_name: Garmin Mexican Fitness Coach SFT Dataset
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# 🇲🇽 Garmin Mexican Fitness Coach SFT Dataset

Dataset sintético bilingüe de alta fidelidad para el ajuste fino instruccional (**Supervised Fine-Tuning / LoRA**) de modelos de lenguaje pequeños (e.g., *Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, SmolLM2*), diseñado para dotarlos de la personalidad, modismos y tono enérgico de un **Coach de Alto Rendimiento Mexicano** con fundamento fisiológico estricto (**Firstbeat Technologies & Garmin Connect**).

---

## 📊 Resumen del Dataset

- **Tamaño total:** {total_examples} ejemplos estructurados en formato **ChatML** (`system`, `user`, `assistant`).
- **Idiomas & Estilo:**
  - **Español (ES):** 50% de ejemplos con jerga fitness y modismos mexicanos auténticos (*carnal, mi rey, machín, al tiro, chamba, a reventar la barra, quemar llanta*). Rango calibrado: 240 a 360 tokens.
  - **Inglés (EN):** 50% de ejemplos con tono de coach bilingüe/hispanoamericano (*my friend, engine, get after it, dialed in, beast mode*). Rango calibrado: 190 a 280 tokens.
- **Métricas Biométricas Cubiertas:**
  1. `hrv_rmssd`: Variabilidad de la frecuencia cardíaca nocturna, balance autonómico simpático/parasimpático.
  2. `sleep_score`: Puntuación y arquitectura del sueño (sueño profundo N3, REM, estrés nocturno, SpO2, respiración).
  3. `daily_avg_stress`: Estrés diario promedio, horas en reposo restaurativo y carga alostática.
  4. `resting_heart_rate`: Frecuencia cardíaca en reposo basal y volumen sistólico miocárdico.
  5. `total_steps`: Pasos diarios, gasto calórico activo, zonas de frecuencia cardíaca y deuda de recuperación (EPOC).
- **Equilibrio de Escenarios (3 Tiers de Intensidad):**
  - **Alto Rendimiento / Ejercicio Fuerte (High Intensity):** Récords personales (PRs), series de VO2max en pista, tempo en umbral de lactato, trail running con desnivel y circuitos metabólicos.
  - **Base Aeróbica / Mantenimiento (Moderate Base):** Rodajes suaves en Z2, hipertrofia submáxima controlada y consistencia metabólica.
  - **Descarga / Fatiga / Alertas (Recovery / Fatigue):** Desvelo severo, sobreentrenamiento, estrés laboral y marcadores tempranos de infección.

---

## 🏛️ Estructura de Tres Secciones Obligatorias

Cada respuesta del asistente sigue una arquitectura pedagógica y clínica inmutable:

1. `### 1. El Diagnóstico Rápido`: Evaluación empática, directa y con jerga mexicana del estado biológico actual del atleta.
2. `### 2. La Explicación Fisiológica (Lo que dice la ciencia)`: Fundamentación científica basada en literatura médica de Firstbeat (tono vagal, aclaramiento de lactato, hormona de crecimiento en sueño profundo, ATP-PCr).
3. `### 3. La Chamba de Hoy (Plan de Acción)`: Prescripción operativa de 4 a 5 puntos claros con intensidades, tiempos de pausa, técnica, nutrición y descanso.

---

## 💻 Ejemplo de Entrenamiento con TRL (`SFTTrainer`)

```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

dataset = load_dataset("{repo_id}", split="train")

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def format_prompts(batch):
    formatted = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append(text)
    return {{"text": formatted}}

formatted_dataset = dataset.map(format_prompts, batched=True)

training_args = TrainingArguments(
    output_dir="./lora_mexican_coach",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    fp16=True,
)

trainer = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained(model_id),
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

trainer.train()
```

---

## 📜 Licencia y Cita
- **Licencia:** Apache 2.0
- **Autor:** Gerardo Mayel Fernández Alamilla
- **Repositorio:** [garmin-personalized-agent](https://github.com/GerardoMayel/garmin-personalized-agent)
"""


def upload_dataset_to_hub(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    repo_id: str | None = None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Uploads the synthetic dataset and dataset card to Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN no encontrado en las variables de entorno ni en .env.")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Archivo de dataset no encontrado: {dataset_path}")

    # Count examples
    total_examples = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_examples += 1

    print(f"📦 Dataset cargado: {dataset_path} ({total_examples} ejemplos)")

    # Resolve repo id
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name")

    if not repo_id:
        repo_id = f"{username}/{DEFAULT_REPO_NAME}"

    print(f"👤 Autenticado en Hugging Face como: {username}")
    print(f"🎯 Repositorio destino: {repo_id} (Privado: {private})")

    # Generate Dataset Card
    card_content = generate_dataset_card_content(repo_id, total_examples)

    if dry_run:
        print("🔍 [DRY-RUN] Dataset Card generado correctamente:")
        print("-" * 60)
        print(card_content[:500] + "\n...")
        print("-" * 60)
        print("✅ Simulación completada con éxito. No se subieron archivos al Hub.")
        return f"https://huggingface.co/datasets/{repo_id}"

    # 1. Create or ensure repository exists
    print(f"🚀 Creando/verificando repositorio en Hugging Face: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # 2. Upload train.jsonl
    print("📤 Subiendo train.jsonl al Hub...")
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"feat: add {total_examples} synthetic ChatML training examples",
    )

    # 3. Upload README.md (Dataset Card)
    print("📝 Subiendo README.md (Dataset Card) al Hub...")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as tmp_file:
        tmp_file.write(card_content)
        tmp_card_path = tmp_file.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="docs: add comprehensive dataset card with YAML metadata and TRL guide",
        )
    finally:
        if os.path.exists(tmp_card_path):
            os.remove(tmp_card_path)

    hub_url = f"https://huggingface.co/datasets/{repo_id}"
    print("\n" + "=" * 80)
    print(f"🎉 ¡Dataset publicado con éxito en Hugging Face!")
    print(f"🔗 URL: {hub_url}")
    print("=" * 80 + "\n")
    return hub_url


def main() -> None:
    """CLI Parser."""
    parser = argparse.ArgumentParser(description="Subidor de datasets sintéticos a Hugging Face Hub.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=f"Ruta al archivo .jsonl (def: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Nombre completo del repo (e.g., GerardoMayel/garmin-mexican-fitness-coach-sft)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear el repositorio como privado en Hugging Face",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular sin subir archivos",
    )
    args = parser.parse_args()

    upload_dataset_to_hub(
        dataset_path=Path(args.dataset_path),
        repo_id=args.repo_id,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
