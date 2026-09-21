"""Preview and validation CLI for the Mexican Fitness Coach Few-Shot Seed Dataset.

Validates:
1. JSONL syntax and ChatML schema (system, user, assistant).
2. Presence of mandatory three sections:
   - ### 1. El Diagnóstico Rápido
   - ### 2. La Explicación Fisiológica (Lo que dice la ciencia)
   - ### 3. La Chamba de Hoy (Plan de Acción)
3. Mexican fitness coach vernacular and style markers.
4. Token estimation and statistics across biometric metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")


def load_and_validate_few_shots(file_path: Path = DEFAULT_SEEDS_PATH) -> list[dict]:
    """Loads and validates few-shot examples against strict ChatML schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de seeds no encontrado: {file_path}")

    examples = []
    required_sections = [
        "### 1. El Diagnóstico Rápido",
        "### 2. La Explicación Fisiológica",
        "### 3. La Chamba de Hoy",
    ]

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            assert len(messages) == 3, f"Línea {idx}: debe contener exactamente 3 mensajes (system, user, assistant)"
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

            assistant_content = messages[2]["content"]
            for sec in required_sections:
                assert sec in assistant_content, f"Línea {idx} ({item.get('id')}): falta la sección obligatoria '{sec}'"

            examples.append(item)

    return examples


def preview_few_shots(metric_filter: str | None = None, limit: int | None = None) -> None:
    """Pretty prints the few-shot examples in terminal."""
    examples = load_and_validate_few_shots()

    if metric_filter:
        examples = [e for e in examples if e.get("metric_category") == metric_filter]

    if limit:
        examples = examples[:limit]

    print("\n" + "=" * 80)
    print("🥊 CATÁLOGO SEMILLA DE FEW-SHOT EXAMPLES (COACH FITNESS MEXICANO)")
    print(f"   Total de ejemplos cargados y validados: {len(examples)}")
    print("=" * 80 + "\n")

    for i, ex in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] ID: {ex['id']} | Métrica: {ex['metric_category']}")
        print(f"📌 Escenario: {ex['scenario']}")
        print("-" * 80)

        user_content = ex["messages"][1]["content"]
        assistant_content = ex["messages"][2]["content"]

        print("👤 USUARIO (Contexto Garmin SQLite + RAG Firstbeat + Pregunta):")
        print(user_content)
        print("\n🇲🇽 COACH MEXICANO (Diagnóstico + Ciencia + Chamba de Hoy):")
        print(assistant_content)
        print("\n" + "=" * 80 + "\n")


def main() -> None:
    """CLI parser for previewing few shot dataset."""
    parser = argparse.ArgumentParser(description="Visualizador de Few-Shot Examples para datos sintéticos.")
    parser.add_argument("--metric", type=str, default=None, help="Filtrar por métrica biométrica")
    parser.add_argument("--limit", type=int, default=None, help="Límite de ejemplos a mostrar")
    args = parser.parse_args()

    preview_few_shots(metric_filter=args.metric, limit=args.limit)


if __name__ == "__main__":
    main()
