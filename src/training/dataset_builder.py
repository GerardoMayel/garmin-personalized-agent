#!/usr/bin/env python3
"""Dataset Builder and Formatter for Garmin Coach SFT Fine-Tuning.

Loads, validates, splits, and formats ChatML conversation data for
Supervised Fine-Tuning (SFT / LoRA) with Hugging Face TRL and Transformers.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger

DEFAULT_LOCAL_PATH = Path("data/synthetic/synthetic_dataset_400.jsonl")
DEFAULT_HF_REPO = "GerardoMayel/garmin-mexican-fitness-coach-sft"


def load_raw_dataset(
    path: Path | str | None = None,
    use_hub: bool = False,
    repo_id: str = DEFAULT_HF_REPO,
) -> list[dict[str, Any]]:
    """Loads raw ChatML dataset from local JSONL or Hugging Face Hub.

    Args:
        path: Optional local path to JSONL file. Defaults to DEFAULT_LOCAL_PATH.
        use_hub: If True, forces download from Hugging Face Hub.
        repo_id: Hugging Face dataset repository identifier.

    Returns:
        List of validated dictionary examples.
    """
    file_path = Path(path) if path else DEFAULT_LOCAL_PATH

    if not use_hub and file_path.exists():
        logger.info(f"Cargando dataset local desde: {file_path}")
        examples: list[dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    examples.append(data)
                except json.JSONDecodeError as err:
                    logger.warning(f"Error decodificando línea {line_idx} en {file_path}: {err}")
        logger.info(f"Cargados {len(examples)} ejemplos locales.")
        return examples

    logger.info(f"Cargando dataset desde Hugging Face Hub: {repo_id}")
    hf_ds = load_dataset(repo_id, split="train")
    examples = [dict(row) for row in hf_ds]
    logger.info(f"Cargados {len(examples)} ejemplos desde Hugging Face.")
    return examples


def validate_chatml_example(example: dict[str, Any]) -> bool:
    """Validates that an example adheres to the expected ChatML coaching schema.

    Checks:
        - Must contain 'messages' list with 3 turns: system, user, assistant.
        - System prompt, user query, and assistant response must be non-empty strings.
    """
    if "messages" not in example or not isinstance(example["messages"], list):
        return False
    messages = example["messages"]
    if len(messages) != 3:
        return False

    expected_roles = ["system", "user", "assistant"]
    for msg, expected_role in zip(messages, expected_roles):
        if not isinstance(msg, dict):
            return False
        if msg.get("role") != expected_role:
            return False
        if not msg.get("content") or not isinstance(msg["content"], str):
            return False

    return True


def split_dataset(
    examples: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically splits examples into train and validation sets.

    Args:
        examples: Full list of raw examples.
        val_ratio: Fraction of examples allocated to validation (e.g. 0.1 for 10%).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_examples, val_examples).
    """
    valid_examples = [ex for ex in examples if validate_chatml_example(ex)]
    if len(valid_examples) < len(examples):
        dropped = len(examples) - len(valid_examples)
        logger.warning(f"Se descartaron {dropped} ejemplos con formato ChatML inválido.")

    rng = random.Random(seed)
    shuffled = list(valid_examples)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_examples = shuffled[:val_size]
    train_examples = shuffled[val_size:]

    logger.info(
        f"Dataset dividido: {len(train_examples)} entrenamiento ({100 - val_ratio*100:.0f}%), "
        f"{len(val_examples)} validación ({val_ratio*100:.0f}%)."
    )
    return train_examples, val_examples


def compute_dataset_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates distribution statistics across languages, metrics, tiers, and token counts."""
    total = len(examples)
    if total == 0:
        return {"total": 0}

    languages: dict[str, int] = {}
    metrics: dict[str, int] = {}
    tiers: dict[str, int] = {}
    token_lengths: list[int] = []

    for ex in examples:
        lang = ex.get("language", "unknown")
        languages[lang] = languages.get(lang, 0) + 1

        metric = ex.get("metric_category", "unknown")
        metrics[metric] = metrics.get(metric, 0) + 1

        tier = ex.get("intensity_tier") or ex.get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1

        # Token length approximation from assistant content
        if "messages" in ex and len(ex["messages"]) >= 3:
            content = ex["messages"][2]["content"]
            approx_tokens = ex.get("approx_tokens", int(len(content.split()) * 1.33))
            token_lengths.append(approx_tokens)

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    min_tokens = min(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0

    return {
        "total_examples": total,
        "languages": languages,
        "tiers": tiers,
        "metrics": metrics,
        "token_stats": {
            "avg_tokens": round(avg_tokens, 1),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        },
    }


def build_hf_dataset_dict(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    tokenizer: Any | None = None,
) -> DatasetDict:
    """Builds a Hugging Face DatasetDict with train and validation splits.

    Optionally formats conversations into a single 'text' field if a tokenizer is provided.
    """
    def _to_record(ex: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": ex.get("id", ""),
            "metric_category": ex.get("metric_category", ""),
            "tier": ex.get("tier", ""),
            "language": ex.get("language", "es"),
            "messages": ex["messages"],
        }
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            record["text"] = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        return record

    train_data = [_to_record(ex) for ex in train_examples]
    val_data = [_to_record(ex) for ex in val_examples]

    ds_train = Dataset.from_list(train_data)
    ds_val = Dataset.from_list(val_data)

    return DatasetDict({"train": ds_train, "validation": ds_val})


def main() -> None:
    """CLI to inspect and validate dataset."""
    parser = argparse.ArgumentParser(description="Dataset Builder y Validador de SFT.")
    parser.add_argument("--path", type=str, default=str(DEFAULT_LOCAL_PATH), help="Ruta al archivo JSONL local.")
    parser.add_argument("--hub", action="store_true", help="Cargar desde Hugging Face Hub.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proporción de validación (def: 0.1).")
    args = parser.parse_args()

    examples = load_raw_dataset(path=args.path, use_hub=args.hub)
    stats = compute_dataset_metrics(examples)

    print("\n" + "=" * 70)
    print("📊 REPORTE DE ESTADÍSTICAS DEL DATASET SFT")
    print("=" * 70)
    print(f"Total de ejemplos: {stats['total_examples']}")
    print(f"Distribución por Idioma: {stats['languages']}")
    print(f"Distribución por Nivel de Esfuerzo (Tiers): {stats['tiers']}")
    print(f"Métricas Biométricas: {stats['metrics']}")
    print(f"Estadísticas de Tokens: {stats['token_stats']}")
    print("=" * 70 + "\n")

    train_ex, val_ex = split_dataset(examples, val_ratio=args.val_ratio)
    print(f"✅ División completada con éxito: {len(train_ex)} Train | {len(val_ex)} Val")


if __name__ == "__main__":
    main()
