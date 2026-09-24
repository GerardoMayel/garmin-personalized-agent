"""Unit tests for Fine-Tuning and SFT training module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.dataset_builder import (
    compute_dataset_metrics,
    load_raw_dataset,
    split_dataset,
    validate_chatml_example,
)
from src.training.evaluate_model import evaluate_batch, evaluate_single_response
from src.training.train_lora import build_lora_config, load_config, resolve_device_and_dtype

CONFIG_PATH = Path("configs/fine_tuning_lora.yaml")
DATASET_PATH = Path("data/synthetic/synthetic_dataset_400.jsonl")
SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")


def test_fine_tuning_config_structure():
    """Verify that fine_tuning_lora.yaml contains all required sections and parameters."""
    cfg = load_config(CONFIG_PATH)

    assert "model" in cfg
    assert "dataset" in cfg
    assert "lora" in cfg
    assert "training" in cfg

    # Model assertions
    assert "base_model_name" in cfg["model"]
    assert isinstance(cfg["model"]["base_model_name"], str)

    # LoRA assertions
    lora = cfg["lora"]
    assert lora["r"] > 0
    assert lora["lora_alpha"] > 0
    assert len(lora["target_modules"]) > 0

    # Training assertions
    train_cfg = cfg["training"]
    assert train_cfg["num_train_epochs"] >= 1
    assert train_cfg["learning_rate"] > 0
    assert "output_dir" in train_cfg


def test_lora_config_instantiation():
    """Verify that PEFT LoraConfig is constructed properly from configuration."""
    cfg = load_config(CONFIG_PATH)
    peft_cfg = build_lora_config(cfg["lora"])

    assert peft_cfg.r == cfg["lora"]["r"]
    assert peft_cfg.lora_alpha == cfg["lora"]["lora_alpha"]
    assert set(peft_cfg.target_modules) == set(cfg["lora"]["target_modules"])


def test_dataset_builder_loading_and_splitting():
    """Verify raw dataset loading, validation, and splitting."""
    examples = load_raw_dataset(path=SEEDS_PATH)
    assert len(examples) == 12

    for ex in examples:
        assert validate_chatml_example(ex)

    train_ex, val_ex = split_dataset(examples, val_ratio=0.25, seed=42)
    assert len(train_ex) == 9
    assert len(val_ex) == 3


def test_dataset_builder_metrics_computation():
    """Verify metrics calculation for bilingual distribution and tokens."""
    examples = load_raw_dataset(path=SEEDS_PATH)
    metrics = compute_dataset_metrics(examples)

    assert metrics["total_examples"] == 12
    assert "es" in metrics["languages"]
    assert "en" in metrics["languages"]
    assert metrics["token_stats"]["avg_tokens"] > 150


def test_evaluate_response_rubrics():
    """Verify single response and batch evaluation on section compliance and slang."""
    valid_es = (
        "¡Qué onda, carnal!\n\n"
        "### 1. El Diagnóstico Rápido\nTodo en orden mi rey.\n\n"
        "### 2. La Explicación Fisiológica\nFirstbeat muestra tu tono vagal excelente.\n\n"
        "### 3. La Chamba de Hoy\nDale suave 30 minutos a paso conversacional."
    )
    res_es = evaluate_single_response(valid_es, language="es")
    assert res_es["has_all_sections"]
    assert res_es["has_vernacular"]
    assert "carnal" in res_es["matched_markers"]

    valid_en = (
        "What's up, my friend!\n\n"
        "### 1. Quick Diagnosis\nYour engine is completely primed today.\n\n"
        "### 2. Physiological Breakdown\nScience proves your parasympathetic nervous system is recovered.\n\n"
        "### 3. Today's Work\nHit the track for a smooth run."
    )
    res_en = evaluate_single_response(valid_en, language="en")
    assert res_en["has_all_sections"]
    assert res_en["has_vernacular"]
    assert "my friend" in res_en["matched_markers"]

    batch_res = evaluate_batch([{"content": valid_es, "language": "es"}, {"content": valid_en, "language": "en"}])
    assert batch_res["total_evaluated"] == 2
    assert batch_res["section_compliance_rate"] == 100.0
    assert batch_res["vernacular_marker_rate"] == 100.0


def test_device_and_dtype_resolution():
    """Verify hardware detection logic."""
    cfg = load_config(CONFIG_PATH)
    device, dtype, bnb = resolve_device_and_dtype(cfg["model"])

    assert device in ("mps", "cuda", "cpu")
    assert isinstance(dtype, torch.dtype)
