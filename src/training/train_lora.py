#!/usr/bin/env python3
"""LoRA / PEFT Supervised Fine-Tuning Trainer for Garmin Fitness Coach SLM.

Trains Small Language Models (e.g., Qwen 2.5 1.5B/3B, Llama 3.2 1B/3B) using TRL's
SFTTrainer and PEFT LoRA adapters on the bilingual Garmin coach dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import DatasetDict
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.training.dataset_builder import (
    build_hf_dataset_dict,
    load_raw_dataset,
    split_dataset,
)

DEFAULT_CONFIG_PATH = Path("configs/fine_tuning_lora.yaml")


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_device_and_dtype(
    model_cfg: dict[str, Any],
) -> tuple[str, torch.dtype, BitsAndBytesConfig | None]:
    """Determines optimal compute device, torch dtype, and quantization config.

    Supports:
        - Apple Silicon (MPS): torch.float16 or torch.bfloat16.
        - NVIDIA GPU (CUDA): torch.bfloat16 or float16 with optional 4-bit BitsAndBytes.
        - CPU fallback: torch.float32.
    """
    use_4bit = model_cfg.get("use_4bit_quantization", False)
    bnb_config: BitsAndBytesConfig | None = None

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("CUDA disponible: Cuantización 4-bit (NF4) activada.")
        else:
            logger.info(f"CUDA disponible: Precisión nativa {dtype}.")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        logger.info("Apple Silicon (MPS) detectado: Usando aceleración Metal en float16.")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("GPU no disponible: Ejecutando en CPU en float32.")

    return device, dtype, bnb_config


def prepare_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    bnb_config: BitsAndBytesConfig | None = None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Loads tokenizer and base causal language model."""
    logger.info(f"Cargando tokenizador para: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Cargando modelo base: {model_name}")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }

    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif device in ("cuda", "mps"):
        # For non-quantized MPS / single-GPU, device mapping
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    """Instantiates PEFT LoraConfig from dictionary."""
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def train(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    model_id_override: str | None = None,
    output_dir_override: str | None = None,
    epochs_override: int | None = None,
    max_steps_override: int | None = None,
    dry_run: bool = False,
) -> str:
    """Executes the LoRA Supervised Fine-Tuning pipeline.

    Args:
        config_path: Path to YAML fine-tuning config.
        model_id_override: Optional base model ID to override config.
        output_dir_override: Optional output directory for checkpoints.
        epochs_override: Optional training epochs override.
        max_steps_override: Optional max training steps limit (useful for quick checks).
        dry_run: If True, loads and formats data and model without executing trainer.train().

    Returns:
        Path string to the saved LoRA adapter checkpoint.
    """
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})

    model_name = model_id_override or model_cfg.get("base_model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = output_dir_override or train_cfg.get("output_dir", "./checkpoints/lora_garmin_mexican_coach")
    num_epochs = epochs_override or train_cfg.get("num_train_epochs", 3)
    max_steps = max_steps_override if max_steps_override is not None else -1

    logger.info("=" * 70)
    logger.info("🚀 INICIANDO PIPELINE DE FINE-TUNING LORA (SFT)")
    logger.info("=" * 70)
    logger.info(f"Modelo Base: {model_name}")
    logger.info(f"Directorio de Salida: {output_dir}")
    logger.info(f"Épocas: {num_epochs} | Pasos Máximos: {max_steps}")

    # 1. Device and Precision
    device, dtype, bnb_config = resolve_device_and_dtype(model_cfg)

    # 2. Tokenizer & Base Model
    model, tokenizer = prepare_model_and_tokenizer(
        model_name=model_name,
        device=device,
        dtype=dtype,
        bnb_config=bnb_config,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # 3. LoRA PEFT Config
    peft_config = build_lora_config(lora_cfg)
    logger.info(f"Configuración LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # 4. Dataset Loading & Preparation
    dataset_path = dataset_cfg.get("local_path", "data/synthetic/synthetic_dataset_400.jsonl")
    val_ratio = dataset_cfg.get("val_split_ratio", 0.1)
    seed = dataset_cfg.get("seed", 42)

    raw_examples = load_raw_dataset(path=dataset_path)
    train_raw, val_raw = split_dataset(raw_examples, val_ratio=val_ratio, seed=seed)

    dataset_dict = build_hf_dataset_dict(train_raw, val_raw, tokenizer=tokenizer)
    logger.info(f"Dataset HF preparado: {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} val")

    if dry_run:
        logger.info("🔍 [DRY-RUN] Verificación completada. Muestra del prompt formateado:")
        sample_text = dataset_dict["train"][0].get("text", "")
        print("\n" + "-" * 60)
        print(sample_text[:500] + "\n...")
        print("-" * 60)
        logger.info("✅ Simulación finalizada exitosamente. No se ejecutó el entrenamiento.")
        return output_dir

    # 5. SFTConfig & TrainingArguments
    max_seq_length = dataset_cfg.get("max_seq_length", 2048)
    use_fp16 = (device == "cuda" and not torch.cuda.is_bf16_supported()) or (device == "mps")
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 25),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        fp16=use_fp16,
        bf16=use_bf16,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # 6. SFTTrainer Initialization
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("🏋️ Iniciando proceso de entrenamiento...")
    trainer.train()

    # 7. Save Adapter & Tokenizer
    logger.info(f"💾 Guardando adaptador LoRA en: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"🎉 ¡Entrenamiento completado exitosamente! Checkpoint listo en {output_dir}")
    return output_dir


def main() -> None:
    """CLI Entrypoint for LoRA SFT training."""
    parser = argparse.ArgumentParser(description="Entrenador LoRA / PEFT para Coach Mexicano.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Ruta al YAML de configuración.")
    parser.add_argument("--model-id", type=str, default=None, help="Modelo base en Hugging Face.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio destino de checkpoints.")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas.")
    parser.add_argument("--max-steps", type=int, default=None, help="Máximo de pasos de entrenamiento.")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin entrenar.")
    args = parser.parse_args()

    train(
        config_path=args.config,
        model_id_override=args.model_id,
        output_dir_override=args.output_dir,
        epochs_override=args.epochs,
        max_steps_override=args.max_steps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
