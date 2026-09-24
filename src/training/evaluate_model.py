#!/usr/bin/env python3
"""Evaluation and Benchmark Suite for Garmin Mexican Fitness Coach Models.

Evaluates base and fine-tuned models on:
1. Strict 3-section structure compliance (Quick Diagnosis, Physiological Breakdown, Today's Work).
2. Authentic coaching vernacular markers (Mexican fitness slang in ES, bilingual energy in EN).
3. Token length calibration (strictly within 200 - 350 tokens).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

# Required Section Headers
SECTIONS_ES = [
    "### 1. El Diagnóstico Rápido",
    "### 2. La Explicación Fisiológica",
    "### 3. La Chamba de Hoy",
]

SECTIONS_EN = [
    "### 1. Quick Diagnosis",
    "### 2. Physiological Breakdown",
    "### 3. Today's Work",
]

# Vernacular Markers
SLANG_ES = [
    "carnal",
    "mi rey",
    "machín",
    "al tiro",
    "chamba",
    "paliza",
    "máquina",
    "pila",
    "chingón",
    "a darle",
]

SLANG_EN = [
    "my friend",
    "engine",
    "dialed in",
    "get after it",
    "beast mode",
    "barbell",
    "track",
    "work",
]


def evaluate_single_response(content: str, language: str = "es") -> dict[str, Any]:
    """Evaluates a single model response against quality and compliance rubrics."""
    lang = language.lower()
    sections = SECTIONS_EN if lang == "en" else SECTIONS_ES
    slang_words = SLANG_EN if lang == "en" else SLANG_ES

    # 1. Section Compliance
    missing_sections = [s for s in sections if s not in content]
    has_all_sections = len(missing_sections) == 0

    # 2. Vernacular Markers
    content_lower = content.lower()
    matched_markers = [w for w in slang_words if w in content_lower]
    has_vernacular = len(matched_markers) > 0

    # 3. Token Length Calibration (Target: 200 - 350 tokens)
    word_count = len(content.split())
    approx_tokens = int(word_count * 1.33)
    in_token_range = 190 <= approx_tokens <= 360

    return {
        "language": lang,
        "word_count": word_count,
        "approx_tokens": approx_tokens,
        "has_all_sections": has_all_sections,
        "missing_sections": missing_sections,
        "has_vernacular": has_vernacular,
        "matched_markers": matched_markers,
        "in_token_range": in_token_range,
    }


def evaluate_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates metrics across multiple responses."""
    total = len(records)
    if total == 0:
        return {"total_evaluated": 0}

    section_passed = 0
    vernacular_passed = 0
    token_range_passed = 0
    all_criteria_passed = 0
    token_counts: list[int] = []

    for rec in records:
        content = ""
        lang = rec.get("language", "es")

        # Extract assistant response from ChatML messages or raw string
        if "messages" in rec:
            for m in rec["messages"]:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    break
        elif "response" in rec:
            content = rec["response"]
        elif "content" in rec:
            content = rec["content"]

        eval_res = evaluate_single_response(content, language=lang)

        if eval_res["has_all_sections"]:
            section_passed += 1
        if eval_res["has_vernacular"]:
            vernacular_passed += 1
        if eval_res["in_token_range"]:
            token_range_passed += 1
        if (
            eval_res["has_all_sections"]
            and eval_res["has_vernacular"]
            and eval_res["in_token_range"]
        ):
            all_criteria_passed += 1

        token_counts.append(eval_res["approx_tokens"])

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    return {
        "total_evaluated": total,
        "section_compliance_rate": round(section_passed / total * 100, 2),
        "vernacular_marker_rate": round(vernacular_passed / total * 100, 2),
        "token_range_compliance_rate": round(token_range_passed / total * 100, 2),
        "overall_pass_rate": round(all_criteria_passed / total * 100, 2),
        "avg_tokens": round(avg_tokens, 1),
    }


def evaluate_jsonl_file(file_path: Path | str) -> dict[str, Any]:
    """Loads a JSONL file and evaluates all assistant records."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    logger.info(f"Evaluando {len(records)} registros desde: {path}")
    return evaluate_batch(records)


def main() -> None:
    """CLI Entrypoint for Model and Dataset Evaluation."""
    parser = argparse.ArgumentParser(description="Evaluador de Calidad y Formato del Coach Mexicano.")
    parser.add_argument(
        "--file",
        type=str,
        default="data/synthetic/seed_few_shots.jsonl",
        help="Archivo JSONL a evaluar (def: seed_few_shots.jsonl)",
    )
    args = parser.parse_args()

    results = evaluate_jsonl_file(args.file)

    print("\n" + "=" * 70)
    print(f"📊 RESULTADOS DE EVALUACIÓN: {args.file}")
    print("=" * 70)
    print(f"Total Registros Evaluados:        {results['total_evaluated']}")
    print(f"Cumplimiento Estructura (3 Sec):  {results['section_compliance_rate']}%")
    print(f"Presencia de Jerga/Modismos:      {results['vernacular_marker_rate']}%")
    print(f"Rango de Tokens (200-350 tok):     {results['token_range_compliance_rate']}%")
    print(f"Aprobación Total de Criterios:    {results['overall_pass_rate']}%")
    print(f"Promedio de Tokens:               {results['avg_tokens']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
