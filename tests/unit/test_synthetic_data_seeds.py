"""Unit test for validating the Mexican Fitness Coach Few-Shot Seed Dataset."""

from __future__ import annotations

from pathlib import Path

from src.training.preview_few_shots import load_and_validate_few_shots

SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")


def test_seed_few_shots_syntax_and_schema():
    """Should load all seed examples and validate ChatML schema and sections."""
    examples = load_and_validate_few_shots(SEEDS_PATH)
    assert len(examples) == 15, f"Expected 15 seed examples, found {len(examples)}"

    metrics = {e["metric_category"] for e in examples}
    expected_metrics = {"hrv_rmssd", "sleep_score", "daily_avg_stress", "resting_heart_rate", "total_steps"}
    assert metrics == expected_metrics, f"Missing metrics in seeds: {expected_metrics - metrics}"

    for ex in examples:
        assert ex["id"].startswith("seed_")
        assert len(ex["messages"]) == 3
        # Ensure authentic Mexican vernacular markers
        content = ex["messages"][2]["content"].lower()
        has_slang = any(
            w in content
            for w in [
                "carnal",
                "mi rey",
                "chingón",
                "paliza",
                "chamba",
                "machín",
                "pila",
                "máquina",
                "brother",
                "al tiro",
            ]
        )
        assert has_slang, f"Example {ex['id']} is missing Mexican coaching vernacular markers"


def test_seed_few_shots_includes_peak_performance_scenarios():
    """Ensure catalog includes high-intensity, PR, and peak performance scenarios."""
    examples = load_and_validate_few_shots(SEEDS_PATH)
    scenarios = [e["scenario"].lower() for e in examples]

    # Verify high intensity keywords in scenarios
    has_vo2max = any("vo2max" in s or "series" in s for s in scenarios)
    has_pr_or_heavy = any("pr" in s or "pesada" in s or "récord" in s for s in scenarios)
    has_lactate_or_tempo = any("umbral de lactato" in s or "tempo" in s for s in scenarios)
    has_hiit_or_functional = any("hiit" in s or "circuito" in s for s in scenarios)

    assert has_vo2max, "Missing VO2Max / track interval scenario"
    assert has_pr_or_heavy, "Missing PR / heavy strength scenario"
    assert has_lactate_or_tempo, "Missing lactate threshold / tempo run scenario"
    assert has_hiit_or_functional, "Missing high-intensity HIIT / functional circuit scenario"

