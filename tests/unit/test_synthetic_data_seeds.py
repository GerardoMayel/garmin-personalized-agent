"""Unit test for validating the Mexican Fitness Coach Few-Shot Seed Dataset."""

from __future__ import annotations

from pathlib import Path

from src.training.preview_few_shots import load_and_validate_few_shots

SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")


def test_seed_few_shots_syntax_and_schema():
    """Should load all seed examples and validate ChatML schema and sections."""
    examples = load_and_validate_few_shots(SEEDS_PATH)
    assert len(examples) == 10, f"Expected 10 seed examples, found {len(examples)}"

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
