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


def test_models_registry_structure():
    """Verify models_registry.json schema, active models, and probe latencies."""
    import json
    registry_path = Path("data/synthetic/models_registry.json")
    assert registry_path.exists(), "models_registry.json must exist after probe"

    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_active_models" in data
    assert data["total_active_models"] >= 5
    assert len(data["models"]) == data["total_active_models"]

    for m in data["models"]:
        assert "model_id" in m
        assert m["status"] == "active"
        assert "probe_latency_ms" in m
        assert m["probe_latency_ms"] > 0


def test_upload_dataset_to_hub_dry_run():
    """Verify Hugging Face Hub uploader generates valid Dataset Card in dry run."""
    from src.training.upload_to_huggingface import upload_dataset_to_hub

    url = upload_dataset_to_hub(
        dataset_path=SEEDS_PATH,
        repo_id="GerardoMayel/test-garmin-coach-sft",
        dry_run=True,
    )
    assert url == "https://huggingface.co/datasets/GerardoMayel/test-garmin-coach-sft"


