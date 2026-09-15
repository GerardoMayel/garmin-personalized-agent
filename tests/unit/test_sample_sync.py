"""Unit tests for the sample_sync lightweight script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.sample_sync import authenticate_client, fetch_sample, load_credentials


def test_load_credentials_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Should exit when credentials are missing."""
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.delenv("GARMIN_EMAIL", raising=False)
    monkeypatch.delenv("GARMIN_PASSWORD", raising=False)

    with pytest.raises(SystemExit):
        load_credentials(env_file=empty_env)


def test_load_credentials_success(monkeypatch: pytest.MonkeyPatch):
    """Should return credentials and path when set."""
    monkeypatch.setenv("GARMIN_EMAIL", "runner@example.com")
    monkeypatch.setenv("GARMIN_PASSWORD", "secretpass")
    monkeypatch.setenv("GARMIN_TOKEN_STORE", "/tmp/tokens")

    email, password, token_store = load_credentials()
    assert email == "runner@example.com"
    assert password == "secretpass"
    assert token_store == Path("/tmp/tokens")


def test_fetch_sample_success(tmp_path: Path):
    """Should query Garmin endpoints and save sample json files."""
    mock_client = MagicMock()
    mock_client.get_full_name.return_value = "Test Athlete"
    mock_client.get_sleep_data.return_value = {
        "dailySleepDTO": {"sleepScores": {"overall": {"value": 88}}, "sleepTimeSeconds": 25200}
    }
    mock_client.get_hrv_data.return_value = {
        "hrvSummary": {"lastNightAvg": 55, "status": "BALANCED"}
    }
    mock_client.get_stress_data.return_value = {"avgStressLevel": 28}
    mock_client.get_user_summary.return_value = {"totalSteps": 12000, "restingHeartRate": 52}
    mock_client.get_activities.return_value = [
        {
            "activityId": 999888,
            "activityName": "Trail Run",
            "activityType": {"typeKey": "running"},
            "distance": 8500,
            "duration": 2800,
        }
    ]

    output_dir = tmp_path / "sample_out"
    bundle = fetch_sample(mock_client, output_dir=output_dir)

    assert bundle["user_name"] == "Test Athlete"
    assert bundle["sleep"]["score"] == 88
    assert bundle["hrv"]["lastNightAvg"] == 55
    assert bundle["activity"]["id"] == 999888

    assert (output_dir / "sample_sleep.json").exists()
    assert (output_dir / "sample_hrv.json").exists()
    assert (output_dir / "sample_stress.json").exists()
    assert (output_dir / "sample_daily_summary.json").exists()
    assert (output_dir / "sample_activity_999888.json").exists()
