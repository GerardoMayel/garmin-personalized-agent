"""Unit tests for GarminDataIngestor client and ingestion pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectTooManyRequestsError,
)

from src.ingestion.garmin_sync import GarminDataIngestor


class TestGarminDataIngestor:
    """Test suite for GarminDataIngestor authentication, biometrics, and activity sync."""

    def test_missing_credentials_raises_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Should raise ValueError if credentials are not in environment or arguments."""
        monkeypatch.delenv("GARMIN_EMAIL", raising=False)
        monkeypatch.delenv("GARMIN_PASSWORD", raising=False)

        with pytest.raises(ValueError, match="GARMIN_EMAIL y GARMIN_PASSWORD"):
            GarminDataIngestor(raw_data_dir=str(tmp_path))

    def test_cached_token_authentication(self, tmp_path: Path):
        """Should succeed when login with token directory succeeds."""
        mock_client = MagicMock()
        token_dir = tmp_path / "tokens"

        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(tmp_path / "raw"),
            tokenstore_dir=str(token_dir),
            client=mock_client,
        )

        assert ingestor.email == "test@example.com"
        assert ingestor.raw_data_dir == tmp_path / "raw"

    def test_auth_fallback_dumps_tokens(self, tmp_path: Path):
        """When cached login fails, should perform full login and dump session with garth."""
        mock_client = MagicMock()
        mock_client.login.side_effect = [
            GarminConnectAuthenticationError("Expired"),
            None,
        ]
        token_dir = tmp_path / "tokens"

        with patch("src.ingestion.garmin_sync.Garmin", return_value=mock_client):
            ingestor = GarminDataIngestor(
                email="test@example.com",
                password="secure_password",
                raw_data_dir=str(tmp_path / "raw"),
                tokenstore_dir=str(token_dir),
            )

            assert mock_client.login.call_count == 2
            assert ingestor.tokenstore_dir == token_dir
            mock_client.garth.dump.assert_called_once_with(token_dir.as_posix())

    def test_rate_limit_error_re_raised(self, tmp_path: Path):
        """Rate limit error should be propagated to avoid hitting API repeatedly."""
        mock_client = MagicMock()
        mock_client.login.side_effect = GarminConnectTooManyRequestsError("429 Too Many Requests")

        with patch("src.ingestion.garmin_sync.Garmin", return_value=mock_client):
            with pytest.raises(GarminConnectTooManyRequestsError):
                GarminDataIngestor(
                    email="test@example.com",
                    password="secure_password",
                    raw_data_dir=str(tmp_path / "raw"),
                    tokenstore_dir=str(tmp_path / "tokens"),
                )

    def test_sync_daily_biometrics_creates_expected_files(self, tmp_path: Path):
        """Should query all endpoints and write valid JSON files in data/raw/YYYY-MM-DD/."""
        mock_client = MagicMock()
        mock_client.get_sleep_data.return_value = {"sleep_score": 85, "deep_sleep_seconds": 5400}
        mock_client.get_hrv_data.return_value = {"hrv_status": "BALANCED", "last_night_avg": 58}
        mock_client.get_stress_data.return_value = {"stress_levels": [20, 25, 30]}
        mock_client.get_user_summary.return_value = {"steps": 10500, "resting_hr": 48}
        mock_client.get_max_metrics.return_value = {"vo2_max_running": 54}
        mock_client.get_fitnessage_data.return_value = {
            "chronologicalAge": 40,
            "fitnessAge": 34.76,
            "achievableFitnessAge": 34.50,
            "components": {
                "bodyFat": {"value": 16.6},
                "rhr": {"value": 57},
            },
            "lastUpdated": "2026-09-14T00:00:00.0",
        }

        raw_dir = tmp_path / "raw"
        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(raw_dir),
            client=mock_client,
        )

        target_date = date(2026, 9, 14)
        results = ingestor.sync_daily_biometrics(target_date)

        day_dir = raw_dir / "2026-09-14"
        assert day_dir.exists()

        assert (day_dir / "sleep.json").exists()
        assert (day_dir / "hrv.json").exists()
        assert (day_dir / "stress.json").exists()
        assert (day_dir / "daily_summary.json").exists()
        assert (day_dir / "max_metrics.json").exists()
        assert (day_dir / "fitness_age.json").exists()

        # Check content integrity
        with open(day_dir / "sleep.json", encoding="utf-8") as f:
            data = json.load(f)
            assert data["sleep_score"] == 85

        assert all(results.values())

    def test_sync_daily_biometrics_resilient_to_partial_failure(self, tmp_path: Path):
        """If one endpoint fails, others should still succeed without crashing."""
        mock_client = MagicMock()
        mock_client.get_sleep_data.side_effect = Exception("No sleep data for date")
        mock_client.get_hrv_data.return_value = {"hrv_status": "BALANCED"}
        mock_client.get_stress_data.return_value = {"stress": 22}
        mock_client.get_user_summary.return_value = {"steps": 8000}
        mock_client.get_max_metrics.return_value = {"vo2_max": 52}
        mock_client.get_fitnessage_data.return_value = {"fitnessAge": 35.0}

        raw_dir = tmp_path / "raw"
        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(raw_dir),
            client=mock_client,
        )

        results = ingestor.sync_daily_biometrics(date(2026, 9, 14))

        assert results["sleep"] is False
        assert results["hrv"] is True
        assert (raw_dir / "2026-09-14" / "hrv.json").exists()
        assert not (raw_dir / "2026-09-14" / "sleep.json").exists()

    def test_sync_activities_downloads_metadata_and_fit(self, tmp_path: Path):
        """Should download activity metadata and binary .fit payload."""
        mock_client = MagicMock()
        mock_client.get_activities.return_value = [
            {
                "activityId": 123456789,
                "activityName": "Morning Interval Run",
                "startTimeLocal": "2026-09-14 07:30:00",
                "distance": 10000.0,
            }
        ]
        mock_client.download_activity.return_value = b"MOCK_FIT_BINARY_DATA"

        raw_dir = tmp_path / "raw"
        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(raw_dir),
            client=mock_client,
        )

        activities = ingestor.sync_activities(limit=5, download_fit=True)

        assert len(activities) == 1
        act_dir = raw_dir / "2026-09-14" / "activities"
        assert act_dir.exists()

        summary_file = act_dir / "activity_123456789_summary.json"
        assert summary_file.exists()
        with open(summary_file, encoding="utf-8") as f:
            data = json.load(f)
            assert data["activityId"] == 123456789

        fit_file = act_dir / "activity_123456789.zip"
        assert fit_file.exists()
        assert fit_file.read_bytes() == b"MOCK_FIT_BINARY_DATA"

    def test_run_sync_window(self, tmp_path: Path):
        """Should execute biometrics and activities sync over a multi-day window starting at T-1."""
        mock_client = MagicMock()
        mock_client.get_sleep_data.return_value = {}
        mock_client.get_hrv_data.return_value = {}
        mock_client.get_stress_data.return_value = {}
        mock_client.get_user_summary.return_value = {}
        mock_client.get_max_metrics.return_value = {}
        mock_client.get_fitnessage_data.return_value = {}
        mock_client.get_activities.return_value = []

        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(tmp_path / "raw"),
            client=mock_client,
        )

        ingestor.run_sync_window(days_back=3, sync_fit=False)

        assert mock_client.get_sleep_data.call_count == 3
        mock_client.get_activities.assert_called_once_with(0, 6)

    def test_sync_daily_biometrics_blocks_today(self, tmp_path: Path):
        """Should reject date.today() by default under the 'día vencido' rule."""
        mock_client = MagicMock()
        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(tmp_path / "raw"),
            client=mock_client,
        )

        res = ingestor.sync_daily_biometrics(date.today(), allow_today=False)
        assert res == {}
        mock_client.get_sleep_data.assert_not_called()

    def test_purge_unclosed_or_future_dates(self, tmp_path: Path):
        """Should purge any raw directory with date >= today."""
        mock_client = MagicMock()
        raw_dir = tmp_path / "raw"
        today_dir = raw_dir / date.today().isoformat()
        today_dir.mkdir(parents=True, exist_ok=True)
        (today_dir / "daily_summary.json").write_text("{}", encoding="utf-8")

        mock_db = MagicMock()
        mock_db.delete_records_on_or_after.return_value = {"daily_summaries": 1}

        ingestor = GarminDataIngestor(
            email="test@example.com",
            password="secure_password",
            raw_data_dir=str(raw_dir),
            client=mock_client,
            db=mock_db,
        )

        purged = ingestor.purge_unclosed_or_future_dates()
        assert date.today().isoformat() in purged
        assert not today_dir.exists()
        mock_db.delete_records_on_or_after.assert_called_once_with(date.today().isoformat())

