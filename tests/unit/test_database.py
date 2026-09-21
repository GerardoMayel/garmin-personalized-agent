"""Unit tests for SQLite Historical Database and data ingestion operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.database import GarminDatabase


@pytest.fixture
def temp_db(tmp_path: Path) -> GarminDatabase:
    """Create a temporary database instance for isolated testing."""
    db_file = tmp_path / "test_history.db"
    return GarminDatabase(db_path=db_file)


class TestGarminDatabase:
    """Test suite for SQLite schema creation, upserting, backfill, and queries."""

    def test_init_db_creates_tables(self, temp_db: GarminDatabase):
        """Should create all required tables and indices."""
        counts = temp_db.count_records()
        assert "daily_summaries" in counts
        assert "sleep_records" in counts
        assert "hrv_records" in counts
        assert "stress_records" in counts
        assert "max_metrics" in counts
        assert "fitness_age_records" in counts
        assert "activities" in counts
        assert all(count == 0 for count in counts.values())

    def test_upsert_daily_summary_insert_and_update(self, temp_db: GarminDatabase):
        """Should insert and update daily summary on conflict."""
        payload = {
            "calendarDate": "2026-09-14",
            "totalSteps": 12000,
            "totalDistanceMeters": 9500.0,
            "activeKilocalories": 550.0,
            "totalKilocalories": 2400.0,
            "restingHeartRate": 52,
            "minHeartRate": 48,
            "maxHeartRate": 165,
            "averageStressLevel": 25,
            "maxStressLevel": 80,
            "floorsAscended": 12.0,
            "vigorousIntensityMinutes": 35,
            "moderateIntensityMinutes": 20,
        }

        assert temp_db.upsert_daily_summary(payload) is True
        assert temp_db.count_records()["daily_summaries"] == 1

        # Update steps on same date
        payload["totalSteps"] = 14500
        assert temp_db.upsert_daily_summary(payload) is True
        assert temp_db.count_records()["daily_summaries"] == 1

        with temp_db.get_connection() as conn:
            row = conn.execute(
                "SELECT total_steps, resting_heart_rate FROM daily_summaries WHERE calendar_date = '2026-09-14'"
            ).fetchone()
            assert row["total_steps"] == 14500
            assert row["resting_heart_rate"] == 52

    def test_upsert_sleep(self, temp_db: GarminDatabase):
        """Should insert sleep record with score and sleep stages."""
        payload = {
            "dailySleepDTO": {
                "calendarDate": "2026-09-14",
                "sleepTimeSeconds": 26000,
                "deepSleepSeconds": 6000,
                "lightSleepSeconds": 13000,
                "remSleepSeconds": 7000,
                "awakeSleepSeconds": 0,
                "averageSpO2Value": 98.0,
                "averageRespirationValue": 14.5,
                "avgSleepStress": 15.0,
                "avgHeartRate": 58.0,
                "sleepScores": {"overall": {"value": 90}},
            }
        }

        assert temp_db.upsert_sleep(payload) is True
        assert temp_db.count_records()["sleep_records"] == 1

        with temp_db.get_connection() as conn:
            row = conn.execute(
                "SELECT sleep_score, deep_sleep_seconds FROM sleep_records WHERE calendar_date = '2026-09-14'"
            ).fetchone()
            assert row["sleep_score"] == 90
            assert row["deep_sleep_seconds"] == 6000

    def test_upsert_hrv(self, temp_db: GarminDatabase):
        """Should insert HRV record with baseline and rMSSD values."""
        payload = {
            "hrvSummary": {
                "calendarDate": "2026-09-14",
                "lastNightAvg": 45,
                "weeklyAvg": 42,
                "lastNight5MinHigh": 52,
                "status": "BALANCED",
                "baseline": {"lowUpper": 35, "balancedLow": 38, "balancedUpper": 50},
                "feedbackPhrase": "HRV_BALANCED",
            }
        }

        assert temp_db.upsert_hrv(payload) is True
        assert temp_db.count_records()["hrv_records"] == 1

        with temp_db.get_connection() as conn:
            row = conn.execute(
                "SELECT last_night_avg, status, baseline_balanced_low FROM hrv_records WHERE calendar_date = '2026-09-14'"
            ).fetchone()
            assert row["last_night_avg"] == 45
            assert row["status"] == "BALANCED"
            assert row["baseline_balanced_low"] == 38

    def test_upsert_activity(self, temp_db: GarminDatabase):
        """Should insert activity with summary and fit zip path."""
        payload = {
            "activityId": 123456,
            "activityName": "Evening Tempo Run",
            "startTimeLocal": "2026-09-14 18:30:00",
            "activityType": {"typeKey": "running"},
            "distance": 8000.0,
            "duration": 2400.0,
            "elapsedDuration": 2450.0,
            "elevationGain": 45.0,
            "averageSpeed": 3.33,
            "maxSpeed": 4.12,
            "averageHR": 158,
            "maxHR": 174,
            "calories": 620.0,
        }

        assert temp_db.upsert_activity(payload, fit_zip_path="/data/activity_123456.zip") is True
        assert temp_db.count_records()["activities"] == 1

        with temp_db.get_connection() as conn:
            row = conn.execute("SELECT * FROM activities WHERE activity_id = 123456").fetchone()
            assert row["activity_name"] == "Evening Tempo Run"
            assert row["calendar_date"] == "2026-09-14"
            assert row["fit_zip_path"] == "/data/activity_123456.zip"

    def test_upsert_fitness_age(self, temp_db: GarminDatabase):
        """Should insert fitness age metrics with biological gap calculation."""
        payload = {
            "chronologicalAge": 40,
            "fitnessAge": 34.76,
            "achievableFitnessAge": 34.50,
            "components": {
                "bodyFat": {"value": 16.6},
                "rhr": {"value": 57},
                "vigorousMinutesAvg": {"value": 38.6, "potentialAge": 33.9},
                "vigorousDaysAvg": {"value": 1.3},
            },
            "lastUpdated": "2026-09-14T00:00:00.0",
        }

        assert temp_db.upsert_fitness_age(payload, calendar_date="2026-09-14") is True
        assert temp_db.count_records()["fitness_age_records"] == 1

        with temp_db.get_connection() as conn:
            row = conn.execute("SELECT * FROM fitness_age_records WHERE calendar_date = '2026-09-14'").fetchone()
            assert row["chronological_age"] == 40.0
            assert row["fitness_age"] == 34.76
            assert row["fitness_age_gap"] == 5.24
            assert row["achievable_fitness_age"] == 34.50
            assert row["body_fat_pct"] == 16.6
            assert row["rhr_component"] == 57.0
            assert row["target_potential_age"] == 33.9

    def test_ingest_raw_directory_and_timeseries_query(
        self, temp_db: GarminDatabase, tmp_path: Path
    ):
        """Should scan folder hierarchy and populate tables, queryable via timeseries join."""
        raw_root = tmp_path / "raw"
        day_dir = raw_root / "2026-09-14"
        day_dir.mkdir(parents=True)

        # Write daily summary
        (day_dir / "daily_summary.json").write_text(
            json.dumps({"calendarDate": "2026-09-14", "totalSteps": 10000, "restingHeartRate": 50}),
            encoding="utf-8",
        )
        # Write sleep
        (day_dir / "sleep.json").write_text(
            json.dumps(
                {
                    "dailySleepDTO": {
                        "calendarDate": "2026-09-14",
                        "sleepScores": {"overall": {"value": 85}},
                    }
                }
            ),
            encoding="utf-8",
        )
        # Write hrv
        (day_dir / "hrv.json").write_text(
            json.dumps(
                {
                    "hrvSummary": {
                        "calendarDate": "2026-09-14",
                        "lastNightAvg": 48,
                        "status": "BALANCED",
                    }
                }
            ),
            encoding="utf-8",
        )

        stats = temp_db.ingest_raw_directory(raw_root)
        assert stats["daily_summaries"] == 1
        assert stats["sleep_records"] == 1
        assert stats["hrv_records"] == 1

        # Query timeseries
        ts = temp_db.get_biometrics_timeseries("2026-09-10", "2026-09-15")
        assert len(ts) == 1
        record = ts[0]
        assert record["calendar_date"] == "2026-09-14"
        assert record["total_steps"] == 10000
        assert record["resting_heart_rate"] == 50
        assert record["sleep_score"] == 85
        assert record["hrv_rmssd"] == 48

    def test_build_consolidated_actuals_and_unified_timeline(self, temp_db: GarminDatabase):
        """Should build consolidated actuals, upsert forecasts, and query unified timeline."""
        import pandas as pd

        # 1. Insert daily actuals in base tables
        temp_db.upsert_daily_summary({
            "calendarDate": "2026-09-14",
            "totalSteps": 11500,
            "restingHeartRate": 53,
            "averageStressLevel": 28,
            "totalKilocalories": 2300.0,
            "activeKilocalories": 450.0,
            "floorsAscended": 10.0,
        })
        temp_db.upsert_sleep({
            "dailySleepDTO": {
                "calendarDate": "2026-09-14",
                "sleepScores": {"overall": {"value": 88}},
                "sleepTimeSeconds": 28000,
                "deepSleepSeconds": 6500,
                "remSleepSeconds": 7200,
                "averageSpO2Value": 97.0,
            }
        })
        temp_db.upsert_hrv({
            "hrvSummary": {
                "calendarDate": "2026-09-14",
                "lastNightAvg": 46,
                "status": "BALANCED",
            }
        })
        temp_db.upsert_fitness_age({
            "chronologicalAge": 40.0,
            "fitnessAge": 34.5,
            "achievableFitnessAge": 34.0,
            "components": {"bodyFat": {"value": 16.5}, "rhr": {"value": 53}},
        }, calendar_date="2026-09-14")

        # 2. Build consolidated actuals
        rows = temp_db.build_consolidated_actuals()
        assert rows == 1
        assert temp_db.count_records()["consolidated_daily_actuals"] == 1

        actuals = temp_db.get_consolidated_actuals("2026-09-01", "2026-09-20")
        assert len(actuals) == 1
        act = actuals[0]
        assert act["calendar_date"] == "2026-09-14"
        assert act["total_steps"] == 11500
        assert act["resting_heart_rate"] == 53
        assert act["sleep_score"] == 88
        assert act["hrv_rmssd"] == 46
        assert act["fitness_age"] == 34.5
        assert act["fitness_age_gap"] == 5.5

        # 3. Upsert forecasts
        forecast_data = [
            {
                "target_date": "2026-09-21",
                "metric": "resting_heart_rate",
                "predicted_mean": 52.4,
                "ci_lower": 50.0,
                "ci_upper": 55.0,
                "model_name": "SARIMAX(1,0,1)",
                "mae": 1.2,
                "rmse": 1.5,
                "model_type": "time_series",
                "generated_at": "2026-09-20T23:59:00",
            },
            {
                "target_date": "2026-09-21",
                "metric": "total_steps",
                "predicted_mean": 11000.0,
                "ci_lower": 9500.0,
                "ci_upper": 12500.0,
                "model_name": "SARIMAX(1,1,1)",
                "mae": 800.0,
                "rmse": 1000.0,
                "model_type": "time_series",
                "generated_at": "2026-09-20T23:59:00",
            },
        ]
        df_forecast = pd.DataFrame(forecast_data)
        fc_rows = temp_db.upsert_consolidated_forecasts(df_forecast)
        assert fc_rows == 2
        assert temp_db.count_records()["consolidated_biometric_forecasts"] == 2

        # 4. Query unified timeline view
        timeline = temp_db.get_unified_timeline(start_date="2026-09-14", end_date="2026-09-22")
        assert len(timeline) == 2
        # Date 2026-09-14 is ACTUAL
        assert timeline[0]["calendar_date"] == "2026-09-14"
        assert timeline[0]["record_type"] == "ACTUAL"
        assert timeline[0]["resting_heart_rate"] == 53
        assert timeline[0]["total_steps"] == 11500

        # Date 2026-09-21 is FORECAST
        assert timeline[1]["calendar_date"] == "2026-09-21"
        assert timeline[1]["record_type"] == "FORECAST"
        assert timeline[1]["resting_heart_rate"] == 52.4
        assert timeline[1]["total_steps"] == 11000.0
