"""Unit tests for DVC Clean Dataset Manager."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.analytics.dvc_manager import GarminDVCManager


@pytest.fixture
def mock_garmin_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with minimal biometric tables."""
    db_path = tmp_path / "test_garmin.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create daily_summaries
    cur.execute(
        """
        CREATE TABLE daily_summaries (
            calendar_date TEXT PRIMARY KEY,
            total_steps INTEGER,
            total_distance_meters REAL,
            active_kilocalories REAL,
            total_kilocalories REAL,
            resting_heart_rate INTEGER,
            min_heart_rate INTEGER,
            max_heart_rate INTEGER,
            avg_stress_level INTEGER,
            max_stress_level INTEGER,
            floors_ascended INTEGER,
            vigorous_minutes INTEGER,
            moderate_minutes INTEGER,
            updated_at TIMESTAMP
        )
        """
    )
    # Create sleep_records
    cur.execute(
        """
        CREATE TABLE sleep_records (
            calendar_date TEXT PRIMARY KEY,
            sleep_score INTEGER,
            total_sleep_seconds INTEGER,
            deep_sleep_seconds INTEGER,
            light_sleep_seconds INTEGER,
            rem_sleep_seconds INTEGER,
            awake_sleep_seconds INTEGER,
            avg_spo2 REAL,
            lowest_spo2 REAL,
            avg_respiration REAL,
            avg_sleep_stress REAL,
            updated_at TIMESTAMP
        )
        """
    )
    # Create hrv_records
    cur.execute(
        """
        CREATE TABLE hrv_records (
            calendar_date TEXT PRIMARY KEY,
            last_night_avg REAL,
            weekly_avg REAL,
            baseline_low REAL,
            baseline_balanced_low REAL,
            baseline_balanced_upper REAL,
            updated_at TIMESTAMP
        )
        """
    )
    # Create stress_records
    cur.execute(
        """
        CREATE TABLE stress_records (
            calendar_date TEXT PRIMARY KEY,
            avg_stress_level INTEGER,
            max_stress_level INTEGER,
            rest_stress_duration_sec INTEGER,
            activity_stress_duration_sec INTEGER,
            low_stress_duration_sec INTEGER,
            medium_stress_duration_sec INTEGER,
            high_stress_duration_sec INTEGER,
            updated_at TIMESTAMP
        )
        """
    )
    # Create max_metrics
    cur.execute(
        """
        CREATE TABLE max_metrics (
            calendar_date TEXT PRIMARY KEY,
            vo2_max_running REAL,
            vo2_max_precise REAL,
            fitness_age REAL,
            updated_at TIMESTAMP
        )
        """
    )
    # Create fitness_age_records
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fitness_age_records (
            calendar_date TEXT PRIMARY KEY,
            chronological_age REAL,
            fitness_age REAL,
            achievable_fitness_age REAL,
            fitness_age_gap REAL,
            body_fat_pct REAL,
            rhr_component REAL,
            vigorous_minutes_avg REAL,
            vigorous_days_avg REAL,
            target_potential_age REAL,
            updated_at TIMESTAMP
        )
        """
    )
    # Create activities
    cur.execute(
        """
        CREATE TABLE activities (
            activity_id TEXT PRIMARY KEY,
            calendar_date TEXT,
            activity_name TEXT,
            activity_type TEXT,
            distance_meters REAL,
            duration_seconds REAL,
            elapsed_duration_seconds REAL,
            elevation_gain_meters REAL,
            avg_speed_mps REAL,
            max_speed_mps REAL,
            avg_hr REAL,
            max_hr REAL,
            calories REAL,
            fit_zip_path TEXT,
            start_time_local TEXT,
            updated_at TIMESTAMP
        )
        """
    )

    # Insert 3 days
    for d in ["2026-09-01", "2026-09-02", "2026-09-03"]:
        cur.execute(
            "INSERT INTO daily_summaries (calendar_date, total_steps, active_kilocalories, total_kilocalories, resting_heart_rate, avg_stress_level) VALUES (?, 8000, 400, 2200, 52, 25)",
            (d,),
        )
        cur.execute(
            "INSERT INTO sleep_records (calendar_date, sleep_score, total_sleep_seconds, deep_sleep_seconds) VALUES (?, 85, 28800, 7200)",
            (d,),
        )
        cur.execute(
            "INSERT INTO hrv_records (calendar_date, last_night_avg, weekly_avg) VALUES (?, 60.0, 58.0)",
            (d,),
        )

    # Insert activities: running and gym
    cur.execute(
        "INSERT INTO activities (activity_id, calendar_date, activity_type, avg_hr, max_hr, calories, duration_seconds) VALUES ('act1', '2026-09-01', 'running', 155.0, 175.0, 450, 2400)"
    )
    cur.execute(
        "INSERT INTO activities (activity_id, calendar_date, activity_type, avg_hr, max_hr, calories, duration_seconds) VALUES ('act2', '2026-09-02', 'strength_training', 110.0, 140.0, 300, 3000)"
    )

    conn.commit()
    conn.close()
    return db_path


class TestGarminDVCManager:
    """Test suite for DVC clean dataset management."""

    def test_extract_clean_dataframe(self, mock_garmin_db: Path, tmp_path: Path):
        """Should extract, merge, and impute all required activity and biometric fields."""
        manager = GarminDVCManager(db_path=mock_garmin_db, dvc_dir=tmp_path / "dvc")
        df = manager.extract_clean_dataframe()

        assert len(df) == 3
        assert "calendar_date" in df.columns
        assert "resting_heart_rate" in df.columns
        assert "running_avg_hr" in df.columns
        assert "gym_avg_hr" in df.columns
        assert "walking_avg_hr" in df.columns
        assert "total_sleep_hours" in df.columns
        assert "resting_kilocalories" in df.columns
        assert "total_kilocalories" in df.columns

        # Verify sleep hours: 28800 / 3600 = 8.0 hrs
        assert df["total_sleep_hours"].iloc[0] == 8.0
        # Verify resting calories = 2200 - 400 = 1800
        assert df["resting_kilocalories"].iloc[0] == 1800.0

    def test_update_clean_dataset_append_only(self, mock_garmin_db: Path, tmp_path: Path):
        """Should append new records without overwriting past history."""
        dvc_dir = tmp_path / "dvc"
        manager = GarminDVCManager(db_path=mock_garmin_db, dvc_dir=dvc_dir)
        df1 = manager.update_clean_dataset()
        assert len(df1) == 3

        # Re-run: should keep 3
        df2 = manager.update_clean_dataset()
        assert len(df2) == 3

        # Add a 4th day to DB
        conn = sqlite3.connect(mock_garmin_db)
        conn.cursor().execute(
            "INSERT INTO daily_summaries (calendar_date, total_steps, active_kilocalories, total_kilocalories, resting_heart_rate, avg_stress_level) VALUES ('2026-09-04', 10000, 500, 2400, 50, 20)"
        )
        conn.commit()
        conn.close()

        df3 = manager.update_clean_dataset()
        assert len(df3) == 4
        assert "2026-09-04" in df3["calendar_date"].values
