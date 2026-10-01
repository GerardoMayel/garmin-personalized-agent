"""Unit tests for BiWeeklyAverageManager."""

from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.analytics.bi_weekly_averages import BiWeeklyAverageManager


@pytest.fixture
def temp_dbs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Sets up temporary historical and predictions SQLite databases."""
    hist_db = tmp_path / "garmin_history.db"
    fore_db = tmp_path / "weekly_biometric_forecasts.db"
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1. Populate temporary garmin_history.db
    conn_hist = sqlite3.connect(hist_db)
    cursor_h = conn_hist.cursor()
    cursor_h.execute(
        """
        CREATE TABLE consolidated_daily_actuals (
            calendar_date TEXT PRIMARY KEY,
            total_steps INTEGER,
            daily_avg_stress INTEGER,
            resting_heart_rate INTEGER,
            sleep_score INTEGER,
            total_sleep_seconds INTEGER,
            hrv_rmssd REAL,
            total_kilocalories REAL,
            active_kilocalories REAL,
            resting_kilocalories REAL,
            fitness_age REAL,
            fitness_age_gap REAL
        );
        """
    )
    cursor_h.execute(
        """
        CREATE TABLE activities (
            activity_id INTEGER PRIMARY KEY,
            calendar_date TEXT,
            activity_type TEXT,
            avg_hr REAL
        );
        """
    )

    # Sample historical days: 2026-09-24 to 2026-09-30
    # Include some zeros and NaNs to test user's zero-filtering rule
    hist_rows = [
        ("2026-09-24", 10000, 30, 60, 80, 25200, 40.0, 2200.0, 400.0, 1800.0, 34.5, 5.5),
        ("2026-09-25", 8000, 32, 58, 75, 21600, 38.0, 2100.0, 300.0, 1800.0, 34.5, 5.5),
        ("2026-09-26", 0, None, 0, None, None, None, 0.0, 0.0, 0.0, 34.5, 5.5),  # Off day/zeros
        ("2026-09-27", 12000, 28, 59, 85, 28800, 42.0, 2400.0, 600.0, 1800.0, 34.5, 5.5),
    ]
    cursor_h.executemany(
        """
        INSERT INTO consolidated_daily_actuals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        hist_rows,
    )

    # Activity HRs
    act_rows = [
        (1, "2026-09-24", "running", 160.0),
        (2, "2026-09-25", "strength_training", 120.0),
        (3, "2026-09-27", "running", 170.0),
    ]
    cursor_h.executemany("INSERT INTO activities VALUES (?, ?, ?, ?)", act_rows)
    conn_hist.commit()
    conn_hist.close()

    # 2. Populate temporary weekly_biometric_forecasts.db
    conn_fore = sqlite3.connect(fore_db)
    cursor_f = conn_fore.cursor()
    cursor_f.execute(
        """
        CREATE TABLE biometric_forecasts (
            target_date TEXT,
            metric TEXT,
            predicted_value REAL,
            PRIMARY KEY (target_date, metric)
        );
        """
    )

    # Forecast days: 2026-10-01 to 2026-10-07
    fore_dates = [f"2026-10-0{i}" for i in range(1, 8)]
    fore_rows = []
    for d in fore_dates:
        fore_rows.append((d, "total_steps", 9000.0))
        fore_rows.append((d, "daily_avg_stress", 29.0))
        fore_rows.append((d, "resting_heart_rate", 58.0))
        fore_rows.append((d, "sleep_score", 82.0))
        fore_rows.append((d, "total_sleep_hours", 7.5))
        fore_rows.append((d, "hrv_rmssd", 41.0))
        fore_rows.append((d, "total_kilocalories", 2300.0))
        fore_rows.append((d, "active_kilocalories", 450.0))
        fore_rows.append((d, "resting_kilocalories", 1850.0))
        fore_rows.append((d, "fitness_age", 34.6))
        fore_rows.append((d, "fitness_age_gap", 5.4))
        fore_rows.append((d, "walking_avg_hr", 115.0))
        fore_rows.append((d, "running_avg_hr", 165.0))
        fore_rows.append((d, "gym_avg_hr", 118.0))

    cursor_f.executemany("INSERT INTO biometric_forecasts VALUES (?, ?, ?)", fore_rows)
    conn_fore.commit()
    conn_fore.close()

    return hist_db, fore_db, pred_dir


def test_window_dates_calculation(temp_dbs: tuple[Path, Path, Path]) -> None:
    hist_db, fore_db, pred_dir = temp_dbs
    mgr = BiWeeklyAverageManager(db_path=hist_db, forecast_db_path=fore_db, predictions_dir=pred_dir)

    anchor = date(2026, 10, 1)
    hist_start, hist_end, fore_start, fore_end = mgr._get_window_dates(anchor)

    # Historical: 7 closed days ending yesterday
    assert hist_start == "2026-09-24"
    assert hist_end == "2026-09-30"
    # Forecast: 7 days starting today (anchor)
    assert fore_start == "2026-10-01"
    assert fore_end == "2026-10-07"


def test_clean_average_zeros_and_nulls_rule(temp_dbs: tuple[Path, Path, Path]) -> None:
    hist_db, fore_db, pred_dir = temp_dbs
    mgr = BiWeeklyAverageManager(db_path=hist_db, forecast_db_path=fore_db, predictions_dir=pred_dir)

    # 1. Zeros and None must be excluded from average and denominator
    vals = [100.0, 0.0, None, np.nan, 200.0]
    avg, count = mgr._compute_clean_average(vals, "total_steps")
    assert avg == 150.0
    assert count == 2  # Divided by 2, NOT by 5 or 7!

    # 2. All zeros or nulls return None, 0
    empty_vals = [0.0, None, np.nan]
    avg_empty, count_empty = mgr._compute_clean_average(empty_vals, "daily_avg_stress")
    assert avg_empty is None
    assert count_empty == 0


def test_compute_and_save_averages(temp_dbs: tuple[Path, Path, Path]) -> None:
    hist_db, fore_db, pred_dir = temp_dbs
    mgr = BiWeeklyAverageManager(db_path=hist_db, forecast_db_path=fore_db, predictions_dir=pred_dir)

    record = mgr.compute_and_save_averages(anchor_date="2026-10-01")

    assert record["anchor_date"] == "2026-10-01"
    assert record["history_window_start"] == "2026-09-24"
    assert record["history_window_end"] == "2026-09-30"
    assert record["forecast_window_start"] == "2026-10-01"
    assert record["forecast_window_end"] == "2026-10-07"

    # Steps: 10000, 8000, 0 (ignored), 12000 -> (10000 + 8000 + 12000) / 3 = 10000.0
    assert record["avg_total_steps_last_7d"] == 10000.0
    assert record["valid_days_hist_total_steps"] == 3
    assert record["forecast_total_steps_next_7d"] == 9000.0
    assert record["valid_days_forecast_total_steps"] == 7

    # Running HR: 160, 170 -> 165.0 (2 days)
    assert record["avg_running_avg_hr_last_7d"] == 165.0
    assert record["valid_days_hist_running_avg_hr"] == 2

    # Verify table in SQLite databases
    for db_path in (fore_db, hist_db):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT anchor_date, avg_total_steps_last_7d, forecast_total_steps_next_7d FROM bi_weekly_average")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "2026-10-01"
        assert rows[0][1] == 10000.0
        assert rows[0][2] == 9000.0

    # Verify parquet and csv export
    parquet_file = pred_dir / "bi_weekly_average.parquet"
    csv_file = pred_dir / "bi_weekly_average.csv"
    assert parquet_file.exists()
    assert csv_file.exists()

    df_parquet = pd.read_parquet(parquet_file)
    assert len(df_parquet) == 1
    assert df_parquet["anchor_date"].iloc[0] == "2026-10-01"


def test_idempotent_single_record_per_day(temp_dbs: tuple[Path, Path, Path]) -> None:
    """Verifies that running multiple times for the same anchor_date updates cleanly without duplicate rows."""
    hist_db, fore_db, pred_dir = temp_dbs
    mgr = BiWeeklyAverageManager(db_path=hist_db, forecast_db_path=fore_db, predictions_dir=pred_dir)

    # First run
    mgr.compute_and_save_averages(anchor_date="2026-10-01")
    # Second run
    mgr.compute_and_save_averages(anchor_date="2026-10-01")

    conn = sqlite3.connect(fore_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bi_weekly_average WHERE anchor_date = '2026-10-01'")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 1  # Exactly 1 record per anchor date!
