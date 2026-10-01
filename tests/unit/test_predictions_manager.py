"""Unit tests for BiometricPredictionsManager."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.predictions_manager import (
    BiometricPredictionsManager,
    get_biweekly_target_dates,
)


@pytest.fixture
def mock_clean_features(tmp_path: Path) -> Path:
    """Create a temporary parquet file with 14 days of clean features."""
    dates = pd.date_range("2026-09-01", periods=14, freq="D")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "calendar_date": dates.strftime("%Y-%m-%d"),
            "resting_heart_rate": 52.0 + np.random.normal(0, 1.0, 14),
            "running_avg_hr": 150.0 + np.random.normal(0, 3.0, 14),
            "gym_avg_hr": 110.0 + np.random.normal(0, 2.0, 14),
            "walking_avg_hr": 115.0 + np.random.normal(0, 2.0, 14),
            "total_sleep_hours": 7.5 + np.random.normal(0, 0.4, 14),
            "sleep_score": 82.0 + np.random.normal(0, 3.0, 14),
            "active_kilocalories": 450.0 + np.random.normal(0, 50.0, 14),
            "resting_kilocalories": 1850.0 + np.random.normal(0, 10.0, 14),
            "total_kilocalories": 2300.0 + np.random.normal(0, 50.0, 14),
            "total_steps": 8500.0 + np.random.normal(0, 800.0, 14),
            "daily_avg_stress": 24.0 + np.random.normal(0, 2.0, 14),
            "hrv_rmssd": 58.0 + np.random.normal(0, 3.0, 14),
        }
    )
    p = tmp_path / "garmin_clean_features.parquet"
    df.to_parquet(p, index=False)
    return p


class TestPredictionsManager:
    """Test suite for bi-weekly rolling predictions and lock mechanism."""

    def test_get_biweekly_target_dates(self):
        """Should calculate remainder of current week + entire next week."""
        # 2026-09-18 is a Friday (weekday=4)
        base_friday = date(2026, 9, 18)

        # 1. With include_today=True: Friday 18 through Sunday 27 (10 dates)
        dates_fri = get_biweekly_target_dates(base_friday, include_today=True)
        assert len(dates_fri) == 10
        assert dates_fri[0] == date(2026, 9, 18)
        assert dates_fri[-1] == date(2026, 9, 27)

        # 2. With include_today=False: Saturday 19 through Sunday 27 (9 dates)
        dates_tomorrow = get_biweekly_target_dates(base_friday, include_today=False)
        assert len(dates_tomorrow) == 9
        assert dates_tomorrow[0] == date(2026, 9, 19)
        assert dates_tomorrow[-1] == date(2026, 9, 27)

        # 3. Monday 2026-09-21: Monday 21 through Sunday Oct 4 (14 dates)
        base_monday = date(2026, 9, 21)
        dates_mon = get_biweekly_target_dates(base_monday, include_today=True)
        assert len(dates_mon) == 14
        assert dates_mon[0] == date(2026, 9, 21)
        assert dates_mon[-1] == date(2026, 10, 4)

    def test_generate_and_lock_predictions(self, mock_clean_features: Path, tmp_path: Path):
        """Should generate predictions, save in SQLite, and permanently lock them from overwrite."""
        pred_dir = tmp_path / "predictions"
        manager = BiometricPredictionsManager(
            features_file=mock_clean_features,
            predictions_dir=pred_dir,
        )

        metrics_subset = ["resting_heart_rate", "total_sleep_hours"]
        preds_1 = manager.generate_and_update_forecasts(
            metrics=metrics_subset, reference_date=date(2026, 9, 14)
        )

        assert len(preds_1) > 0
        assert "is_locked" in preds_1.columns
        assert preds_1["is_locked"].all()

        # Check SQLite persistence and metadata
        assert manager.db_path.exists()
        meta_df = manager.get_forecast_metadata()
        assert not meta_df.empty
        assert "run_id" in meta_df.columns
        assert meta_df["new_records_added"].iloc[0] > 0

        # Re-running immediately must NOT change existing records (Strict Immutability Rule)
        preds_2 = manager.generate_and_update_forecasts(
            metrics=metrics_subset, reference_date=date(2026, 9, 14)
        )
        assert len(preds_2) == len(preds_1)
        pd.testing.assert_frame_equal(preds_1, preds_2, check_dtype=False)
