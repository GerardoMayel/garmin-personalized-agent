"""Unit tests for time series forecasting and recovery prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.time_series_models import (
    GarminProphetForecaster,
    SleepRecoveryPredictor,
)


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate 14-day sample time series."""
    np.random.seed(42)
    dates = pd.date_range("2026-09-01", periods=14, freq="D")
    return pd.DataFrame(
        {
            "calendar_date": dates.astype(str),
            "resting_heart_rate": 50 + np.sin(np.arange(14)) * 3 + np.random.normal(0, 0.5, 14),
            "total_steps": 8000 + np.random.normal(0, 1000, 14),
            "daily_avg_stress": 25 + np.random.normal(0, 3, 14),
            "sleep_score": 80 + np.random.normal(0, 5, 14),
            "prev_day_steps": 8000 + np.random.normal(0, 1000, 14),
            "prev_day_stress": 25 + np.random.normal(0, 3, 14),
            "prev_day_active_cals": 400 + np.random.normal(0, 50, 14),
            "deep_sleep_ratio": np.random.uniform(0.15, 0.25, 14),
            "hrv_to_baseline_ratio": np.random.uniform(0.9, 1.1, 14),
            "stress_balance_ratio": np.random.uniform(1.0, 2.5, 14),
            "acwr_steps": np.random.uniform(0.9, 1.2, 14),
        }
    )


class TestGarminProphetForecaster:
    """Test suite for Prophet forecaster."""

    def test_fit_and_predict(self, sample_timeseries_df):
        """Should fit Prophet model and generate future predictions with confidence intervals."""
        forecaster = GarminProphetForecaster(weekly_seasonality=False)
        forecaster.fit(sample_timeseries_df, target_col="resting_heart_rate")
        assert forecaster.is_fitted is True

        future_df = forecaster.predict(periods=5)
        assert len(future_df) == len(sample_timeseries_df) + 5
        assert "ds" in future_df.columns
        assert "yhat" in future_df.columns
        assert "yhat_lower" in future_df.columns
        assert "yhat_upper" in future_df.columns

    def test_evaluate_chronological(self, sample_timeseries_df):
        """Should calculate MAE, RMSE, and MAPE on holdout period."""
        forecaster = GarminProphetForecaster()
        metrics = forecaster.evaluate_chronological(
            sample_timeseries_df, target_col="resting_heart_rate", test_size=2
        )
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape_pct" in metrics
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= 0.0

    def test_predict_without_fit_raises_error(self):
        """Should raise RuntimeError if predict called before fit."""
        forecaster = GarminProphetForecaster()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            forecaster.predict(periods=5)


class TestSleepRecoveryPredictor:
    """Test suite for SleepRecoveryPredictor."""

    def test_fit_and_predict_random_forest(self, sample_timeseries_df):
        """Should train recovery regressor and output predictions and feature importances."""
        predictor = SleepRecoveryPredictor(model_type="rf", random_state=42)
        predictor.fit(sample_timeseries_df, target_col="sleep_score")
        assert predictor.is_fitted is True

        preds = predictor.predict(sample_timeseries_df)
        assert len(preds) == len(sample_timeseries_df)
        assert len(predictor.feature_importances_) > 0
        assert np.isclose(sum(predictor.feature_importances_.values()), 1.0, atol=1e-2)

    def test_predict_without_fit_raises_error(self, sample_timeseries_df):
        """Should raise RuntimeError if predict called before fit."""
        predictor = SleepRecoveryPredictor()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            predictor.predict(sample_timeseries_df)
