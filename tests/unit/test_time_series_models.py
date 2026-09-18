"""Unit tests for time series forecasting and recovery prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.time_series_models import (
    EnsembleBiometricForecaster,
    GarminProphetForecaster,
    HoltWintersForecaster,
    SleepRecoveryPredictor,
    compute_physiological_bounds,
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


class TestPhysiologicalBounds:
    """Test suite for domain bounds calculation."""

    def test_resting_heart_rate_bounds(self):
        """Should enforce clinical floor >= 38.0 bpm and reasonable cap."""
        series = pd.Series([52.0, 50.0, 49.0, 48.0, 51.0, 53.0])
        floor, cap = compute_physiological_bounds(series, "resting_heart_rate")
        assert floor >= 38.0
        assert cap <= 115.0
        assert floor < cap

    def test_stress_and_sleep_bounds(self):
        """Should strictly enforce [0, 100] for stress and sleep score."""
        series = pd.Series([25.0, 35.0, 40.0, 30.0])
        floor_stress, cap_stress = compute_physiological_bounds(series, "daily_avg_stress")
        assert floor_stress == 0.0
        assert cap_stress == 100.0

        floor_sleep, cap_sleep = compute_physiological_bounds(series, "sleep_score")
        assert floor_sleep == 0.0
        assert cap_sleep == 100.0


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

    def test_downward_trend_respects_floor(self):
        """Should not project sub-human heart rates even under steep historical decline."""
        dates = pd.date_range("2026-09-01", periods=10, freq="D")
        declining_hr = [58.0, 56.0, 54.0, 52.0, 50.0, 48.0, 46.0, 44.0, 42.0, 40.0]
        df = pd.DataFrame(
            {
                "calendar_date": dates.astype(str),
                "resting_heart_rate": declining_hr,
            }
        )
        forecaster = GarminProphetForecaster(
            weekly_seasonality=False,
            bounds=(38.0, 70.0),
        )
        forecaster.fit(df, target_col="resting_heart_rate")
        preds = forecaster.predict(periods=14)

        # Future forecasts must strictly obey floor >= 38.0
        future_yhat = preds.tail(14)["yhat"].values
        assert np.all(future_yhat >= 38.0), f"Found sub-floor prediction: {future_yhat.min()}"

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


class TestHoltWintersForecaster:
    """Test suite for Holt-Winters damped ETS forecaster."""

    def test_fit_and_predict(self, sample_timeseries_df):
        """Should fit Holt-Winters model and output damped forecasts bounded by limits."""
        hw = HoltWintersForecaster(bounds=(38.0, 100.0))
        hw.fit(sample_timeseries_df, target_col="resting_heart_rate")
        assert hw.is_fitted is True

        forecast = hw.predict(periods=7)
        assert len(forecast) == len(sample_timeseries_df) + 7
        assert "yhat" in forecast.columns
        assert "yhat_lower" in forecast.columns
        assert "yhat_upper" in forecast.columns

        # Verify bounds clamping
        assert forecast["yhat"].min() >= 38.0
        assert forecast["yhat"].max() <= 100.0

    def test_evaluate_chronological(self, sample_timeseries_df):
        """Should perform chronological split evaluation."""
        hw = HoltWintersForecaster()
        metrics = hw.evaluate_chronological(
            sample_timeseries_df, target_col="resting_heart_rate", test_size=2
        )
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0.0


class TestEnsembleBiometricForecaster:
    """Test suite for hybrid ensemble forecaster."""

    def test_ensemble_fit_and_predict(self, sample_timeseries_df):
        """Should blend Prophet and Holt-Winters predictions into bounded ensemble."""
        ensemble = EnsembleBiometricForecaster(bounds=(38.0, 100.0))
        ensemble.fit(sample_timeseries_df, target_col="resting_heart_rate")
        assert ensemble.is_fitted is True

        preds = ensemble.predict(periods=7)
        assert len(preds) == len(sample_timeseries_df) + 7
        assert np.all(preds["yhat"] >= 38.0)
        assert np.all(preds["yhat"] <= 100.0)

    def test_ensemble_evaluate(self, sample_timeseries_df):
        """Should evaluate ensemble chronological performance."""
        ensemble = EnsembleBiometricForecaster()
        metrics = ensemble.evaluate_chronological(
            sample_timeseries_df, target_col="resting_heart_rate", test_size=2
        )
        assert "mae" in metrics
        assert "rmse" in metrics


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
