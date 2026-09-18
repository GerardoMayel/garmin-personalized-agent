"""Unit tests for physiological anomaly detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.anomaly_detection import (
    AnomalySeverity,
    PhysiologicalAnomalyDetector,
)


@pytest.fixture
def synthetic_biometrics_df() -> pd.DataFrame:
    """Generate synthetic biometric dataframe for testing."""
    np.random.seed(42)
    dates = pd.date_range("2026-09-01", periods=15, freq="D")
    df = pd.DataFrame(
        {
            "calendar_date": dates.astype(str),
            "resting_heart_rate": np.random.normal(52, 2, 15),
            "hrv_rmssd": np.random.normal(65, 5, 15),
            "daily_avg_stress": np.random.normal(25, 4, 15),
            "sleep_score": np.random.normal(82, 4, 15),
            "total_steps": np.random.normal(9000, 1000, 15),
        }
    )
    # Inject an acute outlier on day 10 (illness/overtraining pattern: low HRV, high resting HR, high stress)
    df.loc[10, "resting_heart_rate"] = 72.0
    df.loc[10, "hrv_rmssd"] = 32.0
    df.loc[10, "daily_avg_stress"] = 55.0
    return df


class TestPhysiologicalAnomalyDetector:
    """Test suite for PhysiologicalAnomalyDetector."""

    def test_fit_and_detect_detects_injected_outlier(self, synthetic_biometrics_df):
        """Should fit models and identify the acute outlier day."""
        detector = PhysiologicalAnomalyDetector(contamination=0.15, random_state=42)
        detector.fit(synthetic_biometrics_df)
        assert detector.is_fitted is True

        results = detector.detect(synthetic_biometrics_df)
        assert len(results) == len(synthetic_biometrics_df)

        # Day 10 must be flagged as an anomaly with contributing factors
        outlier_res = results[10]
        assert outlier_res.is_anomaly is True
        assert outlier_res.severity in (AnomalySeverity.ACUTE_ALERT, AnomalySeverity.MILD_ANOMALY)
        assert len(outlier_res.contributing_factors) > 0

    def test_missing_features_raises_value_error(self):
        """Should raise ValueError if required features are not present."""
        df = pd.DataFrame({"calendar_date": ["2026-09-01"], "unknown_col": [123]})
        detector = PhysiologicalAnomalyDetector(features=["resting_heart_rate"])
        with pytest.raises(ValueError, match="None of the required features"):
            detector.fit(df)

    def test_to_dataframe(self, synthetic_biometrics_df):
        """Should convert AnomalyPoint list to well-structured DataFrame."""
        detector = PhysiologicalAnomalyDetector(random_state=42)
        results = detector.detect(synthetic_biometrics_df)
        df_res = detector.to_dataframe(results)

        assert isinstance(df_res, pd.DataFrame)
        assert len(df_res) == len(synthetic_biometrics_df)
        assert "calendar_date" in df_res.columns
        assert "is_anomaly" in df_res.columns
        assert "severity" in df_res.columns
        assert "isolation_forest_score" in df_res.columns
