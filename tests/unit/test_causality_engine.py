"""Unit tests for physiological causality and temporal impact engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.causality_engine import (
    CausalityTestResult,
    CrossCorrelationResult,
    PhysiologicalCausalityEngine,
)


@pytest.fixture
def causal_synthetic_df() -> pd.DataFrame:
    """Generate time series with synthetic directional causality (X causes Y with lag 1)."""
    np.random.seed(42)
    n = 20
    x = np.random.normal(30, 5, n)
    y = np.zeros(n)
    y[0] = 60
    for t in range(1, n):
        # Y is explicitly dependent on X_{t-1}
        y[t] = 0.5 * y[t - 1] - 0.4 * x[t - 1] + np.random.normal(0, 1)

    return pd.DataFrame(
        {
            "calendar_date": pd.date_range("2026-09-01", periods=n, freq="D").astype(str),
            "cause_x": x,
            "effect_y": y,
            "constant_col": np.ones(n) * 10.0,
        }
    )


class TestPhysiologicalCausalityEngine:
    """Test suite for PhysiologicalCausalityEngine."""

    def test_compute_cross_correlation(self, causal_synthetic_df):
        """Should calculate normalized CCF and identify peak lag correlation."""
        engine = PhysiologicalCausalityEngine()
        res = engine.compute_cross_correlation(
            causal_synthetic_df, "cause_x", "effect_y", max_lags=3
        )

        assert isinstance(res, CrossCorrelationResult)
        assert len(res.lags) == 7  # -3 to +3
        assert len(res.correlations) == 7
        assert -1.0 <= res.peak_correlation <= 1.0

    def test_test_granger_causality_valid(self, causal_synthetic_df):
        """Should execute Granger causality test and produce valid result."""
        engine = PhysiologicalCausalityEngine(significance_level=0.10)
        res = engine.test_granger_causality(causal_synthetic_df, "cause_x", "effect_y", max_lag=2)

        assert isinstance(res, CausalityTestResult)
        assert res.cause_variable == "cause_x"
        assert res.effect_variable == "effect_y"
        assert 0.0 <= res.p_value <= 1.0
        assert res.f_statistic >= 0.0
        assert isinstance(res.is_significant, (bool, np.bool_))
        assert len(res.interpretation) > 0

    def test_constant_series_handled_safely(self, causal_synthetic_df):
        """Should return non-significant test when column has 0 variance without crashing."""
        engine = PhysiologicalCausalityEngine()
        res = engine.test_granger_causality(
            causal_synthetic_df, "constant_col", "effect_y", max_lag=2
        )

        assert res.is_significant is False
        assert res.p_value == 1.0
        assert "Insufficient variance" in res.interpretation

    def test_missing_column_raises_value_error(self, causal_synthetic_df):
        """Should raise ValueError if variables are not in DataFrame."""
        engine = PhysiologicalCausalityEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.compute_cross_correlation(causal_synthetic_df, "cause_x", "unknown_y")
