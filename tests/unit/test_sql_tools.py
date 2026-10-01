"""Unit tests for deterministic SQL-backed Garmin tools."""

from __future__ import annotations

import pytest

from src.agents.tools import get_garmin_actuals_tool, get_garmin_forecasts_tool
from src.tools.garmin_sql_tools import (
    get_garmin_actuals,
    get_garmin_forecasts,
)


class TestGarminSQLTools:
    """Test suite for get_garmin_actuals and get_garmin_forecasts tools."""

    def test_get_garmin_actuals_default_single_day(self):
        """Should retrieve the latest closed single day actuals with all expected metric keys."""
        res = get_garmin_actuals(days=1)
        assert res["status"] == "success"
        assert res["period"]["total_days_retrieved"] == 1
        assert len(res["records"]) == 1

        record = res["records"][0]
        # Verify metric keys present
        assert "resting_heart_rate" in record
        assert "hrv_rmssd" in record
        assert "daily_avg_stress" in record
        assert "sleep_score" in record
        assert "total_sleep_hours" in record
        assert "total_steps" in record
        assert "active_kilocalories" in record

        # Verify clinical extras present
        assert "deep_sleep_hours" in record
        assert "rem_sleep_hours" in record
        assert "avg_spo2" in record
        assert "avg_respiration" in record
        assert "activity_count" in record
        assert "total_activity_distance_km" in record

    def test_get_garmin_actuals_multi_day_summary(self):
        """Should retrieve 3 days and compute valid mathematical summary."""
        res = get_garmin_actuals(days=3)
        assert res["status"] == "success"
        assert res["period"]["total_days_retrieved"] == 3
        assert len(res["records"]) == 3

        summary = res["summary"]
        assert "avg_resting_heart_rate" in summary
        assert "avg_hrv_rmssd" in summary
        assert "avg_daily_stress" in summary
        assert "avg_sleep_score" in summary
        assert "avg_total_sleep_hours" in summary
        assert "total_steps_period" in summary
        assert summary["avg_resting_heart_rate"] > 0
        assert summary["total_steps_period"] > 0

    def test_get_garmin_actuals_excludes_biological_age(self):
        """Must strictly exclude fitness age, chronological age, and gap."""
        res = get_garmin_actuals(days=5)
        assert res["status"] == "success"
        for rec in res["records"]:
            assert "fitness_age" not in rec
            assert "chronological_age" not in rec
            assert "fitness_age_gap" not in rec

    def test_get_garmin_forecasts_specific_metric(self):
        """Should retrieve forecasts for resting_heart_rate with horizon and statements."""
        res = get_garmin_forecasts(metric="resting_heart_rate", horizon_days=7)
        assert res["status"] == "success"
        assert res["horizon_days_requested"] == 7
        assert len(res["predictions"]) == 7

        first_pred = res["predictions"][0]
        assert first_pred["metric"] == "resting_heart_rate"
        assert first_pred["days_ahead"] == 1
        assert first_pred["predicted_value"] > 0
        assert first_pred["ci_lower"] <= first_pred["predicted_value"] <= first_pred["ci_upper"]
        assert first_pred["is_locked"] is True

        # Check statement
        assert len(res["summary_statements"]) == 7
        assert "El pronóstico para dentro de 1 día(s)" in res["summary_statements"][0]
        assert "resting_heart_rate" in res["summary_statements"][0]

    def test_get_garmin_forecasts_all_metrics(self):
        """Should retrieve multi-metric forecasts and exclude biological age."""
        res = get_garmin_forecasts(metric=None, horizon_days=3)
        assert res["status"] == "success"
        assert len(res["predictions"]) > 0

        metrics = res["metrics_included"]
        assert "resting_heart_rate" in metrics
        assert "total_steps" in metrics
        assert "fitness_age" not in metrics
        assert "fitness_age_gap" not in metrics

    def test_get_garmin_forecasts_blocks_biological_age_query(self):
        """Should reject querying fitness_age or fitness_age_gap."""
        res = get_garmin_forecasts(metric="fitness_age")
        assert res["status"] == "error"
        assert "excluida" in res["message"]
        assert len(res["predictions"]) == 0

    def test_langchain_tool_wrappers(self):
        """LangChain/LangGraph tool decorators should execute and return identical schema."""
        actuals_tool_res = get_garmin_actuals_tool.invoke({"days": 2})
        assert actuals_tool_res["status"] == "success"
        assert actuals_tool_res["period"]["total_days_retrieved"] == 2

        forecasts_tool_res = get_garmin_forecasts_tool.invoke({"metric": "sleep_score", "horizon_days": 4})
        assert forecasts_tool_res["status"] == "success"
        assert len(forecasts_tool_res["predictions"]) == 4
