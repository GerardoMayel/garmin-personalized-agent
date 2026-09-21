"""Time Series Forecasting and Predictive Recovery Modeling for Garmin Telemetry.

Features:
1. GarminProphetForecaster: Bayesian structural time series forecasting with Meta Prophet,
   with anomaly filtering, logistic growth, and physiological saturation boundaries [floor, cap].
2. HoltWintersForecaster: Classical damped additive Exponential Smoothing (ETS) with
   physiological homeostasis dampening and bound clamping.
3. EnsembleBiometricForecaster: Multi-model ensemble combining Bayesian Structural &
   Damped Exponential Smoothing for robust, clinically valid biometrics forecasting.
4. SleepRecoveryPredictor: Supervised machine learning (Random Forest / Gradient Boosting)
   predicting nocturnal sleep quality and recovery readiness from daytime exertion and stress.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.common.logger import get_logger

logger = get_logger(__name__)


def compute_physiological_bounds(
    series: pd.Series,
    metric_name: str,
) -> tuple[float, float]:
    """Calculate personalized, physiologically viable saturation bounds [floor, cap].

    Combines empirical athlete telemetry percentiles with clinical safety boundaries
    to prevent biologically implausible forecasts (e.g., fatal bradycardia or impossible stress).

    Args:
        series: Historical time-series observations.
        metric_name: Name of target metric.

    Returns:
        tuple[float, float]: (floor, cap) bounds.
    """
    valid = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(valid) == 0:
        return (0.0, 100.0)

    obs_min = float(np.min(valid))
    obs_max = float(np.max(valid))
    obs_q01 = float(np.percentile(valid, 1))
    obs_q99 = float(np.percentile(valid, 99))

    if metric_name == "resting_heart_rate":
        # Clinical baseline floor: Human resting HR below ~38 bpm without severe pathology is rare.
        base_floor = max(38.0, obs_q01 - 3.0)
        floor = min(base_floor, obs_min - 0.5)
        base_cap = min(115.0, max(obs_max + 6.0, obs_q99 + 4.0))
        cap = max(base_cap, obs_max + 0.5)
        if cap - floor < 5.0:
            cap = floor + 10.0

    elif metric_name == "running_avg_hr":
        # Aerobic/Anaerobic running: floor ~115 bpm, cap ~195 bpm
        floor = max(110.0, min(obs_min - 5.0, obs_q01 - 3.0))
        cap = min(200.0, max(obs_max + 10.0, obs_q99 + 8.0))
        if cap - floor < 10.0:
            cap = floor + 20.0

    elif metric_name == "gym_avg_hr":
        # Strength training sessions: floor ~80 bpm, cap ~170 bpm
        floor = max(80.0, min(obs_min - 5.0, obs_q01 - 3.0))
        cap = min(175.0, max(obs_max + 10.0, obs_q99 + 8.0))
        if cap - floor < 10.0:
            cap = floor + 20.0

    elif metric_name == "walking_avg_hr":
        # Walking sessions: floor ~75 bpm, cap ~145 bpm
        floor = max(75.0, min(obs_min - 5.0, obs_q01 - 3.0))
        cap = min(150.0, max(obs_max + 8.0, obs_q99 + 6.0))
        if cap - floor < 10.0:
            cap = floor + 15.0

    elif metric_name == "total_sleep_hours":
        # Human sleep duration: minimum 3.5 - 4.0 hrs, max 12.0 - 13.0 hrs
        floor = max(3.5, min(obs_min - 0.5, obs_q01 - 0.5))
        cap = min(13.0, max(obs_max + 1.0, obs_q99 + 1.0))
        if cap - floor < 2.0:
            cap = floor + 3.0

    elif metric_name == "active_kilocalories":
        floor = 0.0
        cap = max(obs_max * 1.5, 3000.0)

    elif metric_name == "resting_kilocalories":
        floor = max(900.0, min(obs_min - 50.0, 1100.0))
        cap = min(2800.0, max(obs_max + 100.0, 2400.0))
        if cap - floor < 100.0:
            cap = floor + 200.0

    elif metric_name == "total_kilocalories":
        floor = max(1100.0, min(obs_min - 100.0, 1400.0))
        cap = max(obs_max * 1.4, 4500.0)

    elif metric_name == "hrv_rmssd":
        # HRV RMSSD (ms): Minimum physiological floor ~15 ms
        floor = max(15.0, min(obs_min * 0.8, obs_q01 - 5.0))
        floor = min(floor, obs_min - 0.5)
        cap = max(obs_max * 1.35, obs_q99 + 15.0)
        if cap - floor < 10.0:
            cap = floor + 15.0

    elif metric_name == "daily_avg_stress":
        # Autonomic stress: Empirical baseline bounds around mu +/- 1.5 sigma
        # Athlete's empirical telemetry ranges between 20 and 36, centering organically around ~26-29.
        obs_mean = float(np.mean(valid))
        obs_std = float(np.std(valid)) if len(valid) > 1 else 3.0
        if np.isnan(obs_std) or obs_std <= 0:
            obs_std = 3.0

        floor = max(18.0, min(obs_min, round(obs_mean - 1.5 * obs_std, 1)))
        cap = min(38.0, max(obs_max, round(obs_mean + 1.5 * obs_std, 1)))
        if cap - floor < 6.0:
            cap = min(40.0, floor + 8.0)

    elif metric_name == "sleep_score":
        # Garmin standard sleep score scale (0-100) with clinical minimum floor
        floor = max(30.0, min(obs_min - 10.0, 50.0))
        cap = 100.0

    elif metric_name == "total_steps":
        floor = 0.0
        cap = max(obs_max * 1.5, 30000.0)

    elif metric_name == "fitness_age":
        # Fitness age (years): bounded by achievable potential floor (~33.5 - 34.0) and chronological age cap (~40.5)
        floor = max(28.0, min(obs_min - 0.5, 33.5))
        cap = max(obs_max + 1.0, 41.0)

    elif metric_name == "fitness_age_gap":
        # Biological rejuvenation gap (years younger): Chronological Age - Fitness Age
        # Empirical gap is ~5.1 - 5.3 years, target potential ~6.1 years
        floor = max(0.0, min(obs_min - 1.0, 3.5))
        cap = max(obs_max + 1.5, 7.5)

    else:
        # Generic heuristic
        if obs_min >= 0:
            floor = max(0.0, obs_min * 0.8)
            floor = min(floor, max(0.0, obs_min - 0.5))
        else:
            floor = obs_min * 1.2
        cap = max(obs_max * 1.25, floor + 5.0)

    return (round(floor, 2), round(cap, 2))


def get_anomaly_dates_for_target(
    df: pd.DataFrame,
    target_col: str,
) -> set[str]:
    """Identify dates with anomalous observations to exclude from baseline model training.

    Combines targeted multi-feature anomaly detection (Isolation Forest / LOF)
    with univariate Tukey IQR filtering on the specific target metric.
    """
    anomaly_dates: set[str] = set()

    # Keyword mapping from target_col to contributing factor keywords
    feature_keywords = {
        "daily_avg_stress": ["stress"],
        "resting_heart_rate": ["resting hr", "heart rate"],
        "running_avg_hr": ["running", "heart rate"],
        "gym_avg_hr": ["gym", "heart rate"],
        "walking_avg_hr": ["walking", "heart rate"],
        "hrv_rmssd": ["hrv", "rmssd"],
        "sleep_score": ["sleep"],
        "total_sleep_hours": ["sleep"],
        "total_steps": ["step", "activity"],
        "active_kilocalories": ["calorie", "cals", "activity"],
        "total_kilocalories": ["calorie", "cals"],
        "resting_kilocalories": ["calorie", "resting"],
    }
    keywords = feature_keywords.get(target_col, [target_col.replace("_", " ")])

    # 1. Multi-feature anomaly detection (targeted to target_col)
    try:
        from src.analytics.anomaly_detection import PhysiologicalAnomalyDetector

        detector = PhysiologicalAnomalyDetector()
        detected_points = detector.detect(df)
        for pt in detected_points:
            if pt.is_anomaly:
                # Attribute anomaly to target_col only if target metric is a contributing factor
                is_target_affected = any(
                    any(kw in factor.lower() for kw in keywords)
                    for factor in pt.contributing_factors
                )
                if is_target_affected:
                    anomaly_dates.add(pt.calendar_date)
    except Exception as exc:
        logger.debug(f"Multi-feature anomaly detection skipped during TS prep: {exc}")

    # 2. Univariate Tukey IQR rule on target_col (excluding slow-moving adaptation metrics)
    if target_col in df.columns and "calendar_date" in df.columns and target_col not in ("fitness_age", "fitness_age_gap"):
        vals = pd.to_numeric(df[target_col], errors="coerce")
        q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:
            low_cutoff = q25 - 2.5 * iqr
            high_cutoff = q75 + 2.5 * iqr
            target_outliers = df[vals.lt(low_cutoff) | vals.gt(high_cutoff)][
                "calendar_date"
            ].astype(str)
            anomaly_dates.update(target_outliers.tolist())

    return anomaly_dates


@dataclass
class ForecastResult:
    """Encapsulates time series forecasting outputs and validation metrics."""

    target: str
    forecast_df: pd.DataFrame
    metrics: dict[str, float]
    model_name: str = "Prophet"
    bounds: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert forecast summary to dictionary."""
        tail = self.forecast_df.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return {
            "target": self.target,
            "model": self.model_name,
            "bounds": {"floor": self.bounds[0], "cap": self.bounds[1]} if self.bounds else None,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "forecast_sample": tail.to_dict(orient="records"),
        }


class GarminProphetForecaster:
    """Bayesian time series forecasting engine using Meta Prophet.

    Supports:
    - Anomaly filtering (masks outlier days to avoid skewing baseline dynamics)
    - Logistic growth with physiological and personalized saturation bounds [floor, cap]
    - Post-prediction clamping ensuring outputs strictly respect biological viability.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False,
        growth: Literal["linear", "logistic", "flat"] = "logistic",
        bounds: tuple[float, float] | None = None,
        filter_anomalies: bool = True,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.bounds = bounds
        self.filter_anomalies = filter_anomalies

        self.model: Prophet | None = None
        self.target_col: str | None = None
        self.regressors: list[str] = []
        self.is_fitted: bool = False
        self.masked_anomalies_: list[str] = []

    def _prepare_prophet_df(
        self,
        df: pd.DataFrame,
        target_col: str,
        regressors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Format input DataFrame into Prophet standard columns (ds, y, floor, cap)."""
        if "calendar_date" not in df.columns:
            raise ValueError("Input DataFrame must contain 'calendar_date' column.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        cols_to_keep = ["calendar_date", target_col]
        active_regressors = []
        if regressors:
            for reg in regressors:
                if reg in df.columns and reg != target_col:
                    cols_to_keep.append(reg)
                    active_regressors.append(reg)

        p_df = df[cols_to_keep].copy()
        p_df["ds"] = pd.to_datetime(p_df["calendar_date"])
        p_df["y"] = pd.to_numeric(p_df[target_col], errors="coerce")

        # Sort chronologically
        p_df = p_df.sort_values("ds").reset_index(drop=True)

        # 1. Anomaly Filtering
        if self.filter_anomalies:
            anomaly_dates = get_anomaly_dates_for_target(df, target_col)
            # Only mask if we don't end up masking more than 50% of the data
            if 0 < len(anomaly_dates) < len(p_df) * 0.5:
                mask = p_df["calendar_date"].astype(str).isin(anomaly_dates)
                self.masked_anomalies_ = p_df.loc[mask, "calendar_date"].astype(str).tolist()
                p_df.loc[mask, "y"] = np.nan
                logger.info(
                    f"Prophet: Masked {len(self.masked_anomalies_)} anomalous days for '{target_col}'."
                )

        # 2. Physiological Bounds
        if self.bounds is None:
            self.bounds = compute_physiological_bounds(df[target_col], target_col)

        floor, cap = self.bounds
        p_df["floor"] = floor
        p_df["cap"] = cap

        # For non-null training observations, ensure strict inclusion in (floor, cap) for logistic logit domain
        valid_mask = p_df["y"].notna()
        if valid_mask.any():
            p_df.loc[valid_mask, "y"] = p_df.loc[valid_mask, "y"].clip(
                lower=floor + 0.1, upper=cap - 0.1
            )

        for reg in active_regressors:
            p_df[reg] = pd.to_numeric(p_df[reg], errors="coerce").ffill().bfill()

        self.regressors = active_regressors
        return p_df.drop(columns=["calendar_date"])

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        regressors: list[str] | None = None,
    ) -> GarminProphetForecaster:
        """Fit Prophet forecasting model on historical data with bounds and anomaly filtering."""
        p_df = self._prepare_prophet_df(df, target_col, regressors)
        self.target_col = target_col

        # Weekly seasonality in Prophet requires >= 14 observations; disable for ultra-short series
        effective_weekly = self.weekly_seasonality
        if len(p_df) < 14 and effective_weekly:
            effective_weekly = False
            logger.info(
                f"Prophet: Disabled weekly seasonality for '{target_col}' (observations={len(p_df)} < 14)."
            )

        effective_growth = (
            "flat" if target_col == "daily_avg_stress" and self.growth == "logistic" else self.growth
        )

        self.model = Prophet(
            growth=effective_growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            weekly_seasonality=effective_weekly,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        for reg in self.regressors:
            self.model.add_regressor(reg)

        self.model.fit(p_df)
        self.is_fitted = True
        logger.info(
            f"Prophet forecaster fitted for target '{target_col}' (growth={effective_growth}, "
            f"bounds={self.bounds}, anomalies_masked={len(self.masked_anomalies_)})."
        )
        return self

    def predict(
        self,
        periods: int = 7,
        freq: str = "D",
        future_regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts over future horizon bounded by physiological constraints."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        if self.bounds:
            future["floor"] = self.bounds[0]
            future["cap"] = self.bounds[1]

        if self.regressors:
            if future_regressors is not None:
                for reg in self.regressors:
                    if reg in future_regressors.columns:
                        future[reg] = future_regressors[reg].values
                    else:
                        future[reg] = 0.0
            else:
                for reg in self.regressors:
                    future[reg] = 0.0

        forecast = self.model.predict(future)

        # Enforce physiological bounds strictly on predictions and confidence intervals
        if self.bounds:
            floor, cap = self.bounds
            forecast["yhat"] = forecast["yhat"].clip(lower=floor, upper=cap)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=floor, upper=cap)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=floor, upper=cap)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate_chronological(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: int = 3,
    ) -> dict[str, float]:
        """Perform chronological train/test split evaluation."""
        if len(df) <= test_size + 2:
            test_size = max(1, len(df) // 4)

        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()

        eval_forecaster = GarminProphetForecaster(
            growth=self.growth,
            bounds=self.bounds or compute_physiological_bounds(df[target_col], target_col),
            filter_anomalies=self.filter_anomalies,
            weekly_seasonality=False,
            yearly_seasonality=False,
            daily_seasonality=False,
        )
        eval_forecaster.fit(train_df, target_col)

        forecast = eval_forecaster.predict(periods=test_size, freq="D")

        y_true = test_df[target_col].to_numpy(dtype=float)
        y_pred = forecast.tail(test_size)["yhat"].to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

        metrics = {"mae": mae, "rmse": rmse, "mape_pct": mape}
        logger.info(
            f"Chronological eval for {target_col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%"
        )
        return metrics


class HoltWintersForecaster:
    """Holt-Winters Exponential Smoothing (ETS) with Damped Trend and Physiological Clamping.

    Particularly suited for biological metrics due to its damped trend (capturing homeostatic
    equilibrium and preventing runaway linear divergence).
    """

    def __init__(
        self,
        bounds: tuple[float, float] | None = None,
        filter_anomalies: bool = True,
        damped_trend: bool = True,
    ) -> None:
        self.bounds = bounds
        self.filter_anomalies = filter_anomalies
        self.damped_trend = damped_trend

        self.model: Any = None
        self.fitted_res_: Any = None
        self.target_col: str | None = None
        self.is_fitted: bool = False
        self.last_date_: pd.Timestamp | None = None
        self.history_dates_: pd.DatetimeIndex | None = None
        self.residual_sigma_: float = 1.0

    def fit(self, df: pd.DataFrame, target_col: str) -> HoltWintersForecaster:
        """Fit Holt-Winters Exponential Smoothing model."""
        if "calendar_date" not in df.columns:
            raise ValueError("Input DataFrame must contain 'calendar_date' column.")
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not found in DataFrame.")

        self.target_col = target_col
        sub_df = df[["calendar_date", target_col]].copy()
        sub_df["ds"] = pd.to_datetime(sub_df["calendar_date"])
        sub_df = sub_df.sort_values("ds").reset_index(drop=True)

        if self.bounds is None:
            self.bounds = compute_physiological_bounds(sub_df[target_col], target_col)

        # Anomaly handling: interpolate over flagged anomaly dates
        y_vals = pd.to_numeric(sub_df[target_col], errors="coerce").copy()
        if self.filter_anomalies:
            anomaly_dates = get_anomaly_dates_for_target(df, target_col)
            if 0 < len(anomaly_dates) < len(sub_df) * 0.5:
                mask = sub_df["calendar_date"].astype(str).isin(anomaly_dates)
                y_vals.loc[mask] = np.nan
                logger.info(
                    f"Holt-Winters: Interpolating {len(anomaly_dates)} anomaly dates for '{target_col}'."
                )

        # Interpolate missing values cleanly
        y_clean = y_vals.interpolate(method="linear").ffill().bfill().to_numpy(dtype=float)

        # Weekly seasonality if >= 14 observations
        seasonal = "add" if len(y_clean) >= 14 else None
        seasonal_periods = 7 if seasonal else None
        trend = None if target_col == "daily_avg_stress" else "add"
        damped = self.damped_trend if trend is not None else False

        self.model = ExponentialSmoothing(
            y_clean,
            trend=trend,
            damped_trend=damped,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        self.fitted_res_ = self.model.fit(optimized=True)
        self.is_fitted = True
        self.last_date_ = sub_df["ds"].iloc[-1]
        self.history_dates_ = pd.DatetimeIndex(sub_df["ds"])

        residuals = y_clean - self.fitted_res_.fittedvalues
        sigma = float(np.nanstd(residuals))
        self.residual_sigma_ = sigma if sigma > 1e-4 else 1.0

        logger.info(
            f"HoltWintersForecaster fitted on {len(y_clean)} days for '{target_col}' "
            f"(bounds={self.bounds}, residual_sigma={self.residual_sigma_:.2f})."
        )
        return self

    def predict(self, periods: int = 7, freq: str = "D") -> pd.DataFrame:
        """Generate forecasts and prediction intervals bounded by physiological limits."""
        if not self.is_fitted or self.fitted_res_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        raw_forecast = self.fitted_res_.forecast(periods)
        future_dates = pd.date_range(
            start=self.last_date_ + pd.Timedelta(days=1), periods=periods, freq=freq
        )

        # In-sample fitted
        hist_fitted = self.fitted_res_.fittedvalues
        all_ds = (list(self.history_dates_) if self.history_dates_ is not None else []) + list(
            future_dates
        )
        all_yhat = np.concatenate([hist_fitted, raw_forecast])

        # Standard error expansion over horizon
        future_steps = np.arange(1, periods + 1)
        future_se = self.residual_sigma_ * np.sqrt(future_steps)
        future_lower = raw_forecast - 1.96 * future_se
        future_upper = raw_forecast + 1.96 * future_se

        hist_lower = hist_fitted - 1.96 * self.residual_sigma_
        hist_upper = hist_fitted + 1.96 * self.residual_sigma_

        all_lower = np.concatenate([hist_lower, future_lower])
        all_upper = np.concatenate([hist_upper, future_upper])

        forecast_df = pd.DataFrame(
            {
                "ds": all_ds,
                "yhat": all_yhat,
                "yhat_lower": all_lower,
                "yhat_upper": all_upper,
            }
        )

        # Clamp to bounds
        if self.bounds:
            floor, cap = self.bounds
            forecast_df["yhat"] = forecast_df["yhat"].clip(lower=floor, upper=cap)
            forecast_df["yhat_lower"] = forecast_df["yhat_lower"].clip(lower=floor, upper=cap)
            forecast_df["yhat_upper"] = forecast_df["yhat_upper"].clip(lower=floor, upper=cap)

        return forecast_df

    def evaluate_chronological(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: int = 3,
    ) -> dict[str, float]:
        """Perform chronological train/test split evaluation."""
        if len(df) <= test_size + 2:
            test_size = max(1, len(df) // 4)

        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()

        eval_model = HoltWintersForecaster(
            bounds=self.bounds or compute_physiological_bounds(df[target_col], target_col),
            filter_anomalies=self.filter_anomalies,
            damped_trend=self.damped_trend,
        )
        eval_model.fit(train_df, target_col)
        forecast = eval_model.predict(periods=test_size, freq="D")

        y_true = test_df[target_col].to_numpy(dtype=float)
        y_pred = forecast.tail(test_size)["yhat"].to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

        metrics = {"mae": mae, "rmse": rmse, "mape_pct": mape}
        logger.info(
            f"Holt-Winters chronological eval for {target_col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%"
        )
        return metrics


class EnsembleBiometricForecaster:
    """Ensemble Forecaster combining Meta Prophet and Holt-Winters Damped ETS.

    Combines the flexibility of Bayesian structural time-series (changepoints, calendar patterns)
    with the homeostatic stability of damped exponential smoothing.
    """

    def __init__(
        self,
        bounds: tuple[float, float] | None = None,
        filter_anomalies: bool = True,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        self.bounds = bounds
        self.filter_anomalies = filter_anomalies
        self.weights = weights

        self.prophet = GarminProphetForecaster(
            bounds=self.bounds,
            filter_anomalies=self.filter_anomalies,
        )
        self.holt_winters = HoltWintersForecaster(
            bounds=self.bounds,
            filter_anomalies=self.filter_anomalies,
        )
        self.is_fitted = False
        self.target_col: str | None = None

    def fit(self, df: pd.DataFrame, target_col: str) -> EnsembleBiometricForecaster:
        """Fit both Prophet and Holt-Winters forecasters."""
        self.target_col = target_col
        if self.bounds is None:
            self.bounds = compute_physiological_bounds(df[target_col], target_col)
            self.prophet.bounds = self.bounds
            self.holt_winters.bounds = self.bounds

        self.prophet.fit(df, target_col)
        self.holt_winters.fit(df, target_col)
        self.is_fitted = True
        logger.info(
            f"EnsembleBiometricForecaster fitted for '{target_col}' with weights={self.weights}."
        )
        return self

    def predict(self, periods: int = 7, freq: str = "D") -> pd.DataFrame:
        """Generate weighted ensemble predictions bounded by physiological limits."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting.")

        p_df = self.prophet.predict(periods=periods, freq=freq)
        hw_df = self.holt_winters.predict(periods=periods, freq=freq)

        # Align by ds
        merged = pd.merge(p_df, hw_df, on="ds", suffixes=("_prophet", "_hw")).sort_values("ds")

        w_p, w_hw = self.weights
        total_w = w_p + w_hw
        w_p, w_hw = w_p / total_w, w_hw / total_w

        merged["yhat"] = w_p * merged["yhat_prophet"] + w_hw * merged["yhat_hw"]
        merged["yhat_lower"] = w_p * merged["yhat_lower_prophet"] + w_hw * merged["yhat_lower_hw"]
        merged["yhat_upper"] = w_p * merged["yhat_upper_prophet"] + w_hw * merged["yhat_upper_hw"]

        if self.bounds:
            floor, cap = self.bounds
            merged["yhat"] = merged["yhat"].clip(lower=floor, upper=cap)
            merged["yhat_lower"] = merged["yhat_lower"].clip(lower=floor, upper=cap)
            merged["yhat_upper"] = merged["yhat_upper"].clip(lower=floor, upper=cap)

        return merged[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate_chronological(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: int = 3,
    ) -> dict[str, float]:
        """Chronological train/test split evaluation for ensemble."""
        if len(df) <= test_size + 2:
            test_size = max(1, len(df) // 4)

        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()

        eval_ensemble = EnsembleBiometricForecaster(
            bounds=self.bounds or compute_physiological_bounds(df[target_col], target_col),
            filter_anomalies=self.filter_anomalies,
            weights=self.weights,
        )
        eval_ensemble.fit(train_df, target_col)
        forecast = eval_ensemble.predict(periods=test_size, freq="D")

        y_true = test_df[target_col].to_numpy(dtype=float)
        y_pred = forecast.tail(test_size)["yhat"].to_numpy(dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

        metrics = {"mae": mae, "rmse": rmse, "mape_pct": mape}
        logger.info(
            f"Ensemble chronological eval for {target_col}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%"
        )
        return metrics


class SleepRecoveryPredictor:
    """Predictive model for overnight sleep quality and recovery readiness."""

    DEFAULT_FEATURES = [
        "prev_day_steps",
        "prev_day_stress",
        "prev_day_active_cals",
        "deep_sleep_ratio",
        "hrv_to_baseline_ratio",
        "stress_balance_ratio",
        "acwr_steps",
    ]

    def __init__(
        self,
        features: list[str] | None = None,
        model_type: str = "rf",
        random_state: int = 42,
    ) -> None:
        self.features = features or self.DEFAULT_FEATURES
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()

        if model_type == "gb":
            self.model: Any = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=50, max_depth=4, random_state=random_state
            )

        self.is_fitted = False
        self.feature_names_: list[str] = []
        self.feature_importances_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "sleep_score") -> SleepRecoveryPredictor:
        """Fit recovery prediction model."""
        available_cols = [c for c in self.features if c in df.columns]
        if not available_cols:
            raise ValueError(f"None of features {self.features} found in DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not in DataFrame.")

        X = df[available_cols].ffill().bfill().to_numpy(dtype=float)
        y = df[target_col].ffill().bfill().to_numpy(dtype=float)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_names_ = available_cols

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = {
                available_cols[i]: float(self.model.feature_importances_[i])
                for i in range(len(available_cols))
            }

        logger.info(
            f"SleepRecoveryPredictor trained on {len(df)} samples predicting '{target_col}'."
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict recovery score for new instances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = df[self.feature_names_].ffill().bfill().to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)
        return np.asarray(self.model.predict(X_scaled), dtype=float)


def main() -> None:
    """CLI runner for biometric forecasting and recovery prediction."""
    parser = argparse.ArgumentParser(
        description="Garmin Bounded Time Series Forecasting & Recovery Prediction."
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("data/processed/garmin_ml_features.parquet"),
        help="Path to feature parquet file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="resting_heart_rate",
        choices=[
            "resting_heart_rate",
            "hrv_rmssd",
            "daily_avg_stress",
            "sleep_score",
            "total_steps",
        ],
        help="Target metric to forecast.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["prophet", "holt_winters", "ensemble"],
        help="Forecasting algorithm to execute.",
    )
    parser.add_argument("--days", type=int, default=7, help="Horizon days to forecast.")
    parser.add_argument(
        "--no-filter-anomalies",
        dest="filter_anomalies",
        action="store_false",
        default=True,
        help="Disable automatic anomaly filtering before fitting.",
    )
    parser.add_argument(
        "--predict-sleep", action="store_true", help="Run sleep recovery regression."
    )

    args = parser.parse_args()

    file_path: Path = args.features_file
    if not file_path.exists():
        csv_path = file_path.with_suffix(".csv")
        if csv_path.exists():
            file_path = csv_path
        else:
            print(f"Error: features file not found: {file_path}")
            sys.exit(1)

    df = pd.read_parquet(file_path) if file_path.suffix == ".parquet" else pd.read_csv(file_path)

    bounds = compute_physiological_bounds(df[args.target], args.target)

    print("\n" + "=" * 70)
    print("📈 Garmin Biometric Forecasting Engine")
    print(f"   • Target:       {args.target}")
    print(f"   • Model:        {args.model.upper()}")
    print(f"   • Bounds:       [Floor: {bounds[0]}, Cap: {bounds[1]}]")
    print(f"   • Filter Anom:  {args.filter_anomalies}")
    print("=" * 70)

    # Instantiate forecaster based on CLI choice
    if args.model == "prophet":
        forecaster: Any = GarminProphetForecaster(
            bounds=bounds, filter_anomalies=args.filter_anomalies
        )
    elif args.model == "holt_winters":
        forecaster = HoltWintersForecaster(bounds=bounds, filter_anomalies=args.filter_anomalies)
    else:
        forecaster = EnsembleBiometricForecaster(
            bounds=bounds, filter_anomalies=args.filter_anomalies
        )

    metrics = forecaster.evaluate_chronological(df, target_col=args.target, test_size=2)
    print(
        f"Validation Performance: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape_pct']:.1f}%"
    )

    forecaster.fit(df, target_col=args.target)
    future_preds = forecaster.predict(periods=args.days)

    print(f"\n🔮 Next {args.days} Days Forecast ({args.target}):")
    tail = future_preds.tail(args.days)
    for _, row in tail.iterrows():
        date_str = str(row["ds"])[:10]
        yhat = row["yhat"]
        low, high = row["yhat_lower"], row["yhat_upper"]
        print(f"  • {date_str}: {yhat:6.1f}  (CI: [{low:5.1f}, {high:5.1f}])")

    if args.predict_sleep and "sleep_score" in df.columns:
        print("\n" + "-" * 70)
        print("🛌 Sleep Recovery Predictor (Random Forest Feature Importances):")
        predictor = SleepRecoveryPredictor()
        predictor.fit(df, target_col="sleep_score")
        sorted_imp = sorted(
            predictor.feature_importances_.items(), key=lambda x: x[1], reverse=True
        )
        for feat, imp in sorted_imp:
            print(f"  • {feat:25s}: {imp * 100:5.1f}%")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
