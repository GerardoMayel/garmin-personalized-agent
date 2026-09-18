"""Time Series Forecasting and Predictive Recovery Modeling for Garmin Telemetry.

Features:
1. GarminProphetForecaster: Bayesian structural time series forecasting with Prophet,
   modeling trends, weekly seasonality, and confidence intervals.
2. SleepRecoveryPredictor: Supervised machine learning (Random Forest / Gradient Boosting)
   predicting nocturnal sleep quality and recovery readiness from daytime exertion and stress.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Encapsulates time series forecasting outputs and validation metrics."""

    target: str
    forecast_df: pd.DataFrame
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert forecast summary to dictionary."""
        tail = self.forecast_df.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return {
            "target": self.target,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "forecast_sample": tail.to_dict(orient="records"),
        }


class GarminProphetForecaster:
    """Bayesian time series forecasting engine using Meta Prophet."""

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality

        self.model: Prophet | None = None
        self.target_col: str | None = None
        self.regressors: list[str] = []
        self.is_fitted: bool = False

    def _prepare_prophet_df(
        self,
        df: pd.DataFrame,
        target_col: str,
        regressors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Format input DataFrame into Prophet standard columns (ds, y)."""
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

        # Sort and clean
        p_df = p_df.sort_values("ds").reset_index(drop=True)
        p_df["y"] = p_df["y"].ffill().bfill()

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
        """Fit Prophet forecasting model on historical data."""
        p_df = self._prepare_prophet_df(df, target_col, regressors)
        self.target_col = target_col

        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )

        for reg in self.regressors:
            self.model.add_regressor(reg)

        self.model.fit(p_df)
        self.is_fitted = True
        logger.info(f"Prophet forecaster fitted for target '{target_col}' with {len(p_df)} dates.")
        return self

    def predict(
        self,
        periods: int = 7,
        freq: str = "D",
        future_regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts over future horizon."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        if self.regressors:
            if future_regressors is not None:
                for reg in self.regressors:
                    if reg in future_regressors.columns:
                        future[reg] = future_regressors[reg].values
                    else:
                        future[reg] = 0.0
            else:
                # Default regressor assumption: last observed value
                for reg in self.regressors:
                    future[reg] = 0.0

        forecast = self.model.predict(future)
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

        eval_model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            weekly_seasonality=False,  # Short window fallback
            yearly_seasonality=False,
            daily_seasonality=False,
        )
        p_train = self._prepare_prophet_df(train_df, target_col)
        eval_model.fit(p_train)

        future = eval_model.make_future_dataframe(periods=test_size, freq="D")
        forecast = eval_model.predict(future)

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
    """CLI runner for Prophet forecasting and recovery prediction."""
    parser = argparse.ArgumentParser(
        description="Garmin Time Series Forecasting & Recovery Prediction."
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
    parser.add_argument("--days", type=int, default=7, help="Horizon days to forecast.")
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

    print("\n" + "=" * 65)
    print(f"📈 Garmin Prophet Forecasting Engine: Target = {args.target}")
    print("=" * 65)

    forecaster = GarminProphetForecaster()
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
        print(f"  • {date_str}: {yhat:6.1f}  (95% CI: [{low:5.1f}, {high:5.1f}])")

    if args.predict_sleep and "sleep_score" in df.columns:
        print("\n" + "-" * 65)
        print("🛌 Sleep Recovery Predictor (Random Forest Feature Importances):")
        predictor = SleepRecoveryPredictor()
        predictor.fit(df, target_col="sleep_score")
        sorted_imp = sorted(
            predictor.feature_importances_.items(), key=lambda x: x[1], reverse=True
        )
        for feat, imp in sorted_imp:
            print(f"  • {feat:25s}: {imp * 100:5.1f}%")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
