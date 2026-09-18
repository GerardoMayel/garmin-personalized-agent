"""Biometric Rolling Bi-Weekly Predictions Manager for Garmin Telemetry.

Features:
1. Multi-metric horizon projection:
   - Heart rates: resting, running, gym, walking.
   - Sleep: total sleep hours, sleep score.
   - Calories: active, resting, total.
   - Autonomic & load: steps, daily stress, HRV.
2. Horizon calculation:
   - Remainder of current week (today through Sunday).
   - Entire subsequent week (Monday through Sunday).
3. Strict Immutability & Lock Rule:
   - Once a prediction is recorded for a given (target_date, metric), it is NEVER overwritten.
   - Subsequent runs (daily/weekly) only predict and append missing future dates.
4. Storage:
   - Local: data/processed/predictions/weekly_biometric_forecasts.parquet and .csv.
   - Cloud: Cloudflare R2 under processed/predictions/.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.time_series_models import (
    EnsembleBiometricForecaster,
    compute_physiological_bounds,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DVC_PATH = Path("data/dvc/garmin_clean_features.parquet")
DEFAULT_PRED_DIR = Path("data/processed/predictions")
DEFAULT_PRED_PARQUET = DEFAULT_PRED_DIR / "weekly_biometric_forecasts.parquet"
DEFAULT_PRED_CSV = DEFAULT_PRED_DIR / "weekly_biometric_forecasts.csv"

SUPPORTED_PREDICTION_METRICS = [
    # Heart Rates
    "resting_heart_rate",
    "running_avg_hr",
    "gym_avg_hr",
    "walking_avg_hr",
    # Sleep
    "total_sleep_hours",
    "sleep_score",
    # Calories
    "active_kilocalories",
    "resting_kilocalories",
    "total_kilocalories",
    # Load & Autonomic
    "total_steps",
    "daily_avg_stress",
    "hrv_rmssd",
]


def get_biweekly_target_dates(current_date: date | None = None) -> list[date]:
    """Compute the target date horizon: remainder of current week + entire next week.

    Returns:
        list[date]: Dates from tomorrow up to the Sunday of next week.
    """
    base_date = current_date or date.today()

    # Days left in current week (Monday=0 ... Sunday=6)
    days_to_sunday = 6 - base_date.weekday()
    current_week_sunday = base_date + timedelta(days=days_to_sunday)
    next_week_sunday = current_week_sunday + timedelta(days=7)

    # We forecast starting from tomorrow (or today if base_date is historical)
    start_forecast = base_date + timedelta(days=1)
    if start_forecast > next_week_sunday:
        return []

    target_dates = []
    curr = start_forecast
    while curr <= next_week_sunday:
        target_dates.append(curr)
        curr += timedelta(days=1)

    return target_dates


class BiometricPredictionsManager:
    """Manages generation, immutability, and persistence of rolling biometric forecasts."""

    def __init__(
        self,
        features_file: Path = DEFAULT_DVC_PATH,
        predictions_dir: Path = DEFAULT_PRED_DIR,
    ) -> None:
        self.features_file = Path(features_file)
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.predictions_dir / "weekly_biometric_forecasts.parquet"
        self.csv_path = self.predictions_dir / "weekly_biometric_forecasts.csv"

    def load_existing_predictions(self) -> pd.DataFrame:
        """Load registered predictions table if present, else empty structured DataFrame."""
        if self.parquet_path.exists():
            return pd.read_parquet(self.parquet_path)

        return pd.DataFrame(
            columns=[
                "forecast_generated_date",
                "target_date",
                "metric",
                "predicted_value",
                "ci_lower",
                "ci_upper",
                "model_name",
                "is_locked",
            ]
        )

    def generate_and_update_forecasts(
        self,
        metrics: list[str] | None = None,
        reference_date: date | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for unrecorded horizon dates and lock them into the table.

        Existing predictions are preserved and never overwritten.
        """
        if not self.features_file.exists():
            raise FileNotFoundError(f"Clean features file not found: {self.features_file}")

        df_history = pd.read_parquet(self.features_file)
        df_history["calendar_date"] = pd.to_datetime(df_history["calendar_date"])
        df_history = df_history.sort_values("calendar_date").reset_index(drop=True)

        latest_history_date = df_history["calendar_date"].iloc[-1].date()
        effective_base_date = reference_date or latest_history_date

        target_dates = get_biweekly_target_dates(effective_base_date)
        if not target_dates:
            logger.warning("No target dates computed for bi-weekly horizon.")
            return self.load_existing_predictions()

        active_metrics = metrics or SUPPORTED_PREDICTION_METRICS
        existing_df = self.load_existing_predictions()

        # Set of already locked (target_date, metric)
        if not existing_df.empty:
            locked_keys = set(
                zip(
                    existing_df["target_date"].astype(str),
                    existing_df["metric"].astype(str),
                    strict=False,
                )
            )
        else:
            locked_keys = set()

        new_records: list[dict[str, Any]] = []
        gen_date_str = str(effective_base_date)

        # Horizon length in days
        horizon_days = (target_dates[-1] - effective_base_date).days

        logger.info(
            f"Evaluating forecasts from {target_dates[0]} to {target_dates[-1]} ({len(target_dates)} dates)."
        )

        for metric in active_metrics:
            if metric not in df_history.columns:
                logger.warning(f"Metric '{metric}' not in clean features DataFrame; skipping.")
                continue

            # Check if all target dates are already locked for this metric
            unrecorded_dates = [d for d in target_dates if (str(d), metric) not in locked_keys]
            if not unrecorded_dates:
                logger.info(
                    f"Metric '{metric}': All {len(target_dates)} horizon dates already locked. Skipping recalculation."
                )
                continue

            bounds = compute_physiological_bounds(df_history[metric], metric)
            ensemble = EnsembleBiometricForecaster(bounds=bounds, filter_anomalies=True)

            try:
                ensemble.fit(df_history, target_col=metric)
                pred_df = ensemble.predict(periods=horizon_days)
                pred_df["ds_date"] = pd.to_datetime(pred_df["ds"]).dt.date

                # Extract only the unrecorded target dates
                for t_date in unrecorded_dates:
                    match_row = pred_df[pred_df["ds_date"] == t_date]
                    if not match_row.empty:
                        row = match_row.iloc[0]
                        new_records.append(
                            {
                                "forecast_generated_date": gen_date_str,
                                "target_date": str(t_date),
                                "metric": metric,
                                "predicted_value": round(float(row["yhat"]), 2),
                                "ci_lower": round(float(row["yhat_lower"]), 2),
                                "ci_upper": round(float(row["yhat_upper"]), 2),
                                "model_name": "Ensemble_Prophet_HoltWinters",
                                "is_locked": True,
                            }
                        )
            except Exception as exc:
                logger.error(f"Failed to generate forecast for '{metric}': {exc}", exc_info=True)

        if new_records:
            new_df = pd.DataFrame(new_records)
            combined_df = (
                pd.concat([existing_df, new_df], ignore_index=True)
                .sort_values(["metric", "target_date"])
                .reset_index(drop=True)
            )
            logger.info(
                f"PredictionsManager: Appended {len(new_records)} newly locked predictions across {len(active_metrics)} metrics."
            )
        else:
            combined_df = existing_df
            logger.info("PredictionsManager: No new predictions needed (all dates locked).")

        # Persist locally
        combined_df.to_parquet(self.parquet_path, index=False)
        combined_df.to_csv(self.csv_path, index=False)
        return combined_df


def main() -> None:
    """CLI runner for biometric predictions manager."""
    parser = argparse.ArgumentParser(
        description="Garmin Biometric Rolling Bi-Weekly Predictions Manager."
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=DEFAULT_DVC_PATH,
        help="Path to clean DVC parquet file.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Output directory for predictions.",
    )
    parser.add_argument("--sync-r2", action="store_true", help="Sync to Cloudflare R2 bucket.")

    args = parser.parse_args()

    manager = BiometricPredictionsManager(
        features_file=args.features_file,
        predictions_dir=args.predictions_dir,
    )
    preds_df = manager.generate_and_update_forecasts()

    print("\n" + "=" * 75)
    print("📈 Garmin Rolling Bi-Weekly Locked Predictions Table")
    print(f"   • Total Forecast Records: {len(preds_df)}")
    print(f"   • Metrics Count:          {preds_df['metric'].nunique()}")
    print(
        f"   • Target Dates Span:      {preds_df['target_date'].min()} to {preds_df['target_date'].max()}"
    )
    print(f"   • Parquet Saved to:       {args.predictions_dir / 'weekly_biometric_forecasts.parquet'}")
    print(f"   • CSV Saved to:           {args.predictions_dir / 'weekly_biometric_forecasts.csv'}")
    print("=" * 75)

    # Sample display
    print("\n📋 Sample Forecast Preview (Next 5 entries per metric):")
    sample_preview = preds_df.groupby("metric").head(2)[
        ["target_date", "metric", "predicted_value", "ci_lower", "ci_upper", "is_locked"]
    ]
    print(sample_preview.to_string(index=False))
    print("=" * 75 + "\n")

    if args.sync_r2:
        try:
            from src.common.r2_storage import R2StorageClient

            r2 = R2StorageClient()
            if r2.is_configured():
                print("☁️ Syncing predictions to Cloudflare R2...")
                r2.sync_predictions(predictions_dir=args.predictions_dir)
                print("✅ Predictions sync to Cloudflare R2 complete.")
            else:
                print("⚠️ Cloudflare R2 not configured; skipping cloud sync.")
        except Exception as e:
            print(f"❌ Error syncing predictions to R2: {e}")


if __name__ == "__main__":
    main()
