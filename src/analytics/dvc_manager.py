"""DVC Clean Dataset Manager for Garmin Biometric Telemetry.

Extracts, imputes, and version-controls a single immutable, clean dataset
from the raw and processed SQLite database, ensuring:
1. Append-only history: new dates are added without modifying/overwriting past validated records.
2. Derivation of activity-specific heart rates (running, gym, walking).
3. Split caloric expenditure (active vs resting vs total).
4. Sleep duration (hours) and quality metrics.
5. Autonomic recovery indicators (stress, HRV RMSSD).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("data/processed/garmin_history.db")
DEFAULT_DVC_DIR = Path("data/dvc")
CLEAN_PARQUET_PATH = DEFAULT_DVC_DIR / "garmin_clean_features.parquet"
CLEAN_CSV_PATH = DEFAULT_DVC_DIR / "garmin_clean_features.csv"


class GarminDVCManager:
    """Manages the clean, versioned DVC dataset extracted from Garmin SQLite telemetry."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        dvc_dir: Path = DEFAULT_DVC_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.dvc_dir = Path(dvc_dir)
        self.dvc_dir.mkdir(parents=True, exist_ok=True)

    def extract_clean_dataframe(self) -> pd.DataFrame:
        """Extract and merge all biometric, activity, sleep, stress, and calorie records."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)

        query = """
        WITH daily_act AS (
            SELECT
                calendar_date,
                COUNT(*) AS daily_activity_count,
                SUM(duration_seconds) AS total_activity_duration_sec,
                SUM(distance_meters) AS total_activity_distance_m,
                SUM(calories) AS total_activity_calories,
                AVG(avg_hr) AS avg_activity_hr,
                MAX(max_hr) AS max_activity_hr
            FROM activities
            GROUP BY calendar_date
        )
        SELECT
            d.calendar_date,
            d.total_steps,
            d.total_distance_meters,
            d.active_kilocalories,
            d.total_kilocalories,
            d.resting_heart_rate,
            d.min_heart_rate,
            d.max_heart_rate,
            d.avg_stress_level AS daily_avg_stress,
            d.max_stress_level AS daily_max_stress,
            d.vigorous_minutes,
            d.moderate_minutes,
            s.sleep_score,
            s.total_sleep_seconds,
            s.deep_sleep_seconds,
            s.light_sleep_seconds,
            s.rem_sleep_seconds,
            s.awake_sleep_seconds,
            s.avg_spo2,
            s.lowest_spo2,
            s.avg_respiration,
            s.avg_sleep_stress,
            h.last_night_avg AS hrv_rmssd,
            h.weekly_avg AS hrv_weekly_avg,
            h.baseline_low AS hrv_baseline_low,
            h.baseline_balanced_low AS hrv_baseline_balanced_low,
            h.baseline_balanced_upper AS hrv_baseline_balanced_upper,
            st.rest_stress_duration_sec,
            st.activity_stress_duration_sec,
            st.low_stress_duration_sec,
            st.medium_stress_duration_sec,
            st.high_stress_duration_sec,
            m.vo2_max_running,
            m.fitness_age,
            COALESCE(a.daily_activity_count, 0) AS daily_activity_count,
            COALESCE(a.total_activity_duration_sec, 0) AS total_activity_duration_sec,
            COALESCE(a.total_activity_distance_m, 0) AS total_activity_distance_m,
            COALESCE(a.total_activity_calories, 0) AS total_activity_calories,
            a.avg_activity_hr,
            a.max_activity_hr
        FROM daily_summaries d
        LEFT JOIN sleep_records s ON d.calendar_date = s.calendar_date
        LEFT JOIN hrv_records h ON d.calendar_date = h.calendar_date
        LEFT JOIN stress_records st ON d.calendar_date = st.calendar_date
        LEFT JOIN max_metrics m ON d.calendar_date = m.calendar_date
        LEFT JOIN daily_act a ON d.calendar_date = a.calendar_date
        ORDER BY d.calendar_date ASC;
        """

        df_base = pd.read_sql_query(query, conn)

        # 2. Activity specific HRs (Running, Gym, Walking)
        act_df = pd.read_sql_query(
            """
            SELECT
                calendar_date,
                activity_type,
                avg_hr,
                max_hr,
                calories,
                duration_seconds
            FROM activities
            """,
            conn,
        )
        conn.close()

        act_pivots: dict[str, dict[str, float]] = {}
        for _, row in act_df.iterrows():
            c_date = str(row["calendar_date"])
            atype = str(row["activity_type"]).lower()
            hr = float(row["avg_hr"]) if pd.notna(row["avg_hr"]) else np.nan

            if c_date not in act_pivots:
                act_pivots[c_date] = {
                    "running_avg_hr": np.nan,
                    "gym_avg_hr": np.nan,
                    "walking_avg_hr": np.nan,
                }

            if pd.notna(hr):
                if atype in ("running", "treadmill_running"):
                    act_pivots[c_date]["running_avg_hr"] = hr
                elif atype in ("strength_training", "gym", "fitness_equipment"):
                    act_pivots[c_date]["gym_avg_hr"] = hr
                elif atype in ("walking", "hiking"):
                    act_pivots[c_date]["walking_avg_hr"] = hr

        act_summary = pd.DataFrame.from_dict(act_pivots, orient="index")
        if not act_summary.empty:
            act_summary = act_summary.reset_index().rename(columns={"index": "calendar_date"})
            df_merged = df_base.merge(act_summary, on="calendar_date", how="left")
        else:
            df_merged = df_base.copy()
            df_merged["running_avg_hr"] = np.nan
            df_merged["gym_avg_hr"] = np.nan
            df_merged["walking_avg_hr"] = np.nan

        df_merged["calendar_date"] = pd.to_datetime(df_merged["calendar_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        df_merged = df_merged.sort_values("calendar_date").reset_index(drop=True)

        # Derived metrics
        # 1. Sleep hours
        df_merged["total_sleep_hours"] = (
            df_merged["total_sleep_seconds"].fillna(0.0) / 3600.0
        ).round(2)

        # 2. Calorie split: resting = total - active
        df_merged["active_kilocalories"] = pd.to_numeric(
            df_merged["active_kilocalories"], errors="coerce"
        ).fillna(300.0)
        df_merged["total_kilocalories"] = pd.to_numeric(
            df_merged["total_kilocalories"], errors="coerce"
        ).fillna(2100.0)
        df_merged["resting_kilocalories"] = (
            df_merged["total_kilocalories"] - df_merged["active_kilocalories"]
        ).clip(lower=800.0)

        # 3. Activity HR interpolation (running ~145, gym ~110, walking ~115 if rest days)
        # Using observed athlete baselines
        obs_run = df_merged["running_avg_hr"].dropna().mean()
        obs_gym = df_merged["gym_avg_hr"].dropna().mean()
        obs_walk = df_merged["walking_avg_hr"].dropna().mean()

        def_run = float(obs_run) if pd.notna(obs_run) else 145.0
        def_gym = float(obs_gym) if pd.notna(obs_gym) else 110.0
        def_walk = float(obs_walk) if pd.notna(obs_walk) else 115.0

        df_merged["running_avg_hr"] = (
            df_merged["running_avg_hr"].ffill().bfill().fillna(def_run).round(1)
        )
        df_merged["gym_avg_hr"] = (
            df_merged["gym_avg_hr"].ffill().bfill().fillna(def_gym).round(1)
        )
        df_merged["walking_avg_hr"] = (
            df_merged["walking_avg_hr"].ffill().bfill().fillna(def_walk).round(1)
        )

        # 4. Fill biometrics with domain baseline defaults
        df_merged["resting_heart_rate"] = (
            df_merged["resting_heart_rate"].ffill().bfill().fillna(58.0)
        )
        df_merged["hrv_rmssd"] = df_merged["hrv_rmssd"].ffill().bfill().fillna(55.0)
        df_merged["daily_avg_stress"] = df_merged["daily_avg_stress"].ffill().bfill().fillna(25.0)
        df_merged["sleep_score"] = df_merged["sleep_score"].ffill().bfill().fillna(80.0)
        df_merged["total_steps"] = df_merged["total_steps"].fillna(5000.0)

        # 5. Workload dynamics (ACWR, rolling windows)
        df_merged["steps_roll_7d"] = df_merged["total_steps"].rolling(7, min_periods=1).mean()
        df_merged["acwr_steps"] = (
            df_merged["total_steps"].rolling(3, min_periods=1).mean()
            / (df_merged["steps_roll_7d"] + 1e-4)
        ).round(2)

        return df_merged

    def update_clean_dataset(self) -> pd.DataFrame:
        """Extract current database records and perform an append-only merge on DVC table.

        Existing dates are preserved and new dates are seamlessly appended.
        """
        extracted_df = self.extract_clean_dataframe()
        parquet_path = self.dvc_dir / "garmin_clean_features.parquet"
        csv_path = self.dvc_dir / "garmin_clean_features.csv"

        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            existing_dates = set(existing_df["calendar_date"].astype(str))
            new_rows = extracted_df[
                ~extracted_df["calendar_date"].astype(str).isin(existing_dates)
            ]

            if not new_rows.empty:
                combined = (
                    pd.concat([existing_df, new_rows], ignore_index=True)
                    .sort_values("calendar_date")
                    .reset_index(drop=True)
                )
                logger.info(
                    f"DVC: Appended {len(new_rows)} new dates to clean dataset. Total: {len(combined)} rows."
                )
            else:
                combined = existing_df
                logger.info(f"DVC: Dataset is up to date with {len(combined)} records.")
        else:
            combined = extracted_df
            logger.info(f"DVC: Initialized clean dataset with {len(combined)} records.")

        combined.to_parquet(parquet_path, index=False)
        combined.to_csv(csv_path, index=False)
        return combined


def main() -> None:
    """CLI runner for DVC clean dataset generation."""
    parser = argparse.ArgumentParser(description="Garmin DVC Clean Dataset Manager.")
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite database."
    )
    parser.add_argument(
        "--dvc-dir", type=Path, default=DEFAULT_DVC_DIR, help="Destination DVC directory."
    )
    parser.add_argument("--sync-r2", action="store_true", help="Sync to Cloudflare R2 bucket.")

    args = parser.parse_args()

    manager = GarminDVCManager(db_path=args.db_path, dvc_dir=args.dvc_dir)
    clean_df = manager.update_clean_dataset()

    print("\n" + "=" * 65)
    print("📦 Garmin DVC Clean Dataset Successfully Updated")
    print(f"   • Total Days:   {len(clean_df)}")
    print(f"   • Date Range:   {clean_df['calendar_date'].min()} to {clean_df['calendar_date'].max()}")
    print(f"   • Total Cols:   {len(clean_df.columns)}")
    print(f"   • Saved to:     {args.dvc_dir / 'garmin_clean_features.parquet'}")
    print("=" * 65 + "\n")

    if args.sync_r2:
        try:
            from src.common.r2_storage import R2StorageClient

            r2 = R2StorageClient()
            if r2.is_configured():
                print("☁️ Syncing DVC dataset to Cloudflare R2...")
                r2.sync_dvc_dataset(dvc_dir=args.dvc_dir)
                print("✅ DVC sync to Cloudflare R2 complete.")
            else:
                print("⚠️ Cloudflare R2 not configured; skipping cloud sync.")
        except Exception as e:
            print(f"❌ Error syncing to R2: {e}")


if __name__ == "__main__":
    main()
