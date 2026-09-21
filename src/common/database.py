"""SQLite Historical Database for Garmin Physiological Telemetry.

Manages relational persistence, schema definitions, indexing, and ingestion
from raw JSON payloads and live Garmin Connect sync pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("GarminDatabase")

DEFAULT_DB_PATH = Path("data/processed/garmin_history.db")


class GarminDatabase:
    """Historical SQLite relational store for daily biometrics and activities."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_db_path = os.getenv("GARMIN_DB_PATH")
        self.db_path = Path(db_path or env_db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding SQLite connection with Row factory."""
        conn = sqlite3.connect(self.db_path.as_posix(), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Daily Summary
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    calendar_date TEXT PRIMARY KEY,
                    total_steps INTEGER,
                    total_distance_meters REAL,
                    active_kilocalories REAL,
                    total_kilocalories REAL,
                    resting_heart_rate INTEGER,
                    min_heart_rate INTEGER,
                    max_heart_rate INTEGER,
                    avg_stress_level INTEGER,
                    max_stress_level INTEGER,
                    floors_ascended REAL,
                    vigorous_minutes INTEGER,
                    moderate_minutes INTEGER,
                    updated_at TEXT
                )
                """
            )

            # 2. Sleep Records
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sleep_records (
                    calendar_date TEXT PRIMARY KEY,
                    sleep_score INTEGER,
                    total_sleep_seconds INTEGER,
                    deep_sleep_seconds INTEGER,
                    light_sleep_seconds INTEGER,
                    rem_sleep_seconds INTEGER,
                    awake_sleep_seconds INTEGER,
                    nap_seconds INTEGER,
                    avg_spo2 REAL,
                    lowest_spo2 REAL,
                    avg_respiration REAL,
                    avg_sleep_stress REAL,
                    avg_heart_rate REAL,
                    sleep_start_local TEXT,
                    sleep_end_local TEXT,
                    updated_at TEXT
                )
                """
            )

            # 3. HRV Records
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hrv_records (
                    calendar_date TEXT PRIMARY KEY,
                    last_night_avg REAL,
                    weekly_avg REAL,
                    last_night_5min_high REAL,
                    status TEXT,
                    baseline_low REAL,
                    baseline_balanced_low REAL,
                    baseline_balanced_upper REAL,
                    feedback_phrase TEXT,
                    updated_at TEXT
                )
                """
            )

            # 4. Stress Records
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stress_records (
                    calendar_date TEXT PRIMARY KEY,
                    avg_stress_level INTEGER,
                    max_stress_level INTEGER,
                    rest_stress_duration_sec INTEGER,
                    activity_stress_duration_sec INTEGER,
                    low_stress_duration_sec INTEGER,
                    medium_stress_duration_sec INTEGER,
                    high_stress_duration_sec INTEGER,
                    updated_at TEXT
                )
                """
            )

            # 5. VO2 Max & Max Metrics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS max_metrics (
                    calendar_date TEXT PRIMARY KEY,
                    vo2_max_running REAL,
                    vo2_max_precise REAL,
                    fitness_age INTEGER,
                    updated_at TEXT
                )
                """
            )

            # 6. Activities
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS activities (
                    activity_id INTEGER PRIMARY KEY,
                    calendar_date TEXT,
                    activity_name TEXT,
                    activity_type TEXT,
                    distance_meters REAL,
                    duration_seconds REAL,
                    elapsed_duration_seconds REAL,
                    elevation_gain_meters REAL,
                    avg_speed_mps REAL,
                    max_speed_mps REAL,
                    avg_hr INTEGER,
                    max_hr INTEGER,
                    calories REAL,
                    fit_zip_path TEXT,
                    start_time_local TEXT,
                    updated_at TEXT
                )
                """
            )

            # Indices for analytical querying
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_activities_date ON activities(calendar_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_activities_type ON activities(activity_type)"
            )

    # -------------------------------------------------------------------------
    # Upsert Operations (Idempotent)
    # -------------------------------------------------------------------------

    def upsert_daily_summary(self, data: dict[str, Any]) -> bool:
        """Upsert daily summary record from Garmin user_summary JSON."""
        date_str = data.get("calendarDate")
        if not date_str:
            return False

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_summaries (
                    calendar_date, total_steps, total_distance_meters, active_kilocalories,
                    total_kilocalories, resting_heart_rate, min_heart_rate, max_heart_rate,
                    avg_stress_level, max_stress_level, floors_ascended, vigorous_minutes,
                    moderate_minutes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calendar_date) DO UPDATE SET
                    total_steps = excluded.total_steps,
                    total_distance_meters = excluded.total_distance_meters,
                    active_kilocalories = excluded.active_kilocalories,
                    total_kilocalories = excluded.total_kilocalories,
                    resting_heart_rate = excluded.resting_heart_rate,
                    min_heart_rate = excluded.min_heart_rate,
                    max_heart_rate = excluded.max_heart_rate,
                    avg_stress_level = excluded.avg_stress_level,
                    max_stress_level = excluded.max_stress_level,
                    floors_ascended = excluded.floors_ascended,
                    vigorous_minutes = excluded.vigorous_minutes,
                    moderate_minutes = excluded.moderate_minutes,
                    updated_at = excluded.updated_at
                """,
                (
                    date_str,
                    data.get("totalSteps"),
                    data.get("totalDistanceMeters"),
                    data.get("activeKilocalories"),
                    data.get("totalKilocalories"),
                    data.get("restingHeartRate"),
                    data.get("minHeartRate"),
                    data.get("maxHeartRate"),
                    data.get("averageStressLevel"),
                    data.get("maxStressLevel"),
                    data.get("floorsAscended"),
                    data.get("vigorousIntensityMinutes"),
                    data.get("moderateIntensityMinutes"),
                    now,
                ),
            )
        return True

    def upsert_sleep(self, data: dict[str, Any]) -> bool:
        """Upsert sleep record from Garmin sleep JSON."""
        dto = data.get("dailySleepDTO") or data
        date_str = dto.get("calendarDate")
        if not date_str:
            return False

        sleep_scores = dto.get("sleepScores", {})
        overall_score = sleep_scores.get("overall", {}).get("value")
        if overall_score is None and "overallScore" in dto:
            overall_score = dto.get("overallScore", {}).get("value")

        start_local = dto.get("sleepStartTimestampLocal")
        end_local = dto.get("sleepEndTimestampLocal")

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sleep_records (
                    calendar_date, sleep_score, total_sleep_seconds, deep_sleep_seconds,
                    light_sleep_seconds, rem_sleep_seconds, awake_sleep_seconds, nap_seconds,
                    avg_spo2, lowest_spo2, avg_respiration, avg_sleep_stress, avg_heart_rate,
                    sleep_start_local, sleep_end_local, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calendar_date) DO UPDATE SET
                    sleep_score = excluded.sleep_score,
                    total_sleep_seconds = excluded.total_sleep_seconds,
                    deep_sleep_seconds = excluded.deep_sleep_seconds,
                    light_sleep_seconds = excluded.light_sleep_seconds,
                    rem_sleep_seconds = excluded.rem_sleep_seconds,
                    awake_sleep_seconds = excluded.awake_sleep_seconds,
                    nap_seconds = excluded.nap_seconds,
                    avg_spo2 = excluded.avg_spo2,
                    lowest_spo2 = excluded.lowest_spo2,
                    avg_respiration = excluded.avg_respiration,
                    avg_sleep_stress = excluded.avg_sleep_stress,
                    avg_heart_rate = excluded.avg_heart_rate,
                    sleep_start_local = excluded.sleep_start_local,
                    sleep_end_local = excluded.sleep_end_local,
                    updated_at = excluded.updated_at
                """,
                (
                    date_str,
                    overall_score,
                    dto.get("sleepTimeSeconds"),
                    dto.get("deepSleepSeconds"),
                    dto.get("lightSleepSeconds"),
                    dto.get("remSleepSeconds"),
                    dto.get("awakeSleepSeconds"),
                    dto.get("napTimeSeconds"),
                    dto.get("averageSpO2Value"),
                    dto.get("lowestSpO2Value"),
                    dto.get("averageRespirationValue"),
                    dto.get("avgSleepStress"),
                    dto.get("avgHeartRate"),
                    str(start_local) if start_local else None,
                    str(end_local) if end_local else None,
                    now,
                ),
            )
        return True

    def upsert_hrv(self, data: dict[str, Any]) -> bool:
        """Upsert HRV record from Garmin hrv JSON."""
        summary = data.get("hrvSummary") or data
        date_str = summary.get("calendarDate")
        if not date_str:
            return False

        baseline = summary.get("baseline", {})
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO hrv_records (
                    calendar_date, last_night_avg, weekly_avg, last_night_5min_high,
                    status, baseline_low, baseline_balanced_low, baseline_balanced_upper,
                    feedback_phrase, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calendar_date) DO UPDATE SET
                    last_night_avg = excluded.last_night_avg,
                    weekly_avg = excluded.weekly_avg,
                    last_night_5min_high = excluded.last_night_5min_high,
                    status = excluded.status,
                    baseline_low = excluded.baseline_low,
                    baseline_balanced_low = excluded.baseline_balanced_low,
                    baseline_balanced_upper = excluded.baseline_balanced_upper,
                    feedback_phrase = excluded.feedback_phrase,
                    updated_at = excluded.updated_at
                """,
                (
                    date_str,
                    summary.get("lastNightAvg"),
                    summary.get("weeklyAvg"),
                    summary.get("lastNight5MinHigh"),
                    summary.get("status"),
                    baseline.get("lowUpper"),
                    baseline.get("balancedLow"),
                    baseline.get("balancedUpper"),
                    summary.get("feedbackPhrase"),
                    now,
                ),
            )
        return True

    def upsert_stress(self, data: dict[str, Any]) -> bool:
        """Upsert daily stress record from Garmin stress JSON."""
        date_str = data.get("calendarDate")
        if not date_str:
            return False

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO stress_records (
                    calendar_date, avg_stress_level, max_stress_level, rest_stress_duration_sec,
                    activity_stress_duration_sec, low_stress_duration_sec,
                    medium_stress_duration_sec, high_stress_duration_sec, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calendar_date) DO UPDATE SET
                    avg_stress_level = excluded.avg_stress_level,
                    max_stress_level = excluded.max_stress_level,
                    rest_stress_duration_sec = excluded.rest_stress_duration_sec,
                    activity_stress_duration_sec = excluded.activity_stress_duration_sec,
                    low_stress_duration_sec = excluded.low_stress_duration_sec,
                    medium_stress_duration_sec = excluded.medium_stress_duration_sec,
                    high_stress_duration_sec = excluded.high_stress_duration_sec,
                    updated_at = excluded.updated_at
                """,
                (
                    date_str,
                    data.get("avgStressLevel"),
                    data.get("maxStressLevel"),
                    data.get("restStressDuration"),
                    data.get("activityStressDuration"),
                    data.get("lowStressDuration"),
                    data.get("mediumStressDuration"),
                    data.get("highStressDuration"),
                    now,
                ),
            )
        return True

    def upsert_max_metrics(self, data: Any) -> bool:
        """Upsert VO2 Max metrics from Garmin max_metrics JSON."""
        entry = data[0] if isinstance(data, list) and data else data
        if not isinstance(entry, dict):
            return False

        generic = entry.get("generic", {})
        date_str = generic.get("calendarDate") or entry.get("calendarDate")
        if not date_str:
            return False

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO max_metrics (
                    calendar_date, vo2_max_running, vo2_max_precise, fitness_age, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(calendar_date) DO UPDATE SET
                    vo2_max_running = excluded.vo2_max_running,
                    vo2_max_precise = excluded.vo2_max_precise,
                    fitness_age = excluded.fitness_age,
                    updated_at = excluded.updated_at
                """,
                (
                    date_str,
                    generic.get("vo2MaxValue"),
                    generic.get("vo2MaxPreciseValue"),
                    generic.get("fitnessAge"),
                    now,
                ),
            )
        return True

    def upsert_activity(self, data: dict[str, Any], fit_zip_path: str | None = None) -> bool:
        """Upsert activity summary and local .fit.zip location."""
        act_id = data.get("activityId")
        if not act_id:
            return False

        start_time = data.get("startTimeLocal", "")
        cal_date = start_time[:10] if start_time else None
        act_type = data.get("activityType", {}).get("typeKey")

        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO activities (
                    activity_id, calendar_date, activity_name, activity_type, distance_meters,
                    duration_seconds, elapsed_duration_seconds, elevation_gain_meters,
                    avg_speed_mps, max_speed_mps, avg_hr, max_hr, calories, fit_zip_path,
                    start_time_local, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(activity_id) DO UPDATE SET
                    calendar_date = excluded.calendar_date,
                    activity_name = excluded.activity_name,
                    activity_type = excluded.activity_type,
                    distance_meters = excluded.distance_meters,
                    duration_seconds = excluded.duration_seconds,
                    elapsed_duration_seconds = excluded.elapsed_duration_seconds,
                    elevation_gain_meters = excluded.elevation_gain_meters,
                    avg_speed_mps = excluded.avg_speed_mps,
                    max_speed_mps = excluded.max_speed_mps,
                    avg_hr = excluded.avg_hr,
                    max_hr = excluded.max_hr,
                    calories = excluded.calories,
                    fit_zip_path = COALESCE(excluded.fit_zip_path, activities.fit_zip_path),
                    start_time_local = excluded.start_time_local,
                    updated_at = excluded.updated_at
                """,
                (
                    act_id,
                    cal_date,
                    data.get("activityName"),
                    act_type,
                    data.get("distance"),
                    data.get("duration"),
                    data.get("elapsedDuration"),
                    data.get("elevationGain"),
                    data.get("averageSpeed"),
                    data.get("maxSpeed"),
                    data.get("averageHR"),
                    data.get("maxHR"),
                    data.get("calories"),
                    fit_zip_path,
                    start_time,
                    now,
                ),
            )
        return True

    # -------------------------------------------------------------------------
    # Batch Backfill from data/raw/
    # -------------------------------------------------------------------------

    def ingest_raw_directory(self, raw_dir: Path = Path("data/raw")) -> dict[str, int]:
        """Scan data/raw/ and backfill all JSON snapshots into SQLite."""
        stats = {
            "daily_summaries": 0,
            "sleep_records": 0,
            "hrv_records": 0,
            "stress_records": 0,
            "max_metrics": 0,
            "activities": 0,
        }

        if not raw_dir.exists():
            logger.warning(f"Raw directory does not exist: {raw_dir}")
            return stats

        logger.info(f"Scanning raw data partitions in: {raw_dir.resolve()}")

        for day_dir in sorted(raw_dir.iterdir()):
            if not day_dir.is_dir() or day_dir.name in {"sample", ".git"}:
                continue

            # 1. Daily summary
            sum_file = day_dir / "daily_summary.json"
            if sum_file.exists():
                try:
                    with open(sum_file, encoding="utf-8") as f:
                        if self.upsert_daily_summary(json.load(f)):
                            stats["daily_summaries"] += 1
                except Exception as e:
                    logger.debug(f"Could not load {sum_file}: {e}")

            # 2. Sleep
            sleep_file = day_dir / "sleep.json"
            if sleep_file.exists():
                try:
                    with open(sleep_file, encoding="utf-8") as f:
                        if self.upsert_sleep(json.load(f)):
                            stats["sleep_records"] += 1
                except Exception as e:
                    logger.debug(f"Could not load {sleep_file}: {e}")

            # 3. HRV
            hrv_file = day_dir / "hrv.json"
            if hrv_file.exists():
                try:
                    with open(hrv_file, encoding="utf-8") as f:
                        if self.upsert_hrv(json.load(f)):
                            stats["hrv_records"] += 1
                except Exception as e:
                    logger.debug(f"Could not load {hrv_file}: {e}")

            # 4. Stress
            stress_file = day_dir / "stress.json"
            if stress_file.exists():
                try:
                    with open(stress_file, encoding="utf-8") as f:
                        if self.upsert_stress(json.load(f)):
                            stats["stress_records"] += 1
                except Exception as e:
                    logger.debug(f"Could not load {stress_file}: {e}")

            # 5. Max Metrics
            metrics_file = day_dir / "max_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, encoding="utf-8") as f:
                        if self.upsert_max_metrics(json.load(f)):
                            stats["max_metrics"] += 1
                except Exception as e:
                    logger.debug(f"Could not load {metrics_file}: {e}")

            # 6. Activities subfolder
            act_dir = day_dir / "activities"
            if act_dir.exists() and act_dir.is_dir():
                for act_json in act_dir.glob("activity_*_summary.json"):
                    try:
                        act_id = act_json.stem.split("_")[1]
                        zip_file = act_dir / f"activity_{act_id}.zip"
                        fit_path = zip_file.as_posix() if zip_file.exists() else None
                        with open(act_json, encoding="utf-8") as f:
                            if self.upsert_activity(json.load(f), fit_zip_path=fit_path):
                                stats["activities"] += 1
                    except Exception as e:
                        logger.debug(f"Could not load {act_json}: {e}")

        logger.info(f"Backfill complete! Ingested records: {stats}")
        return stats

    # -------------------------------------------------------------------------
    # Analytical Query Helpers
    # -------------------------------------------------------------------------

    def get_biometrics_timeseries(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Retrieve unified daily biometric timeline for ML and plotting."""
        with self.get_connection() as conn:
            query = """
                SELECT
                    d.calendar_date,
                    d.total_steps,
                    d.resting_heart_rate,
                    d.active_kilocalories,
                    s.sleep_score,
                    s.total_sleep_seconds,
                    s.deep_sleep_seconds,
                    s.rem_sleep_seconds,
                    s.avg_spo2,
                    s.avg_respiration,
                    h.last_night_avg AS hrv_rmssd,
                    h.status AS hrv_status,
                    st.avg_stress_level,
                    m.vo2_max_running
                FROM daily_summaries d
                LEFT JOIN sleep_records s ON d.calendar_date = s.calendar_date
                LEFT JOIN hrv_records h ON d.calendar_date = h.calendar_date
                LEFT JOIN stress_records st ON d.calendar_date = st.calendar_date
                LEFT JOIN max_metrics m ON d.calendar_date = m.calendar_date
                WHERE d.calendar_date BETWEEN ? AND ?
                ORDER BY d.calendar_date ASC
            """
            cursor = conn.execute(query, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]

    def get_daily_summary(self, calendar_date: str) -> dict[str, Any] | None:
        """Fetch daily summary record by date."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_summaries WHERE calendar_date = ?", (calendar_date,)
            ).fetchone()
            return dict(row) if row else None

    def delete_records_on_or_after(self, min_date: str) -> dict[str, int]:
        """Delete all telemetry and activity records on or after min_date."""
        deleted: dict[str, int] = {}
        tables = [
            "daily_summaries",
            "sleep_records",
            "hrv_records",
            "stress_records",
            "max_metrics",
            "activities",
        ]
        with self.get_connection() as conn:
            for tbl in tables:
                cursor = conn.execute(
                    f"DELETE FROM {tbl} WHERE calendar_date >= ?", (min_date,)
                )
                deleted[tbl] = cursor.rowcount
        logger.info(f"Registros eliminados a partir de {min_date}: {deleted}")
        return deleted

    def count_records(self) -> dict[str, int]:
        """Count rows in all tables."""
        tables = [
            "daily_summaries",
            "sleep_records",
            "hrv_records",
            "stress_records",
            "max_metrics",
            "activities",
        ]
        counts = {}
        with self.get_connection() as conn:
            for t in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {t}")
                counts[t] = cursor.fetchone()[0]
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Garmin SQLite Historical Database Manager.")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Scan data/raw/ and backfill all historical JSON snapshots into SQLite",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Custom SQLite database file path (default: data/processed/garmin_history.db)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display count of records in all database tables",
    )

    args = parser.parse_args()
    db = GarminDatabase(db_path=args.db_path)

    if args.backfill:
        db.ingest_raw_directory()

    counts = db.count_records()
    print("\n" + "=" * 50)
    print(f"📊 Garmin SQLite Database Status: {db.db_path}")
    print("=" * 50)
    for table, count in counts.items():
        print(f"  • {table:<22}: {count:>4} records")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
