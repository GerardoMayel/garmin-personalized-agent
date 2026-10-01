"""Motor de Promedios Bi-Semanales (Histórico Pasado 7 Días vs Pronóstico Futuro 7 Días).

Este módulo calcula diariamente a fecha ancla T0 (hoy) una fila consolidada en la tabla
'bi_weekly_average' dentro de la base de datos de predicciones (weekly_biometric_forecasts.db)
y en la base histórica (garmin_history.db).

Reglas de Negocio:
1. Fecha Ancla: T0 = hoy (date.today() o fecha especificada).
2. Ventana Histórica (7 días cerrados): [T0 - 7 días, T0 - 1 día].
3. Ventana Pronóstico (7 días futuros): [T0, T0 + 6 días] (hoy + 6 días siguientes = 7 días).
4. Regla de Ceros y Nulos: Los ceros y valores nulos NO promedian ni suman al divisor.
   El promedio se calcula dividiendo únicamente entre el número de días con valores válidos y reales disponibles.
5. Unicidad: Existe un único registro por fecha ancla (anchor_date es PRIMARY KEY).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sqlite3
import sys
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("BiWeeklyAverages")

DEFAULT_DB_PATH = Path("data/processed/garmin_history.db")
DEFAULT_FORECAST_DB_PATH = Path("data/processed/predictions/weekly_biometric_forecasts.db")
DEFAULT_PREDICTIONS_DIR = Path("data/processed/predictions")

# Las 14 métricas biométricas monitoreadas
BIOMETRIC_METRICS = [
    "total_steps",
    "daily_avg_stress",
    "resting_heart_rate",
    "sleep_score",
    "total_sleep_hours",
    "hrv_rmssd",
    "total_kilocalories",
    "active_kilocalories",
    "resting_kilocalories",
    "fitness_age",
    "fitness_age_gap",
    "walking_avg_hr",
    "running_avg_hr",
    "gym_avg_hr",
]


class BiWeeklyAverageManager:
    """Administrador de cálculo, persistencia y exportación de promedios bi-semanales."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        forecast_db_path: Path | str = DEFAULT_FORECAST_DB_PATH,
        predictions_dir: Path | str = DEFAULT_PREDICTIONS_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.forecast_db_path = Path(forecast_db_path)
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def _get_window_dates(self, anchor: date) -> tuple[str, str, str, str]:
        """Calcula las ventanas de 7 días históricos y 7 días proyectados."""
        hist_start = (anchor - timedelta(days=7)).strftime("%Y-%m-%d")
        hist_end = (anchor - timedelta(days=1)).strftime("%Y-%m-%d")
        fore_start = anchor.strftime("%Y-%m-%d")
        fore_end = (anchor + timedelta(days=6)).strftime("%Y-%m-%d")
        return hist_start, hist_end, fore_start, fore_end

    def _fetch_historical_telemetry(
        self, start_date: str, end_date: str
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """Extrae telemetría histórica diaria y frecuencias cardíacas de actividades."""
        if not self.db_path.exists():
            logger.warning(f"Base de datos histórica no encontrada: {self.db_path}")
            return pd.DataFrame(), {}

        conn = sqlite3.connect(self.db_path)
        try:
            # 1. Datos diarios consolidados
            query_actuals = """
                SELECT * FROM consolidated_daily_actuals
                WHERE calendar_date >= ? AND calendar_date <= ?
                ORDER BY calendar_date ASC
            """
            df_actuals = pd.read_sql_query(query_actuals, conn, params=[start_date, end_date])

            # Calcular total_sleep_hours si no existe en la tabla
            if (
                "total_sleep_hours" not in df_actuals.columns
                and "total_sleep_seconds" in df_actuals.columns
            ):
                df_actuals["total_sleep_hours"] = (
                    df_actuals["total_sleep_seconds"].fillna(0.0) / 3600.0
                )

            # 2. Actividades específicas para FC por deporte
            query_acts = """
                SELECT calendar_date, activity_type, avg_hr
                FROM activities
                WHERE calendar_date >= ? AND calendar_date <= ?
            """
            df_acts = pd.read_sql_query(query_acts, conn, params=[start_date, end_date])

            act_hrs: dict[str, dict[str, float]] = {
                "running_avg_hr": {},
                "gym_avg_hr": {},
                "walking_avg_hr": {},
            }
            for _, r in df_acts.iterrows():
                atype = str(r["activity_type"]).lower()
                c_date = str(r["calendar_date"])
                hr = float(r["avg_hr"]) if pd.notna(r["avg_hr"]) else None
                if hr and hr > 0:
                    if atype in ("running", "treadmill_running"):
                        act_hrs["running_avg_hr"][c_date] = hr
                    elif atype in ("strength_training", "gym", "fitness_equipment"):
                        act_hrs["gym_avg_hr"][c_date] = hr
                    elif atype in ("walking", "hiking"):
                        act_hrs["walking_avg_hr"][c_date] = hr

            return df_actuals, act_hrs
        finally:
            conn.close()

    def _fetch_forecast_telemetry(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extrae pronósticos futuros de la tabla biometric_forecasts."""
        if not self.forecast_db_path.exists():
            logger.warning(f"Base de predicciones no encontrada: {self.forecast_db_path}")
            return pd.DataFrame()

        conn = sqlite3.connect(self.forecast_db_path)
        try:
            query = """
                SELECT target_date, metric, predicted_value
                FROM biometric_forecasts
                WHERE target_date >= ? AND target_date <= ?
                ORDER BY target_date ASC
            """
            return pd.read_sql_query(query, conn, params=[start_date, end_date])
        finally:
            conn.close()

    @staticmethod
    def _compute_clean_average(values: list[float] | pd.Series, metric_name: str) -> tuple[float | None, int]:
        """Calcula promedio estricto excluyendo ceros y nulos.

        Regla del usuario: 'los ceros no promedian, solo los valores existentes,
        por lo que no siempre dividimos entre 7 sino por lo valores disponibles'.
        """
        valid_vals: list[float] = []
        for v in values:
            if v is None or pd.isna(v):
                continue
            val_float = float(v)
            # Para fitness_age_gap, un valor 0 exacto es no-informativo (sin brecha)
            if metric_name == "fitness_age_gap":
                if val_float != 0.0:
                    valid_vals.append(val_float)
            else:
                # Métricas fisiológicas estrictamente positivas (> 0)
                if val_float > 0.0:
                    valid_vals.append(val_float)

        count = len(valid_vals)
        if count == 0:
            return None, 0

        avg = round(float(np.mean(valid_vals)), 2)
        return avg, count

    def compute_bi_weekly_average(
        self, anchor_date: str | date | None = None
    ) -> dict[str, Any]:
        """Calcula el resumen de promedios bi-semanales para la fecha ancla T0."""
        if anchor_date is None:
            anchor = date.today()
        elif isinstance(anchor_date, str):
            anchor = datetime.strptime(anchor_date, "%Y-%m-%d").date()
        else:
            anchor = anchor_date

        anchor_str = anchor.strftime("%Y-%m-%d")
        hist_start, hist_end, fore_start, fore_end = self._get_window_dates(anchor)

        logger.info(
            f"Calculando promedios bi-semanales para fecha ancla T0={anchor_str} | "
            f"Histórico (7d): [{hist_start}..{hist_end}] | "
            f"Forecast (7d): [{fore_start}..{fore_end}]"
        )

        df_actuals, act_hrs = self._fetch_historical_telemetry(hist_start, hist_end)
        df_forecast = self._fetch_forecast_telemetry(fore_start, fore_end)

        record: dict[str, Any] = {
            "anchor_date": anchor_str,
            "history_window_start": hist_start,
            "history_window_end": hist_end,
            "forecast_window_start": fore_start,
            "forecast_window_end": fore_end,
        }

        for m in BIOMETRIC_METRICS:
            # 1. Promedio histórico (últimos 7 días cerrados)
            if m in act_hrs:
                hist_series = list(act_hrs[m].values())
            elif not df_actuals.empty and m in df_actuals.columns:
                hist_series = df_actuals[m].dropna().tolist()
            else:
                hist_series = []

            hist_avg, hist_count = self._compute_clean_average(hist_series, m)

            # 2. Promedio proyectado (siguientes 7 días)
            if not df_forecast.empty:
                fore_series = df_forecast[df_forecast["metric"] == m]["predicted_value"].dropna().tolist()
            else:
                fore_series = []

            fore_avg, fore_count = self._compute_clean_average(fore_series, m)

            # Asignar columnas estándar
            record[f"avg_{m}_last_7d"] = hist_avg
            record[f"forecast_{m}_next_7d"] = fore_avg
            record[f"valid_days_hist_{m}"] = hist_count
            record[f"valid_days_forecast_{m}"] = fore_count

            # Columnas alias convenientes para consultas ágiles
            if m == "total_steps":
                record["avg_steps_last_7d"] = hist_avg
                record["forecast_steps_next_7d"] = fore_avg
            elif m == "daily_avg_stress":
                record["avg_stress_last_7d"] = hist_avg
                record["forecast_stress_next_7d"] = fore_avg
            elif m == "resting_heart_rate":
                record["avg_resting_hr_last_7d"] = hist_avg
                record["forecast_resting_hr_next_7d"] = fore_avg
            elif m == "total_sleep_hours":
                record["avg_sleep_hours_last_7d"] = hist_avg
                record["forecast_sleep_hours_next_7d"] = fore_avg
            elif m == "total_kilocalories":
                record["avg_total_calories_last_7d"] = hist_avg
                record["forecast_total_calories_next_7d"] = fore_avg
            elif m == "active_kilocalories":
                record["avg_active_calories_last_7d"] = hist_avg
                record["forecast_active_calories_next_7d"] = fore_avg

        record["created_at"] = datetime.now().isoformat()
        return record

    def _ensure_tables_and_views(self, conn: sqlite3.Connection) -> None:
        """Crea la tabla bi_weekly_average y la vista de desglose si no existen."""
        cursor = conn.cursor()

        # Construir DDL con columnas tipadas
        columns_ddl = [
            "anchor_date TEXT PRIMARY KEY",
            "history_window_start TEXT NOT NULL",
            "history_window_end TEXT NOT NULL",
            "forecast_window_start TEXT NOT NULL",
            "forecast_window_end TEXT NOT NULL",
        ]

        for m in BIOMETRIC_METRICS:
            columns_ddl.append(f"avg_{m}_last_7d REAL")
            columns_ddl.append(f"forecast_{m}_next_7d REAL")
            columns_ddl.append(f"valid_days_hist_{m} INTEGER DEFAULT 0")
            columns_ddl.append(f"valid_days_forecast_{m} INTEGER DEFAULT 0")

        # Alias útiles
        aliases = [
            ("avg_steps_last_7d", "REAL"),
            ("forecast_steps_next_7d", "REAL"),
            ("avg_stress_last_7d", "REAL"),
            ("forecast_stress_next_7d", "REAL"),
            ("avg_resting_hr_last_7d", "REAL"),
            ("forecast_resting_hr_next_7d", "REAL"),
            ("avg_sleep_hours_last_7d", "REAL"),
            ("forecast_sleep_hours_next_7d", "REAL"),
            ("avg_total_calories_last_7d", "REAL"),
            ("forecast_total_calories_next_7d", "REAL"),
            ("avg_active_calories_last_7d", "REAL"),
            ("forecast_active_calories_next_7d", "REAL"),
        ]
        for col, c_type in aliases:
            columns_ddl.append(f"{col} {c_type}")

        columns_ddl.append("created_at TEXT NOT NULL")

        ddl = f"""
        CREATE TABLE IF NOT EXISTS bi_weekly_average (
            {', '.join(columns_ddl)}
        );
        """
        cursor.execute(ddl)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bi_weekly_anchor ON bi_weekly_average(anchor_date);")
        conn.commit()

    def save_to_database(self, record: dict[str, Any]) -> None:
        """Inserta o actualiza el registro en weekly_biometric_forecasts.db y garmin_history.db."""
        for target_db in (self.forecast_db_path, self.db_path):
            if not target_db.parent.exists():
                target_db.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(target_db)
            try:
                self._ensure_tables_and_views(conn)
                cursor = conn.cursor()

                cols = list(record.keys())
                placeholders = ", ".join(["?"] * len(cols))
                col_names = ", ".join(cols)

                query = f"""
                    INSERT OR REPLACE INTO bi_weekly_average ({col_names})
                    VALUES ({placeholders})
                """
                cursor.execute(query, [record[c] for c in cols])
                conn.commit()
                logger.info(f"Guardado exitoso en {target_db.name} | anchor_date={record['anchor_date']}")
            finally:
                conn.close()

    def export_data_files(self) -> tuple[Path, Path]:
        """Exporta la tabla completa bi_weekly_average a Parquet y CSV."""
        conn = sqlite3.connect(self.forecast_db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM bi_weekly_average ORDER BY anchor_date ASC", conn)
        finally:
            conn.close()

        parquet_path = self.predictions_dir / "bi_weekly_average.parquet"
        csv_path = self.predictions_dir / "bi_weekly_average.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        logger.info(
            f"Archivos exportados: {parquet_path.name} ({len(df)} filas), {csv_path.name} ({len(df)} filas)"
        )
        return parquet_path, csv_path

    def compute_and_save_averages(
        self, anchor_date: str | date | None = None
    ) -> dict[str, Any]:
        """Flujo integral: cálculo, guardado en SQLite y exportación a archivos."""
        record = self.compute_bi_weekly_average(anchor_date=anchor_date)
        self.save_to_database(record)
        self.export_data_files()
        return record


def main() -> None:
    """CLI para ejecución manual o en cron."""
    parser = argparse.ArgumentParser(
        description="Generador de Promedios Bi-Semanales (Pasado 7d vs Futuro 7d)."
    )
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=None,
        help="Fecha ancla T0 en formato YYYY-MM-DD (por defecto hoy).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Ruta a garmin_history.db.",
    )
    parser.add_argument(
        "--forecast-db-path",
        type=Path,
        default=DEFAULT_FORECAST_DB_PATH,
        help="Ruta a weekly_biometric_forecasts.db.",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="Sincronizar tabla y exportaciones con Cloudflare R2.",
    )
    args = parser.parse_args()

    manager = BiWeeklyAverageManager(
        db_path=args.db_path,
        forecast_db_path=args.forecast_db_path,
    )
    res = manager.compute_and_save_averages(anchor_date=args.anchor_date)

    print("\n" + "=" * 70)
    print("📊 REPORTE DE PROMEDIOS BI-SEMANALES (T0 vs PASADO Y FUTURO)")
    print("=" * 70)
    print(f"Fecha Ancla (T0)      : {res['anchor_date']}")
    print(f"Ventana Histórica (7d): {res['history_window_start']} -> {res['history_window_end']}")
    print(f"Ventana Forecast  (7d): {res['forecast_window_start']} -> {res['forecast_window_end']}")
    print("-" * 70)
    print(f"{'Métrica':<25} | {'Hist 7d':<10} | {'Días':<5} | {'Fore 7d':<10} | {'Días':<5}")
    print("-" * 70)

    for m in BIOMETRIC_METRICS:
        h_val = res.get(f"avg_{m}_last_7d")
        h_days = res.get(f"valid_days_hist_{m}", 0)
        f_val = res.get(f"forecast_{m}_next_7d")
        f_days = res.get(f"valid_days_forecast_{m}", 0)

        h_str = f"{h_val:.2f}" if h_val is not None else "N/A"
        f_str = f"{f_val:.2f}" if f_val is not None else "N/A"
        print(f"{m:<25} | {h_str:<10} | {h_days:<5} | {f_str:<10} | {f_days:<5}")

    print("=" * 70)

    if args.r2_sync:
        try:
            from src.common.r2_storage import R2StorageClient

            r2 = R2StorageClient()
            if r2.is_configured():
                print("☁️  Sincronizando tabla y predicciones con Cloudflare R2...")
                r2.sync_predictions(remote_prefix="forecast")
                r2.backup_database(local_db_path=args.db_path)
                print("✅ Sincronización con Cloudflare R2 completada con éxito.")
            else:
                print("⚠️  R2 no configurado; omitiendo subida.")
        except Exception as e:
            print(f"❌ Error sincronizando con R2: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
