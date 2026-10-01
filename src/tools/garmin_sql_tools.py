"""Deterministic SQL-backed analytical tools (Function Calling) for Garmin biometrics.

Provides secure, parameterized query tools to retrieve:
1. Real Biometric Actuals (consolidated_daily_actuals) across configurable windows,
   bringing forecasted equivalents plus 4 high-value clinical recovery metrics (sleep phases,
   oximetry/respiration, stress duration zones, and athletic cardiovascular load),
   strictly excluding biological age.
2. Biometric Forecasts (consolidated_biometric_forecasts) across future horizons,
   calculating days ahead, confidence intervals, and natural language diagnostic summaries,
   strictly excluding biological age.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger("GarminSQLTools")

DEFAULT_DB_PATH = Path("data/processed/garmin_history.db")

# Metrics strictly excluded from coach tools as per requirements
EXCLUDED_METRICS = {"fitness_age", "chronological_age", "fitness_age_gap"}


def _get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Creates a read-only or standard connection with Row factory."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        raise FileNotFoundError(f"Base de datos SQLite no encontrada en {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# =============================================================================
# GRUPO 1: DATOS REALES (Actuals)
# =============================================================================


def get_garmin_actuals(
    days: int = 1,
    end_date: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recupera la telemetría biométrica real consolidada de Garmin para los últimos N días cerrados.

    Incluye todas las métricas que tienen pronóstico futuro (FC reposo, HRV, estrés diario,
    puntuación de sueño, pasos, calorías) más 4 métricas clínicas adicionales de alto valor
    (arquitectura detallada del sueño, oximetría y respiración, zonas temporales de estrés y
    carga de actividades deportivas). Excluye la edad biológica.

    Args:
        days: Número de días cerrados a consultar (default: 1 para ayer/último día cerrado).
        end_date: Fecha final de corte YYYY-MM-DD (default: último día cerrado disponible).
        db_path: Ruta opcional a la base de datos SQLite.

    Returns:
        Diccionario estructurado con periodo, lista de registros diarios y métricas agregadas.
    """
    days = max(1, min(days, 90))
    conn = _get_db_connection(db_path)

    try:
        # Determinar fecha final efectiva si no se proporciona
        if not end_date:
            row_latest = conn.execute(
                "SELECT MAX(calendar_date) as max_date FROM consolidated_daily_actuals"
            ).fetchone()
            if not row_latest or not row_latest["max_date"]:
                return {
                    "status": "empty",
                    "message": "No hay datos consolidados en la base de datos.",
                    "records": [],
                }
            effective_end = row_latest["max_date"]
        else:
            effective_end = end_date

        end_dt = datetime.strptime(effective_end, "%Y-%m-%d").date()
        start_dt = end_dt - timedelta(days=days - 1)
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()

        # Query parametrizada con desglose explícito (sin edad biológica)
        query = """
            SELECT
                calendar_date,
                -- 1. Métricas con pronóstico futuro equivalente
                resting_heart_rate,
                hrv_rmssd,
                hrv_weekly_avg,
                hrv_status,
                daily_avg_stress,
                daily_max_stress,
                sleep_score,
                ROUND(total_sleep_seconds / 3600.0, 2) AS total_sleep_hours,
                total_steps,
                active_kilocalories,
                resting_kilocalories,
                total_kilocalories,
                -- 2. Extra Clínico 1: Arquitectura de Fases de Sueño
                deep_sleep_seconds,
                ROUND(deep_sleep_seconds / 3600.0, 2) AS deep_sleep_hours,
                rem_sleep_seconds,
                ROUND(rem_sleep_seconds / 3600.0, 2) AS rem_sleep_hours,
                light_sleep_seconds,
                ROUND(light_sleep_seconds / 3600.0, 2) AS light_sleep_hours,
                awake_sleep_seconds,
                avg_sleep_stress,
                -- 3. Extra Clínico 2: Oximetría y Respiración Nocturna
                avg_spo2,
                lowest_spo2,
                avg_respiration,
                -- 4. Extra Clínico 3: Distribución Temporal de Zonas de Estrés
                rest_stress_duration_sec,
                ROUND(rest_stress_duration_sec / 3600.0, 2) AS rest_stress_hours,
                low_stress_duration_sec,
                medium_stress_duration_sec,
                high_stress_duration_sec,
                ROUND(high_stress_duration_sec / 3600.0, 2) AS high_stress_hours,
                -- 5. Extra Clínico 4: Carga y Frecuencia de Actividades Deportivas
                activity_count,
                ROUND(total_activity_duration_sec / 60.0, 1) AS total_activity_minutes,
                ROUND(total_activity_distance_m / 1000.0, 2) AS total_activity_distance_km,
                total_activity_calories,
                avg_activity_hr,
                max_activity_hr
            FROM consolidated_daily_actuals
            WHERE calendar_date BETWEEN ? AND ?
            ORDER BY calendar_date ASC
        """
        cursor = conn.execute(query, (start_str, end_str))
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return {
                "status": "empty",
                "period": {"start_date": start_str, "end_date": end_str, "days_requested": days},
                "records": [],
                "summary": {},
            }

        # Calcular agregaciones y promedios del periodo
        valid_rhr = [r["resting_heart_rate"] for r in rows if r["resting_heart_rate"] is not None]
        valid_hrv = [r["hrv_rmssd"] for r in rows if r["hrv_rmssd"] is not None]
        valid_stress = [r["daily_avg_stress"] for r in rows if r["daily_avg_stress"] is not None]
        valid_sleep = [r["sleep_score"] for r in rows if r["sleep_score"] is not None]
        valid_sleep_h = [r["total_sleep_hours"] for r in rows if r["total_sleep_hours"] is not None]
        valid_deep_h = [r["deep_sleep_hours"] for r in rows if r["deep_sleep_hours"] is not None]
        valid_rem_h = [r["rem_sleep_hours"] for r in rows if r["rem_sleep_hours"] is not None]
        valid_steps = [r["total_steps"] for r in rows if r["total_steps"] is not None]
        valid_act_kcal = [r["active_kilocalories"] for r in rows if r["active_kilocalories"] is not None]
        valid_tot_kcal = [r["total_kilocalories"] for r in rows if r["total_kilocalories"] is not None]
        valid_rest_stress_h = [r["rest_stress_hours"] for r in rows if r["rest_stress_hours"] is not None]
        valid_spo2 = [r["avg_spo2"] for r in rows if r["avg_spo2"] is not None]
        valid_resp = [r["avg_respiration"] for r in rows if r["avg_respiration"] is not None]
        total_activities = sum(r["activity_count"] or 0 for r in rows)

        summary = {
            "avg_resting_heart_rate": round(sum(valid_rhr) / len(valid_rhr), 1) if valid_rhr else None,
            "avg_hrv_rmssd": round(sum(valid_hrv) / len(valid_hrv), 1) if valid_hrv else None,
            "latest_hrv_status": rows[-1].get("hrv_status"),
            "avg_daily_stress": round(sum(valid_stress) / len(valid_stress), 1) if valid_stress else None,
            "avg_sleep_score": round(sum(valid_sleep) / len(valid_sleep), 1) if valid_sleep else None,
            "avg_total_sleep_hours": round(sum(valid_sleep_h) / len(valid_sleep_h), 2) if valid_sleep_h else None,
            "avg_deep_sleep_hours": round(sum(valid_deep_h) / len(valid_deep_h), 2) if valid_deep_h else None,
            "avg_rem_sleep_hours": round(sum(valid_rem_h) / len(valid_rem_h), 2) if valid_rem_h else None,
            "avg_rest_stress_hours": round(sum(valid_rest_stress_h) / len(valid_rest_stress_h), 2) if valid_rest_stress_h else None,
            "avg_spo2": round(sum(valid_spo2) / len(valid_spo2), 1) if valid_spo2 else None,
            "avg_respiration": round(sum(valid_resp) / len(valid_resp), 1) if valid_resp else None,
            "avg_total_steps": round(sum(valid_steps) / len(valid_steps), 0) if valid_steps else None,
            "total_steps_period": sum(valid_steps) if valid_steps else 0,
            "avg_active_kilocalories": round(sum(valid_act_kcal) / len(valid_act_kcal), 0) if valid_act_kcal else None,
            "avg_total_kilocalories": round(sum(valid_tot_kcal) / len(valid_tot_kcal), 0) if valid_tot_kcal else None,
            "total_activities_count": total_activities,
        }

        return {
            "status": "success",
            "period": {
                "start_date": start_str,
                "end_date": end_str,
                "total_days_retrieved": len(rows),
            },
            "records": rows,
            "summary": summary,
        }

    finally:
        conn.close()


# =============================================================================
# GRUPO 2: PRONÓSTICOS Y PROYECCIONES (Forecasts)
# =============================================================================


def get_garmin_forecasts(
    metric: str | None = None,
    horizon_days: int = 14,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recupera proyecciones biométricas futuras y calcula días hacia adelante y resúmenes de intervalo.

    Permite consultar una métrica en particular o todas las métricas biométricas proyectadas
    para el horizonte de días especificado. Excluye estrictamente la edad biológica.

    Args:
        metric: Nombre de métrica opcional (ej. 'resting_heart_rate', 'hrv_rmssd', 'daily_avg_stress',
                'sleep_score', 'total_steps', 'active_kilocalories', 'total_kilocalories').
        horizon_days: Límite de días hacia adelante a proyectar (default: 14).
        db_path: Ruta opcional a la base de datos SQLite.

    Returns:
        Diccionario estructurado con lista de proyecciones, horizonte, días relativos y frases resumen.
    """
    if metric and metric in EXCLUDED_METRICS:
        return {
            "status": "error",
            "message": f"La métrica '{metric}' está excluida de las herramientas de consulta general.",
            "predictions": [],
        }

    conn = _get_db_connection(db_path)
    try:
        # Obtener la fecha base de corte de datos reales para calcular días hacia adelante
        row_max_actual = conn.execute(
            "SELECT MAX(calendar_date) as max_date FROM consolidated_daily_actuals"
        ).fetchone()
        base_ref_date_str = row_max_actual["max_date"] if row_max_actual and row_max_actual["max_date"] else date.today().isoformat()
        base_ref_date = datetime.strptime(base_ref_date_str, "%Y-%m-%d").date()

        # Construir consulta con exclusión estricta de edad biológica
        excluded_placeholders = ",".join(f"'{m}'" for m in EXCLUDED_METRICS)
        where_clauses = [f"metric NOT IN ({excluded_placeholders})"]
        params: list[Any] = []

        if metric:
            where_clauses.append("metric = ?")
            params.append(metric)

        query = f"""
            SELECT
                forecast_generated_date,
                target_date,
                metric,
                predicted_value,
                ci_lower,
                ci_upper,
                model_name,
                is_locked,
                updated_at
            FROM consolidated_biometric_forecasts
            WHERE {" AND ".join(where_clauses)}
            ORDER BY target_date ASC, metric ASC
        """
        cursor = conn.execute(query, params)
        raw_rows = [dict(r) for r in cursor.fetchall()]

        if not raw_rows:
            return {
                "status": "empty",
                "message": "No se encontraron pronósticos registrados en la base de datos.",
                "predictions": [],
            }

        # Filtrar por horizonte relativo respecto a la fecha base y enriquecer
        predictions = []
        summary_statements = []

        for r in raw_rows:
            target_dt = datetime.strptime(r["target_date"], "%Y-%m-%d").date()
            days_ahead = (target_dt - base_ref_date).days
            if days_ahead <= 0 or days_ahead > horizon_days:
                continue

            pred_item = {
                "target_date": r["target_date"],
                "metric": r["metric"],
                "predicted_value": r["predicted_value"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "days_ahead": days_ahead,
                "model_name": r["model_name"],
                "forecast_generated_date": r["forecast_generated_date"],
                "is_locked": bool(r["is_locked"]),
            }
            predictions.append(pred_item)

            # Generar frase diagnóstica en lenguaje natural
            unit = _get_metric_unit(r["metric"])
            statement = (
                f"El pronóstico para dentro de {days_ahead} día(s) ({r['target_date']}) "
                f"de {r['metric']} es de {r['predicted_value']}{unit} "
                f"(IC 95%: [{r['ci_lower']} a {r['ci_upper']}]{unit}), proyectado por {r['model_name']}."
            )
            summary_statements.append(statement)

        distinct_metrics = sorted(list({p["metric"] for p in predictions}))

        return {
            "status": "success",
            "base_reference_date": base_ref_date_str,
            "horizon_days_requested": horizon_days,
            "total_predictions": len(predictions),
            "metrics_included": distinct_metrics,
            "predictions": predictions,
            "summary_statements": summary_statements,
        }

    finally:
        conn.close()


def _get_metric_unit(metric: str) -> str:
    """Devuelve la unidad de medida para el resumen."""
    units = {
        "resting_heart_rate": " ppm",
        "running_avg_hr": " ppm",
        "walking_avg_hr": " ppm",
        "gym_avg_hr": " ppm",
        "hrv_rmssd": " ms",
        "daily_avg_stress": " pts",
        "sleep_score": " pts",
        "total_sleep_hours": " hrs",
        "total_steps": " pasos",
        "active_kilocalories": " kcal",
        "resting_kilocalories": " kcal",
        "total_kilocalories": " kcal",
    }
    return units.get(metric, "")
