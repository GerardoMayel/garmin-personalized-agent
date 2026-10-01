"""Agentic Tool Definitions for Garmin Personal Insight Agent.

Exposes deterministic analytical tools for LangGraph / LangChain agents and FastAPI endpoints:
- get_garmin_actuals_tool: Retrieves consolidated real telemetry (with 4 clinical extras).
- get_garmin_forecasts_tool: Retrieves locked biometric forecasts with horizon calculation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.tools.garmin_sql_tools import (
    get_garmin_actuals,
    get_garmin_forecasts,
)


@tool
def get_garmin_actuals_tool(days: int = 1, end_date: str | None = None) -> dict[str, Any]:
    """Consulta la telemetría real consolidada de Garmin para los últimos N días cerrados.

    Retorna métricas reales: frecuencia cardíaca en reposo (RHR), variabilidad (HRV rMSSD),
    estrés diario, puntuación de sueño, pasos, calorías y 4 métricas clínicas de alto valor
    (arquitectura de fases de sueño profundo/REM, oximetría SpO2/respiración, horas en zonas
    de estrés y carga de actividades deportivas). Excluye edad biológica.

    Args:
        days: Número de días cerrados a consultar (1 para ayer, 3, 7 o 14 días).
        end_date: Fecha final opcional en formato YYYY-MM-DD.
    """
    return get_garmin_actuals(days=days, end_date=end_date)


@tool
def get_garmin_forecasts_tool(metric: str | None = None, horizon_days: int = 14) -> dict[str, Any]:
    """Consulta los pronósticos biométricos futuros generados por los modelos de Machine Learning.

    Retorna las predicciones, intervalos de confianza al 95%, cálculo de días hacia adelante
    y frases resumen para una métrica específica o para todas las métricas biométricas
    (RHR, HRV, estrés, sueño, pasos, calorías). Excluye edad biológica.

    Args:
        metric: Nombre de la métrica opcional ('resting_heart_rate', 'hrv_rmssd', 'daily_avg_stress',
                'sleep_score', 'total_steps', 'active_kilocalories', etc.).
        horizon_days: Días hacia adelante a proyectar (default: 14).
    """
    return get_garmin_forecasts(metric=metric, horizon_days=horizon_days)


__all__ = [
    "get_garmin_actuals",
    "get_garmin_forecasts",
    "get_garmin_actuals_tool",
    "get_garmin_forecasts_tool",
]
