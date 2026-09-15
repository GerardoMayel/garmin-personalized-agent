#!/usr/bin/env python3
"""Garmin Connect Lightweight Sample Extraction Script.

Use this script to quickly test authentication and retrieve a single-day sample
of physiological telemetry (Sleep, HRV, Stress, Activity) before running
the full multi-day ingestion pipeline.

Usage:
    python -m src.ingestion.sample_sync
    # or
    python src/ingestion/sample_sync.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from src.common.logger import get_logger

logger = get_logger("GarminSample")


def load_credentials(env_file: Path | None = None) -> tuple[str, str, Path]:
    """Load and validate credentials from .env."""
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    token_store_str = os.getenv("GARMIN_TOKEN_STORE", "~/.garminconnect")
    token_store_path = Path(os.path.expanduser(token_store_str))

    if not email or not password:
        logger.error("❌ Faltan credenciales en el archivo .env")
        logger.error("Asegúrate de definir GARMIN_EMAIL y GARMIN_PASSWORD en tu archivo .env")
        sys.exit(1)

    return email, password, token_store_path


def authenticate_client(email: str, password: str, token_store: Path) -> Garmin:
    """Authenticate to Garmin Connect with token caching."""
    client = Garmin(email, password)

    if token_store.exists() and any(token_store.iterdir()):
        logger.info(f"🔑 Intentando restaurar sesión desde tokens en: {token_store}")
        try:
            client.login(token_store.as_posix())
            logger.info("✅ Sesión autenticada exitosamente con tokens en caché.")
            return client
        except (FileNotFoundError, GarminConnectAuthenticationError) as e:
            logger.warning(f"⚠️ Tokens expirados o inválidos ({e}). Intentando login completo...")
        except GarminConnectTooManyRequestsError:
            logger.error("❌ Error 429: Límite de peticiones de Garmin alcanzado (Rate Limit).")
            logger.error("Espera unos minutos antes de volver a intentarlo.")
            sys.exit(1)
        except GarminConnectConnectionError as e:
            logger.error(f"❌ Error de conexión de red con Garmin: {e}")
            sys.exit(1)

    logger.info("🔐 Iniciando sesión con correo y contraseña...")
    try:
        client.login()
        token_store.mkdir(parents=True, exist_ok=True)
        if hasattr(client, "garth") and hasattr(client.garth, "dump"):
            client.garth.dump(token_store.as_posix())
            logger.info(f"💾 Tokens guardados en caché local: {token_store}")
        logger.info("✅ Autenticación inicial completada exitosamente.")
        return client
    except GarminConnectAuthenticationError as e:
        logger.error(f"❌ Error de autenticación: Credenciales incorrectas o bloqueo MFA: {e}")
        sys.exit(1)
    except GarminConnectTooManyRequestsError:
        logger.error("❌ Error 429: Garmin ha limitado temporalmente las peticiones por IP.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error inesperado durante el login: {e}")
        sys.exit(1)


def save_sample_json(data: Any, target_file: Path) -> None:
    """Save data to JSON file with indentation."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def fetch_sample(client: Garmin, output_dir: Path = Path("data/raw/sample")) -> dict[str, Any]:
    """Fetch sample telemetry for 1 day and 1 activity."""
    # Target date: yesterday usually has complete sleep and daily data
    today = date.today()
    yesterday = today - timedelta(days=1)
    target_date_str = yesterday.isoformat()

    logger.info(f"\n📡 Obteniendo datos de muestra para la fecha: {target_date_str}")
    sample_bundle: dict[str, Any] = {"date": target_date_str}

    # 1. Perfil de Usuario
    try:
        user_name = client.get_full_name()
        logger.info(f"👤 Usuario Garmin conectado: {user_name}")
        sample_bundle["user_name"] = user_name
    except Exception as e:
        logger.warning(f"No se pudo obtener el nombre de usuario: {e}")

    # 2. Datos de Sueño
    try:
        sleep_data = client.get_sleep_data(target_date_str)
        save_sample_json(sleep_data, output_dir / "sample_sleep.json")
        sleep_score = None
        if isinstance(sleep_data, dict):
            sleep_score = sleep_data.get("dailySleepDTO", {}).get("sleepScores", {}).get(
                "overall", {}
            ).get("value") or sleep_data.get("sleepScores", {}).get("overall", {}).get("value")
            duration_sec = sleep_data.get("dailySleepDTO", {}).get("sleepTimeSeconds")
            duration_hrs = round(duration_sec / 3600, 1) if duration_sec else "N/A"
            logger.info(f"💤 Sueño: Puntuación {sleep_score}/100 | Duración: {duration_hrs}h")
        sample_bundle["sleep"] = {
            "score": sleep_score,
            "file": str(output_dir / "sample_sleep.json"),
        }
    except Exception as e:
        logger.warning(f"No se pudieron descargar datos de sueño: {e}")

    # 3. Datos de HRV (Variabilidad de la Frecuencia Cardíaca)
    try:
        hrv_data = client.get_hrv_data(target_date_str)
        save_sample_json(hrv_data, output_dir / "sample_hrv.json")
        hrv_summary = hrv_data.get("hrvSummary", {}) if isinstance(hrv_data, dict) else {}
        last_night_avg = hrv_summary.get("lastNightAvg")
        status = hrv_summary.get("status")
        logger.info(f"❤️ HRV nocturna: {last_night_avg} ms (rMSSD) | Estado: {status}")
        sample_bundle["hrv"] = {"lastNightAvg": last_night_avg, "status": status}
    except Exception as e:
        logger.warning(f"No se pudieron descargar datos de HRV: {e}")

    # 4. Estrés y Body Battery
    try:
        stress_data = client.get_stress_data(target_date_str)
        save_sample_json(stress_data, output_dir / "sample_stress.json")
        avg_stress = None
        if isinstance(stress_data, dict):
            avg_stress = stress_data.get("avgStressLevel")
            logger.info(f"⚡ Estrés promedio del día: {avg_stress}/100")
        sample_bundle["stress"] = {"avg_stress": avg_stress}
    except Exception as e:
        logger.warning(f"No se pudieron descargar datos de estrés: {e}")

    # 5. Resumen Diario de Actividad
    try:
        summary_data = client.get_user_summary(target_date_str)
        save_sample_json(summary_data, output_dir / "sample_daily_summary.json")
        if isinstance(summary_data, dict):
            steps = summary_data.get("totalSteps")
            rhr = summary_data.get("restingHeartRate")
            logger.info(f"🚶 Pasos: {steps} | FC en reposo: {rhr} bpm")
            sample_bundle["daily_summary"] = {"steps": steps, "restingHeartRate": rhr}
    except Exception as e:
        logger.warning(f"No se pudo descargar el resumen diario: {e}")

    # 6. Última Actividad
    try:
        activities = client.get_activities(0, 1)
        if activities:
            act = activities[0]
            act_id = act.get("activityId")
            act_name = act.get("activityName")
            act_type = act.get("activityType", {}).get("typeKey")
            distance_m = act.get("distance", 0)
            duration_s = act.get("duration", 0)
            dist_km = round(distance_m / 1000, 2) if distance_m else 0
            dur_min = round(duration_s / 60, 1) if duration_s else 0
            save_sample_json(act, output_dir / f"sample_activity_{act_id}.json")
            logger.info(
                f"🏃 Última actividad: '{act_name}' ({act_type}) - {dist_km} km en {dur_min} min"
            )
            sample_bundle["activity"] = {
                "id": act_id,
                "name": act_name,
                "type": act_type,
                "distance_km": dist_km,
                "duration_min": dur_min,
            }
        else:
            logger.info("ℹ️ No se encontraron actividades registradas recientemente.")
    except Exception as e:
        logger.warning(f"No se pudo descargar la última actividad: {e}")

    return sample_bundle


def main() -> None:
    print("\n" + "=" * 60)
    print(" 🧪 Garmin Connect - Test de Conexión y Extracción de Muestra")
    print("=" * 60)

    email, password, token_store = load_credentials()
    output_sample_dir = Path("data/raw/sample")

    client = authenticate_client(email, password, token_store)
    _ = fetch_sample(client, output_dir=output_sample_dir)

    print("\n" + "-" * 60)
    print(" 🎉 ¡Prueba exitosa! Muestra guardada en:")
    print(f"    📁 {output_sample_dir.resolve()}")
    print("    Archivos generados:")
    for file_path in output_sample_dir.glob("*.json"):
        print(f"     • {file_path.name}")
    print("-" * 60)
    print(" Ya puedes ejecutar el pipeline completo de sincronización con confianza:")
    print("   uv run python -m src.ingestion.garmin_sync")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
