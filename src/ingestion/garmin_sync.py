"""Garmin Connect Data Ingestion Module.

Automates session persistence, downloading daily health metrics (Sleep, HRV,
Stress, Body Battery, Max Metrics/VO2 Max), activity metadata, and raw .FIT
telemetry packages, and persists them into the historical SQLite database.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from src.common.database import GarminDatabase
from src.common.logger import get_logger

load_dotenv()
logger = get_logger("GarminIngestor")


class GarminDataIngestor:
    """Orchestrates authentication, biometrics extraction, and SQLite database persistence."""

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        raw_data_dir: str = "data/raw",
        tokenstore_dir: Optional[str] = None,
        client: Optional[Any] = None,
        db: Optional[GarminDatabase] = None,
    ) -> None:
        self.email = email or os.getenv("GARMIN_EMAIL")
        self.password = password or os.getenv("GARMIN_PASSWORD")
        self.raw_data_dir = Path(raw_data_dir)
        self.db = db if db is not None else GarminDatabase()

        env_token_store = os.getenv("GARMIN_TOKEN_STORE", "~/.garminconnect")
        resolved_token_dir = tokenstore_dir or env_token_store
        self.tokenstore_dir = Path(os.path.expanduser(resolved_token_dir))

        if client is not None:
            self.client = client
        else:
            if not self.email or not self.password:
                raise ValueError(
                    "Las variables GARMIN_EMAIL y GARMIN_PASSWORD deben estar definidas en el archivo .env"
                )
            self.client = Garmin(self.email, self.password)
            self._authenticate()

    def _authenticate(self) -> None:
        """Autenticación con persistencia de tokens de sesión para evitar bloqueos."""
        try:
            self.client.login(self.tokenstore_dir.as_posix())
            logger.info("Autenticado exitosamente mediante tokens en caché.")
        except (FileNotFoundError, GarminConnectAuthenticationError):
            logger.info("Tokens no encontrados o expirados. Iniciando sesión completa...")
            self.client.login()
            self.tokenstore_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(self.client, "garth") and hasattr(self.client.garth, "dump"):
                self.client.garth.dump(self.tokenstore_dir.as_posix())
            logger.info("Nueva sesión almacenada en caché local.")
        except GarminConnectTooManyRequestsError:
            logger.error("Error: Límite de peticiones alcanzado (Rate Limit). Intenta más tarde.")
            raise
        except GarminConnectConnectionError as e:
            logger.error(f"Error de conexión con Garmin Connect: {e}")
            raise

    def _save_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Guarda un diccionario como archivo JSON formateado con identación."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def sync_daily_biometrics(self, target_date: date) -> Dict[str, bool]:
        """Descarga métricas biomédicas para una fecha y las persiste en JSON y SQLite."""
        date_str = target_date.isoformat()
        day_dir = self.raw_data_dir / date_str
        logger.info(f"Sincronizando métricas para la fecha: {date_str}")
        results: Dict[str, bool] = {}

        # 1. Sueño detallado
        try:
            sleep_data = self.client.get_sleep_data(date_str)
            self._save_json(sleep_data, day_dir / "sleep.json")
            self.db.upsert_sleep(sleep_data)
            results["sleep"] = True
        except Exception as e:
            logger.warning(f"No se pudieron obtener datos de sueño: {e}")
            results["sleep"] = False

        # 2. HRV nocturna
        try:
            hrv_data = self.client.get_hrv_data(date_str)
            self._save_json(hrv_data, day_dir / "hrv.json")
            self.db.upsert_hrv(hrv_data)
            results["hrv"] = True
        except Exception as e:
            logger.warning(f"No se pudieron obtener datos de HRV: {e}")
            results["hrv"] = False

        # 3. Estrés y Body Battery
        try:
            stress_data = self.client.get_stress_data(date_str)
            self._save_json(stress_data, day_dir / "stress.json")
            self.db.upsert_stress(stress_data)
            results["stress"] = True
        except Exception as e:
            logger.warning(f"No se pudieron obtener datos de estrés: {e}")
            results["stress"] = False

        # 4. Resumen diario
        try:
            summary = self.client.get_user_summary(date_str)
            self._save_json(summary, day_dir / "daily_summary.json")
            self.db.upsert_daily_summary(summary)
            results["daily_summary"] = True
        except Exception as e:
            logger.warning(f"No se pudo obtener el resumen diario de actividad: {e}")
            results["daily_summary"] = False

        # 5. VO2 Max y métricas de entrenamiento
        try:
            max_metrics = self.client.get_max_metrics(date_str)
            self._save_json(max_metrics, day_dir / "max_metrics.json")
            self.db.upsert_max_metrics(max_metrics)
            results["max_metrics"] = True
        except Exception as e:
            logger.warning(f"No se pudieron obtener las métricas de VO2 Max: {e}")
            results["max_metrics"] = False

        return results

    def sync_activities(self, limit: int = 10, download_fit: bool = True) -> List[Dict[str, Any]]:
        """Descarga el resumen de actividades recientes, archivos .fit y actualiza la base de datos."""
        logger.info(f"Obteniendo las últimas {limit} actividades...")
        activities = self.client.get_activities(0, limit)
        downloaded: List[Dict[str, Any]] = []

        for act in activities:
            act_id = act.get("activityId")
            if not act_id:
                continue

            start_time = act.get("startTimeLocal", "")[:10]
            target_date_str = start_time if start_time else "unknown_date"
            act_dir = self.raw_data_dir / target_date_str / "activities"
            act_dir.mkdir(parents=True, exist_ok=True)

            # Guardar metadata estructurada
            self._save_json(act, act_dir / f"activity_{act_id}_summary.json")

            fit_path = act_dir / f"activity_{act_id}.zip"
            # Descargar archivo binario .fit
            if download_fit and not fit_path.exists():
                try:
                    logger.info(
                        f"Descargando archivo .fit de la actividad {act_id} ({act.get('activityName')})"
                    )
                    dl_fmt = getattr(
                        getattr(self.client, "ActivityDownloadFormat", None),
                        "ORIGINAL",
                        getattr(Garmin, "ActivityDownloadFormat", None).ORIGINAL
                        if hasattr(Garmin, "ActivityDownloadFormat")
                        else 1,
                    )
                    fit_data = self.client.download_activity(act_id, dl_fmt=dl_fmt)
                    with open(fit_path, "wb") as f:
                        f.write(fit_data)
                except Exception as e:
                    logger.warning(f"Error al descargar .fit para la actividad {act_id}: {e}")

            # Upsert into SQLite
            fit_zip_loc = fit_path.as_posix() if fit_path.exists() else None
            self.db.upsert_activity(act, fit_zip_path=fit_zip_loc)
            downloaded.append(act)

        return downloaded

    def run_sync_window(self, days_back: int = 7, sync_fit: bool = True) -> None:
        """Ejecuta una sincronización completa para una ventana móvil de N días hacia atrás."""
        today = date.today()
        for i in range(days_back):
            current_date = today - timedelta(days=i)
            self.sync_daily_biometrics(current_date)

        self.sync_activities(limit=days_back * 2, download_fit=sync_fit)
        logger.info("Sincronización completada exitosamente.")


if __name__ == "__main__":
    ingestor = GarminDataIngestor()
    ingestor.run_sync_window(days_back=7, sync_fit=True)
