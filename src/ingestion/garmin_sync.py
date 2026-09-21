"""Garmin Connect Data Ingestion Module.

Automates session persistence, downloading daily health metrics (Sleep, HRV,
Stress, Body Battery, Max Metrics/VO2 Max), activity metadata, and raw .FIT
telemetry packages, and persists them into the historical SQLite database.
"""

from __future__ import annotations

import json
import os
import shutil
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

from src.common.database import GarminDatabase
from src.common.logger import get_logger

load_dotenv()
logger = get_logger("GarminIngestor")


class GarminDataIngestor:
    """Orchestrates authentication, biometrics extraction, and SQLite database persistence."""

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        raw_data_dir: str = "data/raw",
        tokenstore_dir: str | None = None,
        client: Any | None = None,
        db: GarminDatabase | None = None,
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

    def _save_json(self, data: dict[str, Any], output_path: Path) -> None:
        """Guarda un diccionario como archivo JSON formateado con identación."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def sync_daily_biometrics(
        self, target_date: date, allow_today: bool = False
    ) -> dict[str, bool]:
        """Descarga métricas biomédicas para una fecha y las persiste en JSON y SQLite.

        Respeta estrictamente la regla de 'día vencido': si target_date >= date.today()
        y allow_today es False, se rechaza la descarga para evitar persistir días inconclusos.
        """
        if not allow_today and target_date >= date.today():
            logger.warning(
                f"Omitiendo sincronización de {target_date}: regla de 'día vencido' activa. "
                f"Solo se sincronizan días cerrados (< {date.today()})."
            )
            return {}

        date_str = target_date.isoformat()
        day_dir = self.raw_data_dir / date_str
        logger.info(f"Sincronizando métricas para la fecha: {date_str}")
        results: dict[str, bool] = {}

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

        # 6. Edad de forma física (Fitness Age 2.0)
        try:
            fitness_age_data = self.client.get_fitnessage_data(date_str)
            self._save_json(fitness_age_data, day_dir / "fitness_age.json")
            self.db.upsert_fitness_age(fitness_age_data, calendar_date=date_str)
            results["fitness_age"] = True
        except Exception as e:
            logger.warning(f"No se pudieron obtener las métricas de edad física: {e}")
            results["fitness_age"] = False

        return results

    def is_day_complete_and_valid(self, target_date: date) -> tuple[bool, str]:
        """Verifica si los datos raw y la persistencia en SQLite para una fecha están completos y sin nulos corruptos."""
        if target_date >= date.today():
            return False, "fecha en curso o futura (regla de día vencido)"

        date_str = target_date.isoformat()
        day_dir = self.raw_data_dir / date_str

        # 1. Comprobar existencia del directorio
        if not day_dir.exists() or not day_dir.is_dir():
            return False, "directorio raw no existe"

        # 2. Comprobar presencia de los 6 archivos raw esenciales
        expected_files = [
            "daily_summary.json",
            "stress.json",
            "sleep.json",
            "hrv.json",
            "max_metrics.json",
            "fitness_age.json",
        ]
        for fname in expected_files:
            fpath = day_dir / fname
            if not fpath.exists():
                return False, f"archivo raw '{fname}' ausente"

        # 3. Validar contenido de daily_summary.json, stress.json y fitness_age.json
        try:
            with open(day_dir / "daily_summary.json", "r", encoding="utf-8") as f:
                daily_summary = json.load(f)
            steps = daily_summary.get("totalSteps")
            avg_stress = daily_summary.get("averageStressLevel")
            if steps is None or avg_stress is None:
                return False, "métricas clave nulas en daily_summary.json (totalSteps o averageStressLevel)"
        except Exception as e:
            return False, f"error parseando daily_summary.json: {e}"

        try:
            with open(day_dir / "stress.json", "r", encoding="utf-8") as f:
                stress_data = json.load(f)
            if stress_data.get("avgStressLevel") is None and stress_data.get("maxStressLevel") is None:
                return False, "stress.json no contiene niveles de estrés válidos"
        except Exception as e:
            return False, f"error parseando stress.json: {e}"

        try:
            with open(day_dir / "fitness_age.json", "r", encoding="utf-8") as f:
                fitness_age_data = json.load(f)
            if fitness_age_data.get("fitnessAge") is None:
                return False, "fitness_age.json no contiene fitnessAge válido"
        except Exception as e:
            return False, f"error parseando fitness_age.json: {e}"

        # 4. Validar persistencia en base de datos SQLite
        row = self.db.get_daily_summary(date_str)
        if not row:
            return False, "registro ausente en tabla daily_summaries de SQLite"
        if row.get("avg_stress_level") is None or row.get("total_steps") is None:
            return False, "registro en SQLite con columnas clave nulas"

        return True, "completo y consolidado"

    def purge_unclosed_or_future_dates(self) -> list[str]:
        """Elimina particiones raw y registros de SQLite correspondientes a fechas en curso o futuras."""
        today_str = date.today().isoformat()
        purged: list[str] = []

        # 1. Limpieza de SQLite
        del_db_stats = self.db.delete_records_on_or_after(today_str)
        logger.info(f"Purga de registros SQLite >= {today_str}: {del_db_stats}")

        # 2. Limpieza de data/raw/
        if self.raw_data_dir.exists():
            for item in self.raw_data_dir.iterdir():
                if item.is_dir() and item.name >= today_str:
                    try:
                        date.fromisoformat(item.name)
                        shutil.rmtree(item)
                        purged.append(item.name)
                        logger.warning(f"Partición raw no cerrada eliminada: {item.as_posix()}")
                    except ValueError:
                        continue

        return purged

    def sync_activities(self, limit: int = 10, download_fit: bool = True) -> list[dict[str, Any]]:
        """Descarga el resumen de actividades recientes, archivos .fit y actualiza la base de datos."""
        logger.info(f"Obteniendo las últimas {limit} actividades...")
        activities = self.client.get_activities(0, limit)
        downloaded: list[dict[str, Any]] = []

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
                    dl_fmt_cls = getattr(Garmin, "ActivityDownloadFormat", None)
                    dl_fmt = getattr(dl_fmt_cls, "ORIGINAL", 1) if dl_fmt_cls else 1
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

    def reconcile_and_repair_window(
        self, days_back: int = 15, sync_fit: bool = True, force: bool = False
    ) -> dict[str, list[str]]:
        """Inspecciona y repara automáticamente cualquier día con archivos faltantes o métricas nulas en la ventana móvil."""
        logger.info(
            f"Iniciando reconciliación y autorreparación de los últimos {days_back} días a día vencido..."
        )
        stats: dict[str, list[str]] = {
            "skipped": [],
            "repaired": [],
            "failed": [],
            "purged": [],
        }

        # Purgar cualquier partición o registro >= hoy
        stats["purged"] = self.purge_unclosed_or_future_dates()

        yesterday = date.today() - timedelta(days=1)
        for i in range(days_back):
            current_date = yesterday - timedelta(days=i)
            date_str = current_date.isoformat()

            is_valid, reason = self.is_day_complete_and_valid(current_date)
            if is_valid and not force:
                logger.info(f"Día {date_str} verificado y consolidado ({reason}). Omitiendo.")
                stats["skipped"].append(date_str)
                continue

            logger.info(
                f"Día {date_str} requiere reconciliación/reparación ({reason}). Descargando de Garmin Connect..."
            )
            try:
                res = self.sync_daily_biometrics(current_date, allow_today=False)
                if any(res.values()):
                    stats["repaired"].append(date_str)
                else:
                    stats["failed"].append(date_str)
            except Exception as e:
                logger.error(f"Error reparando el día {date_str}: {e}")
                stats["failed"].append(date_str)

        self.sync_activities(limit=days_back * 2, download_fit=sync_fit)
        logger.info(f"Reconciliación y autorreparación completada: {stats}")
        return stats

    def run_sync_window(self, days_back: int = 15, sync_fit: bool = True) -> None:
        """Ejecuta una sincronización completa para una ventana móvil a día vencido (T-1 hacia atrás)."""
        yesterday = date.today() - timedelta(days=1)
        for i in range(days_back):
            current_date = yesterday - timedelta(days=i)
            self.sync_daily_biometrics(current_date)

        self.sync_activities(limit=days_back * 2, download_fit=sync_fit)
        logger.info("Sincronización a día vencido completada exitosamente.")


if __name__ == "__main__":
    from src.ingestion.sync_pipeline import main

    main()
