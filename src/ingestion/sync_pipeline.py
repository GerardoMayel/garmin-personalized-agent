"""Garmin Data Sync Pipeline Entrypoint.

CLI command and entrypoint for executing automated or scheduled Garmin Connect
data synchronization, supporting rolling window sync, single-date backfills,
and persisting to SQLite and raw data partitions.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from src.common.database import GarminDatabase
from src.common.logger import get_logger
from src.ingestion.garmin_sync import GarminDataIngestor

logger = get_logger("SyncPipeline")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the Garmin sync pipeline."""
    parser = argparse.ArgumentParser(
        description="Garmin Connect Ingestion Pipeline: sync daily biometrics, activities, and .fit files a día vencido."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=15,
        help="Number of rolling days to synchronize a día vencido (default: 15 for self-healing reconciliation).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to synchronize in YYYY-MM-DD format (must be < today).",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        default=False,
        help="Skip downloading binary .fit activity archives (downloads JSON metadata only).",
    )
    parser.add_argument(
        "--reconcile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit and self-heal missing/corrupted days in the window (default: True).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download of all days in the window regardless of status.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw JSON and .fit files (default: data/raw).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to historical SQLite database (default: data/processed/garmin_history.db).",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        default=False,
        help="Upload SQLite database and raw partitions to Cloudflare R2 after sync.",
    )
    return parser.parse_args(args)


def run_pipeline(
    days_back: int = 15,
    target_date: str | None = None,
    sync_fit: bool = True,
    reconcile: bool = True,
    force: bool = False,
    raw_dir: str = "data/raw",
    db_path: str | None = None,
    r2_sync: bool = False,
) -> int:
    """Executes the data synchronization pipeline.

    Returns:
        0 on success, 1 on error.
    """
    logger.info("Iniciando pipeline de sincronización de Garmin Connect...")
    try:
        db = GarminDatabase(db_path=db_path) if db_path else GarminDatabase()
        ingestor = GarminDataIngestor(raw_data_dir=raw_dir, db=db)

        if target_date:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
            if parsed_date >= date.today():
                logger.warning(
                    f"Fecha objetivo {target_date} rechazada: regla de 'día vencido' activa. "
                    f"Solo se permite sincronizar días cerrados (< {date.today()})."
                )
                return 1

            logger.info(f"Modo fecha específica seleccionada: {parsed_date}")
            res = ingestor.sync_daily_biometrics(parsed_date)
            logger.info(f"Resultados biométricos para {parsed_date}: {res}")
            ingestor.sync_activities(limit=10, download_fit=sync_fit)
        elif reconcile:
            logger.info(
                f"Modo reconciliación autorreparable seleccionado: analizando últimos {days_back} días a día vencido"
            )
            reconcile_stats = ingestor.reconcile_and_repair_window(
                days_back=days_back, sync_fit=sync_fit, force=force
            )
            logger.info(f"Reconciliación completada: {reconcile_stats}")
        else:
            logger.info(f"Modo ventana móvil seleccionada: sincronizando últimos {days_back} días a día vencido")
            ingestor.run_sync_window(days_back=days_back, sync_fit=sync_fit)

        logger.info("Pipeline de sincronización finalizado exitosamente.")

        # Actualizar dataset limpio DVC y tabla de predicciones bloqueadas
        try:
            from src.analytics.dvc_manager import GarminDVCManager
            from src.analytics.predictions_manager import BiometricPredictionsManager

            target_db_path = Path(db_path) if db_path else Path("data/processed/garmin_history.db")
            dvc_mgr = GarminDVCManager(db_path=target_db_path)
            clean_df = dvc_mgr.update_clean_dataset()
            logger.info(f"Dataset limpio DVC actualizado: {len(clean_df)} registros disponibles.")

            pred_mgr = BiometricPredictionsManager()
            preds_df = pred_mgr.generate_and_update_forecasts()
            logger.info(
                f"Tabla de predicciones quincenales actualizada: {len(preds_df)} registros bloqueados."
            )
        except Exception as e:
            logger.warning(f"Error actualizando dataset DVC o predicciones: {e}")

        if r2_sync:
            try:
                from src.common.r2_storage import R2StorageClient

                r2 = R2StorageClient()
                if r2.is_configured():
                    logger.info(
                        "Sincronizando base de datos, DVC, predicciones y particiones raw con Cloudflare R2..."
                    )
                    r2.backup_database(
                        local_db_path=db_path or Path("data/processed/garmin_history.db")
                    )
                    r2.sync_raw_directory(raw_dir=raw_dir)
                    r2.sync_dvc_dataset()
                    r2.sync_predictions(remote_prefix="forecast")
                    r2.sync_artifacts()
                    logger.info("Sincronización con Cloudflare R2 completada con éxito.")
                else:
                    logger.warning("R2 no configurado; omitiendo subida a la nube.")
            except Exception as e:
                logger.warning(f"Error sincronizando con R2: {e}")

        return 0

    except Exception as e:
        logger.error(f"Fallo crítico en el pipeline de sincronización: {e}", exc_info=True)
        return 1


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    sync_fit = not args.no_fit
    exit_code = run_pipeline(
        days_back=args.days_back,
        target_date=args.date,
        sync_fit=sync_fit,
        reconcile=args.reconcile,
        force=args.force,
        raw_dir=args.raw_dir,
        db_path=args.db_path,
        r2_sync=args.r2_sync,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
