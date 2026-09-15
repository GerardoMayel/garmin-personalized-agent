"""Garmin Data Sync Pipeline Entrypoint.

CLI command and entrypoint for executing automated or scheduled Garmin Connect
data synchronization, supporting rolling window sync, single-date backfills,
and persisting to SQLite and raw data partitions.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from src.common.database import GarminDatabase
from src.common.logger import get_logger
from src.ingestion.garmin_sync import GarminDataIngestor

logger = get_logger("SyncPipeline")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the Garmin sync pipeline."""
    parser = argparse.ArgumentParser(
        description="Garmin Connect Ingestion Pipeline: sync daily biometrics, activities, and .fit files."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=2,
        help="Number of rolling days to synchronize (default: 2 to sync yesterday and today).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date to synchronize in YYYY-MM-DD format (overrides --days-back).",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        default=False,
        help="Skip downloading binary .fit activity archives (downloads JSON metadata only).",
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
    return parser.parse_args(args)


def run_pipeline(
    days_back: int = 2,
    target_date: str | None = None,
    sync_fit: bool = True,
    raw_dir: str = "data/raw",
    db_path: str | None = None,
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
            logger.info(f"Modo fecha específica seleccionada: {parsed_date}")
            res = ingestor.sync_daily_biometrics(parsed_date)
            logger.info(f"Resultados biométricos para {parsed_date}: {res}")
            ingestor.sync_activities(limit=10, download_fit=sync_fit)
        else:
            logger.info(f"Modo ventana móvil seleccionada: sincronizando últimos {days_back} días")
            ingestor.run_sync_window(days_back=days_back, sync_fit=sync_fit)

        logger.info("Pipeline de sincronización finalizado exitosamente.")
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
        raw_dir=args.raw_dir,
        db_path=args.db_path,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
