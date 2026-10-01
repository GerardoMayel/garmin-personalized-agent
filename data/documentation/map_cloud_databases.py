#!/usr/bin/env python3
"""Generador de Documentación y Mapeo de Bases de Datos SQLite en Cloud Storage.

Este script se conecta al almacenamiento Cloudflare R2 ('garmin-personal-data'),
escanea todas las bases de datos SQLite (.db, .sqlite, .sqlite3), extrae sus
esquemas técnicos, tipos de datos, claves primarias, relaciones y un TOP 3 de
ejemplos reales para cada columna.

Genera automáticamente los siguientes archivos de texto en data/documentation/:
1. 01_catalogo_bases_y_tablas.txt: Inventario de bases de datos, tablas y vistas.
2. 02_diagrama_entidad_relacion.txt: Diagrama Entidad-Relación en formato texto (ASCII).
3. 03_mapeo_columnas_tipos_y_muestras.txt: Diccionario detallado de tipos técnicos y ejemplos.
4. 00_resumen_esquema_consolidado.txt: Documento consolidado con el esquema integral.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

try:
    from src.common.logger import get_logger
    from src.common.r2_storage import R2StorageClient

    logger = get_logger("DBDocumentation")
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("DBDocumentation")  # type: ignore[assignment]
    R2StorageClient = None  # type: ignore[assignment, misc]


# =====================================================================
# Modelos de Datos en Memoria para Análisis
# =====================================================================

class ColumnInfo:
    """Información técnica y muestras de una columna."""

    def __init__(
        self,
        cid: int,
        name: str,
        declared_type: str,
        notnull: bool,
        default_value: Any,
        pk: int,
        samples: list[str],
    ) -> None:
        self.cid = cid
        self.name = name
        self.declared_type = declared_type or "DYNAMIC/TEXT"
        self.notnull = notnull
        self.default_value = default_value
        self.pk = pk
        self.samples = samples

    @property
    def pk_badge(self) -> str:
        if self.pk == 1:
            return "[PK]"
        elif self.pk > 1:
            return f"[PK-{self.pk}]"
        return "    "

    @property
    def constraint_str(self) -> str:
        constraints = []
        if self.pk > 0:
            constraints.append("PRIMARY KEY" if self.pk == 1 else f"PK (part {self.pk})")
        if self.notnull:
            constraints.append("NOT NULL")
        if self.default_value is not None:
            constraints.append(f"DEFAULT {self.default_value}")
        return ", ".join(constraints) if constraints else "NULLABLE"


class TableInfo:
    """Información completa de una tabla o vista."""

    def __init__(
        self,
        name: str,
        type_: str,
        sql: str,
        row_count: int,
        columns: list[ColumnInfo],
        foreign_keys: list[dict[str, Any]],
    ) -> None:
        self.name = name
        self.type_ = type_  # 'table' o 'view'
        self.sql = sql
        self.row_count = row_count
        self.columns = columns
        self.foreign_keys = foreign_keys

    @property
    def pk_columns(self) -> list[str]:
        return [c.name for c in sorted(self.columns, key=lambda x: x.pk) if c.pk > 0]


class DatabaseAnalysis:
    """Análisis completo de una base de datos SQLite."""

    def __init__(
        self,
        filename: str,
        remote_key: str,
        size_bytes: int,
        source: str,
        tables: list[TableInfo],
    ) -> None:
        self.filename = filename
        self.remote_key = remote_key
        self.size_bytes = size_bytes
        self.source = source  # 'Cloudflare R2' o 'Local Cache'
        self.tables = tables

    @property
    def table_count(self) -> int:
        return sum(1 for t in self.tables if t.type_ == "table")

    @property
    def view_count(self) -> int:
        return sum(1 for t in self.tables if t.type_ == "view")

    @property
    def total_rows(self) -> int:
        return sum(t.row_count for t in self.tables if t.type_ == "table")


# =====================================================================
# Extractor y Analizador de SQLite
# =====================================================================

def format_sample_value(val: Any) -> str:
    """Convierte un valor de muestra a un formato textual legible y conciso."""
    if val is None:
        return "NULL"
    if isinstance(val, (bytes, bytearray)):
        return f"<BLOB {len(val)}B: 0x{val[:8].hex()}...>"
    if isinstance(val, float):
        if val.is_integer():
            return f"{int(val)}"
        return f"{val:.4g}"
    val_str = str(val).strip()
    # Reemplazar saltos de línea y tabulaciones para formato en tablas txt
    val_str = val_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Truncar si es demasiado extenso (ej. JSON largo de parámetros)
    if len(val_str) > 42:
        return f"'{val_str[:39]}...'"
    if isinstance(val, str):
        return f"'{val_str}'"
    return val_str


def analyze_sqlite_file(
    file_path: Path,
    filename: str,
    remote_key: str,
    size_bytes: int,
    source: str,
    max_samples: int = 3,
) -> DatabaseAnalysis:
    """Conecta a una base SQLite y analiza todas sus tablas, tipos y ejemplos."""
    conn = sqlite3.connect(file_path)
    cur = conn.cursor()

    # Obtener todas las tablas y vistas excluyendo las internas de SQLite
    cur.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
        ORDER BY type ASC, name ASC
        """
    )
    raw_entities = cur.fetchall()

    tables: list[TableInfo] = []

    for name, entity_type, sql in raw_entities:
        # 1. Total de registros
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{name}"')
            row_count = cur.fetchone()[0]
        except sqlite3.Error as err:
            logger.warning(f"No se pudo contar filas en {name}: {err}")
            row_count = 0

        # 2. Columnas y tipos técnicos
        cur.execute(f'PRAGMA table_info("{name}")')
        col_tuples = cur.fetchall()

        columns: list[ColumnInfo] = []
        for cid, col_name, col_type, notnull, dflt, pk in col_tuples:
            # 3. Top N ejemplos reales (distintos y no nulos)
            samples: list[str] = []
            try:
                cur.execute(
                    f'SELECT DISTINCT "{col_name}" FROM "{name}" WHERE "{col_name}" IS NOT NULL LIMIT {max_samples}'
                )
                raw_samples = cur.fetchall()
                samples = [format_sample_value(r[0]) for r in raw_samples]
            except sqlite3.Error as err:
                logger.debug(f"Error extrayendo muestras de {name}.{col_name}: {err}")

            if not samples:
                samples = ["[Sin registros / Todos NULL]"]

            columns.append(
                ColumnInfo(
                    cid=cid,
                    name=col_name,
                    declared_type=col_type or "UNSPECIFIED",
                    notnull=bool(notnull),
                    default_value=dflt,
                    pk=pk,
                    samples=samples,
                )
            )

        # 4. Claves foráneas (si las hay configuradas a nivel DDL)
        foreign_keys: list[dict[str, Any]] = []
        try:
            cur.execute(f'PRAGMA foreign_key_list("{name}")')
            for fk_row in cur.fetchall():
                foreign_keys.append(
                    {
                        "id": fk_row[0],
                        "seq": fk_row[1],
                        "table": fk_row[2],
                        "from": fk_row[3],
                        "to": fk_row[4],
                        "on_update": fk_row[5],
                        "on_delete": fk_row[6],
                    }
                )
        except sqlite3.Error:
            pass

        tables.append(
            TableInfo(
                name=name,
                type_=entity_type,
                sql=sql or "",
                row_count=row_count,
                columns=columns,
                foreign_keys=foreign_keys,
            )
        )

    conn.close()

    return DatabaseAnalysis(
        filename=filename,
        remote_key=remote_key,
        size_bytes=size_bytes,
        source=source,
        tables=tables,
    )


# =====================================================================
# Descubrimiento de Bases en Cloud Storage (Cloudflare R2)
# =====================================================================

def fetch_and_analyze_databases(
    use_local_only: bool = False,
    max_samples: int = 3,
) -> list[DatabaseAnalysis]:
    """Descarga e inspecciona todas las bases SQLite de R2 o del disco local."""
    analyses: list[DatabaseAnalysis] = []

    r2_client = None
    if not use_local_only and R2StorageClient is not None:
        client_instance = R2StorageClient()
        if client_instance.is_configured():
            ok, msg = client_instance.test_connection()
            if ok:
                r2_client = client_instance
            else:
                logger.warning(f"R2 no alcanzable ({msg}). Se usará almacenamiento local.")
        else:
            logger.warning("Credenciales de R2 incompletas. Se buscará en base de datos local.")

    if r2_client is not None:
        logger.info(f"Escaneando bases SQLite en Cloudflare R2 bucket: '{r2_client.bucket_name}'...")
        paginator = r2_client._client.get_paginator("list_objects_v2")

        remote_db_keys: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=r2_client.bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = Path(key).suffix.lower()
                # Filtrar solo archivos con extensiones de bases de datos relacionales SQLite
                if ext in [".db", ".sqlite", ".sqlite3"] and obj["Size"] > 0:
                    remote_db_keys.append({"key": key, "size": obj["Size"], "modified": obj.get("LastModified")})

        logger.info(f"Se encontraron {len(remote_db_keys)} bases de datos SQLite en Cloudflare R2.")

        with tempfile.TemporaryDirectory(prefix="r2_sqlite_inspect_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for item in remote_db_keys:
                key = item["key"]
                size = item["size"]
                filename = Path(key).name
                local_dest = tmp_path / f"{Path(key).stem}_{size}.db"

                logger.info(f"Descargando 'r2://{r2_client.bucket_name}/{key}' ({size:,} bytes)...")
                try:
                    r2_client._client.download_file(r2_client.bucket_name, key, str(local_dest))
                    db_analysis = analyze_sqlite_file(
                        file_path=local_dest,
                        filename=filename,
                        remote_key=key,
                        size_bytes=size,
                        source=f"Cloudflare R2 (r2://{r2_client.bucket_name}/{key})",
                        max_samples=max_samples,
                    )
                    analyses.append(db_analysis)
                    logger.info(
                        f"Mapeada base '{filename}': {db_analysis.table_count} tablas, "
                        f"{db_analysis.view_count} vistas, {db_analysis.total_rows:,} registros."
                    )
                except Exception as err:
                    logger.error(f"Error procesando base remota '{key}': {err}")

    # Fallback si no se encontró nada en R2 o se forzó local
    if not analyses:
        logger.info("Buscando bases de datos SQLite locales en 'data/'...")
        local_db_paths = [
            PROJECT_ROOT / "data/processed/garmin_history.db",
            PROJECT_ROOT / "data/processed/predictions/weekly_biometric_forecasts.db",
            PROJECT_ROOT / "data/garmin_personal.db",
        ]
        for p in local_db_paths:
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                logger.info(f"Analizando base local: {p}")
                analyses.append(
                    analyze_sqlite_file(
                        file_path=p,
                        filename=p.name,
                        remote_key=str(p.relative_to(PROJECT_ROOT)),
                        size_bytes=p.stat().st_size,
                        source="Almacenamiento Local (Fallback)",
                        max_samples=max_samples,
                    )
                )

    return analyses


# =====================================================================
# Generadores de Formato de Texto (Diagramas y Tablas)
# =====================================================================

def generate_catalog_doc(databases: list[DatabaseAnalysis]) -> str:
    """Genera 01_catalogo_bases_y_tablas.txt: inventario de bases, tablas y tamaños."""
    lines: list[str] = []
    sep_double = "=" * 90
    sep_single = "-" * 90

    lines.append(sep_double)
    lines.append("        CATÁLOGO E INVENTARIO DE BASES DE DATOS SQLITE EN CLOUD STORAGE")
    lines.append("                  GARMIN PERSONAL INSIGHT AGENT (CLOUD R2)")
    lines.append(sep_double)
    lines.append(f"Fecha de Generación: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Total de Bases de Datos SQLite Analizadas: {len(databases)}")
    lines.append("")

    # Tabla resumen general
    lines.append("RESUMEN DE BASES DE DATOS:")
    lines.append(sep_single)
    lines.append(
        f"{'#':<3} | {'Nombre Archivo':<32} | {'Ubicación en Storage':<32} | {'Tamaño':<10} | {'Tablas':<6} | {'Vistas':<6}"
    )
    lines.append(sep_single)

    total_tables = 0
    total_views = 0
    total_bytes = 0

    for idx, db in enumerate(databases, 1):
        size_str = f"{db.size_bytes / 1024:.1f} KB"
        lines.append(
            f"{idx:<3} | {db.filename:<32} | {db.remote_key:<32} | {size_str:<10} | {db.table_count:<6} | {db.view_count:<6}"
        )
        total_tables += db.table_count
        total_views += db.view_count
        total_bytes += db.size_bytes

    lines.append(sep_single)
    lines.append(
        f"{'TOT':<3} | {len(databases)} Base(s) de Datos          | {'Bucket garmin-personal-data':<32} | "
        f"{total_bytes / 1024:.1f} KB   | {total_tables:<6} | {total_views:<6}"
    )
    lines.append("")
    lines.append("")

    # Detalle por base de datos y desglose de tablas
    for idx, db in enumerate(databases, 1):
        lines.append(sep_double)
        lines.append(f"[{idx}] BASE DE DATOS: {db.filename}")
        lines.append(f"    Ruta Cloud: {db.remote_key}")
        lines.append(f"    Origen:     {db.source}")
        lines.append(f"    Tamaño:     {db.size_bytes:,} bytes ({db.size_bytes / 1024:.1f} KB)")
        lines.append(f"    Estructura: {db.table_count} Tablas relacionales, {db.view_count} Vistas SQL")
        lines.append(f"    Registros:  {db.total_rows:,} filas totales almacenadas")
        lines.append(sep_single)
        lines.append(
            f"    {'Tipo':<8} | {'Nombre de Elemento':<35} | {'Cols':<5} | {'Filas':<8} | {'Clave Primaria [PK]':<25}"
        )
        lines.append("    " + "-" * 86)

        for table in db.tables:
            pk_display = ", ".join(table.pk_columns) if table.pk_columns else "[Sin PK explícita]"
            col_count = len(table.columns)
            row_count_str = f"{table.row_count:,}"
            lines.append(
                f"    {table.type_.upper():<8} | {table.name:<35} | {col_count:<5} | {row_count_str:<8} | {pk_display:<25}"
            )

        lines.append("")
        lines.append("")

    lines.append(sep_double)
    lines.append("FIN DEL CATÁLOGO DE BASES DE DATOS")
    lines.append(sep_double)
    return "\n".join(lines)


def generate_er_diagram_doc(databases: list[DatabaseAnalysis]) -> str:
    """Genera 02_diagrama_entidad_relacion.txt: Diagramas ER en ASCII y relaciones."""
    lines: list[str] = []
    sep_double = "=" * 90
    sep_single = "-" * 90

    lines.append(sep_double)
    lines.append("           DIAGRAMA ENTIDAD-RELACIÓN EN TEXTO (ASCII ER DIAGRAM)")
    lines.append("                  GARMIN PERSONAL INSIGHT AGENT (CLOUD R2)")
    lines.append(sep_double)
    lines.append(f"Fecha de Generación: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("Formato: Diagrama Textual con Cajas de Entidades, Claves Primarias/Foráneas y Cardinalidades")
    lines.append("")

    for db in databases:
        lines.append(sep_single)
        lines.append(f"BASE DE DATOS: {db.filename} (Ruta Storage: {db.remote_key})")
        lines.append(sep_single)
        lines.append("")

        # Caso 1: garmin_history.db (Arquitectura Biométrico-Longitudinal y Tablas Consolidadas)
        if "garmin_history" in db.filename or "garmin_personal" in db.filename:
            lines.append("""
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MAPA DE RELACIONES LONGITUDINALES Y CONSOLIDACIÓN (CLAVE MAESTRA: calendar_date)       │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [ TABLAS ATÓMICAS DIARIAS ]                   [ TABLA HUB MAESTRA CONSOLIDADA ]
                                                             
  ┌───────────────────────────────┐                          ┌─────────────────────────────────────┐
  │ daily_summaries               │                          │ consolidated_daily_actuals          │
  ├───────────────────────────────┤                          ├─────────────────────────────────────┤
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───►│ * [PK] calendar_date (TEXT)         │
  │   total_steps (INTEGER)       │                          │   total_steps (INTEGER)             │
  │   total_distance_meters (REAL)│                          │   resting_heart_rate (INTEGER)      │
  │   active_kilocalories (REAL)  │                          │   hrv_rmssd (REAL)                  │
  │   resting_heart_rate (INTEGER)│                          │   daily_avg_stress (INTEGER)        │
  └───────────────────────────────┘                          │   sleep_score (INTEGER)             │
                                                             │   deep_sleep_hours (REAL)           │
  ┌───────────────────────────────┐                          │   avg_spo2 (REAL)                   │
  │ sleep_records                 │                          │   active_kilocalories (REAL)        │
  ├───────────────────────────────┤                          │   fitness_age (REAL)                │
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───►│   ... (+39 métricas aplanadas)      │
  │   sleep_score (INTEGER)       │                          └──────────────────┬──────────────────┘
  │   deep_sleep_seconds (INTEGER)│                                             │
  │   rem_sleep_seconds (INTEGER) │                                             │
  │   avg_spo2 (REAL)             │                                             │
  └───────────────────────────────┘                                             │
                                                                                │
  ┌───────────────────────────────┐                                             │
  │ hrv_records                   │                                             │
  ├───────────────────────────────┤                                             │
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───────────────────────┤
  │   last_night_avg (REAL)       │                                             │
  │   weekly_avg (REAL)           │                                             │
  │   status (TEXT)               │                                             │
  └───────────────────────────────┘                                             │
                                                                                │
  ┌───────────────────────────────┐                                             │
  │ stress_records                │                                             │
  ├───────────────────────────────┤                                             │
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───────────────────────┤
  │   avg_stress_level (INTEGER)  │                                             │
  │   max_stress_level (INTEGER)  │                                             │
  │   rest_stress_duration (INT)  │                                             │
  └───────────────────────────────┘                                             │
                                                                                │
  ┌───────────────────────────────┐                                             │
  │ fitness_age_records           │                                             │
  ├───────────────────────────────┤                                             │
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───────────────────────┤
  │   fitness_age (REAL)          │                                             │
  │   fitness_age_gap (REAL)      │                                             │
  └───────────────────────────────┘                                             │
                                                                                │
  ┌───────────────────────────────┐                                             │
  │ max_metrics                   │                                             │
  ├───────────────────────────────┤                                             │
  │ * [PK] calendar_date (TEXT)   │───( 1 : 1 Consolida )───────────────────────┤
  │   vo2_max_running (REAL)      │                                             │
  │   vo2_max_precise (REAL)      │                                             │
  └───────────────────────────────┘                                             │
                                                                                │
  ┌───────────────────────────────┐                                             │
  │ activities                    │                                             │
  ├───────────────────────────────┤                                             │
  │ * [PK] activity_id (INTEGER)  │                                             │
  │   [FK] calendar_date (TEXT)   │───( N : 1 Agrega )──────────────────────────┘
  │   activity_name (TEXT)        │
  │   activity_type (TEXT)        │
  │   distance_meters (REAL)      │
  │   duration_seconds (REAL)     │
  └───────────────────────────────┘
                                                 │ (Unión Temporal Histórica)
                                                 ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ VISTA SQL: unified_biometrics_timeline                                                 │
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ * calendar_date (TEXT)                                                                 │
  │   record_type (TEXT: 'ACTUAL' | 'FORECAST')                                            │
  │   resting_heart_rate, hrv_rmssd, daily_avg_stress, sleep_score, total_steps, calorías... │
  └──────────────────────────────────────────────▲─────────────────────────────────────────┘
                                                 │ (Empalme de Proyecciones Futuras)
  ┌──────────────────────────────────────────────┴─────────────────────────────────────────┐
  │ consolidated_biometric_forecasts                                                       │
  ├────────────────────────────────────────────────────────────────────────────────────────┤
  │ * [PK] target_date (TEXT)            (Fecha proyectada a futuro)                       │
  │ * [PK] metric (TEXT)                 (Métrica pronosticada: RHR, HRV, Estrés, Pasos...) │
  │   forecast_generated_date (TEXT)     (Fecha de origen de telemetría cerrada)           │
  │   predicted_value (REAL)             (Valor central proyectado por el ensamble ML)     │
  │   ci_lower, ci_upper (REAL)          (Intervalos de confianza al 95%)                  │
  │   model_name (TEXT)                  (Ensemble_Prophet_HoltWinters, etc.)              │
  └────────────────────────────────────────────────────────────────────────────────────────┘
""")

        # Caso 2: weekly_biometric_forecasts.db (Modelos de Series Temporales y Ensamble)
        elif "forecast" in db.filename:
            lines.append("""
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ARQUITECTURA DE SERIES TEMPORALES, METADATOS Y PREDICCIONES FUTURAS                   │
└────────────────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────┐
  │ forecast_metadata                             │
  ├───────────────────────────────────────────────┤
  │ * [PK] run_id (INTEGER)                       │
  │   forecast_generated_date (TEXT)              │◄─────────────────────────────────────┐
  │   horizon_start (TEXT)                        │                                      │
  │   horizon_end (TEXT)                          │                                      │
  │   total_horizon_days (INTEGER)                │                                      │
  │   metrics_count (INTEGER)                     │                                      │
  │   metrics_list (TEXT: JSON array)             │                                      │
  │   total_locked_records (INTEGER)              │                                      │
  │   created_at (TEXT)                           │                                      │
  └───────────────────────┬───────────────────────┘                                      │
                          │                                                              │
                          │ ( 1 : N ) Cada corrida de pronóstico genera predicciones     │
                          │           para múltiples métricas y días hacia adelante      │
                          ▼                                                              │
  ┌───────────────────────────────────────────────┐                                      │
  │ biometric_forecasts                           │                                      │
  ├───────────────────────────────────────────────┤                                      │
  │ * [PK] target_date (TEXT)                     │                                      │
  │ * [PK] metric (TEXT)                          │                                      │
  │   [FK] forecast_generated_date (TEXT)         │──────────────────────────────────────┘
  │   predicted_value (REAL)                      │
  │   ci_lower (REAL)                             │ (Intervalo de Confianza Inferior 95%)
  │   ci_upper (REAL)                             │ (Intervalo de Confianza Superior 95%)
  │   model_name (TEXT)                           │ (Algoritmo: Ensemble_Prophet_HoltWinters)
  │   is_locked (INTEGER: 1=Locked, 0=Draft)      │ (Garantiza inmutabilidad histórica)
  │   created_at (TEXT)                           │
  └───────────────────────────────────────────────┘
""")

        # Detalle de Tablas Individuales (Representación en Cajas ASCII)
        lines.append("DETALLE DE ENTIDADES Y CAMPOS CLAVE:")
        lines.append("")

        for t in db.tables:
            badge = "[VISTA]" if t.type_ == "view" else "[TABLA]"
            pk_info = ", ".join(t.pk_columns) if t.pk_columns else "Sin PK"
            header = f"+--- {badge} {t.name} (Filas: {t.row_count:,} | PK: {pk_info}) "
            box_width = 86
            box_top = header + "-" * max(0, box_width - len(header)) + "+"
            box_bot = "+" + "-" * (box_width - 1) + "+"

            lines.append(box_top)
            for c in t.columns:
                pk_mark = "[PK] " if c.pk > 0 else "     "
                null_mark = "NOT NULL" if c.notnull else "NULL    "
                type_str = f"({c.declared_type})"
                c_line = f"|  {pk_mark}{c.name:<32} {type_str:<18} {null_mark}"
                c_line = c_line + " " * max(0, box_width - len(c_line) - 1) + "|"
                lines.append(c_line)
            lines.append(box_bot)
            lines.append("")

    lines.append(sep_double)
    lines.append("FIN DEL DIAGRAMA ENTIDAD-RELACIÓN")
    lines.append(sep_double)
    return "\n".join(lines)


def generate_columns_and_samples_doc(
    databases: list[DatabaseAnalysis],
    max_samples: int = 3,
) -> str:
    """Genera 03_mapeo_columnas_tipos_y_muestras.txt con nomenclatura técnica y top 3 muestras."""
    lines: list[str] = []
    sep_double = "=" * 105
    sep_single = "-" * 105

    lines.append(sep_double)
    lines.append("        MAPEO DETALLADO DE COLUMNAS, TIPOS DE DATOS TÉCNICOS Y EJEMPLOS REALES")
    lines.append("                      GARMIN PERSONAL INSIGHT AGENT (CLOUD R2)")
    lines.append(sep_double)
    lines.append(f"Fecha de Generación: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Muestras Extraídas por Columna: TOP {max_samples} valores reales distintos no nulos")
    lines.append("")

    for db_idx, db in enumerate(databases, 1):
        lines.append(sep_double)
        lines.append(f"BASE DE DATOS [{db_idx}/{len(databases)}]: {db.filename}")
        lines.append(f"Ubicación en Storage: {db.remote_key}")
        lines.append(f"Tamaño Físico:        {db.size_bytes:,} bytes ({db.size_bytes / 1024:.1f} KB)")
        lines.append(sep_double)
        lines.append("")

        for t_idx, t in enumerate(db.tables, 1):
            entity_label = "VISTA SQL" if t.type_ == "view" else "TABLA"
            lines.append(sep_single)
            lines.append(
                f"[{db_idx}.{t_idx}] {entity_label}: {t.name} (Total Registros: {t.row_count:,} | Columnas: {len(t.columns)})"
            )
            if t.pk_columns:
                lines.append(f"      Clave Primaria: {', '.join(t.pk_columns)}")
            lines.append(sep_single)

            lines.append(
                f"{'#':<3} | {'Columna':<30} | {'Tipo Técnico':<14} | {'Restricción':<18} | {'TOP 3 Ejemplos del Dato Contenido':<32}"
            )
            lines.append("-" * 105)

            for col in t.columns:
                samples_str = ", ".join(col.samples)
                lines.append(
                    f"{col.cid:<3} | {col.name:<30} | {col.declared_type:<14} | {col.constraint_str:<18} | {samples_str}"
                )

            lines.append("")
            lines.append("")

    lines.append(sep_double)
    lines.append("FIN DEL MAPEO DE COLUMNAS Y TIPOS TÉCNICOS")
    lines.append(sep_double)
    return "\n".join(lines)


def generate_master_doc(
    catalog_text: str,
    er_diagram_text: str,
    columns_text: str,
) -> str:
    """Genera 00_resumen_esquema_consolidado.txt que integra los 3 documentos."""
    lines: list[str] = []
    lines.append("=" * 105)
    lines.append("           ESQUEMA INTEGRAL CONSOLIDADO DE BASES DE DATOS SQLITE (CLOUD R2)")
    lines.append("                        GARMIN PERSONAL INSIGHT AGENT")
    lines.append("=" * 105)
    lines.append("")
    lines.append("Este documento reúne:")
    lines.append("  1. Catálogo e Inventario de Bases de Datos y Tablas.")
    lines.append("  2. Diagrama Entidad-Relación Textual (ASCII ERD).")
    lines.append("  3. Diccionario Completo de Columnas, Tipos Técnicos y Top 3 Muestras.")
    lines.append("")
    lines.append("\n" + "=" * 105 + "\n")
    lines.append(catalog_text)
    lines.append("\n" + "=" * 105 + "\n")
    lines.append(er_diagram_text)
    lines.append("\n" + "=" * 105 + "\n")
    lines.append(columns_text)
    return "\n".join(lines)


# =====================================================================
# Orquestador Principal y Punto de Entrada CLI
# =====================================================================

def generate_database_documentation(
    output_dir: Path,
    use_local_only: bool = False,
    max_samples: int = 3,
) -> dict[str, Path]:
    """Ejecuta el pipeline completo de análisis y exporta los archivos .txt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Iniciando análisis de bases de datos para documentar en: {output_dir}")

    databases = fetch_and_analyze_databases(use_local_only=use_local_only, max_samples=max_samples)

    if not databases:
        logger.error("No se encontraron bases de datos SQLite para analizar.")
        return {}

    # 1. Generar contenidos
    logger.info("Generando Catálogo e Inventario de tablas...")
    catalog_content = generate_catalog_doc(databases)

    logger.info("Generando Diagrama Entidad-Relación en formato texto...")
    er_content = generate_er_diagram_doc(databases)

    logger.info("Generando Mapeo de columnas, tipos técnicos y ejemplos de datos...")
    columns_content = generate_columns_and_samples_doc(databases, max_samples=max_samples)

    logger.info("Generando Documento consolidado maestro...")
    master_content = generate_master_doc(catalog_content, er_content, columns_content)

    # 2. Guardar archivos de texto
    files_map = {
        "catalog": output_dir / "01_catalogo_bases_y_tablas.txt",
        "er_diagram": output_dir / "02_diagrama_entidad_relacion.txt",
        "columns_mapping": output_dir / "03_mapeo_columnas_tipos_y_muestras.txt",
        "master": output_dir / "00_resumen_esquema_consolidado.txt",
    }

    files_map["catalog"].write_text(catalog_content, encoding="utf-8")
    files_map["er_diagram"].write_text(er_content, encoding="utf-8")
    files_map["columns_mapping"].write_text(columns_content, encoding="utf-8")
    files_map["master"].write_text(master_content, encoding="utf-8")

    for key, file_path in files_map.items():
        logger.info(f"Guardado: {file_path.name} ({file_path.stat().st_size:,} bytes)")

    return files_map


def main() -> None:
    """CLI para ejecutar la documentación de bases de datos."""
    parser = argparse.ArgumentParser(
        description="Mapeador y Generador de Documentación de Bases SQLite en Cloud Storage."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data/documentation"),
        help="Directorio destino para guardar los archivos TXT (def: data/documentation)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Analizar únicamente bases SQLite locales sin conectar a Cloudflare R2.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Número de valores de muestra a extraer por columna (def: 3)",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generated_files = generate_database_documentation(
        output_dir=out_path,
        use_local_only=args.local,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 80)
    print("🎉 DOCUMENTACIÓN DE BASES DE DATOS GENERADA CON ÉXITO")
    print("=" * 80)
    print(f"Directorio de salida: {out_path.resolve()}")
    for key, path in generated_files.items():
        print(f"  - [{key.upper()}] {path.name} ({path.stat().st_size:,} bytes)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
