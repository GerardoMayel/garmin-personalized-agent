"""Unit tests for Cloud SQLite Database Documentation Generator."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from data.documentation.map_cloud_databases import (
    analyze_sqlite_file,
    format_sample_value,
    generate_catalog_doc,
    generate_columns_and_samples_doc,
    generate_database_documentation,
    generate_er_diagram_doc,
)


def test_format_sample_value():
    """Verify safe formatting and string truncation for diverse SQLite datatypes."""
    assert format_sample_value(None) == "NULL"
    assert format_sample_value(123) == "123"
    assert format_sample_value(45.0) == "45"
    assert format_sample_value(51.724) == "51.72"
    assert format_sample_value("2026-09-01") == "'2026-09-01'"
    assert format_sample_value(b"\x00\x01\x02\x03") == "<BLOB 4B: 0x00010203...>"

    long_str = "a" * 100
    formatted_long = format_sample_value(long_str)
    assert len(formatted_long) <= 45
    assert formatted_long.endswith("...'")


@pytest.fixture
def sample_sqlite_db():
    """Creates a temporary SQLite database with tables, views, and distinct data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            height_m REAL DEFAULT 1.75
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE metrics (
            calendar_date TEXT,
            metric_type TEXT,
            value REAL,
            PRIMARY KEY (calendar_date, metric_type)
        )
        """
    )
    cur.execute("CREATE VIEW v_users_summary AS SELECT user_id, username FROM users")

    cur.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        [
            (1, "atleta_01", "atleta1@garmin.com", 1.80),
            (2, "atleta_02", "atleta2@garmin.com", 1.72),
            (3, "atleta_03", None, 1.68),
        ],
    )
    cur.executemany(
        "INSERT INTO metrics VALUES (?, ?, ?)",
        [
            ("2026-09-01", "hrv_rmssd", 45.2),
            ("2026-09-02", "hrv_rmssd", 48.0),
            ("2026-09-03", "hrv_rmssd", 43.1),
        ],
    )
    conn.commit()
    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


def test_analyze_sqlite_file(sample_sqlite_db: Path):
    """Verify extraction of table structures, technical data types, and sample data."""
    analysis = analyze_sqlite_file(
        file_path=sample_sqlite_db,
        filename="test_database.db",
        remote_key="test/test_database.db",
        size_bytes=sample_sqlite_db.stat().st_size,
        source="Test Environment",
        max_samples=3,
    )

    assert analysis.filename == "test_database.db"
    assert analysis.table_count == 2
    assert analysis.view_count == 1
    assert analysis.total_rows == 6

    tables_by_name = {t.name: t for t in analysis.tables}
    assert "users" in tables_by_name
    assert "metrics" in tables_by_name
    assert "v_users_summary" in tables_by_name

    # Validate users table
    users_t = tables_by_name["users"]
    assert users_t.row_count == 3
    assert users_t.pk_columns == ["user_id"]

    cols_by_name = {c.name: c for c in users_t.columns}
    assert cols_by_name["user_id"].declared_type == "INTEGER"
    assert cols_by_name["user_id"].pk == 1
    assert "1" in cols_by_name["user_id"].samples

    assert cols_by_name["username"].declared_type == "TEXT"
    assert cols_by_name["username"].notnull is True
    assert "'atleta_01'" in cols_by_name["username"].samples

    # Validate composite PK in metrics
    metrics_t = tables_by_name["metrics"]
    assert metrics_t.pk_columns == ["calendar_date", "metric_type"]


def test_documentation_generation_pipeline(sample_sqlite_db: Path):
    """Verify formatting of catalog, ER diagram, and column dictionary."""
    analysis = analyze_sqlite_file(
        file_path=sample_sqlite_db,
        filename="test_database.db",
        remote_key="test/test_database.db",
        size_bytes=sample_sqlite_db.stat().st_size,
        source="Test Environment",
        max_samples=3,
    )

    catalog_txt = generate_catalog_doc([analysis])
    assert "test_database.db" in catalog_txt
    assert "users" in catalog_txt
    assert "metrics" in catalog_txt

    er_txt = generate_er_diagram_doc([analysis])
    assert "test_database.db" in er_txt
    assert "users" in er_txt
    assert "[PK]" in er_txt

    columns_txt = generate_columns_and_samples_doc([analysis], max_samples=3)
    assert "users" in columns_txt
    assert "user_id" in columns_txt
    assert "atleta_01" in columns_txt


def test_end_to_end_local_generation():
    """Verify full CLI execution in a temporary output directory using local fallback."""
    with tempfile.TemporaryDirectory() as tmp_out:
        out_path = Path(tmp_out)
        files = generate_database_documentation(
            output_dir=out_path,
            use_local_only=True,
            max_samples=3,
        )

        assert "catalog" in files
        assert "er_diagram" in files
        assert "columns_mapping" in files
        assert "master" in files

        for k, p in files.items():
            assert p.exists(), f"File {p} was not created"
            assert p.stat().st_size > 0, f"File {p} is empty"
