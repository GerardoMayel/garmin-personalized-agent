"""Unit tests for RAG Chunker, Schemas, DocumentLedger, and Language Detection."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.sync_garmin_metric_descriptions import (
    GARMIN_METRIC_DEFINITIONS,
    sync_garmin_metric_descriptions,
)
from src.rag.language_detector import detect_language
from src.rag.ledger import DocumentLedger
from src.rag.loader_and_chunker import (
    KnowledgeBaseChunker,
    clean_pdf_text,
    tag_physiological_metrics,
)
from src.rag.schemas import PARQUET_CHUNK_SCHEMA, ChunkRecord


def test_chunk_record_validation():
    """Valida la creación y esquema de un ChunkRecord."""
    rec = ChunkRecord(
        chunk_id="test_doc_c001",
        doc_id="test_doc",
        chunk_index=1,
        text="Sample physiological text regarding heart rate variability.",
        tipo_fuente="dispositivos_garmin_sensores",
        idioma="en",
        token_count=10,
        char_count=60,
        doc_title="Test Document",
        file_hash="a1b2c3d4e5f6",
        metadata_json=json.dumps({"metric": "hrv"}),
    )

    assert rec.chunk_id == "test_doc_c001"
    assert rec.token_count == 10
    meta = rec.get_metadata_dict()
    assert meta.get("metric") == "hrv"


def test_parquet_schema_compatibility(tmp_path: Path):
    """Valida que una lista de ChunkRecord se escriba y lea con PyArrow respetando el esquema."""
    pydict = {
        "chunk_id": ["c1", "c2"],
        "doc_id": ["d1", "d1"],
        "chunk_index": [0, 1],
        "text": ["Text 1", "Text 2"],
        "tipo_fuente": ["source_a", "source_a"],
        "idioma": ["en", "es"],
        "token_count": [15, 20],
        "char_count": [6, 6],
        "doc_title": ["Doc 1", "Doc 1"],
        "file_hash": ["hash1", "hash1"],
        "metadata_json": ["{}", "{}"],
        "created_at": ["2026-09-18T20:00:00Z", "2026-09-18T20:00:01Z"],
    }

    table = pa.Table.from_pydict(pydict, schema=PARQUET_CHUNK_SCHEMA)
    out_file = tmp_path / "test_chunks.parquet"
    pq.write_table(table, out_file)

    assert out_file.exists()
    read_table = pq.read_table(out_file)
    assert len(read_table) == 2
    assert read_table.schema.names == PARQUET_CHUNK_SCHEMA.names


def test_language_detection():
    """Valida la detección de idiomas con NLP tradicional y fallbacks."""
    text_en = "Heart rate variability reflects autonomic nervous system balance, parasympathetic tone, and recovery."
    text_es = "La variabilidad del ritmo cardíaco evalúa el estado del sistema nervioso autónomo y la recuperación."

    assert detect_language(text_en) == "en"
    assert detect_language(text_es) == "es"
    assert detect_language("short") == "en"  # fallback en texto muy corto


def test_metric_tagging():
    """Valida la categorización automática de métricas fisiológicas."""
    text = "The Elevate optical PPG sensor measures rMSSD HRV, VO2 max estimation, and nocturnal sleep respiration."
    tags = tag_physiological_metrics(text)

    assert "hrv_rmssd" in tags
    assert "vo2max" in tags
    assert "ppg_sensor" in tags
    assert "sleep" in tags
    assert "respiration" in tags


def test_clean_pdf_text():
    """Valida la remoción de encabezados de Firstbeat y limpieza de saltos rotos."""
    dirty_text = "Firstbeat White Paper\nPage 1 of 5\nPhysio-\nlogical recovery and HRV."
    cleaned = clean_pdf_text(dirty_text)

    assert "Firstbeat White Paper" not in cleaned
    assert "Page 1 of 5" not in cleaned
    assert "Physiological" in cleaned


def test_document_ledger_reconciliation(tmp_path: Path):
    """Valida el cálculo de hash, idempotencia y purga de huérfanos en DocumentLedger."""
    ledger_file = tmp_path / "_ledger.json"
    ledger = DocumentLedger(ledger_path=ledger_file)

    dummy_doc = tmp_path / "doc1.txt"
    dummy_doc.write_text("Contenido inicial del documento", encoding="utf-8")
    hash_v1 = DocumentLedger.compute_sha256(dummy_doc)

    ledger.register_document(
        source="test_source",
        doc_name="doc1.txt",
        file_hash=hash_v1,
        chunk_ids=["doc1_c000", "doc1_c001"],
    )
    ledger.save()

    # Verificar que detecta que no ha cambiado
    assert ledger.is_document_unchanged("test_source", "doc1.txt", hash_v1) is True

    # Modificar documento y verificar que detecta el cambio
    dummy_doc.write_text("Contenido modificado", encoding="utf-8")
    hash_v2 = DocumentLedger.compute_sha256(dummy_doc)
    assert ledger.is_document_unchanged("test_source", "doc1.txt", hash_v2) is False

    # Reconciliar lista activa (si doc1 ya no está presente, purga sus chunks)
    purged = ledger.reconcile_active_documents("test_source", current_active_filenames=set())
    assert purged == ["doc1_c000", "doc1_c001"]
    summary = ledger.get_summary()
    assert summary["sources"]["test_source"]["total_documents"] == 0


def test_sync_garmin_metric_descriptions(tmp_path: Path):
    """Valida la generación del glosario de métricas Garmin en Markdown y JSON."""
    res = sync_garmin_metric_descriptions(output_dir=tmp_path, r2_sync=False)

    json_file = Path(res["json_path"])
    md_file = Path(res["md_path"])

    assert json_file.exists()
    assert md_file.exists()

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_metrics"] == len(GARMIN_METRIC_DEFINITIONS)
        assert data["version"] == "1.0.0"

    md_content = md_file.read_text(encoding="utf-8")
    assert "# Glosario Oficial y Explicaciones de Métricas Garmin Connect" in md_content
    assert "Estado de Variabilidad de la Frecuencia Cardíaca" in md_content


def test_chunker_token_bounds(tmp_path: Path):
    """Valida que los chunks respeten el rango de tokens configurado y generen Parquet."""
    doc_file = tmp_path / "long_doc.md"
    paragraphs = [
        f"Párrafo {i}: Explicación detallada del sensor Garmin Elevate y su algoritmo Firstbeat para estimar la variabilidad cardíaca y el consumo de oxígeno durante el ejercicio extenuante. "
        * 8
        for i in range(10)
    ]
    doc_file.write_text("\n\n".join(paragraphs), encoding="utf-8")

    chunker = KnowledgeBaseChunker(chunk_size_tokens=400, chunk_overlap_tokens=160)
    doc_hash = DocumentLedger.compute_sha256(doc_file)
    chunks = chunker.process_document(
        file_path=doc_file,
        tipo_fuente="descripciones_metricas_garmin",
        doc_title="Long Document",
        file_hash=doc_hash,
    )

    assert len(chunks) > 1
    for c in chunks:
        # Los tokens no deben exceder significativamente el tamaño configurado
        assert c.token_count <= 480
        assert c.tipo_fuente == "descripciones_metricas_garmin"
        assert c.idioma == "es"
