"""Unit tests for Direct-to-Storage incremental ChromaDB indexer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.index_to_chroma import run_direct_to_storage_indexing


def test_indexer_zero_changes_skips_embedding() -> None:
    """Valida que si todos los fragmentos ya están en Chroma con el mismo hash, no se llame a Gemini."""
    mock_chunks = [
        {
            "chunk_id": "doc1_c000",
            "source": "test_src",
            "doc_type": "white_paper",
            "document_id": "doc1",
            "doc_title": "Doc 1",
            "chunk_index": 0,
            "content": "Contenido sin cambios",
            "token_count": 20,
            "char_count": 100,
            "language": "es",
            "source_category": "test_src",
            "tags": ["tag1"],
            "file_hash": "hash_123456",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with (
            patch("src.rag.index_to_chroma.load_chunks_from_parquets", return_value=mock_chunks),
            patch("src.rag.index_to_chroma.restore_vector_db_from_r2", return_value=False),
            patch("src.rag.index_to_chroma.DEFAULT_LOCAL_DIR", tmp_path / "chroma"),
            patch("src.rag.index_to_chroma.DEFAULT_ARCHIVE_PATH", tmp_path / "chroma.tar.gz"),
            patch("src.rag.index_to_chroma.GeminiEmbeddingEngine") as mock_engine,
        ):
            # 1. Primera ejecución: indexa el fragmento
            mock_engine_instance = MagicMock()
            mock_engine_instance.embed_documents.return_value = [[0.1] * 768]
            mock_engine.return_value = mock_engine_instance

            res1 = run_direct_to_storage_indexing(r2_sync=False, force=False)
            assert res1["status"] == "completed"
            assert res1["newly_indexed"] == 1
            assert mock_engine_instance.embed_documents.call_count == 1

            # 2. Segunda ejecución: sin cambios -> debe salir temprano con status 'up_to_date'
            mock_engine_instance.embed_documents.reset_mock()
            res2 = run_direct_to_storage_indexing(r2_sync=False, force=False)
            assert res2["status"] == "up_to_date"
            assert res2["newly_indexed"] == 0
            assert mock_engine_instance.embed_documents.call_count == 0


def test_indexer_purges_orphaned_chunks() -> None:
    """Valida que los fragmentos eliminados de los Parquet se purguen de ChromaDB."""
    chunks_v1 = [
        {
            "chunk_id": "doc1_c000",
            "source": "test_src",
            "doc_type": "white_paper",
            "document_id": "doc1",
            "doc_title": "Doc 1",
            "chunk_index": 0,
            "content": "Doc 1 texto",
            "token_count": 20,
            "char_count": 50,
            "language": "es",
            "source_category": "test_src",
            "tags": [],
            "file_hash": "hash_aaa",
        },
        {
            "chunk_id": "doc2_c000",
            "source": "test_src",
            "doc_type": "white_paper",
            "document_id": "doc2",
            "doc_title": "Doc 2 para eliminar",
            "chunk_index": 0,
            "content": "Doc 2 texto",
            "token_count": 20,
            "char_count": 50,
            "language": "es",
            "source_category": "test_src",
            "tags": [],
            "file_hash": "hash_bbb",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with (
            patch("src.rag.index_to_chroma.restore_vector_db_from_r2", return_value=False),
            patch("src.rag.index_to_chroma.DEFAULT_LOCAL_DIR", tmp_path / "chroma"),
            patch("src.rag.index_to_chroma.DEFAULT_ARCHIVE_PATH", tmp_path / "chroma.tar.gz"),
            patch("src.rag.index_to_chroma.GeminiEmbeddingEngine") as mock_engine,
        ):
            mock_engine_instance = MagicMock()
            mock_engine_instance.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
            mock_engine.return_value = mock_engine_instance

            # Inicializar con doc1 y doc2
            with patch("src.rag.index_to_chroma.load_chunks_from_parquets", return_value=chunks_v1):
                res1 = run_direct_to_storage_indexing(r2_sync=False)
                assert res1["total_in_collection"] == 2

            # Simular eliminación de doc2: sólo queda doc1
            chunks_v2 = [chunks_v1[0]]
            with patch("src.rag.index_to_chroma.load_chunks_from_parquets", return_value=chunks_v2):
                res2 = run_direct_to_storage_indexing(r2_sync=False)
                assert res2["purged_orphans"] == 1
                assert res2["total_in_collection"] == 1


def test_indexer_detects_modified_chunk_hash() -> None:
    """Valida que si el hash SHA-256 de un documento cambia, se re-calcule el embedding."""
    chunk_v1 = {
        "chunk_id": "doc1_c000",
        "source": "test_src",
        "doc_type": "white_paper",
        "document_id": "doc1",
        "doc_title": "Doc 1",
        "chunk_index": 0,
        "content": "Versión 1 del contenido",
        "token_count": 20,
        "char_count": 50,
        "language": "es",
        "source_category": "test_src",
        "tags": [],
        "file_hash": "hash_original",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with (
            patch("src.rag.index_to_chroma.restore_vector_db_from_r2", return_value=False),
            patch("src.rag.index_to_chroma.DEFAULT_LOCAL_DIR", tmp_path / "chroma"),
            patch("src.rag.index_to_chroma.DEFAULT_ARCHIVE_PATH", tmp_path / "chroma.tar.gz"),
            patch("src.rag.index_to_chroma.GeminiEmbeddingEngine") as mock_engine,
        ):
            mock_engine_instance = MagicMock()
            mock_engine_instance.embed_documents.return_value = [[0.1] * 768]
            mock_engine.return_value = mock_engine_instance

            # 1. Indexar versión 1
            with patch("src.rag.index_to_chroma.load_chunks_from_parquets", return_value=[chunk_v1]):
                res1 = run_direct_to_storage_indexing(r2_sync=False)
                assert res1["newly_indexed"] == 1

            # 2. Modificar contenido y hash
            chunk_v2 = dict(chunk_v1)
            chunk_v2["content"] = "Versión 2 actualizada"
            chunk_v2["file_hash"] = "hash_actualizado_999"

            mock_engine_instance.embed_documents.reset_mock()
            mock_engine_instance.embed_documents.return_value = [[0.9] * 768]

            with patch("src.rag.index_to_chroma.load_chunks_from_parquets", return_value=[chunk_v2]):
                res2 = run_direct_to_storage_indexing(r2_sync=False)
                # Debe detectar el cambio de hash y re-indexar
                assert res2["newly_indexed"] == 1
                assert mock_engine_instance.embed_documents.call_count == 1
