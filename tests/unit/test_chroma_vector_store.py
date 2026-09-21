"""Unit tests for ChromaVectorStore (Local and Remote modes)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.vector_store import ChromaVectorStore


def test_local_chroma_lifecycle() -> None:
    """Debe inicializar un cliente local, insertar chunks, consultar y retornar estadísticas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            collection_name="test_biometrics",
            persist_directory=Path(tmpdir),
        )

        assert store.count() == 0

        # Datos sintéticos
        ids = ["c1", "c2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["Documento sobre HRV y rMSSD", "Documento sobre sensor Elevate"]
        metadatas = [
            {"source": "variables_fisiologia_humana", "tags": ["hrv", "rmssd"]},
            {"source": "dispositivos_garmin_sensores", "tags": ["sensor", "ppg"]},
        ]

        count = store.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        assert count == 2
        assert store.count() == 2

        # Probar stats
        stats = store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["sources"]["variables_fisiologia_humana"] == 1
        assert stats["sources"]["dispositivos_garmin_sensores"] == 1

        # Probar query
        results = store.query(query_embedding=[0.1, 0.2, 0.3], n_results=1)
        assert len(results) == 1
        assert results[0]["id"] == "c1"
        assert "HRV" in results[0]["document"]

        # Probar query con filtro where
        filtered = store.query(
            query_embedding=[0.4, 0.5, 0.6],
            n_results=5,
            where={"source": "dispositivos_garmin_sensores"},
        )
        assert len(filtered) == 1
        assert filtered[0]["id"] == "c2"


def test_metadata_normalization() -> None:
    """Debe convertir listas a cadenas y descartar valores None."""
    raw_meta = {
        "str_val": "hola",
        "int_val": 42,
        "list_val": ["tag1", "tag2"],
        "none_val": None,
    }
    clean = ChromaVectorStore._normalize_metadata(raw_meta)
    assert clean["str_val"] == "hola"
    assert clean["int_val"] == 42
    assert clean["list_val"] == "tag1,tag2"
    assert "none_val" not in clean


def test_remote_mode_routing() -> None:
    """Debe enviar peticiones HTTP cuando remote_url está configurado."""
    store = ChromaVectorStore(
        collection_name="remote_coll",
        remote_url="https://mock-space.hf.space",
        auth_token="test_token",
    )

    assert store.is_remote is True

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"upserted_count": 1}
        mock_post.return_value = mock_resp

        store.upsert(
            ids=["x1"],
            embeddings=[[0.1, 0.2]],
            documents=["doc"],
            metadatas=[{"k": "v"}],
        )

        assert mock_post.call_count == 1
        call_url = mock_post.call_args[0][0]
        call_headers = mock_post.call_args[1]["headers"]
        assert call_url == "https://mock-space.hf.space/upsert"
        assert call_headers["Authorization"] == "Bearer test_token"
