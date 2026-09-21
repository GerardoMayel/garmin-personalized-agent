"""Unit tests for Google Gemini Embedding Engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.embeddings import GeminiEmbeddingEngine


def test_init_missing_key() -> None:
    """Debe fallar si GEMINI_API_KEY no está configurada."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        with pytest.raises(ValueError, match="GEMINI_API_KEY no está configurada"):
            GeminiEmbeddingEngine(api_key="")


def test_embed_query_success() -> None:
    """Debe generar el embedding de consulta con la dimensión correcta."""
    engine = GeminiEmbeddingEngine(api_key="mock_key", dimensions=768)
    mock_vector = [0.1] * 768

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": {"values": mock_vector}}
        mock_post.return_value = mock_resp

        vector = engine.embed_query("¿Cómo se calcula el rMSSD?")

        assert len(vector) == 768
        assert vector == mock_vector

        # Verificar payload
        call_args = mock_post.call_args
        assert "RETRIEVAL_QUERY" in str(call_args)
        assert "768" in str(call_args)


def test_embed_documents_batching() -> None:
    """Debe dividir documentos en lotes y retornar lista concatenada de vectores."""
    engine = GeminiEmbeddingEngine(api_key="mock_key", dimensions=768, batch_size=2)
    mock_vector = [0.05] * 768
    texts = ["Texto 1", "Texto 2", "Texto 3"]

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Simula respuesta para batch de 2 y luego batch de 1
        mock_resp.json.side_effect = [
            {"embeddings": [{"values": mock_vector}, {"values": mock_vector}]},
            {"embeddings": [{"values": mock_vector}]},
        ]
        mock_post.return_value = mock_resp

        vectors = engine.embed_documents(texts=texts)

        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)
        assert mock_post.call_count == 2


def test_embed_retry_on_429() -> None:
    """Debe reintentar tras recibir 429 y tener éxito en el segundo intento."""
    engine = GeminiEmbeddingEngine(api_key="mock_key", dimensions=768)
    mock_vector = [0.2] * 768

    with patch("requests.post") as mock_post, patch("time.sleep") as mock_sleep:
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        fail_resp.text = "Rate limited"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"embedding": {"values": mock_vector}}

        mock_post.side_effect = [fail_resp, ok_resp]

        vector = engine.embed_query("Consulta tras rate limit")

        assert vector == mock_vector
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1
