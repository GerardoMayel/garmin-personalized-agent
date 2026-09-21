"""Unit tests for the Hugging Face Space FastAPI application."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from deploy.hf_chroma_space.app import app


def test_hf_space_api_endpoints() -> None:
    """Valida los endpoints raíz, health, upsert, stats y query de la API de Space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIRECTORY": tmpdir}):
            client = TestClient(app)

            # 1. Root
            resp = client.get("/")
            assert resp.status_code == 200
            assert resp.json()["status"] == "online"

            # 2. Health
            resp = client.get("/health")
            assert resp.status_code == 200
            health_data = resp.json()
            assert health_data["status"] == "healthy"
            assert health_data["total_chunks"] == 0

            # 3. Upsert
            upsert_payload = {
                "ids": ["chunk_01", "chunk_02"],
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "documents": ["Contenido sobre HRV y estrés", "Contenido sobre sensores Garmin"],
                "metadatas": [
                    {"source": "variables_fisiologia_humana", "tags": ["hrv", "stress"]},
                    {"source": "dispositivos_garmin_sensores", "tags": ["sensor", "elevate"]},
                ],
            }
            resp = client.post("/upsert", json=upsert_payload)
            assert resp.status_code == 200
            assert resp.json()["upserted_count"] == 2

            # 4. Stats
            resp = client.get("/stats")
            assert resp.status_code == 200
            stats = resp.json()
            assert stats["total_chunks"] == 2
            assert stats["sources"]["variables_fisiologia_humana"] == 1

            # 5. Query
            query_payload = {
                "query_embedding": [0.1, 0.2, 0.3],
                "n_results": 1,
            }
            resp = client.post("/query", json=query_payload)
            assert resp.status_code == 200
            query_data = resp.json()
            assert query_data["total_found"] == 1
            assert query_data["results"][0]["id"] == "chunk_01"


def test_hf_space_auth_enforcement() -> None:
    """Debe rechazar peticiones si AUTH_BEARER_TOKEN está configurado y el token es inválido."""
    with patch.dict("os.environ", {"AUTH_BEARER_TOKEN": "secret_token_123"}):
        client = TestClient(app)

        # Sin token
        resp = client.post(
            "/upsert", json={"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        )
        assert resp.status_code == 401

        # Token incorrecto
        resp = client.post(
            "/upsert",
            headers={"Authorization": "Bearer wrong_token"},
            json={"ids": [], "embeddings": [], "documents": [], "metadatas": []},
        )
        assert resp.status_code == 401

        # Token correcto
        resp = client.post(
            "/upsert",
            headers={"Authorization": "Bearer secret_token_123"},
            json={"ids": [], "embeddings": [], "documents": [], "metadatas": []},
        )
        assert resp.status_code == 200
