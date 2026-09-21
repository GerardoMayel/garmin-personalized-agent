"""ChromaDB Vector Store Client for Remote Hugging Face Space / FastAPI Endpoints.

Interacts strictly via HTTP with the remote ChromaDB FastAPI backend
hosted on Hugging Face Spaces (or containerized service) authenticated via Bearer $HF_TOKEN.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("ChromaVectorStore")

DEFAULT_COLLECTION_NAME = "biometric_knowledge_base"


class ChromaVectorStore:
    """Cliente remoto exclusivo para la base de datos vectorial ChromaDB vía FastAPI."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        remote_url: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.remote_url = remote_url or os.getenv("CHROMA_REMOTE_URL")
        self.auth_token = auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if not self.remote_url:
            raise ValueError(
                "CHROMA_REMOTE_URL no está configurado. El almacenamiento vectorial opera "
                "exclusivamente en modo remoto (Hugging Face Space / FastAPI)."
            )

        self.remote_url = self.remote_url.rstrip("/")
        logger.info(f"ChromaVectorStore inicializado en MODO REMOTO: {self.remote_url}")

    def _get_headers(self) -> dict[str, str]:
        """Genera las cabeceras HTTP necesarias para peticiones al backend remoto."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Normaliza los metadatos para tipos aceptados por ChromaDB (str, int, float, bool)."""
        clean: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ",".join(str(item) for item in v)
            elif v is None:
                continue
            else:
                clean[k] = str(v)
        return clean

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Inserta o actualiza fragmentos con sus vectores y metadatos en el backend remoto."""
        if not ids:
            return 0

        clean_metadatas = [self._normalize_metadata(m) for m in metadatas]
        url = f"{self.remote_url}/upsert"
        payload: dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": clean_metadatas,
        }
        resp = requests.post(url, json=payload, headers=self._get_headers(), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Error en upsert remoto ({resp.status_code}): {resp.text[:300]}"
            )
        result = resp.json()
        return int(result.get("upserted_count", len(ids)))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Busca fragmentos semánticamente más cercanos al vector de consulta en el backend remoto."""
        url = f"{self.remote_url}/query"
        payload: dict[str, Any] = {
            "query_embedding": query_embedding,
            "n_results": n_results,
        }
        if where:
            payload["where"] = where

        resp = requests.post(url, json=payload, headers=self._get_headers(), timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Error en query remoto ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        results: list[dict[str, Any]] = data.get("results", [])
        return results

    def count(self) -> int:
        """Retorna el número total de chunks en la colección remota."""
        stats = self.get_stats()
        return int(stats.get("total_chunks", 0))

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas y desglose por fuente de la colección remota."""
        url = f"{self.remote_url}/stats"
        resp = requests.get(url, headers=self._get_headers(), timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Error en stats remoto ({resp.status_code}): {resp.text[:300]}")
        result: dict[str, Any] = resp.json()
        return result

    def health(self) -> dict[str, Any]:
        """Verifica la conectividad y estado del backend remoto."""
        url = f"{self.remote_url}/health"
        resp = requests.get(url, headers=self._get_headers(), timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Healthcheck remoto falló ({resp.status_code}): {resp.text[:300]}"
            )
        result: dict[str, Any] = resp.json()
        return result

