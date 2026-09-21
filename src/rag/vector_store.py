"""ChromaDB Vector Store Client for Local and Remote Hugging Face Space Endpoints.

Supports:
1. Local Embedded PersistentClient ('data/chroma_db') for offline execution, testing, and CI.
2. Remote HTTP Client against FastAPI running on Hugging Face Spaces with Bearer $HF_TOKEN.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("ChromaVectorStore")

DEFAULT_COLLECTION_NAME = "biometric_knowledge_base"
DEFAULT_LOCAL_PATH = Path("data/chroma_db")


class ChromaVectorStore:
    """Cliente unificado para la base de datos vectorial ChromaDB (Local y Remota)."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        remote_url: str | None = None,
        auth_token: str | None = None,
        persist_directory: Path | str = DEFAULT_LOCAL_PATH,
    ) -> None:
        self.collection_name = collection_name
        self.remote_url = remote_url or os.getenv("CHROMA_REMOTE_URL")
        self.auth_token = auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.persist_directory = Path(persist_directory)

        self.is_remote = bool(self.remote_url and self.remote_url.startswith("http"))
        self._local_client: Any = None
        self._local_collection: Any = None

        if self.is_remote:
            # Limpiar barra final si existe
            assert self.remote_url is not None
            self.remote_url = self.remote_url.rstrip("/")
            logger.info(f"ChromaVectorStore configurado en MODO REMOTO: {self.remote_url}")
        else:
            logger.info(f"ChromaVectorStore configurado en MODO LOCAL: {self.persist_directory}")
            self._init_local_client()

    def _init_local_client(self) -> None:
        """Inicializa el cliente embebido local de ChromaDB."""
        import chromadb

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._local_client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._local_collection = self._local_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_headers(self) -> dict[str, str]:
        """Genera las cabeceras HTTP necesarias para peticiones al Space remoto."""
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
        """Inserta o actualiza fragmentos con sus vectores y metadatos.

        Retorna la cantidad de elementos procesados.
        """
        if not ids:
            return 0

        clean_metadatas = [self._normalize_metadata(m) for m in metadatas]

        if self.is_remote:
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

        # Modo Local
        assert self._local_collection is not None
        self._local_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metadatas,
        )
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Busca fragmentos semánticamente más cercanos al vector de consulta."""
        if self.is_remote:
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

        # Modo Local
        assert self._local_collection is not None
        if self._local_collection.count() == 0:
            return []

        query_args: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._local_collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where

        res = self._local_collection.query(**query_args)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        items: list[dict[str, Any]] = []
        for i in range(len(ids)):
            items.append(
                {
                    "id": ids[i],
                    "document": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                    "distance": float(distances[i]) if distances else 0.0,
                }
            )
        return items

    def count(self) -> int:
        """Retorna el número total de chunks en la colección."""
        if self.is_remote:
            stats = self.get_stats()
            return int(stats.get("total_chunks", 0))

        assert self._local_collection is not None
        return int(self._local_collection.count())

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas y desglose por fuente de la colección."""
        if self.is_remote:
            url = f"{self.remote_url}/stats"
            resp = requests.get(url, headers=self._get_headers(), timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"Error en stats remoto ({resp.status_code}): {resp.text[:300]}")
            result: dict[str, Any] = resp.json()
            return result

        assert self._local_collection is not None
        total = self._local_collection.count()
        breakdown: dict[str, int] = {}
        if total > 0:
            sample = self._local_collection.get(include=["metadatas"])
            metadatas = sample.get("metadatas") or []
            for meta in metadatas:
                if isinstance(meta, dict):
                    src = str(meta.get("source", "unknown"))
                    breakdown[src] = breakdown.get(src, 0) + 1

        return {
            "collection": self.collection_name,
            "total_chunks": total,
            "sources": breakdown,
        }

    def health(self) -> dict[str, Any]:
        """Verifica la conectividad y estado del backend."""
        if self.is_remote:
            url = f"{self.remote_url}/health"
            resp = requests.get(url, headers=self._get_headers(), timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Healthcheck remoto falló ({resp.status_code}): {resp.text[:300]}"
                )
            result: dict[str, Any] = resp.json()
            return result

        assert self._local_collection is not None
        import chromadb

        return {
            "status": "healthy",
            "collection": self.collection_name,
            "total_chunks": self._local_collection.count(),
            "mode": "local",
            "chroma_version": chromadb.__version__,
        }
