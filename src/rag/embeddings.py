"""Google Gemini Dense Embeddings Engine for Biometric Knowledge Base.

Uses Google's text-embedding model ('gemini-embedding-001') via REST API
with 768 output dimensionality, automatic batching, and exponential backoff retry.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("GeminiEmbeddings")

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 50
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0


class GeminiEmbeddingEngine:
    """Motor de embeddings denso utilizando la API de Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY no está configurada en .env ni fue suministrada en inicialización."
            )

        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}"

    def _post_with_retry(self, endpoint_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Ejecuta una petición POST HTTP con reintentos y retroceso exponencial."""
        backoff = INITIAL_BACKOFF
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    endpoint_url,
                    json=payload,
                    timeout=45,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    res_json = response.json()
                    return res_json if isinstance(res_json, dict) else {"data": res_json}

                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # Rate limit (429) o error del servidor (5xx)
                if response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        f"Fallo temporal en Gemini Embeddings (intento {attempt}/{MAX_RETRIES}): "
                        f"{last_error}. Reintentando en {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                # Error 4xx permanente
                raise RuntimeError(f"Error permanente en Gemini Embeddings API: {last_error}")

            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(
                    f"Error de red en Gemini Embeddings (intento {attempt}/{MAX_RETRIES}): {e}. "
                    f"Reintentando en {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(
            f"Se excedió el número máximo de reintentos ({MAX_RETRIES}) en Gemini Embeddings: {last_error}"
        )

    def embed_query(self, text: str) -> list[float]:
        """Genera el embedding para una consulta de búsqueda (taskType=RETRIEVAL_QUERY)."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("El texto de la consulta no puede estar vacío.")

        url = f"{self.base_url}:embedContent?key={self.api_key}"
        payload: dict[str, Any] = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": clean_text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": self.dimensions,
        }

        data = self._post_with_retry(url, payload)
        embedding_obj = data.get("embedding", {})
        values = embedding_obj.get("values", [])
        if not values:
            raise RuntimeError(f"Respuesta inesperada al generar embedding de consulta: {data}")

        return [float(v) for v in values]

    def embed_documents(
        self,
        texts: list[str],
        titles: list[str] | None = None,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Genera embeddings por lotes para fragmentos de documentos (taskType=RETRIEVAL_DOCUMENT)."""
        if not texts:
            return []

        effective_batch_size = batch_size or self.batch_size
        url = f"{self.base_url}:batchEmbedContents?key={self.api_key}"
        all_embeddings: list[list[float]] = []

        total_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size
        logger.info(
            f"Generando embeddings para {len(texts)} documentos en {total_batches} lotes "
            f"de hasta {effective_batch_size} elementos..."
        )

        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i : i + effective_batch_size]
            batch_titles = titles[i : i + effective_batch_size] if titles else None

            requests_payload: list[dict[str, Any]] = []
            for j, text in enumerate(batch_texts):
                req: dict[str, Any] = {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": text.strip()}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": self.dimensions,
                }
                if batch_titles and j < len(batch_titles) and batch_titles[j]:
                    req["title"] = batch_titles[j][:100]
                requests_payload.append(req)

            payload = {"requests": requests_payload}
            response_data = self._post_with_retry(url, payload)
            embeddings_list = response_data.get("embeddings", [])

            if len(embeddings_list) != len(batch_texts):
                raise RuntimeError(
                    f"Inconsistencia en lote {i // effective_batch_size + 1}: "
                    f"se enviaron {len(batch_texts)} textos pero se recibieron {len(embeddings_list)} embeddings."
                )

            for emb_obj in embeddings_list:
                vals = emb_obj.get("values", [])
                if not vals:
                    raise RuntimeError(f"Embedding vacío devuelto en lote: {emb_obj}")
                all_embeddings.append([float(v) for v in vals])

            logger.debug(
                f"Lote {i // effective_batch_size + 1}/{total_batches} procesado "
                f"({len(all_embeddings)}/{len(texts)} acumulados)."
            )

        logger.info(f"Embeddings generados exitosamente para {len(all_embeddings)} documentos.")
        return all_embeddings
