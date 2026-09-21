"""Batch Indexing Pipeline: Parquet Chunks to ChromaDB Vector Store.

Reads processed chunk datasets from all 3 sources (Firstbeat Garmin Devices,
Firstbeat Human Physiology, Garmin Metric Descriptions), computes dense embeddings
via Google Gemini (768-dim), and upserts them into ChromaDB (Local or Remote HF Space).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.common.logger import get_logger
from src.rag.embeddings import GeminiEmbeddingEngine
from src.rag.vector_store import ChromaVectorStore

logger = get_logger("ChromaIndexer")

SOURCE_DATASETS: list[dict[str, Any]] = [
    {
        "source": "dispositivos_garmin_sensores",
        "parquet_path": Path(
            "data/knowledge_base/processed_chunks/dispositivos_garmin_sensores/dataset_v1/part-00001.parquet"
        ),
        "doc_type": "white_paper_sensor",
    },
    {
        "source": "variables_fisiologia_humana",
        "parquet_path": Path(
            "data/knowledge_base/processed_chunks/variables_fisiologia_humana/dataset_v1/part-00001.parquet"
        ),
        "doc_type": "white_paper_physiology",
    },
    {
        "source": "descripciones_metricas_garmin",
        "parquet_path": Path(
            "data/knowledge_base/processed_chunks/descripciones_metricas_garmin/dataset_v1/part-00001.parquet"
        ),
        "doc_type": "official_metric_glossary",
    },
]


def load_chunks_from_parquets() -> list[dict[str, Any]]:
    """Carga y unifica los fragmentos procesados desde los 3 datasets Parquet."""
    all_chunks: list[dict[str, Any]] = []

    for item in SOURCE_DATASETS:
        p_path = Path(str(item["parquet_path"]))
        source_id = str(item["source"])
        doc_type = str(item["doc_type"])

        if not p_path.exists():
            logger.warning(
                f"El archivo Parquet para '{source_id}' no existe en {p_path}. Saltando fuente..."
            )
            continue

        table = pq.read_table(p_path)
        pydict = table.to_pydict()
        num_rows = table.num_rows

        for i in range(num_rows):
            meta_raw = pydict.get("metadata_json", [None])[i]
            parsed_tags: list[str] = []
            if isinstance(meta_raw, str):
                try:
                    loaded_meta = json.loads(meta_raw)
                    if isinstance(loaded_meta, dict):
                        detected = loaded_meta.get("detected_metrics", [])
                        if isinstance(detected, list):
                            parsed_tags = [str(t) for t in detected]
                except Exception:
                    pass

            chunk_source = str(pydict.get("tipo_fuente", [source_id])[i] or source_id)
            all_chunks.append(
                {
                    "chunk_id": str(pydict["chunk_id"][i]),
                    "source": chunk_source,
                    "doc_type": doc_type,
                    "document_id": str(pydict["doc_id"][i]),
                    "doc_title": str(pydict.get("doc_title", [""])[i]),
                    "chunk_index": int(pydict["chunk_index"][i]),
                    "content": str(pydict["text"][i]),
                    "token_count": int(pydict["token_count"][i]),
                    "char_count": int(pydict["char_count"][i]),
                    "language": str(pydict["idioma"][i]),
                    "source_category": chunk_source,
                    "tags": parsed_tags,
                }
            )

        logger.info(f"Cargados {num_rows} chunks desde Parquet de '{source_id}'.")

    return all_chunks


def run_batch_indexing(
    remote_url: str | None = None,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Ejecuta la indexación por lotes de los chunks Parquet hacia ChromaDB."""
    logger.info("=== Iniciando Pipeline de Indexación Batch en ChromaDB ===")

    chunks = load_chunks_from_parquets()
    if not chunks:
        logger.error("No se encontraron fragmentos en los datasets Parquet para indexar.")
        return {"total_indexed": 0, "status": "no_data"}

    logger.info(f"Total de fragmentos unificados a procesar: {len(chunks)}")

    # 1. Inicializar motor de embeddings
    embedding_engine = GeminiEmbeddingEngine(batch_size=batch_size)

    # 2. Inicializar Vector Store (Local o Remoto)
    vector_store = ChromaVectorStore(remote_url=remote_url)

    # 3. Extraer textos y títulos para embedding
    texts = [c["content"] for c in chunks]
    titles = [f"{c['source']}:{c['document_id']}" for c in chunks]

    logger.info("Generando vectores de embedding con Google Gemini (768 dims)...")
    embeddings = embedding_engine.embed_documents(texts=texts, titles=titles, batch_size=batch_size)

    # 4. Preparar lotes para upsert en ChromaDB
    ids = [c["chunk_id"] for c in chunks]
    metadatas: list[dict[str, Any]] = []
    for c in chunks:
        metadatas.append(
            {
                "source": c["source"],
                "doc_type": c["doc_type"],
                "document_id": c["document_id"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "char_count": c["char_count"],
                "language": c["language"],
                "source_category": c["source_category"],
                "tags": ",".join(c["tags"]),
            }
        )

    # 5. Upsert por bloques en ChromaDB
    total_upserted = 0
    upsert_chunk_size = 100
    for i in range(0, len(ids), upsert_chunk_size):
        batch_ids = ids[i : i + upsert_chunk_size]
        batch_embs = embeddings[i : i + upsert_chunk_size]
        batch_docs = texts[i : i + upsert_chunk_size]
        batch_metas = metadatas[i : i + upsert_chunk_size]

        count = vector_store.upsert(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        total_upserted += count
        logger.info(f"Progreso de upsert: {total_upserted}/{len(ids)} fragmentos indexados.")

    stats = vector_store.get_stats()
    logger.info(f"Indexación completada con éxito. Estadísticas de la colección: {stats}")

    return {
        "status": "completed",
        "total_indexed": total_upserted,
        "stats": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indexa los 3 datasets Parquet en ChromaDB con Google Gemini Embeddings."
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Si se especifica, indexa hacia el endpoint remoto configurado en CHROMA_REMOTE_URL.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL directa del endpoint remoto de ChromaDB (sobreescribe CHROMA_REMOTE_URL).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Tamaño de lote para la generación de embeddings en Gemini (default: 50).",
    )
    args = parser.parse_args()

    target_url = args.url
    if args.remote and not target_url:
        import os

        target_url = os.getenv("CHROMA_REMOTE_URL")
        if not target_url:
            logger.error("Se especificó --remote pero CHROMA_REMOTE_URL no está configurada.")
            exit(1)

    run_batch_indexing(remote_url=target_url, batch_size=args.batch_size)
