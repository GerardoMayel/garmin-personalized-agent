"""Batch Indexing Pipeline: Parquet Chunks to ChromaDB Vector Store & Cloudflare R2.

Direct-to-Storage Architecture:
1. Restores existing ChromaDB archive ('data/chroma_db.tar.gz') from Cloudflare R2 if available.
2. Reads processed chunks across all 3 sources (Garmin devices, physiology, metric descriptions).
3. Evaluates idempotency against existing chunk IDs in ChromaDB.
4. Computes 768-dim embeddings for new/missing chunks via Google Gemini.
5. Upserts directly to local ChromaDB ('data/chroma_db').
6. Compresses into 'data/chroma_db.tar.gz' and uploads directly to Cloudflare R2
   ('knowledge_base/vector_db/chroma_db.tar.gz') for read-only serving on Hugging Face Spaces.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, cast

import chromadb
import pyarrow.parquet as pq

from src.common.logger import get_logger
from src.common.r2_storage import R2StorageManager
from src.rag.embeddings import GeminiEmbeddingEngine

logger = get_logger("ChromaIndexer")

COLLECTION_NAME = "biometric_knowledge_base"
DEFAULT_LOCAL_DIR = Path("data/chroma_db")
DEFAULT_ARCHIVE_PATH = Path("data/chroma_db.tar.gz")

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


def restore_vector_db_from_r2(
    archive_path: Path = DEFAULT_ARCHIVE_PATH, target_dir: Path = DEFAULT_LOCAL_DIR
) -> bool:
    """Restaura la base vectorial ChromaDB previa desde Cloudflare R2 si existe."""
    r2 = R2StorageManager()
    if not r2.is_configured():
        logger.info("R2 no configurado en este entorno. Se creará base vectorial desde cero.")
        return False

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if r2.download_vector_db(local_archive=archive_path):
        logger.info(
            f"Índice previo descargado desde R2: {archive_path}. Descomprimiendo en {target_dir.parent}..."
        )
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir.parent)
            logger.info(f"Base vectorial restaurada con éxito en {target_dir}.")
            return True
        except Exception as e:
            logger.warning(f"Error al descomprimir archivo descargado: {e}. Creando nuevo índice.")
            return False

    logger.info(
        "No se encontró índice previo en Cloudflare R2 o bucket vacío. Creando nuevo índice."
    )
    return False


def run_direct_to_storage_indexing(
    r2_sync: bool = True,
    batch_size: int = 15,
    force: bool = False,
) -> dict[str, Any]:
    """Indexa chunks en ChromaDB local de alta velocidad, comprime y sube a Cloudflare R2."""
    logger.info("=== Iniciando Pipeline Direct-to-Storage (ChromaDB + Cloudflare R2) ===")

    # 1. Cargar fragmentos Parquet
    chunks = load_chunks_from_parquets()
    if not chunks:
        logger.error("No se encontraron fragmentos en los datasets Parquet para indexar.")
        return {"total_indexed": 0, "status": "no_data"}

    logger.info(f"Total de fragmentos en Parquet: {len(chunks)}")

    # 2. Restaurar estado previo si existe y no se fuerza reconstrucción
    if not force:
        restore_vector_db_from_r2()
    else:
        if DEFAULT_LOCAL_DIR.exists():
            shutil.rmtree(DEFAULT_LOCAL_DIR)

    # 3. Inicializar ChromaDB local embebido
    DEFAULT_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DEFAULT_LOCAL_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Verificar qué chunks ya están indexados (Idempotencia)
    existing_ids: set[str] = set()
    try:
        data = collection.get(include=[])
        existing_ids = set(data.get("ids", []))
    except Exception as e:
        logger.warning(f"No se pudo consultar IDs existentes: {e}")

    logger.info(f"ChromaDB local contiene actualmente {len(existing_ids)} fragmentos indexados.")
    missing_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    total_indexed = 0
    if missing_chunks:
        logger.info(f"Indexando {len(missing_chunks)} nuevos fragmentos con Gemini Embeddings...")
        embedding_engine = GeminiEmbeddingEngine(batch_size=batch_size)

        texts = [c["content"] for c in missing_chunks]
        titles = [f"{c['source']}:{c['document_id']}" for c in missing_chunks]
        embeddings = embedding_engine.embed_documents(
            texts=texts, titles=titles, batch_size=batch_size
        )

        ids = [c["chunk_id"] for c in missing_chunks]
        metadatas = [
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
            for c in missing_chunks
        ]

        # Upsert en bloques de 100
        for i in range(0, len(ids), 100):
            b_ids = ids[i : i + 100]
            b_embs = embeddings[i : i + 100]
            b_docs = texts[i : i + 100]
            b_metas = metadatas[i : i + 100]

            collection.upsert(
                ids=b_ids,
                embeddings=cast(Any, b_embs),
                documents=b_docs,
                metadatas=cast(Any, b_metas),
            )
            total_indexed += len(b_ids)
            logger.info(f"Progreso de upsert local: {total_indexed}/{len(ids)} fragmentos.")

    else:
        logger.info(
            "Todos los fragmentos ya se encuentran indexados en ChromaDB. Nada nuevo por calcular."
        )

    final_count = collection.count()
    logger.info(f"Total consolidado en ChromaDB local: {final_count} vectores.")

    # 5. Comprimir base vectorial local
    DEFAULT_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Comprimiendo {DEFAULT_LOCAL_DIR} hacia {DEFAULT_ARCHIVE_PATH}...")
    with tarfile.open(DEFAULT_ARCHIVE_PATH, "w:gz") as tar:
        tar.add(str(DEFAULT_LOCAL_DIR), arcname=DEFAULT_LOCAL_DIR.name)

    archive_size_mb = DEFAULT_ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Archivo {DEFAULT_ARCHIVE_PATH} generado ({archive_size_mb:.2f} MB).")

    # 6. Sincronizar a Cloudflare R2
    if r2_sync:
        r2 = R2StorageManager()
        if r2.is_configured():
            logger.info(
                f"Subiendo {DEFAULT_ARCHIVE_PATH} a Cloudflare R2 (knowledge_base/vector_db/)..."
            )
            uploaded = r2.upload_vector_db(local_archive=DEFAULT_ARCHIVE_PATH)
            if uploaded:
                logger.info("✅ Base vectorial ChromaDB sincronizada con éxito en Cloudflare R2.")
                remote_url = os.getenv("CHROMA_REMOTE_URL")
                if remote_url:
                    logger.info(
                        f"Notificando al backend remoto ({remote_url}) para recargar desde R2..."
                    )
                    try:
                        from src.rag.vector_store import ChromaVectorStore

                        store = ChromaVectorStore(remote_url=remote_url)
                        refresh_res = store.refresh_from_r2()
                        logger.info(f"✅ Space remoto actualizado exitosamente: {refresh_res}")
                    except Exception as e:
                        logger.warning(
                            f"Aviso: No se pudo notificar al Space remoto automáticamente: {e}"
                        )
            else:
                logger.error("❌ Falló la subida de la base vectorial a Cloudflare R2.")
        else:
            logger.warning(
                "R2StorageManager no está configurado. Omitiendo subida a Cloudflare R2."
            )

    return {
        "status": "completed",
        "newly_indexed": total_indexed,
        "total_in_collection": final_count,
        "archive_path": str(DEFAULT_ARCHIVE_PATH),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera la base vectorial ChromaDB y la sincroniza con Cloudflare R2 (Direct-to-Storage)."
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        default=True,
        help="Sube el archivo comprimido chroma_db.tar.gz a Cloudflare R2 (default: True).",
    )
    parser.add_argument(
        "--no-r2-sync",
        dest="r2_sync",
        action="store_false",
        help="Omite la subida a Cloudflare R2.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la recreación completa del índice desde cero sin restaurar el previo.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Tamaño de lote para la generación de embeddings en Gemini (default: 15).",
    )
    args = parser.parse_args()

    run_direct_to_storage_indexing(
        r2_sync=args.r2_sync,
        batch_size=args.batch_size,
        force=args.force,
    )
