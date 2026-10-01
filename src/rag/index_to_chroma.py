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
            chunk_file_hash = str(pydict.get("file_hash", [""])[i] or "")
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
                    "file_hash": chunk_file_hash,
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

    # 4. Reconciliación de estado contra ChromaDB (Nuevos, Modificados y Eliminados)
    existing_meta_by_id: dict[str, dict[str, Any]] = {}
    try:
        data = collection.get(include=["metadatas"])
        ex_ids = data.get("ids", [])
        ex_metas = data.get("metadatas") or []
        for cid, meta in zip(ex_ids, ex_metas, strict=False):
            existing_meta_by_id[cid] = meta if isinstance(meta, dict) else {}
    except Exception as e:
        logger.warning(f"No se pudo consultar estado existente de ChromaDB: {e}")

    existing_ids = set(existing_meta_by_id.keys())
    target_ids = {c["chunk_id"] for c in chunks}

    # A. Fragmentos eliminados u obsoletos (chunks que estaban en ChromaDB pero ya no están en Parquets)
    orphaned_ids = list(existing_ids - target_ids)
    if orphaned_ids:
        logger.info(
            f"Detectados {len(orphaned_ids)} fragmentos obsoletos/eliminados en ChromaDB. Purgando..."
        )
        for i in range(0, len(orphaned_ids), 200):
            collection.delete(ids=orphaned_ids[i : i + 200])
        logger.info(f"✓ Purgados {len(orphaned_ids)} fragmentos huérfanos de la colección.")

    # B. Fragmentos nuevos o con documento modificado (hash SHA-256 diferente)
    to_embed_chunks: list[dict[str, Any]] = []
    metadata_backfill_ids: list[str] = []
    metadata_backfill_metas: list[dict[str, Any]] = []

    for c in chunks:
        cid = c["chunk_id"]
        if cid not in existing_meta_by_id:
            to_embed_chunks.append(c)
        else:
            old_meta = existing_meta_by_id[cid]
            old_hash = old_meta.get("file_hash")
            if c["file_hash"] and old_hash and c["file_hash"] != old_hash:
                logger.info(
                    f"Documento modificado detectado para chunk '{cid}' "
                    f"({old_hash[:8]} -> {c['file_hash'][:8]}). Re-indexando..."
                )
                to_embed_chunks.append(c)
            elif not old_hash and c["file_hash"]:
                # Retrocompatibilidad: registrar file_hash en metadata sin recalcular embeddings
                metadata_backfill_ids.append(cid)
                updated_meta = dict(old_meta)
                updated_meta["file_hash"] = c["file_hash"]
                metadata_backfill_metas.append(updated_meta)

    # Actualizar metadatos faltantes en lote (costo 0 API)
    if metadata_backfill_ids:
        logger.info(
            f"Actualizando metadata de trazabilidad (file_hash) para {len(metadata_backfill_ids)} fragmentos sin re-calcular embeddings..."
        )
        for i in range(0, len(metadata_backfill_ids), 200):
            collection.update(
                ids=metadata_backfill_ids[i : i + 200],
                metadatas=cast(Any, metadata_backfill_metas[i : i + 200]),
            )

    logger.info(
        f"Diagnóstico ChromaDB: {len(existing_ids)} existentes en base | "
        f"{len(orphaned_ids)} purgados | "
        f"{len(to_embed_chunks)} nuevos/modificados a vectorizar."
    )

    # C. Si no hay cambios ni nuevos embeddings y no se fuerza reconstrucción: Salir temprano
    has_changes = bool(to_embed_chunks or orphaned_ids or metadata_backfill_ids)
    if not has_changes and not force:
        logger.info(
            "✅ Todos los documentos y fragmentos están 100% sincronizados con ChromaDB. "
            "No se requieren llamadas a Gemini API ni actualización en Cloudflare R2."
        )
        return {
            "status": "up_to_date",
            "newly_indexed": 0,
            "purged_orphans": len(orphaned_ids),
            "total_in_collection": collection.count(),
            "archive_path": str(DEFAULT_ARCHIVE_PATH),
        }

    total_indexed = 0
    if to_embed_chunks:
        logger.info(
            f"Indexando {len(to_embed_chunks)} fragmentos nuevos/modificados con Gemini Embeddings..."
        )
        embedding_engine = GeminiEmbeddingEngine(batch_size=batch_size)

        texts = [c["content"] for c in to_embed_chunks]
        titles = [f"{c['source']}:{c['document_id']}" for c in to_embed_chunks]
        embeddings = embedding_engine.embed_documents(
            texts=texts, titles=titles, batch_size=batch_size
        )

        ids = [c["chunk_id"] for c in to_embed_chunks]
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
                "file_hash": c["file_hash"],
            }
            for c in to_embed_chunks
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
        "purged_orphans": len(orphaned_ids),
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
