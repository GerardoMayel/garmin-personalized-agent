"""Document Ledger and State Reconciliation for RAG Chunk Datasets.

Prevents orphaned chunks, ensures idempotency via SHA-256 fingerprinting,
and tracks changes across all document sources.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger("DocumentLedger")

DEFAULT_LEDGER_PATH = Path("data/knowledge_base/processed_chunks/_ledger.json")


class DocumentLedger:
    """Administra el registro de estado y control de versiones de documentos y chunks."""

    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER_PATH) -> None:
        self.ledger_path = Path(ledger_path)
        self.state: dict[str, Any] = {
            "version": "1.0.0",
            "last_updated": datetime.now(UTC).isoformat(),
            "sources": {},
        }
        self.load()

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Calcula el hash SHA-256 de un archivo en bloques de 64 KB."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load(self) -> None:
        """Carga el ledger desde disco si existe."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "sources" in data:
                        self.state = data
                        logger.debug(f"Ledger cargado desde {self.ledger_path}")
            except Exception as e:
                logger.warning(f"No se pudo cargar el ledger desde {self.ledger_path}: {e}")

    def save(self) -> None:
        """Guarda el estado actual del ledger en disco."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_updated"] = datetime.now(UTC).isoformat()
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        logger.debug(f"Ledger guardado en {self.ledger_path}")

    def is_document_unchanged(self, source: str, doc_name: str, current_hash: str) -> bool:
        """Determina si un documento ya ha sido procesado y su hash no ha cambiado."""
        source_data = self.state.get("sources", {}).get(source, {})
        doc_info = source_data.get("documents", {}).get(doc_name)
        if doc_info and doc_info.get("file_hash") == current_hash:
            return True
        return False

    def register_document(
        self,
        source: str,
        doc_name: str,
        file_hash: str,
        chunk_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Registra o actualiza un documento y sus chunks asociados dentro de una fuente."""
        if "sources" not in self.state:
            self.state["sources"] = {}
        if source not in self.state["sources"]:
            self.state["sources"][source] = {
                "total_documents": 0,
                "total_chunks": 0,
                "documents": {},
            }

        source_dict = self.state["sources"][source]
        source_dict["documents"][doc_name] = {
            "file_hash": file_hash,
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "updated_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }

        # Actualizar agregados de la fuente
        docs = source_dict["documents"]
        source_dict["total_documents"] = len(docs)
        source_dict["total_chunks"] = sum(d.get("chunk_count", 0) for d in docs.values())

    def reconcile_active_documents(
        self, source: str, current_active_filenames: set[str]
    ) -> list[str]:
        """Elimina del ledger documentos obsoletos y retorna los chunk_ids huérfanos purgados."""
        source_data = self.state.get("sources", {}).get(source)
        if not source_data:
            return []

        docs = source_data.get("documents", {})
        tracked_docs = set(docs.keys())
        orphaned_docs = tracked_docs - current_active_filenames
        purged_chunk_ids: list[str] = []

        for doc_name in orphaned_docs:
            orphan_info = docs.pop(doc_name, {})
            chunk_ids = orphan_info.get("chunk_ids", [])
            purged_chunk_ids.extend(chunk_ids)
            logger.info(
                f"Ledger: Purgando documento obsoleto '{doc_name}' ({len(chunk_ids)} chunks huérfanos)"
            )

        source_data["total_documents"] = len(docs)
        source_data["total_chunks"] = sum(d.get("chunk_count", 0) for d in docs.values())
        return purged_chunk_ids

    def get_summary(self) -> dict[str, Any]:
        """Retorna un resumen estructurado del estado de las fuentes."""
        summary: dict[str, Any] = {
            "last_updated": self.state.get("last_updated"),
            "sources": {},
        }
        for src, data in self.state.get("sources", {}).items():
            summary["sources"][src] = {
                "total_documents": data.get("total_documents", 0),
                "total_chunks": data.get("total_chunks", 0),
            }
        return summary
