"""RAG Schemas and Data Models for Document Chunks and Apache Parquet Serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Estructura canónica de un fragmento (chunk) de conocimiento."""

    chunk_id: str = Field(description="Identificador único del chunk (ej. doc_slug_c001)")
    doc_id: str = Field(description="Identificador único del documento de origen")
    chunk_index: int = Field(
        description="Índice ordinal del chunk dentro del documento (0-indexed)"
    )
    text: str = Field(description="Contenido textual del fragmento")
    tipo_fuente: str = Field(
        description="Categoría o fuente de origen (dispositivos_garmin_sensores, variables_fisiologia_humana, descripciones_metricas_garmin)"
    )
    idioma: str = Field(default="en", description="Código de idioma ISO 639-1 (ej. 'en', 'es')")
    token_count: int = Field(description="Número de tokens medidos con tokenizador BPE")
    char_count: int = Field(description="Longitud en caracteres")
    doc_title: str = Field(description="Título legible del documento")
    file_hash: str = Field(description="Hash SHA-256 del archivo fuente para control de cambios")
    metadata_json: str = Field(default="{}", description="Metadatos serializados en JSON")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Timestamp de creación ISO 8601",
    )

    def get_metadata_dict(self) -> dict[str, Any]:
        """Deserializa metadata_json a diccionario."""
        try:
            data = json.loads(self.metadata_json)
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}


PARQUET_CHUNK_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("tipo_fuente", pa.string(), nullable=False),
        pa.field("idioma", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
        pa.field("char_count", pa.int32(), nullable=False),
        pa.field("doc_title", pa.string(), nullable=False),
        pa.field("file_hash", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
    ]
)
