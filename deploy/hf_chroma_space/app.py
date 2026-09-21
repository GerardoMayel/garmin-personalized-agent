"""ChromaDB Vector Backend API for Hugging Face Spaces.

Exposes a unified ChromaDB collection ('biometric_knowledge_base') over FastAPI
with endpoints for health checks, batch upserts, vector queries, and collection statistics.
"""

from __future__ import annotations

import os
import tarfile
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import chromadb
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

COLLECTION_NAME = "biometric_knowledge_base"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")


def sync_chroma_from_r2_if_needed(force: bool = False) -> bool:
    """Descarga y descomprime chroma_db.tar.gz desde Cloudflare R2 si no existe o si se fuerza."""
    persist_dir = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR))
    sqlite_file = persist_dir / "chroma.sqlite3"

    if sqlite_file.exists() and not force:
        return True

    r2_account_id = os.getenv("R2_ACCOUNT_ID")
    r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
    r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    r2_bucket = os.getenv("R2_BUCKET_NAME")

    if not (r2_account_id and r2_access_key and r2_secret_key and r2_bucket):
        return False

    endpoint_url = (
        os.getenv("R2_ENDPOINT_URL") or f"https://{r2_account_id}.r2.cloudflarestorage.com"
    )

    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

        tar_path = Path("/tmp/chroma_db.tar.gz")
        print(f"Descargando chroma_db.tar.gz desde r2://{r2_bucket}/knowledge_base/vector_db/...")
        s3.download_file(r2_bucket, "knowledge_base/vector_db/chroma_db.tar.gz", str(tar_path))

        target_parent = persist_dir.parent
        target_parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_parent)

        if tar_path.exists():
            tar_path.unlink()

        print(f"Base vectorial restaurada exitosamente en {persist_dir}.")
        return True
    except Exception as e:
        print(f"Aviso: No se pudo sincronizar base vectorial desde Cloudflare R2: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: inicializa datos desde R2 al arrancar."""
    sync_chroma_from_r2_if_needed()
    yield


app = FastAPI(
    title="Garmin Biometric ChromaDB Vector Backend",
    description="Vector database service hosting biometric knowledge base chunks for RAG inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_chroma_collection(force_reload: bool = False) -> chromadb.Collection:
    """Obtiene o inicializa la colección persistente en ChromaDB con soporte de recarga."""
    global _client, _collection
    if _collection is None or force_reload:
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)
        os.makedirs(persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def verify_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)] = None,
) -> bool:
    """Verifica autenticación Bearer opcional.

    Si AUTH_BEARER_TOKEN está configurado en el contenedor, se valida el token.
    En Hugging Face Spaces privados, la autenticación principal es aplicada por el proxy inverso de HF con HF_TOKEN.
    """
    expected = os.getenv("AUTH_BEARER_TOKEN")
    if not expected:
        return True
    if not credentials or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Authorization Bearer token.",
        )
    return True


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UpsertRequest(BaseModel):
    ids: list[str] = Field(..., description="Lista de identificadores únicos de chunks.")
    embeddings: list[list[float]] = Field(
        ..., description="Vectores de embedding generados (768 dims)."
    )
    documents: list[str] = Field(..., description="Texto limpio del fragmento.")
    metadatas: list[dict[str, Any]] = Field(
        ..., description="Metadatos estructurados (source, tags, doc_id, etc.)."
    )


class QueryRequest(BaseModel):
    query_embedding: list[float] = Field(
        ..., description="Vector de embedding de la consulta del usuario (768 dims)."
    )
    n_results: int = Field(5, ge=1, le=100, description="Número de resultados a recuperar (top-k).")
    where: dict[str, Any] | None = Field(
        None, description="Filtro de metadatos opcional para Chroma (ej. {'source': 'firstbeat'})."
    )


class QueryResultItem(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    results: list[QueryResultItem]
    total_found: int


class DeleteRequest(BaseModel):
    ids: list[str] = Field(..., description="Lista de IDs a eliminar.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

START_TIME = time.time()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Garmin Biometric ChromaDB Vector Backend",
        "collection": COLLECTION_NAME,
        "docs_url": "/docs",
        "status": "online",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Endpoint de salud para keep-alive en GitHub Actions y monitoreo."""
    collection = get_chroma_collection()
    uptime_seconds = int(time.time() - START_TIME)
    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "uptime_seconds": uptime_seconds,
        "chroma_version": chromadb.__version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/stats")
def collection_stats(
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> dict[str, Any]:
    """Retorna desglose de registros por fuente en la colección."""
    collection = get_chroma_collection()
    total = collection.count()

    # Obtener desglose por source si hay documentos
    breakdown: dict[str, int] = {}
    if total > 0:
        sample = collection.get(include=["metadatas"])
        metadatas = sample.get("metadatas") or []
        for meta in metadatas:
            if isinstance(meta, dict):
                src = str(meta.get("source", "unknown"))
                breakdown[src] = breakdown.get(src, 0) + 1

    return {
        "collection": COLLECTION_NAME,
        "total_chunks": total,
        "sources": breakdown,
    }


@app.post("/upsert")
def upsert_chunks(
    payload: UpsertRequest,
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> dict[str, Any]:
    """Inserta o actualiza un lote de chunks con sus embeddings y metadatos."""
    if not payload.ids:
        return {
            "status": "ok",
            "upserted_count": 0,
            "total_in_collection": get_chroma_collection().count(),
        }

    if len(payload.ids) != len(payload.embeddings) or len(payload.ids) != len(payload.documents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The lengths of ids, embeddings, and documents must match.",
        )

    # Normalizar metadatos para tipos válidos en ChromaDB (str, int, float, bool)
    normalized_metadatas: list[dict[str, Any]] = []
    for meta in payload.metadatas:
        clean_meta: dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = ",".join(str(item) for item in v)
            elif v is None:
                continue
            else:
                clean_meta[k] = str(v)
        normalized_metadatas.append(clean_meta)

    collection = get_chroma_collection()
    collection.upsert(
        ids=payload.ids,
        embeddings=payload.embeddings,
        documents=payload.documents,
        metadatas=normalized_metadatas,
    )

    return {
        "status": "ok",
        "upserted_count": len(payload.ids),
        "total_in_collection": collection.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query_chunks(
    payload: QueryRequest,
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> QueryResponse:
    """Busca los fragmentos más relevantes por similitud de coseno con filtros opcionales."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        return QueryResponse(results=[], total_found=0)

    query_args: dict[str, Any] = {
        "query_embeddings": [payload.query_embedding],
        "n_results": min(payload.n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if payload.where:
        query_args["where"] = payload.where

    response = collection.query(**query_args)

    ids = response.get("ids", [[]])[0]
    docs = response.get("documents", [[]])[0]
    metas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    items: list[QueryResultItem] = []
    for i in range(len(ids)):
        items.append(
            QueryResultItem(
                id=ids[i],
                document=docs[i] if docs else "",
                metadata=metas[i] if metas else {},
                distance=float(distances[i]) if distances else 0.0,
            )
        )

    return QueryResponse(results=items, total_found=len(items))


@app.post("/delete")
def delete_chunks(
    payload: DeleteRequest,
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> dict[str, Any]:
    """Elimina fragmentos específicos por ID."""
    collection = get_chroma_collection()
    collection.delete(ids=payload.ids)
    return {
        "status": "ok",
        "deleted_count": len(payload.ids),
        "total_remaining": collection.count(),
    }


@app.post("/refresh-from-r2")
def refresh_from_r2(
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> dict[str, Any]:
    """Fuerza la descarga y descompresión de la base vectorial desde Cloudflare R2."""
    success = sync_chroma_from_r2_if_needed(force=True)
    collection = get_chroma_collection(force_reload=True)
    return {
        "status": "synced" if success else "failed",
        "total_chunks": collection.count(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
