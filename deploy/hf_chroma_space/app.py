"""Garmin Biometric RAG Backend for Hugging Face Spaces.

Exposes a unified ChromaDB collection ('biometric_knowledge_base') over FastAPI
with full RAG generation, multi-layer guardrails (Rate Limiting, Semantic Domain Filter,
Prompt Injection Guard), batch upserts, vector queries, and R2 synchronization.
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
import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    from guardrails import (
        LLMBudgetTracker,
        check_domain,
        detect_query_language,
        normalize_text,
        rerank_candidates,
        scan_prompt_injection,
    )
except ImportError:
    from deploy.hf_chroma_space.guardrails import (
        LLMBudgetTracker,
        check_domain,
        detect_query_language,
        normalize_text,
        rerank_candidates,
        scan_prompt_injection,
    )

COLLECTION_NAME = "biometric_knowledge_base"
ALLOWED_RAG_SOURCES = ["variables_fisiologia_humana", "dispositivos_garmin_sensores"]
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
DEFAULT_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash-lite")
SIMILARITY_THRESHOLD = 0.52

DOMAIN_REFERENCE_TEXT = (
    "Métricas fisiológicas de Garmin Connect, variabilidad de la frecuencia cardíaca rMSSD, "
    "estrés, descanso, sueño profundo y REM, VO2 max, carga y estado de entrenamiento, "
    "Firstbeat Analytics, pulsaciones, sensores ópticos Elevate PPG, acelerometría, "
    "saturación de oxígeno SpO2, recuperación muscular, fisiología deportiva y rendimiento cardiovascular."
)

llm_budget = LLMBudgetTracker(
    max_per_hour=int(os.getenv("MAX_RAG_CALLS_PER_HOUR", "60")),
    max_per_day=int(os.getenv("MAX_RAG_CALLS_PER_DAY", "100")),
)


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
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=target_parent, filter="data")
        except TypeError:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=target_parent)

        if tar_path.exists():
            tar_path.unlink()

        print(f"Base vectorial restaurada exitosamente en {persist_dir}.")
        return True
    except Exception as e:
        print(f"Aviso: No se pudo sincronizar base vectorial desde Cloudflare R2: {e}")
        return False


_domain_reference_embedding: np.ndarray | None = None


def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY no está configurada en las variables de entorno del servidor.",
        )
    return key


def get_query_embedding(text: str) -> list[float]:
    """Genera vector denso de 768 dimensiones con Gemini para búsqueda semántica."""
    api_key = get_gemini_api_key()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{DEFAULT_EMBEDDING_MODEL}:embedContent?key={api_key}"
    )
    payload = {
        "model": f"models/{DEFAULT_EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text.strip()}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": 768,
    }
    resp = requests.post(
        url, json=payload, headers={"Content-Type": "application/json"}, timeout=30
    )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al generar embedding de consulta ({resp.status_code}): {resp.text[:200]}",
        )
    return resp.json()["embedding"]["values"]


def get_domain_reference_embedding() -> np.ndarray:
    """Obtiene o precarga el vector centroid de referencia del dominio Garmin."""
    global _domain_reference_embedding
    if _domain_reference_embedding is None:
        emb = get_query_embedding(DOMAIN_REFERENCE_TEXT)
        _domain_reference_embedding = np.array(emb, dtype=np.float32)
    return _domain_reference_embedding


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def call_gemini_llm(system_prompt: str, user_content: str) -> str:
    """Llama al LLM de Gemini con prompt de sistema y contexto RAG."""
    api_key = get_gemini_api_key()
    models_to_try = [DEFAULT_LLM_MODEL, "gemini-3.1-flash-lite", "gemini-3.6-flash"]

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1500,
        },
    }

    last_error = ""
    for model_name in models_to_try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        try:
            resp = requests.post(
                url, json=payload, headers={"Content-Type": "application/json"}, timeout=45
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    full_text = "".join(
                        str(p.get("text", "")) for p in parts if "text" in p
                    ).strip()
                    if full_text:
                        return full_text
            last_error = f"{model_name} HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = str(e)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Error en generación con Gemini LLM: {last_error}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: inicializa datos desde R2 al arrancar."""
    sync_chroma_from_r2_if_needed()
    try:
        get_chroma_collection()
    except Exception as e:
        print(f"Aviso al precargar colección ChromaDB: {e}")
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Garmin Biometric RAG Backend",
    description=(
        "Full RAG Backend hosting ChromaDB vector storage, multi-layer guardrails "
        "(Rate Limiting, Semantic Domain Filter, Prompt Injection Guard), and Gemini synthesis "
        "for Garmin physiological and biomechanical telemetry."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description="Pregunta del usuario sobre métricas Garmin o fisiología deportiva.",
    )
    top_k: int = Field(3, ge=1, le=10, description="Número de fragmentos relevantes a recuperar.")
    where: dict[str, Any] | None = Field(
        None,
        description="Filtro opcional de metadatos (ej. {'source': 'variables_fisiologia_humana'}).",
    )


class SourceCitation(BaseModel):
    id: str
    document_id: str
    source: str
    distance: float
    excerpt: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    status: str
    query: str
    answer: str
    domain_similarity: float
    guardrail_status: str
    sources: list[SourceCitation]
    budget_usage: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

START_TIME = time.time()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Garmin Biometric RAG Backend",
        "collection": COLLECTION_NAME,
        "docs_url": "/docs",
        "status": "online",
        "guardrails": [
            "rate_limiting",
            "prompt_injection_guard",
            "domain_intent_classifier",
            "llm_budget_protection",
        ],
        "llm_budget": llm_budget.get_usage(),
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
        "llm_budget": llm_budget.get_usage(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/stats")
def collection_stats(
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> dict[str, Any]:
    """Retorna desglose de registros por fuente en la colección."""
    collection = get_chroma_collection()
    total = collection.count()

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
        "llm_budget": llm_budget.get_usage(),
    }


@app.post("/ask", response_model=AskResponse)
@limiter.limit("15/minute")
@limiter.limit("60/hour")
@limiter.limit("100/day")
def ask_rag(
    request: Request,
    payload: AskRequest,
    _auth: Annotated[bool, Depends(verify_auth)] = True,
) -> AskResponse:
    """Endpoint integral RAG: Guardrails NLP + Embeddings + ChromaDB + Gemini Synthesis + Budget Guard."""
    user_query = payload.query.strip()
    clean_query = normalize_text(user_query)

    # --- Capa 1: Filtro Estricto de Idioma (Solo Español e Inglés permitidos) ---
    lang, is_allowed = detect_query_language(user_query)
    if not is_allowed:
        return AskResponse(
            status="unsupported_language",
            query=user_query,
            answer=(
                "Idioma no permitido. El asistente de telemetría y fisiología de Garmin solo "
                "acepta consultas en español o inglés.\n"
                "Unsupported language. The Garmin telemetry and physiology assistant only "
                "accepts queries in Spanish or English."
            ),
            domain_similarity=0.0,
            guardrail_status="language_rejected",
            sources=[],
            budget_usage=llm_budget.get_usage(),
        )

    # --- Capa 3: Sanitización y Heurísticas NLP contra Prompt Injections ---
    is_malicious, attack_type = scan_prompt_injection(user_query)
    if is_malicious:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Petición bloqueada por políticas de seguridad ({attack_type}).",
        )

    # --- Capa 2: Clasificador de Dominio (Lexicón + Coseno con el Embedding único) ---
    query_vector = get_query_embedding(clean_query)
    ref_vector = get_domain_reference_embedding()
    is_in_domain, domain_sim, match_method = check_domain(
        clean_query, query_vector, ref_vector, threshold=SIMILARITY_THRESHOLD
    )

    if not is_in_domain:
        return AskResponse(
            status="out_of_domain",
            query=user_query,
            answer=(
                "Solo tengo autorización para responder preguntas relacionadas con métricas "
                "fisiológicas, Garmin Connect, Firstbeat Analytics y rendimiento deportivo "
                "(sueño, estrés, VFC/rMSSD, carga de entrenamiento y recuperación)."
            ),
            domain_similarity=round(domain_sim, 4),
            guardrail_status="domain_rejected",
            sources=[],
            budget_usage=llm_budget.get_usage(),
        )

    # --- Flujo RAG: Recuperación de ChromaDB ---
    collection = get_chroma_collection()
    if collection.count() == 0:
        return AskResponse(
            status="empty_knowledge_base",
            query=user_query,
            answer="La base de conocimiento biomédica no contiene documentos indexados actualmente.",
            domain_similarity=round(domain_sim, 4),
            guardrail_status="passed",
            sources=[],
            budget_usage=llm_budget.get_usage(),
        )

    # --- Capa 4: Verificación y Consumo de Presupuesto LLM (Máx 60/hora, 100/día) ---
    budget_ok, budget_msg = llm_budget.check_and_consume()
    if not budget_ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=budget_msg,
        )

    # Restringir estrictamente la recuperación a variables_fisiologia_humana y dispositivos_garmin_sensores
    base_source_filter: dict[str, Any] = {"source": {"$in": ALLOWED_RAG_SOURCES}}
    if payload.where:
        if "source" in payload.where:
            req_src = payload.where["source"]
            if isinstance(req_src, str) and req_src in ALLOWED_RAG_SOURCES:
                source_filter = {"source": req_src}
            elif isinstance(req_src, dict) and "$in" in req_src:
                valid_subset = [s for s in req_src["$in"] if s in ALLOWED_RAG_SOURCES]
                source_filter = {"source": {"$in": valid_subset or ALLOWED_RAG_SOURCES}}
            else:
                source_filter = base_source_filter
        else:
            source_filter = {
                "$and": [
                    base_source_filter,
                    payload.where,
                ]
            }
    else:
        source_filter = base_source_filter

    # Recuperar candidatos expandidos para la fase de Reranking
    candidate_pool_size = min(max(payload.top_k * 3, 10), collection.count())
    query_args: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": candidate_pool_size,
        "where": source_filter,
        "include": ["documents", "metadatas", "distances"],
    }

    query_res = collection.query(**query_args)
    raw_ids = query_res.get("ids", [[]])[0]
    raw_docs = query_res.get("documents", [[]])[0]
    raw_metas = query_res.get("metadatas", [[]])[0]
    raw_dists = query_res.get("distances", [[]])[0]

    candidate_items: list[dict[str, Any]] = []
    for i in range(len(raw_ids)):
        candidate_items.append(
            {
                "id": raw_ids[i],
                "document": raw_docs[i] if raw_docs else "",
                "metadata": raw_metas[i] if raw_metas else {},
                "distance": float(raw_dists[i]) if raw_dists else 0.0,
            }
        )

    # --- Motor de Reranking Híbrido (RRF + BM25) ---
    reranked_results = rerank_candidates(clean_query, candidate_items, top_k=payload.top_k)

    if not reranked_results:
        return AskResponse(
            status="no_matching_evidence",
            query=user_query,
            answer=(
                "No se encontró evidencia biomédica suficiente en las fuentes autorizadas "
                "('variables_fisiologia_humana' y 'dispositivos_garmin_sensores') para responder "
                "con rigor a esta consulta."
            ),
            domain_similarity=round(domain_sim, 4),
            guardrail_status="passed",
            sources=[],
            budget_usage=llm_budget.get_usage(),
        )

    sources_list: list[SourceCitation] = []
    context_blocks: list[str] = []

    for i, cand in enumerate(reranked_results):
        cid = cand["id"]
        doc_txt = str(cand.get("document", ""))
        meta = cand.get("metadata", {}) or {}
        dist = float(cand.get("distance", 0.0))
        rerank_score = float(cand.get("rerank_score", 0.0))

        sources_list.append(
            SourceCitation(
                id=cid,
                document_id=str(meta.get("document_id", "unknown")),
                source=str(meta.get("source", "unknown")),
                distance=round(dist, 4),
                excerpt=doc_txt[:250] + "..." if len(doc_txt) > 250 else doc_txt,
                metadata={**meta, "rerank_score": rerank_score},
            )
        )
        context_blocks.append(
            f"[Documento {i + 1}: {meta.get('document_id')} | Fuente: {meta.get('source')} | Distancia Coseno: {dist:.4f} | Rerank Score: {rerank_score:.5f}]\n{doc_txt}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    if lang == "en":
        lang_instruction = (
            "Language Requirement: The user asked in English. You must provide your complete "
            "response in English, maintaining technical precision and clear explanations."
        )
    else:
        lang_instruction = (
            "Requisito Estricto de Idioma: La respuesta siempre debe darse en español, "
            "independientemente del idioma de los documentos o de cómo se haya formulado la pregunta. "
            "Mantén un tono técnico, pedagógico y riguroso."
        )

    system_prompt = (
        "Eres el Asistente Experto en Fisiología Deportiva y Telemetría de Garmin del usuario.\n"
        "Tu misión es responder rigurosa y pedagógicamente a la consulta basándote exclusivamente en la evidencia "
        "biomédica y tecnológica provista en los fragmentos de contexto (Firstbeat Analytics y sensores Garmin: "
        "variables_fisiologia_humana y dispositivos_garmin_sensores).\n\n"
        f"{lang_instruction}\n\n"
        "Directrices:\n"
        "1. Proporciona explicaciones fisiológicas claras, precisas y accionables para el deportista.\n"
        "2. Cita de forma natural los whitepapers o fuentes relevantes presentes en el contexto.\n"
        "3. Si la respuesta no puede derivarse de la evidencia provista, indícalo con transparencia sin inventar datos.\n"
        "4. Mantén un tono técnico, motivador y profesional."
    )

    user_prompt = (
        f"CONTEXTO BIOMÉDICO RECUPERADO:\n{context_str}\n\n"
        f"PREGUNTA DEL USUARIO:\n{user_query}\n\n"
        "Responde de forma completa, estructurada y basada en el contexto anterior."
    )

    # --- Llamada de Generación LLM ---
    answer = call_gemini_llm(system_prompt=system_prompt, user_content=user_prompt)

    return AskResponse(
        status="success",
        query=user_query,
        answer=answer,
        domain_similarity=round(domain_sim, 4),
        guardrail_status="passed",
        sources=sources_list,
        budget_usage=llm_budget.get_usage(),
    )


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
