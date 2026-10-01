"""Knowledge Base Document Ingestion and BPE Token Chunker.

Processes documents from 3 distinct sources:
1. Firstbeat Garmin Devices & Sensors (PDFs)
2. Firstbeat Human Physiology & Biological Variables (PDFs)
3. Garmin Connect Official Metric Descriptions & User Insights (JSON / MD)

Splits into 400 tokens with 40% overlap (~160 tokens), detects language via hybrid NLP,
reconciles state with DocumentLedger, and writes columnar Apache Parquet datasets to local
and Cloudflare R2.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pypdf
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.logger import get_logger
from src.rag.language_detector import detect_language
from src.rag.ledger import DocumentLedger
from src.rag.schemas import PARQUET_CHUNK_SCHEMA, ChunkRecord

logger = get_logger("RAGChunker")

DEFAULT_CHUNK_SIZE_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 160  # 40% de 400

SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "dispositivos_garmin_sensores": {
        "input_dir": Path("data/knowledge_base/firstbeat/dispositivos_garmin_sensores"),
        "pattern": "*.pdf",
        "output_dir": Path(
            "data/knowledge_base/processed_chunks/dispositivos_garmin_sensores/dataset_v1"
        ),
        "title": "Dispositivos Garmin, Sensores Ópticos y Algoritmos",
    },
    "variables_fisiologia_humana": {
        "input_dir": Path("data/knowledge_base/firstbeat/variables_fisiologia_humana"),
        "pattern": "*.pdf",
        "output_dir": Path(
            "data/knowledge_base/processed_chunks/variables_fisiologia_humana/dataset_v1"
        ),
        "title": "Variables Fisiológicas en el Cuerpo Humano",
    },
    "descripciones_metricas_garmin": {
        "input_dir": Path("data/knowledge_base/descripciones_metricas_garmin"),
        "pattern": "*.md",
        "output_dir": Path(
            "data/knowledge_base/processed_chunks/descripciones_metricas_garmin/dataset_v1"
        ),
        "title": "Glosario y Explicaciones de Métricas Garmin Connect",
    },
}


def clean_pdf_text(raw_text: str) -> str:
    """Limpia el texto extraído de un PDF eliminando encabezados ruidosos y saltos rotos."""
    # Eliminar marcas de agua recurrentes de Firstbeat y numeraciones
    text = re.sub(r"(?i)firstbeat\s+white\s+paper\b", "", raw_text)
    text = re.sub(r"(?i)firstbeat\s+technologies\s+ltd\.?", "", text)
    text = re.sub(r"\bpage\s+\d+\s+(?:of\s+\d+)?\b", "", text, flags=re.I)

    # Corregir palabras divididas por guiones al final de línea (ej. phy- \n siology -> physiology)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Normalizar espacios en blanco y saltos de línea consecutivos
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrae el contenido de un archivo PDF utilizando pypdf con limpieza estructural."""
    pages_text: list[str] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
    except Exception as e:
        logger.error(f"Error extrayendo texto de PDF {pdf_path}: {e}")
        return ""

    joined = "\n\n".join(pages_text)
    return clean_pdf_text(joined)


def extract_text_from_markdown(md_path: Path) -> str:
    """Lee y limpia un archivo Markdown."""
    try:
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        return content.strip()
    except Exception as e:
        logger.error(f"Error leyendo archivo Markdown {md_path}: {e}")
        return ""


def tag_physiological_metrics(text: str) -> list[str]:
    """Clasifica los temas o métricas fisiológicas detectadas en el texto."""
    metrics: list[str] = []
    lower = text.lower()
    mapping = {
        "vo2max": [r"vo2\s*max", r"oxygen consumption", r"consumo.*ox[íi]geno", r"aerobic fitness"],
        "hrv_rmssd": [
            r"hrv\b",
            r"rmssd",
            r"heart rate variability",
            r"variabilidad.*card[íi]aca",
            r"r-r interval",
        ],
        "epoc": [r"epoc\b", r"excess post-exercise", r"training effect", r"efecto.*entrenamiento"],
        "sleep": [r"sleep", r"sueño", r"rem\b", r"slow-wave", r"deep sleep", r"sueño profundo"],
        "stress": [
            r"stress",
            r"estr[ée]s",
            r"body battery",
            r"sympathetic",
            r"parasympathetic",
            r"auton[oó]mic",
        ],
        "ppg_sensor": [
            r"elevate",
            r"photoplethysmograph",
            r"fotopletismograf",
            r"led",
            r"optical sensor",
            r"sensor [oó]ptico",
        ],
        "calories": [
            r"calor[íi]a",
            r"calorie",
            r"energy expenditure",
            r"gasto cal[oó]rico",
            r"bmr\b",
        ],
        "respiration": [r"respiration", r"respiraci[oó]n", r"brpm", r"rsa\b", r"ventilatory"],
        "spo2": [
            r"spo2",
            r"pulse ox",
            r"pulsioximetr[íi]a",
            r"saturaci[oó]n.*ox[íi]geno",
            r"hypoxia",
        ],
    }
    for tag, patterns in mapping.items():
        if any(re.search(pat, lower) for pat in patterns):
            metrics.append(tag)
    return metrics


class KnowledgeBaseChunker:
    """Orquestador de particionado de documentos a chunks de 400 tokens en formato Apache Parquet."""

    def __init__(
        self,
        chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        ledger: DocumentLedger | None = None,
    ) -> None:
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens
        self.ledger = ledger or DocumentLedger()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def count_tokens(self, text: str) -> int:
        """Calcula el número exacto de tokens BPE."""
        return len(self.tokenizer.encode(text))

    def process_document(
        self,
        file_path: Path,
        tipo_fuente: str,
        doc_title: str,
        file_hash: str,
    ) -> list[ChunkRecord]:
        """Extrae, parte en chunks de 400 tokens y genera registros estructurados para un documento."""
        if file_path.suffix.lower() == ".pdf":
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_markdown(file_path)

        if not raw_text:
            logger.warning(f"Texto vacío extraído para {file_path.name}")
            return []

        # Particionado con solapamiento
        text_chunks = self.text_splitter.split_text(raw_text)
        doc_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", file_path.stem.lower()).strip("_")

        records: list[ChunkRecord] = []
        for idx, chunk_txt in enumerate(text_chunks):
            clean_chunk = chunk_txt.strip()
            if not clean_chunk:
                continue

            tok_count = self.count_tokens(clean_chunk)
            idioma = detect_language(clean_chunk)
            tags = tag_physiological_metrics(clean_chunk)

            chunk_id = f"{doc_slug}_c{idx:03d}"
            metadata = {
                "source_file": file_path.name,
                "detected_metrics": tags,
                "chunk_size_target": self.chunk_size,
                "chunk_overlap_target": self.chunk_overlap,
            }

            rec = ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_slug,
                chunk_index=idx,
                text=clean_chunk,
                tipo_fuente=tipo_fuente,
                idioma=idioma,
                token_count=tok_count,
                char_count=len(clean_chunk),
                doc_title=doc_title,
                file_hash=file_hash,
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            )
            records.append(rec)

        return records

    def process_source(
        self,
        source_key: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Procesa todos los documentos de una fuente y escribe el archivo Parquet."""
        cfg = SOURCE_CONFIGS.get(source_key)
        if not cfg:
            raise ValueError(f"Fuente desconocida: {source_key}")

        input_dir: Path = cfg["input_dir"]
        pattern: str = cfg["pattern"]
        output_dir: Path = cfg["output_dir"]
        source_title: str = cfg["title"]

        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "part-00001.parquet"

        if not input_dir.exists():
            logger.warning(f"Directorio de entrada no existe: {input_dir}")
            return {"source": source_key, "total_docs": 0, "total_chunks": 0}

        files = sorted(input_dir.glob(pattern))
        active_filenames = {f.name for f in files}

        logger.info(
            f"Procesando Fuente '{source_key}': {len(files)} documentos encontrados ({pattern})"
        )

        # Reconciliar documentos eliminados en el ledger
        purged = self.ledger.reconcile_active_documents(source_key, active_filenames)
        if purged:
            logger.info(f"Reconciliación: {len(purged)} chunks huérfanos purgados del ledger.")

        all_source_chunks: list[ChunkRecord] = []
        unchanged_count = 0
        processed_count = 0

        for f in files:
            file_hash = DocumentLedger.compute_sha256(f)
            title = f.stem.replace("_", " ").title()

            # Verificación de idempotencia
            if not force and self.ledger.is_document_unchanged(source_key, f.name, file_hash):
                logger.debug(f"  Omitiendo '{f.name}' (sin cambios en hash SHA-256).")
                unchanged_count += 1
            else:
                logger.info(f"  Particionando documento: {f.name}...")
                processed_count += 1

            # Generar chunks para el dataset Parquet consolidado
            chunks = self.process_document(f, source_key, title, file_hash)
            all_source_chunks.extend(chunks)

            # Registrar en el ledger
            chunk_ids = [c.chunk_id for c in chunks]
            self.ledger.register_document(
                source=source_key,
                doc_name=f.name,
                file_hash=file_hash,
                chunk_ids=chunk_ids,
                metadata={"title": title, "tokens": sum(c.token_count for c in chunks)},
            )

        # Escribir Apache Parquet utilizando PyArrow
        if all_source_chunks:
            pydict: dict[str, list[Any]] = {
                "chunk_id": [c.chunk_id for c in all_source_chunks],
                "doc_id": [c.doc_id for c in all_source_chunks],
                "chunk_index": [c.chunk_index for c in all_source_chunks],
                "text": [c.text for c in all_source_chunks],
                "tipo_fuente": [c.tipo_fuente for c in all_source_chunks],
                "idioma": [c.idioma for c in all_source_chunks],
                "token_count": [c.token_count for c in all_source_chunks],
                "char_count": [c.char_count for c in all_source_chunks],
                "doc_title": [c.doc_title for c in all_source_chunks],
                "file_hash": [c.file_hash for c in all_source_chunks],
                "metadata_json": [c.metadata_json for c in all_source_chunks],
                "created_at": [c.created_at for c in all_source_chunks],
            }
            table = pa.Table.from_pydict(pydict, schema=PARQUET_CHUNK_SCHEMA)
            pq.write_table(table, parquet_path, compression="snappy")
            size_kb = parquet_path.stat().st_size / 1024
            logger.info(
                f"  ✓ Dataset Parquet generado: {parquet_path} ({len(table)} chunks, {size_kb:.1f} KB)"
            )
        else:
            logger.warning(f"No se generaron chunks para la fuente {source_key}")

        self.ledger.save()

        return {
            "source": source_key,
            "title": source_title,
            "total_docs": len(files),
            "processed_docs": processed_count,
            "unchanged_docs": unchanged_count,
            "total_chunks": len(all_source_chunks),
            "parquet_path": str(parquet_path),
        }

    def process_all_sources(
        self,
        force: bool = False,
        r2_sync: bool = False,
    ) -> dict[str, Any]:
        """Ejecuta el particionado completo de las 3 fuentes de conocimiento y sincroniza con R2."""
        logger.info("=== Iniciando Pipeline de Chunking de Conocimiento (3 Fuentes) ===")
        results: dict[str, Any] = {"sources": {}, "total_chunks": 0}

        for source_key in SOURCE_CONFIGS:
            res = self.process_source(source_key=source_key, force=force)
            results["sources"][source_key] = res
            results["total_chunks"] += res["total_chunks"]

        # Persistir ledger
        self.ledger.save()
        logger.info(f"Ledger de estado guardado en {self.ledger.ledger_path}")

        if r2_sync:
            from src.common.r2_storage import R2StorageClient

            r2_client = R2StorageClient()
            if r2_client.is_configured():
                logger.info("Sincronizando chunks y ledger con Cloudflare R2...")
                r2_client.sync_processed_chunks()
                results["r2_synced"] = True
            else:
                logger.warning("R2 no configurado. Omitiendo sincronización remota de chunks.")
                results["r2_synced"] = False

        return results


def main() -> None:
    """Punto de entrada CLI para el chunker de RAG."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Particionado a Chunks (400 tokens / 40% overlap) y Exportación a Apache Parquet."
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCE_CONFIGS.keys()),
        default=None,
        help="Procesar únicamente una fuente específica (por defecto procesa las 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar reprocesamiento de documentos omitiendo verificación de hash SHA-256",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="Sincronizar datasets Parquet y ledger con Cloudflare R2 (garmin-personal-data)",
    )

    args = parser.parse_args()

    chunker = KnowledgeBaseChunker()

    print("\n" + "=" * 75)
    print("🧩 RAG PIPELINE: Particionado a Chunks (400 tokens, 40% overlap) -> Parquet")
    print("   • Fuentes: 3 (Dispositivos Garmin, Fisiología Humana, Glosario de Métricas)")
    print(f"   • Forzar reprocesamiento: {'SÍ' if args.force else 'NO (Idempotente por SHA-256)'}")
    print(f"   • Sincronización R2:      {'SÍ' if args.r2_sync else 'NO'}")
    print("=" * 75 + "\n")

    if args.source:
        res = chunker.process_source(args.source, force=args.force)
        print(f"✅ Fuente '{args.source}' procesada: {res['total_chunks']} chunks generados.")
        if args.r2_sync:
            from src.common.r2_storage import R2StorageClient

            r2 = R2StorageClient()
            if r2.is_configured():
                r2.sync_processed_chunks()
    else:
        full_res = chunker.process_all_sources(force=args.force, r2_sync=args.r2_sync)
        print("\n" + "-" * 75)
        print(f"✅ Pipeline completado: {full_res['total_chunks']} chunks generados en total.")
        for src, data in full_res["sources"].items():
            print(
                f"   • {src}: {data['total_docs']} docs -> {data['total_chunks']} chunks ({data.get('parquet_path', '')})"
            )
        print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
