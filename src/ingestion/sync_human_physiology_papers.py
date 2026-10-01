"""Proceso 2: Sincronización de Evidencia Científica de Variables en el Cuerpo Humano.

Recupera y valida documentos técnicos y científicos (máximo 15 PDFs) que explican:
- Comportamiento biológico de las variables en el organismo humano.
- Sistema Nervioso Autónomo: balance Simpático (estrés) vs Parasimpático (recuperación VFC / RMSSD).
- Deuda de oxígeno mitocondrial (EPOC) y homeostasis post-ejercicio.
- Arquitectura biológica del sueño: fases profundas, ligeras y REM, y regulación vagal nocturna.
- Termodinámica humana y gasto calórico según el sustrato oxidado.
- Adaptaciones cardiovasculares y mitocondriales a largo plazo.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger("HumanPhysiologyPapersSync")

DEFAULT_OUTPUT_DIR = Path("data/knowledge_base/firstbeat/variables_fisiologia_humana")
MAX_DOCUMENTS = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class HumanPhysiologyPaper:
    """Metadatos de una publicación científica sobre la fisiología de variables humanas."""

    filename: str
    title: str
    source_url: str
    physiological_topic: str


HUMAN_PHYSIOLOGY_PAPERS: list[HumanPhysiologyPaper] = [
    HumanPhysiologyPaper(
        filename="stress_and_recovery_analysis_hrv.pdf",
        title="Stress and Recovery Analysis Method Based on 24-hour Heart Rate Variability",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/Stress-and-recovery_white-paper_20145.pdf",
        physiological_topic="Fisiología del Sistema Nervioso Autónomo: modulación simpática y parasimpática en VFC",
    ),
    HumanPhysiologyPaper(
        filename="athletes_recovery_analysis_hrv.pdf",
        title="Recovery Analysis for Athletic Training Based on Heart Rate Variability",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/Recovery-white-paper_15.6.20153.pdf",
        physiological_topic="Recuperación neuromuscular, tono vagal nocturno y prevención del sobreentrenamiento",
    ),
    HumanPhysiologyPaper(
        filename="excess_post_exercise_oxygen_consumption_epoc.pdf",
        title="Excess Post-Exercise Oxygen Consumption (EPOC)",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/white_paper_epoc.pdf",
        physiological_topic="Deuda de oxígeno celular, perturbación metabólica y restablecimiento de la homeostasis",
    ),
    HumanPhysiologyPaper(
        filename="aerobic_training_effect_epoc.pdf",
        title="Training Effect - Practical Assessment of Exercise Impact",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/white_paper_training_effect.pdf",
        physiological_topic="Adaptaciones cardiovasculares crónicas, capilarización y biogénesis mitocondrial",
    ),
    HumanPhysiologyPaper(
        filename="anaerobic_training_effect_assessment.pdf",
        title="Anaerobic Training Effect Assessment",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/FFW609US05-171.pdf",
        physiological_topic="Fisiología glucolítica, acumulación de lactato, acidosis y fosfocreatina muscular",
    ),
    HumanPhysiologyPaper(
        filename="energy_expenditure_estimation_hr_hrv.pdf",
        title="Energy Expenditure Estimation Based on Heart Rate and Heart Rate Variability",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/white_paper_energy_expenditure_estimation.pdf",
        physiological_topic="Termodinámica celular y oxidación de sustratos (grasas vs glucógeno) en gasto calórico",
    ),
    HumanPhysiologyPaper(
        filename="oxygen_consumption_estimation.pdf",
        title="Indirect EPOC and Oxygen Consumption Estimation Method",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/white_paper_vo2_estimation.pdf",
        physiological_topic="Cinética del transporte y consumo de oxígeno muscular durante el esfuerzo variable",
    ),
    HumanPhysiologyPaper(
        filename="health_and_fitness_benefits_physical_activity.pdf",
        title="Analysis of Health and Fitness Benefits of Physical Activity",
        source_url="https://www.firstbeat.com/wp-content/uploads/2026/09/Firstbeat-white-paper_analysis-of-health-and-fitness-benefits-of-PA_2018_3.pdf",
        physiological_topic="Beneficios fisiológicos de la actividad: salud endotelial, glucosa y presión arterial",
    ),
    HumanPhysiologyPaper(
        filename="hrv_prediction_individual_adaptation_runners.pdf",
        title="Heart Rate Variability in Prediction of Individual Adaptation to Endurance Training",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/myllymaki_et_al_2010.pdf",
        physiological_topic="Variabilidad individual en la respuesta genética y biológica al ejercicio aeróbico",
    ),
    HumanPhysiologyPaper(
        filename="hynynen_et_al_three_year_stress_study.pdf",
        title="Heart Rate Variability and Recovery in a Three-Year Longitudinal Study",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/hynynen_et_al_three_year.pdf",
        physiological_topic="Efectos biológicos a largo plazo del estrés crónico sobre la función cardíaca autónoma",
    ),
    HumanPhysiologyPaper(
        filename="feldt_et_al_occupational_stress_hrv.pdf",
        title="Psychophysiological Stress and Recovery Dynamics Across Working Populations",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/feldt_et_al_eawop_2007_congress.pdf",
        physiological_topic="Evaluación clínica del agotamiento psicofisiológico y activación simpática sostenida",
    ),
    HumanPhysiologyPaper(
        filename="rusko_acsm_2004_epoc_physiology.pdf",
        title="EPOC Kinetics and Cardiorespiratory Recovery in Exertion",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/rusko_acsm_2004_congress.pdf",
        physiological_topic="Cinética de recuperación de la deuda de oxígeno post-entrenamiento en atletas",
    ),
    HumanPhysiologyPaper(
        filename="rusko_et_al_acsm_2003_training_stimulus.pdf",
        title="Cardiac Autonomic Modulation Following Strenuous Physical Workloads",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/rusko_et_al_acsm_2003_congress-1.pdf",
        physiological_topic="Impacto de cargas intensas en el balance parasimpático agudo y subagudo",
    ),
    HumanPhysiologyPaper(
        filename="rusko_et_al_nes_2006_stress_and_work.pdf",
        title="Autonomic Nervous System Regulation and Physiological Recovery",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/rusko_et_al_nes_2006_congress.pdf",
        physiological_topic="Mecanismos de regulación del sistema nervioso autónomo ante altas demandas físicas",
    ),
    HumanPhysiologyPaper(
        filename="kinnunen_et_al_nes_2006_sleep_recovery.pdf",
        title="Sleep Architecture, Nocturnal Autonomic Regulation and Biological Recovery",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/kinnunen_et_al_nes_2006_congress.pdf",
        physiological_topic="Fisiología del sueño reparador humano y biomarcadores de recuperación durante la noche",
    ),
]


def is_valid_pdf(file_path: Path) -> bool:
    """Verifica que el archivo exista, no esté vacío y contenga la cabecera %PDF-."""
    if not file_path.exists() or file_path.stat().st_size < 1024:
        return False
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def download_pdf(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Descarga de forma segura un archivo PDF verificando su integridad binaria."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".pdf.tmp")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                logger.warning(f"HTTP {status} al descargar {url}")
                return False

            with open(temp_path, "wb") as out_file:
                while chunk := response.read(64 * 1024):
                    out_file.write(chunk)

        if not is_valid_pdf(temp_path):
            logger.error(f"El archivo descargado de {url} no es un PDF válido.")
            if temp_path.exists():
                temp_path.unlink()
            return False

        temp_path.replace(dest_path)
        return True

    except Exception as exc:
        logger.error(f"Error al descargar {url}: {exc}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def sync_human_physiology_papers(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    dry_run: bool = False,
    r2_sync: bool = False,
) -> dict[str, Any]:
    """Ejecuta el Proceso 2 limitando estrictamente a MAX_DOCUMENTS (15)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "already_present": 0,
        "downloaded": 0,
        "failed": 0,
        "total_files": 0,
        "files": [],
    }

    papers_to_sync = HUMAN_PHYSIOLOGY_PAPERS[:MAX_DOCUMENTS]
    logger.info(
        f"Iniciando Proceso 2: Variables en el Cuerpo Humano ({len(papers_to_sync)} documentos, máx: {MAX_DOCUMENTS})"
    )
    logger.info(f"Destino local: {output_dir.resolve()}")

    for paper in papers_to_sync:
        dest_file = output_dir / paper.filename

        if dest_file.exists() and is_valid_pdf(dest_file) and not force:
            logger.info(
                f"  ✓ Presente: {paper.filename} ({dest_file.stat().st_size / 1024:.1f} KB)"
            )
            results["already_present"] += 1
            results["files"].append(paper.filename)
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] Se descargaría: {paper.title} -> {paper.filename}")
            results["downloaded"] += 1
            continue

        logger.info(f"  ↓ Descargando: {paper.title}...")
        if download_pdf(paper.source_url, dest_file):
            size_kb = dest_file.stat().st_size / 1024
            logger.info(f"  ✓ Guardado: {paper.filename} ({size_kb:.1f} KB)")
            results["downloaded"] += 1
            results["files"].append(paper.filename)
        else:
            logger.error(f"  ✗ Falló: {paper.filename}")
            results["failed"] += 1

    results["total_files"] = len(list(output_dir.glob("*.pdf")))

    if r2_sync and not dry_run:
        from src.common.r2_storage import R2StorageClient

        r2_client = R2StorageClient()
        if r2_client.is_configured():
            logger.info("Sincronizando PDFs de Fisiología Humana con Cloudflare R2...")
            r2_stats = {"r2_uploaded": 0, "r2_failed": 0}
            remote_prefix = "knowledge_base/firstbeat/variables_fisiologia_humana"
            for pdf_path in output_dir.glob("*.pdf"):
                remote_key = f"{remote_prefix}/{pdf_path.name}"
                if r2_client.upload_file(pdf_path, remote_key):
                    r2_stats["r2_uploaded"] += 1
                else:
                    r2_stats["r2_failed"] += 1
            results["r2_stats"] = r2_stats
            logger.info(f"Sincronización R2 completada: {r2_stats}")
        else:
            logger.warning("R2 no configurado, omitiendo subida a la nube.")

    return results


def main() -> None:
    """CLI para el Proceso 2: Variables en el Cuerpo Humano."""
    parser = argparse.ArgumentParser(
        description="Proceso 2: Sincronización de Documentos sobre Fisiología Humana y Variables (Máx 15 PDFs)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de destino local (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Forzar re-descarga de documentos existentes"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simular ejecución sin realizar descargas"
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="Sincronizar PDFs con Cloudflare R2 (garmin-personal-data)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧬 FUENTE 2: Análisis de las Variables en el Cuerpo Humano")
    print(f"   • Directorio local:  {args.output_dir.resolve()}")
    print(
        "   • Destino remoto R2: garmin-personal-data/knowledge_base/firstbeat/variables_fisiologia_humana/"
    )
    print(f"   • Límite máx:        {MAX_DOCUMENTS} PDFs")
    print(
        f"   • Modo:              {'SIMULACIÓN (DRY-RUN)' if args.dry_run else 'DESCARGA ACTIVA'}"
    )
    print(f"   • Sincronización R2: {'SÍ' if args.r2_sync else 'NO'}")
    print("=" * 70 + "\n")

    summary = sync_human_physiology_papers(
        output_dir=args.output_dir, force=args.force, dry_run=args.dry_run, r2_sync=args.r2_sync
    )

    print("\n" + "-" * 70)
    print("📊 Resumen del Proceso 2 (Fisiología Humana):")
    print(f"   • Nuevos PDFs descargados: {summary['downloaded']}")
    print(f"   • Ya presentes y válidos:  {summary['already_present']}")
    print(f"   • Descargas fallidas:      {summary['failed']}")
    print(f"   • Total de PDFs en carpeta: {summary['total_files']} (máx: {MAX_DOCUMENTS})")
    if "r2_stats" in summary:
        print(f"   • Subidos a Cloudflare R2: {summary['r2_stats']['r2_uploaded']}")
    print("-" * 70 + "\n")
    print("-" * 70 + "\n")

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
