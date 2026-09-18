"""Proceso 1: Sincronización de Evidencia Científica de Dispositivos Garmin y Sensores.

Recupera y valida documentos técnicos y científicos (máximo 15 PDFs) que explican:
- Sensores físicos: Sensor óptico PPG (Garmin Elevate), acelerometría, pulsioxímetro y altímetro.
- Procesamiento digital de señales (DSP) y detección de picos R-R.
- Eliminación de artefactos de movimiento.
- Algoritmos embebidos en el hardware del reloj (fusión de sensores para sueño, estimación de VO2max en carrera, etc.).
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

logger = get_logger("GarminDevicePapersSync")

DEFAULT_OUTPUT_DIR = Path("data/knowledge_base/firstbeat/dispositivos_garmin_sensores")
MAX_DOCUMENTS = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class GarminDevicePaper:
    """Metadatos de un estudio técnico o validación de sensores de dispositivos Garmin."""

    filename: str
    title: str
    source_url: str
    sensor_topic: str


GARMIN_DEVICE_PAPERS: list[GarminDevicePaper] = [
    GarminDevicePaper(
        filename="accuracy_heart_rate_and_hrv_monitoring.pdf",
        title="Accuracy of Firstbeat Heart Rate and HRV Monitoring",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/white_paper_bodyguard2_final.pdf",
        sensor_topic="Sensor óptico PPG vs ECG: filtrado de artefactos y precisión R-R en movimiento",
    ),
    GarminDevicePaper(
        filename="sleep_analysis_method_based_on_hrv.pdf",
        title="A Sleep Analysis Method Based on Heart Rate Variability",
        source_url="https://www.firstbeat.com/wp-content/uploads/2019/11/A-Sleep-Analysis-Method-Based-on-Heart-Rate-Variability-071119.pdf",
        sensor_topic="Fusión de acelerómetro triaxial de muñeca y VFC óptica para clasificación de sueño",
    ),
    GarminDevicePaper(
        filename="automated_vo2max_estimation.pdf",
        title="Automated VO2max Estimation for Fitness Assessment",
        source_url="https://www.firstbeat.com/wp-content/uploads/2017/06/white_paper_VO2max_30.6.2017.pdf",
        sensor_topic="Algoritmo embebido en reloj combinando velocidad GPS, altitud barométrica y pulso",
    ),
    GarminDevicePaper(
        filename="antila_embec_04_signal_processing.pdf",
        title="Signal Processing and Beat-to-Beat Interval Detection in Wearable Telemetry",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/antila_embec_04_congress_report.pdf",
        sensor_topic="Procesamiento digital de señales (DSP) en sensores de muñeca y portátiles",
    ),
    GarminDevicePaper(
        filename="luhtanen_2007_signal_processing.pdf",
        title="Mathematical Modeling and Signal Processing of Wearable Physiological Data",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/luhtanen_2007_congress.pdf",
        sensor_topic="Filtrado y reconstrucción en tiempo real de intervalos entre latidos en dispositivos ponibles",
    ),
    GarminDevicePaper(
        filename="smolander_et_al_2007_wearable_validation.pdf",
        title="Validation of Portable Wearable Heart Rate and HRV Monitoring vs Clinical Holter",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/smolander_et_al_2007.pdf",
        sensor_topic="Validación clínica de sensores portátiles frente a estándares hospitalarios",
    ),
    GarminDevicePaper(
        filename="smolander_et_al_ecss_2007_sensor_accuracy.pdf",
        title="Evaluation of Beat-to-Beat Detection Accuracy Under Dynamic Movement",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/smolander_et_al_ecss_2007_congress-1.pdf",
        sensor_topic="Exactitud del registro de latidos cardíacos en condiciones de vibración y movimiento deportivo",
    ),
    GarminDevicePaper(
        filename="pulkkinen_at_al_acsm_2004_sensor_reliability.pdf",
        title="Reliability and Validity of Wearable Sensor Telemetry in Everyday Physical Tasks",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/pulkkinen_at_al_acsm_2004_congress-1.pdf",
        sensor_topic="Fiabilidad de sensores fisiológicos en la vida diaria y tareas laborales",
    ),
    GarminDevicePaper(
        filename="pulkkinen_et_al_acsm_2005_heart_rate_accuracy.pdf",
        title="Accuracy Assessment of Wrist and Wearable Heart Rate Monitoring in Field Settings",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/pulkkinen_et_al_acsm_2005_congress.pdf",
        sensor_topic="Precisión de sensores portátiles en carrera y actividades de campo abierto",
    ),
    GarminDevicePaper(
        filename="montgomery_et_al_2009_wearable_monitoring.pdf",
        title="Wearable Physiological Telemetry in Demanding Operational Environments",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/montgomery_et_al_2009.pdf",
        sensor_topic="Comportamiento del sensor y robustez de señal bajo sudoración y temperaturas extremas",
    ),
    GarminDevicePaper(
        filename="wunsch_univ_bayeroit_2005_sensor_report.pdf",
        title="Hardware and Telemetry Evaluation of Wearable Physiological Recorders",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/wunsch_univ_bayeroit_2005_t6_report.pdf",
        sensor_topic="Evaluación de hardware de telemetría y algoritmos de captura de pulso",
    ),
    GarminDevicePaper(
        filename="wunsch_univ_bayeroit_2006_summary.pdf",
        title="Summary Evaluation of Sensor Accuracy and R-R Interval Acquisition",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/wunsch_univ_bayeroit_2006_t6_summary.pdf",
        sensor_topic="Adquisición de R-R por microcontroladores de baja potencia en relojes y pulsómetros",
    ),
    GarminDevicePaper(
        filename="vanttinen_et_al_computer_science_sensor_data.pdf",
        title="Computer Science in Sport: Algorithms for Real-Time Sensor Stream Parsing",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/vanttinen_et_al_international_symposium_computer_science_in_sport_2007.pdf",
        sensor_topic="Estructura computacional para el parseo en tiempo real de flujos telemétricos de sensores",
    ),
    GarminDevicePaper(
        filename="vanttinen_et_al_sensor_telemetry_in_sports.pdf",
        title="Wearable Sensor Telemetry Analysis in Field Team Sports",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/vanttinen_et_al_world_congress_of_science_football_2007.pdf",
        sensor_topic="Captura y robustez de señal en telemetría de sensores durante aceleraciones y colisiones",
    ),
    GarminDevicePaper(
        filename="hayrinen_et_al_telemetry_accuracy_monitoring.pdf",
        title="Telemetry Accuracy and Sensor Calibration for Scientific Monitoring",
        source_url="https://www.firstbeat.com/wp-content/uploads/2015/10/hayrinen_et_al_science_for_success_2007_congress.pdf",
        sensor_topic="Calibración y estándares de precisión en la medición continua de parámetros de esfuerzo",
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


def sync_garmin_device_papers(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    dry_run: bool = False,
    r2_sync: bool = False,
) -> dict[str, Any]:
    """Ejecuta el Proceso 1 limitando estrictamente a MAX_DOCUMENTS (15)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "already_present": 0,
        "downloaded": 0,
        "failed": 0,
        "total_files": 0,
        "files": [],
    }

    papers_to_sync = GARMIN_DEVICE_PAPERS[:MAX_DOCUMENTS]
    logger.info(
        f"Iniciando Proceso 1: Dispositivos Garmin y Sensores ({len(papers_to_sync)} documentos, máx: {MAX_DOCUMENTS})"
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
            logger.info("Sincronizando PDFs de Dispositivos Garmin con Cloudflare R2...")
            r2_stats = {"r2_uploaded": 0, "r2_failed": 0}
            remote_prefix = "knowledge_base/firstbeat/dispositivos_garmin_sensores"
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
    """CLI para el Proceso 1: Dispositivos Garmin y Sensores."""
    parser = argparse.ArgumentParser(
        description="Proceso 1: Sincronización de Documentos sobre Dispositivos y Sensores Garmin (Máx 15 PDFs)."
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
    print("⌚ FUENTE 1: Dispositivos Garmin, Sensores y Algoritmos Embebidos")
    print(f"   • Directorio local:  {args.output_dir.resolve()}")
    print(
        "   • Destino remoto R2: garmin-personal-data/knowledge_base/firstbeat/dispositivos_garmin_sensores/"
    )
    print(f"   • Límite máx:        {MAX_DOCUMENTS} PDFs")
    print(
        f"   • Modo:              {'SIMULACIÓN (DRY-RUN)' if args.dry_run else 'DESCARGA ACTIVA'}"
    )
    print(f"   • Sincronización R2: {'SÍ' if args.r2_sync else 'NO'}")
    print("=" * 70 + "\n")

    summary = sync_garmin_device_papers(
        output_dir=args.output_dir, force=args.force, dry_run=args.dry_run, r2_sync=args.r2_sync
    )

    print("\n" + "-" * 70)
    print("📊 Resumen del Proceso 1 (Dispositivos Garmin):")
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
