"""Proceso 3: Compilación y Sincronización de Descripciones y Explicaciones de Métricas Garmin.

Compila el glosario oficial, descripciones taxonómicas y explicaciones contextuales
de las variables y telemetría que GarminConnect reporta, incluyendo los insights
de sueño, estados de HRV, estrés, Body Battery, VO2 Max y carga de entrenamiento.
Genera versiones estructuradas en Markdown y JSON para ingesta en la base de conocimiento,
y sincroniza con Cloudflare R2 bajo el prefijo:
`knowledge_base/descripciones_metricas_garmin/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.logger import get_logger

logger = get_logger("GarminMetricSync")

DEFAULT_OUTPUT_DIR = Path("data/knowledge_base/descripciones_metricas_garmin")

GARMIN_METRIC_DEFINITIONS: list[dict[str, Any]] = [
    {
        "metric_key": "hrv_status",
        "name_es": "Estado de Variabilidad de la Frecuencia Cardíaca (HRV)",
        "name_en": "Heart Rate Variability (HRV) Status",
        "category": "recuperacion_autonomica",
        "unit": "ms (rMSSD)",
        "description_es": (
            "El estado de HRV de Garmin evalúa el equilibrio del Sistema Nervioso Autónomo (SNA) "
            "comparando el promedio nocturno de rMSSD (raíz cuadrada de las diferencias medias de "
            "intervalos R-R sucesivos) contra una línea base personal calculada durante las últimas "
            "3 a 4 semanas. Se clasifica en cuatro estados: Equilibrado (Balanced), Desequilibrado "
            "(Unbalanced), Bajo (Low) y Pobre (Poor). Un estado Equilibrado indica una óptima "
            "actividad del tono parasimpático y capacidad de recuperación. Un estado Desequilibrado "
            "o Bajo refleja estrés fisiológico acumulado, fatiga simpática, sobreentrenamiento, "
            "consumo de alcohol o el inicio de una infección inmunológica."
        ),
        "user_insights_feedback": [
            {
                "code": "HRV_BALANCED_7",
                "meaning_es": "Línea base equilibrada durante los últimos 7 días. El sistema nervioso autónomo mantiene buena resiliencia y adaptabilidad al entrenamiento.",
            },
            {
                "code": "HRV_UNBALANCED_LOW",
                "meaning_es": "Valores por debajo de la franja normal personal. Indica sobrecarga simpática o recuperación insuficiente; se aconseja priorizar descanso.",
            },
        ],
    },
    {
        "metric_key": "sleep_score_and_architecture",
        "name_es": "Puntuación de Sueño y Arquitectura del Descanso",
        "name_en": "Sleep Score & Sleep Architecture",
        "category": "recuperacion_nocturna",
        "unit": "Puntuación (0-100) y segundos por fase",
        "description_es": (
            "El Sleep Score de Garmin combina la duración total del sueño con la calidad del descanso "
            "evaluada mediante la arquitectura de fases: Sueño Profundo (N3 / Slow-Wave Sleep, 15-25% ideal), "
            "Sueño REM (Movimientos Oculares Rápidos, 20-25% ideal), Sueño Ligero y periodos despierto. "
            "El algoritmo Firstbeat analiza la variabilidad cardíaca y la frecuencia respiratoria durante "
            "la noche para calcular el estrés promedio del sueño (avgSleepStress). Un estrés nocturno < 15 "
            "indica sueño profundamente reparador; estrés > 30 señala microdespertares y recuperación "
            "autonómica comprometida."
        ),
        "user_insights_feedback": [
            {
                "code": "POSITIVE_DEEP",
                "meaning_es": "Duración adecuada de sueño profundo, promoviendo la liberación de hormona del crecimiento y reparación tisular muscular.",
            },
            {
                "code": "STRESS_POS_EXCELLENT_OR_GOOD_SLEEP_VERY_RESTFUL_DAY",
                "meaning_es": "Día muy reparador con bajo estrés general, facilitando un sueño de alta calidad y recuperación psicofísica completa.",
            },
        ],
    },
    {
        "metric_key": "all_day_stress",
        "name_es": "Nivel de Estrés Diario",
        "name_en": "All-Day Stress Score",
        "category": "fisiologia_autonomica",
        "unit": "Escala 0-100",
        "description_es": (
            "Garmin estima el estrés corporal en tiempo real a partir del análisis de los intervalos R-R "
            "del sensor óptico PPG Elevate. La escala de 0 a 100 se segmenta en: 0-25 Reposo (predominio "
            "parasimpático/recuperación), 26-50 Estrés bajo (actividad mental o física ligera), 51-75 Estrés medio "
            "(esfuerzo cognitivo intenso o fatiga) y 76-100 Estrés alto (respuesta simpática de lucha o huida, "
            "agotamiento o enfermedad). Los periodos en reposo recargan el Body Battery."
        ),
        "user_insights_feedback": [
            {
                "code": "STRESS_REST",
                "meaning_es": "Nivel de estrés inferior a 25. El organismo se encuentra en homeostasis y reparación biológica.",
            },
            {
                "code": "STRESS_HIGH_SUSTAINED",
                "meaning_es": "Estrés sostenido superior a 50 durante el día, acelerando el drenaje de reservas biológicas.",
            },
        ],
    },
    {
        "metric_key": "body_battery",
        "name_es": "Body Battery (Batería Corporal)",
        "name_en": "Garmin Body Battery",
        "category": "reserva_energetica",
        "unit": "Puntos (1-100)",
        "description_es": (
            "Body Battery es una métrica de gestión energética propietaria de Firstbeat Analytics que cuantifica "
            "las reservas energéticas del cuerpo en una escala de 1 a 100. Analiza de manera continua la "
            "variabilidad cardíaca, el nivel de estrés, la calidad del sueño y la actividad física registrada. "
            "El descanso reparador nocturno y las pausas diurnas de bajo estrés recargan la batería (+), "
            "mientras que el ejercicio, la digestión pesada, el estrés crónico y el consumo de estimulantes "
            "la descargan (-)."
        ),
        "user_insights_feedback": [
            {
                "code": "BATTERY_CHARGED_HIGH",
                "meaning_es": "Body Battery por encima de 80 al despertar, indicando óptima preparación para exigencia física o cognitiva.",
            },
            {
                "code": "BATTERY_DRAINED_LOW",
                "meaning_es": "Body Battery por debajo de 25, sugiriendo moderar la intensidad de entrenamientos para evitar fatiga acumulada.",
            },
        ],
    },
    {
        "metric_key": "vo2_max",
        "name_es": "VO2 Máximo y Condición Cardiorrespiratoria",
        "name_en": "VO2 Max & Cardiorespiratory Fitness",
        "category": "rendimiento_cardiovascular",
        "unit": "mL/kg/min",
        "description_es": (
            "El VO2 Máximo define el volumen máximo de oxígeno que el cuerpo puede transportar y utilizar "
            "durante un esfuerzo de intensidad máxima. Garmin y Firstbeat calculan automáticamente el VO2 Max "
            "mediante un algoritmo de aprendizaje automático embebido que correlaciona la velocidad o potencia de carrera "
            "con la respuesta de la frecuencia cardíaca submáxima. Filtra automáticamente tramos no estacionarios, "
            "desniveles, paradas y deriva cardíaca térmica para proporcionar estimaciones con un margen de error menor "
            "al 5% respecto a pruebas de laboratorio con análisis de gases respiratorios."
        ),
        "user_insights_feedback": [
            {
                "code": "VO2MAX_EXCELLENT",
                "meaning_es": "Capacidad cardiorrespiratoria en el percentil superior para tu grupo de edad y sexo.",
            },
        ],
    },
    {
        "metric_key": "training_effect_and_epoc",
        "name_es": "Efecto de Entrenamiento (Aeróbico y Anaeróbico) y EPOC",
        "name_en": "Training Effect (Aerobic & Anaerobic) & EPOC",
        "category": "carga_entrenamiento",
        "unit": "Escala 0.0 - 5.0 (y ml/kg EPOC)",
        "description_es": (
            "Training Effect mide el impacto de una actividad física sobre la aptitud aeróbica y anaeróbica. "
            "El Efecto Aeróbico (0.0 a 5.0) cuantifica la acumulación de EPOC (Exceso de Consumo de Oxígeno "
            "Post-Ejercicio), midiendo la perturbación de la homeostasis y el estímulo para mejorar la capacidad "
            "mitocondrial y cardiovascular. El Efecto Anaeróbico (0.0 a 5.0) evalúa la capacidad de generar energía "
            "sin oxígeno mediante intervalos de alta intensidad (HIIT), esprints o series de fuerza, midiendo "
            "la rapidez del incremento de la frecuencia cardíaca y la potencia."
        ),
        "user_insights_feedback": [
            {
                "code": "TE_MAINTAINING",
                "meaning_es": "Puntuación de 2.0 a 2.9: Mantiene la condición cardiovascular actual sin provocar sobrecarga.",
            },
            {
                "code": "TE_IMPROVING",
                "meaning_es": "Puntuación de 3.0 a 3.9: Provoca adaptaciones fisiológicas progresivas que elevan la aptitud aeróbica.",
            },
            {
                "code": "TE_HIGHLY_IMPROVING",
                "meaning_es": "Puntuación de 4.0 a 4.9: Estímulo de sobrecarga de alta magnitud; requiere adecuada recuperación posterior.",
            },
        ],
    },
    {
        "metric_key": "resting_heart_rate",
        "name_es": "Frecuencia Cardíaca en Reposo (RHR)",
        "name_en": "Resting Heart Rate (RHR)",
        "category": "salud_cardiovascular",
        "unit": "ppm (latidos por minuto)",
        "description_es": (
            "La frecuencia cardíaca en reposo es el número de contracciones cardíacas por minuto registradas "
            "durante los periodos de inactividad más profunda del día o justo antes de despertar. Una elevación "
            "sostenida de más de 5-7 ppm sobre el promedio semanal habitual es un marcador temprano de fatiga "
            "cardiovascular, deshidratación, desequilibrio en la recuperación o activación del sistema inmune."
        ),
        "user_insights_feedback": [
            {
                "code": "RHR_OPTIMAL",
                "meaning_es": "Frecuencia en reposo estable o en tendencia descendente, indicando adaptaciones positivas de hipertrofia excéntrica ventricular.",
            },
        ],
    },
    {
        "metric_key": "pulse_ox_spo2",
        "name_es": "Pulsioximetría y Saturación de Oxígeno (SpO2)",
        "name_en": "Pulse Oximetry & SpO2",
        "category": "respiratorio_hematologico",
        "unit": "Porcentaje (%)",
        "description_es": (
            "El sensor óptico de Garmin emite haces combinados de luz roja (660 nm) e infrarroja (940 nm) "
            "a través de la piel para medir la proporción de hemoglobina oxigenada versus desoxigenada en los "
            "capilares. Los valores normales en personas sanas oscilan entre 95% y 100%. Durante el sueño, lecturas "
            "continuas permiten monitorear la hipoxemia nocturna, la aclimatación a la altitud y posibles "
            "eventos de apnea del sueño."
        ),
        "user_insights_feedback": [
            {
                "code": "SPO2_NORMAL",
                "meaning_es": "Saturación arterial estimada estable > 95% a nivel de superficie.",
            },
        ],
    },
    {
        "metric_key": "respiration_rate",
        "name_es": "Frecuencia Respiratoria",
        "name_en": "Respiration Rate",
        "category": "respiratorio_pulmonar",
        "unit": "brpm (respiraciones por minuto)",
        "description_es": (
            "Garmin estima la tasa respiratoria mediante la técnica de arritmia sinusal respiratoria (RSA): "
            "al inhalar, el tono vagal se inhibe transitoriamente y la frecuencia cardíaca se acelera ligeramente; "
            "al exhalar, el tono vagal aumenta y la frecuencia se ralentiza. El algoritmo analiza estas fluctuaciones "
            "micrométricas del intervalo R-R para derivar las respiraciones por minuto durante el día y el sueño. "
            "El rango típico de reposo en adultos sanos es de 12 a 20 brpm."
        ),
        "user_insights_feedback": [
            {
                "code": "RESPIRATION_RESTING_NORMAL",
                "meaning_es": "Frecuencia respiratoria nocturna basal de 12-16 brpm, reflejando adecuada ventilación alveolar.",
            },
        ],
    },
    {
        "metric_key": "calorie_expenditure_split",
        "name_es": "Gasto Calórico: Reposo (BMR) y Calorías Activas",
        "name_en": "Calorie Expenditure: Resting (BMR) & Active",
        "category": "metabolismo_energetico",
        "unit": "kcal (kilocalorías)",
        "description_es": (
            "Garmin desglosa el gasto calórico total en dos componentes termodinámicos esenciales: "
            "1. Calorías en Reposo: Tasa Metabólica Basal (BMR) proyectada matemáticamente mediante la "
            "ecuación de Cunningham o Harris-Benedict ajustada con el perfil biométrico (edad, sexo, peso, "
            "altura y porcentaje graso). "
            "2. Calorías Activas: Energía consumida por encima del BMR derivada del número de pasos, la "
            "intensidad del movimiento medida por acelerómetro y el consumo de oxígeno (VO2) estimado "
            "a partir de la frecuencia cardíaca continua."
        ),
        "user_insights_feedback": [
            {
                "code": "CALORIES_BALANCED",
                "meaning_es": "Equilibrio entre gasto activo y reposo acorde a las metas de entrenamiento y mantenimiento del balance de nitrógeno.",
            },
        ],
    },
]


def generate_markdown_glossary(metrics: list[dict[str, Any]]) -> str:
    """Genera un documento Markdown exhaustivo con el glosario de métricas Garmin."""
    lines = [
        "# Glosario Oficial y Explicaciones de Métricas Garmin Connect",
        "",
        "Documento de referencia técnica que compila las definiciones formales, fundamentos biológicos, "
        "algoritmos embebidos (Firstbeat Analytics) e interpretación de códigos de feedback reportados "
        "en la telemetría del usuario de Garmin Connect.",
        "",
        f"**Fecha de compilación**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Métricas documentadas**: {len(metrics)}",
        "",
        "---",
        "",
    ]

    for m in metrics:
        lines.append(f"## {m['name_es']} ({m['name_en']})")
        lines.append("")
        lines.append(f"- **Clave interna**: `{m['metric_key']}`")
        lines.append(f"- **Categoría fisiológica**: `{m['category']}`")
        lines.append(f"- **Unidad de medida**: {m['unit']}")
        lines.append("")
        lines.append("### Descripción Técnica y Fundamento Fisiológico")
        lines.append(m["description_es"])
        lines.append("")
        if m.get("user_insights_feedback"):
            lines.append("### Interpretación de Insights y Feedback de Garmin Connect")
            for fb in m["user_insights_feedback"]:
                lines.append(f"- **`{fb['code']}`**: {fb['meaning_es']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def sync_garmin_metric_descriptions(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    r2_sync: bool = False,
) -> dict[str, Any]:
    """Compila y almacena las descripciones de métricas de Garmin en local y R2."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "garmin_connect_metric_glossary.json"
    md_path = output_dir / "garmin_connect_metric_glossary.md"

    logger.info(
        f"Iniciando Proceso 3: Compilación de Descripciones de Métricas Garmin ({len(GARMIN_METRIC_DEFINITIONS)} métricas)"
    )

    payload = {
        "title": "Glosario de Métricas y Explicaciones Garmin Connect",
        "compiled_at": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "total_metrics": len(GARMIN_METRIC_DEFINITIONS),
        "metrics": GARMIN_METRIC_DEFINITIONS,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(
        f"  ✓ Archivo JSON guardado: {json_path} ({json_path.stat().st_size / 1024:.1f} KB)"
    )

    md_content = generate_markdown_glossary(GARMIN_METRIC_DEFINITIONS)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(
        f"  ✓ Archivo Markdown guardado: {md_path} ({md_path.stat().st_size / 1024:.1f} KB)"
    )

    results: dict[str, Any] = {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "total_metrics": len(GARMIN_METRIC_DEFINITIONS),
        "r2_uploaded": False,
    }

    if r2_sync:
        from src.common.r2_storage import R2StorageClient

        r2_client = R2StorageClient()
        if r2_client.is_configured():
            logger.info("Sincronizando glosario de métricas Garmin con Cloudflare R2...")
            remote_prefix = "knowledge_base/descripciones_metricas_garmin"
            r2_client.upload_file(
                json_path,
                f"{remote_prefix}/garmin_connect_metric_glossary.json",
            )
            r2_client.upload_file(
                md_path,
                f"{remote_prefix}/garmin_connect_metric_glossary.md",
            )
            results["r2_uploaded"] = True
            logger.info(
                f"  ✓ Glosario subido exitosamente a r2://{r2_client.bucket_name}/{remote_prefix}/"
            )
        else:
            logger.warning(
                "Credenciales de Cloudflare R2 no configuradas. Omitiendo subida remota."
            )

    return results


def main() -> None:
    """Punto de entrada CLI para sincronización del glosario de métricas Garmin."""
    parser = argparse.ArgumentParser(
        description="Proceso 3: Compilación y Sincronización del Glosario de Métricas y Explicaciones Garmin."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de destino local (por defecto: data/knowledge_base/descripciones_metricas_garmin)",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="Sincronizar glosario con Cloudflare R2 (garmin-personal-data)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📊 FUENTE 3: Glosario y Explicaciones de Métricas Garmin Connect")
    print(f"   • Directorio local:  {args.output_dir.resolve()}")
    print(
        "   • Destino remoto R2: garmin-personal-data/knowledge_base/descripciones_metricas_garmin/"
    )
    print(f"   • Métricas:          {len(GARMIN_METRIC_DEFINITIONS)} variables documentadas")
    print(f"   • Sincronización R2: {'SÍ' if args.r2_sync else 'NO'}")
    print("=" * 70 + "\n")

    res = sync_garmin_metric_descriptions(
        output_dir=args.output_dir,
        r2_sync=args.r2_sync,
    )

    print("\n" + "-" * 70)
    print("✅ Compilación finalizada exitosamente:")
    print(f"   • JSON: {res['json_path']}")
    print(f"   • MD:   {res['md_path']}")
    print(f"   • R2:   {'Subido' if res['r2_uploaded'] else 'Local únicamente'}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
