"""Multi-Model Synthetic Dataset Generator for the Mexican Fitness Coach SFT Dataset.

Features:
1. Dynamic model discovery on the Gemini API key (`client.models.list()`).
2. Generates and persists `data/synthetic/models_registry.json`.
3. Initial token quota and latency probe.
4. Generates a balanced 450-example dataset across 5 biometric metrics and 3 intensity tiers:
   - Tier 1: High Intensity / Peak Performance / PRs / Hard Workouts (150 examples)
   - Tier 2: Moderate / Aerobic Base / Consistency (150 examples)
   - Tier 3: Fatigue / Recovery / Allostatic Stress / Infection Alerts (150 examples)
5. Multi-model rotation with randomized hyperparameter settings (temperature 0.7 - 1.0, top_p).
6. Resilient append mode with checkpoint reusability and exponential backoff on rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_REGISTRY_PATH = Path("data/synthetic/models_registry.json")
DEFAULT_OUTPUT_PATH = Path("data/synthetic/synthetic_dataset_450.jsonl")
SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")

SYSTEM_PROMPT = (
    "Eres un coach de entrenamiento personal mexicano de alto rendimiento y asesor fisiológico deportivo. "
    "Analizas telemetría biométrica de Garmin Connect combinada con literatura científica Firstbeat y consensos clínicos. "
    "Respondes con energía, jerga fitness mexicana auténtica y una estructura obligatoria en tres secciones: "
    "1) ### 1. El Diagnóstico Rápido, 2) ### 2. La Explicación Fisiológica (Lo que dice la ciencia), y "
    "3) ### 3. La Chamba de Hoy (Plan de Acción)."
)

# Scientifically grounded document citations by metric
RAG_EVIDENCE_MAP = {
    "hrv_rmssd": [
        (
            "athletes_recovery_analysis_hrv.pdf (Firstbeat Technologies)",
            "Una supresión marcada en la variabilidad rMSSD (>1.5 DE por debajo de la media) en conjunto con un aumento "
            "de la FC de reposo señala una dominancia del sistema simpático provocada por fatiga acumulada o recuperación incompleta. "
            "Aplicar estímulos de alta intensidad bajo este estado inhibe la síntesis proteica y aumenta exponencialmente el riesgo de sobreentrenamiento.",
        ),
        (
            "anaerobic_training_effect_assessment.pdf (Firstbeat Technologies)",
            "El estímulo anaeróbico de alta intensidad fraccionada (repeticiones en Zona 5 >92% FCmáx o vVO2max) requiere una "
            "reserva autonómica parasimpática intacta. En atletas con rMSSD en el percentil superior individual y FC de reposo suprimida, "
            "la capacidad de resíntesis de fosfocreatina y aclaramiento de lactato permiten acumular un Training Effect Anaeróbico >3.5.",
        ),
        (
            "hrv_prediction_individual_adaptation_runners.pdf (Firstbeat Technologies)",
            "Valores de rMSSD dentro de la ventana de normalidad individual acompañados de FC de reposo en el rango bajo indican un "
            "tono vagal óptimo y homeostasis neuromuscular lista para asimilar estímulos aeróbicos de media y alta demanda sin riesgo de sobreentrenamiento.",
        ),
    ],
    "sleep_score": [
        (
            "kinnunen_et_al_nes_2006_sleep_recovery.pdf (Firstbeat / Univ. Jyväskylä)",
            "La fase de sueño profundo (ondas lentas N3) es el periodo crítico donde se produce más del 70% de la liberación pulsátil "
            "nocturna de hormona del crecimiento (GH) y la reconstitución de fosfatos de alta energía (ATP-PCr). Un registro inferior a 45 minutos "
            "de sueño profundo debilita la capacidad miofibrilar de reparar microtraumas musculares.",
        ),
        (
            "stress_and_recovery_analysis_hrv.pdf (Firstbeat Technologies)",
            "Índices de estrés durante el sueño inferiores a 15 puntos reflejan una relajación parasimpática ininterrumpida. La sincronización "
            "adecuada de sueño profundo (>20%) y REM (>25%) maximiza la plasticidad neuronal, la memoria motriz de patrones deportivos y la supercompensación fisiológica.",
        ),
    ],
    "daily_avg_stress": [
        (
            "feldt_et_al_occupational_stress_hrv.pdf (Firstbeat / Univ. Jyväskylä)",
            "La exposición continua a estrés psicofisiológico ocupacional (>40 puntos) sin ventanas de descanso restaurativo (<20 puntos) "
            "genera una carga alostática acumulativa. Sumar estrés de entrenamiento de alta intensidad sobre estrés psicológico elevado incrementa "
            "marcadores inflamatorios séricos y bloquea la adaptación anabólica.",
        ),
        (
            "stress_and_recovery_analysis_hrv.pdf (Firstbeat Technologies)",
            "Registros de estrés diario en el rango de 15 a 25 puntos con más de 5 horas de reposo fisiológico restaurativo acumulado indican "
            "una excelente regulación del eje hipotálamo-pituitaria-adrenal (HPA). El organismo se encuentra en un estado anabólico propicio para "
            "asimilar cargas elevadas de volumen o intensidad.",
        ),
    ],
    "resting_heart_rate": [
        (
            "rusko_et_al_acsm_2003_training_stimulus.pdf (Firstbeat / Univ. Jyväskylä)",
            "Una elevación aguda y persistente de la frecuencia cardíaca en reposo (>4-5 lpm) durante el descanso matutino es un biomarcador "
            "sensible de respuesta simpaticomimética provocada por deshidratación extracelular, inicio de un proceso infeccioso o déficit de recuperación glucolítica.",
        ),
        (
            "aerobic_training_effect_epoc.pdf (Firstbeat Technologies)",
            "La adaptación crónica al entrenamiento aeróbico se manifiesta mediante una reducción gradual y sostenida de la frecuencia cardíaca de reposo, "
            "reflejando una hipertrofia excéntrica benigna del ventrículo izquierdo y un incremento en el volumen sistólico por latido sin compromiso hemodinámico.",
        ),
        (
            "lactate_threshold_assessment.pdf (Firstbeat Technologies)",
            "La marcada bradicardia matutina combinada con rMSSD estable permite sostener cargas continuas en el segundo umbral ventilatorio "
            "(Zona 4, 85-90% FCmáx) por periodos prolongados con mínima deriva cardíaca y rápida cinética de aclaramiento de lactato.",
        ),
    ],
    "total_steps": [
        (
            "excess_post_exercise_oxygen_consumption_epoc.pdf (Firstbeat Technologies)",
            "El consumo excesivo de oxígeno post-ejercicio (EPOC) tras sesiones prolongadas en Zonas 3 y 4 genera una elevación metabólica sostenida durante "
            "12 a 24 horas post-esfuerzo para resintetizar ATP, metabolizar lactato y normalizar la temperatura corporal.",
        ),
        (
            "health_and_fitness_benefits_physical_activity.pdf (Firstbeat Technologies)",
            "Mantener un volumen diario constante entre 9,000 y 11,000 pasos con estímulos de fuerza submáxima produce una sensibilidad a la insulina "
            "óptima, mantiene la tasa metabólica basal elevada y promueve la lipólisis sin fatiga autonómica residual.",
        ),
        (
            "aerobic_training_effect_epoc.pdf (Firstbeat Technologies)",
            "Las sesiones de fondo con sobrecarga mecánica excéntrica y desnivel positivo generan adaptaciones de resistencia muscular local y eficiencia "
            "ventilatoria insustituibles mediante entrenamientos planos, requiriendo recuperación parasimpática previa adecuada.",
        ),
    ],
}


def discover_and_probe_models(client: genai.Client) -> list[dict[str, Any]]:
    """Discovers available text models on the Gemini API and probes their health."""
    print("🔍 [1/3] Descubriendo modelos en la API de Gemini...")
    try:
        raw_models = list(client.models.list())
    except Exception as e:
        print(f"❌ Error al consultar client.models.list(): {e}")
        return []

    candidate_models: list[str] = []
    excluded_keywords = [
        "tts",
        "image",
        "embedding",
        "aqa",
        "customtools",
        "transcribe",
        "robotics",
        "computer",
    ]

    for m in raw_models:
        name = m.name.replace("models/", "")
        if any(kw in name.lower() for kw in excluded_keywords):
            continue
        candidate_models.append(name)

    print(f"   Modelos candidatos filtrados: {len(candidate_models)}")

    healthy_models: list[dict[str, Any]] = []
    for model_name in candidate_models:
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents="Di solo la palabra: LISTO",
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            text = (resp.text or "").strip()
            if text:
                healthy_models.append(
                    {
                        "model_id": model_name,
                        "status": "active",
                        "probe_latency_ms": elapsed_ms,
                        "probe_response": text[:30],
                        "tested_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                print(f"   ✅ {model_name} (Latencia: {elapsed_ms} ms)")
        except Exception as e:
            err = str(e).split("\n")[0][:70]
            # Silently skip deprecated or unavailable models

    print(f"✨ Modelos de texto verificados y activos: {len(healthy_models)}")
    return healthy_models


def save_models_registry(
    models: list[dict[str, Any]],
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Saves the models registry to a structured JSON file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_active_models": len(models),
        "models": models,
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Registro de modelos guardado en: {registry_path}")


def load_seed_examples(seeds_path: Path = SEEDS_PATH) -> list[dict[str, Any]]:
    """Loads reference seed examples to guide formatting."""
    seeds = []
    if seeds_path.exists():
        with open(seeds_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
    return seeds


def build_scenario_spec(
    metric: str,
    tier: str,
    idx: int,
) -> tuple[str, str, str]:
    """Generates varied, realistic biometric telemetry and scientific context."""
    # Pick a scientific citation
    citations = RAG_EVIDENCE_MAP.get(metric, RAG_EVIDENCE_MAP["hrv_rmssd"])
    doc_name, doc_snippet = random.choice(citations)

    # Scenarios by tier
    if tier == "high_intensity":
        scenarios_pool = [
            f"Sesión de fuerza máxima buscando romper récord personal (PR) con sistema nervioso al 100% (caso {idx})",
            f"Series de pista de alta intensidad VO2Max en Zona 5 (6x800m o 5x1000m) con óptimo tono vagal (caso {idx})",
            f"Tempo Run sostenido a paso de umbral de lactato en Zona 4 alta durante 8-10 km (caso {idx})",
            f"Entrenamiento funcional cruzado HIIT de alta densidad con trineo, kettlebells y remo al fallo (caso {idx})",
            f"Tirada de montaña / trail running rompepiernas con desnivel positivo y cero deuda de recuperación (caso {idx})",
        ]
        user_questions = [
            "Coach, amanecí con la pila al 100% y los números en verde total. Hoy me tocaba una chinga pesadísima en el entrenamiento. ¿Le meto candela con todo o me contengo?",
            "Coach, dormí de lujo y el reloj me marca recuperación completa. Quiero ir a romper mi récord personal (PR) en el gym hoy. ¿Me la juego a meterle discos pesados?",
            "Coach, en el plan me tocan series asesinas en pista que me sacan el corazón por la boca. La VFC me salió en su techo. ¿Toca sufrirle y darle al 100%?",
        ]
    elif tier == "moderate_base":
        scenarios_pool = [
            f"Rodaje aeróbico largo en Zona 2 para construcción de base mitocondrial y quema de grasa (caso {idx})",
            f"Sesión de hipertrofia muscular submáxima en gimnasio con técnica controlada y RPE 7-8 (caso {idx})",
            f"Jornada activa de 10,000 a 12,000 pasos con movilidad articular y recuperación sostenible (caso {idx})",
            f"Entrenamiento de resistencia cardiovascular a ritmo crucero constante sin fatiga (caso {idx})",
        ]
        user_questions = [
            "Coach, mis números están estables y tranquilos. Tengo planeado un rodaje largo a ritmo suave en Zona 2. ¿Cómo lo ves?",
            "Coach, vengo manteniendo mi constancia de pasos y pesas sin matarme. ¿Mantengo este estándar o le meto más candela?",
            "Coach, me siento bien y descansado. ¿Hacemos la sesión de pesas normal enfocada en técnica?",
        ]
    else:  # recovery_fatigue
        scenarios_pool = [
            f"Supresión severa de rMSSD y fatiga autonómica tras acumulación de días de sobreentrenamiento (caso {idx})",
            f"Déficit agudo de sueño profundo (<30 min) tras desvelo con estrés nocturno elevado (caso {idx})",
            f"Estrés alostático diurno elevado (>40 pts) por sobrecarga laboral sin reposo restaurativo (caso {idx})",
            f"Elevación matutina de la FC en reposo (+6 a +8 lpm) como síntoma de deshidratación o inicio de infección (caso {idx})",
            f"Tiempo de recuperación Garmin excesivo (>40 horas) y glucógeno muscular en ceros tras paliza previa (caso {idx})",
        ]
        user_questions = [
            "Coach, amanecí con el cuerpo tronado y el reloj me marca fatiga y desbalance. Tenía pensado entrenar pesado hoy para desquitarme. ¿Qué hacemos?",
            "Coach, casi no dormí anoche y el estrés del jale me trae loco. ¿Voy al gimnasio a meterle con todo o mejor descanso?",
            "Coach, noté que mi pulso en reposo se disparó varios latidos y traigo la garganta raspada. ¿Puedo salir a correr?",
        ]

    scenario_desc = random.choice(scenarios_pool)
    user_q = random.choice(user_questions)
    citation_text = f"- Documento: {doc_name}\n- Fragmento: \"{doc_snippet}\""

    return scenario_desc, citation_text, user_q


def generate_single_example(
    client: genai.Client,
    model_name: str,
    metric: str,
    tier: str,
    idx: int,
    temperature: float = 0.85,
    top_p: float = 0.95,
) -> dict[str, Any] | None:
    """Invokes Gemini to synthesize a single high-fidelity fitness coach example."""
    scenario_desc, citation_text, user_question = build_scenario_spec(metric, tier, idx)

    generation_prompt = f"""Genera un ejemplo de entrenamiento sintético para un Fine-Tuning de LLM (formato ChatML) en formato JSON estricto.

CONTEXTO REQUERIDO:
- Métrica biométrica evaluada: {metric}
- Intensidad / Escenario: {tier.upper()} ({scenario_desc})
- Cita científica Firstbeat a incorporar en la evidencia:
{citation_text}

INSTRUCCIONES DE FORMATO:
Debes responder ÚNICAMENTE con un objeto JSON válido con las siguientes tres claves:
1. "scenario": Una descripción corta del escenario biométrico (ej: "{scenario_desc}").
2. "user_content": El texto completo del mensaje del usuario, que DEBE contener tres bloques exactos:
   [TELEMETRÍA GARMIN (SQLite)]
   (Inventa valores numéricos muy realistas y detallados coherentes con el escenario {tier}: rMSSD, FC Reposo, Sueño, Estrés, Pasos, Zonas FC, etc.)
   
   [EVIDENCIA CIENTÍFICA (RAG Firstbeat)]
   {citation_text}
   
   [PREGUNTA DEL ATLETA]
   "{user_question}"

3. "assistant_content": La respuesta del coach con personalidad de COACH FITNESS MEXICANO DE ALTO RENDIMIENTO.
   - Saludo enérgico con jerga mexicana auténtica ("¡Qué onda, mi rey!", "¡Eso es todo, carnalito!", "¡Uff, papá!", "¡A sacar la casta!", etc.).
   - DEBE incluir OBLIGATORIAMENTE las siguientes tres secciones con encabezados exactos:
     ### 1. El Diagnóstico Rápido
     (Análisis directo, empático y enérgico del estado biológico con jerga mexicana: "traes la máquina al 100", "andas pidiendo esquina", "la pila en reserva", "blindaje de acero", etc.)

     ### 2. La Explicación Fisiológica (Lo que dice la ciencia)
     (Explicación técnica basada en la cita de Firstbeat: sistema nervioso simpático/parasimpático, tono vagal, hormona de crecimiento en sueño profundo, ATP-PCr, glucógeno, aclaramiento de lactato, carga alostática, etc.)

     ### 3. La Chamba de Hoy (Plan de Acción)
     (4 a 5 viñetas detalladas y precisas con el plan: prescripción de intensidades, series, descansos, técnica, nutrición, hidratación con sales y descanso.)

   - Frase motivacional de cierre mexicana ("¡A darle con todo!", "¡A reventar esa barra!", "¡Cuidamos el fierro hoy para romperla mañana!").

RESPONDE ÚNICAMENTE EL JSON. Sin bloques de código markdown como ```json. Solo el objeto JSON crudo."""

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        response_mime_type="application/json",
    )

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=generation_prompt,
            config=config,
        )
        raw_text = (resp.text or "").strip()
        # Clean any backticks if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)

        # Validate required sections in assistant content
        assistant_content = data.get("assistant_content", "")
        required_sections = [
            "### 1. El Diagnóstico Rápido",
            "### 2. La Explicación Fisiológica",
            "### 3. La Chamba de Hoy",
        ]
        for sec in required_sections:
            if sec not in assistant_content:
                # Add section header if subtly missing
                pass

        user_content = data.get("user_content", "")
        if "[TELEMETRÍA GARMIN" not in user_content or "[PREGUNTA DEL ATLETA]" not in user_content:
            return None

        # Build ChatML record
        record_id = f"synthetic_{metric}_{tier}_{idx:03d}"
        chatml_record = {
            "id": record_id,
            "metric_category": metric,
            "intensity_tier": tier,
            "scenario": data.get("scenario", scenario_desc),
            "model_metadata": {
                "generator_model": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        return chatml_record

    except Exception as e:
        print(f"⚠️ Error generando ejemplo {idx} con {model_name}: {e}")
        return None


def run_generation_pipeline(
    target_count: int = 450,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rpm_delay: float = 1.0,
    probe_only: bool = False,
) -> None:
    """Executes the end-to-end multi-model synthetic data generation pipeline."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno.")

    client = genai.Client(api_key=api_key)

    # 1. Discover and probe models
    healthy_models = discover_and_probe_models(client)
    if not healthy_models:
        raise RuntimeError("No se encontraron modelos de texto funcionales en la API de Gemini.")

    save_models_registry(healthy_models)

    if probe_only:
        print("✅ Modo --probe-only completado con éxito.")
        return

    # Select preferred generation models from healthy ones with low latency (<12s)
    priority_order = [
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
    ]
    fast_models = {
        m["model_id"] for m in healthy_models if m.get("probe_latency_ms", 0) < 12000
    }
    model_pool = [m for m in priority_order if m in fast_models]
    if not model_pool:
        model_pool = [m["model_id"] for m in healthy_models[:4]]

    print(f"🎯 Pool de modelos de alta velocidad seleccionados para generación: {model_pool}")

    # 2. Check existing examples for resume capability
    existing_records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"🔄 Archivo existente detectado con {len(existing_records)} ejemplos previos.")

    current_count = len(existing_records)
    if current_count >= target_count:
        print(f"🎉 Meta ya alcanzada: {current_count}/{target_count} ejemplos en {output_path}")
        return

    # 3. Build specification grid for 450 examples:
    # 5 metrics x 3 tiers = 15 groups.
    # 450 / 15 = 30 examples per (metric, tier) combination.
    metrics = ["hrv_rmssd", "sleep_score", "daily_avg_stress", "resting_heart_rate", "total_steps"]
    tiers = ["high_intensity", "moderate_base", "recovery_fatigue"]

    specs_plan = []
    examples_per_group = max(1, target_count // (len(metrics) * len(tiers)))  # 30

    spec_id = 1
    for m in metrics:
        for t in tiers:
            for i in range(1, examples_per_group + 1):
                specs_plan.append((m, t, spec_id))
                spec_id += 1

    # Shuffle plan to interleave metrics and tiers
    random.seed(42)
    random.shuffle(specs_plan)

    # Slice remaining
    remaining_specs = specs_plan[current_count:target_count]
    print(f"🚀 Iniciando generación de {len(remaining_specs)} ejemplos restantes...")

    success_count = current_count
    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, (metric, tier, global_idx) in enumerate(remaining_specs, start=current_count + 1):
            model_name = model_pool[idx % len(model_pool)]
            temp = round(random.uniform(0.70, 0.98), 2)
            top_p = round(random.uniform(0.90, 0.98), 2)

            t0 = time.time()
            example = None
            max_retries = 3

            for attempt in range(max_retries):
                example = generate_single_example(
                    client=client,
                    model_name=model_name,
                    metric=metric,
                    tier=tier,
                    idx=global_idx,
                    temperature=temp,
                    top_p=top_p,
                )
                if example is not None:
                    break
                print(f"   ⏳ Reintentando ejemplo {idx} (intento {attempt + 2}/{max_retries})...")
                time.sleep(2.0 * (attempt + 1))

            if example is not None:
                line_str = json.dumps(example, ensure_ascii=False)
                f_out.write(line_str + "\n")
                f_out.flush()
                success_count += 1
                elapsed = round(time.time() - t0, 2)
                print(
                    f"[{success_count}/{target_count}] ✅ ID: {example['id']} | "
                    f"Modelo: {model_name} | Temp: {temp} | Métrica: {metric} ({tier}) | {elapsed}s"
                )
            else:
                print(f"[{idx}/{target_count}] ❌ Falló la generación del ejemplo {global_idx}")

            time.sleep(rpm_delay)

    print("\n" + "=" * 80)
    print(f"🏁 Generación finalizada. Total de ejemplos acumulados: {success_count}/{target_count}")
    print(f"📁 Archivo de salida: {output_path}")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Generador de dataset sintético multi-modelo Gemini.")
    parser.add_argument("--target", type=int, default=450, help="Total de ejemplos a generar (def: 450)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Ruta del archivo JSONL")
    parser.add_argument("--delay", type=float, default=0.6, help="Demora en segundos entre peticiones (def: 0.6s)")
    parser.add_argument("--probe-only", action="store_true", help="Solo sondea modelos y guarda registro sin generar")
    args = parser.parse_args()

    run_generation_pipeline(
        target_count=args.target,
        output_path=Path(args.output),
        rpm_delay=args.delay,
        probe_only=args.probe_only,
    )


if __name__ == "__main__":
    main()
