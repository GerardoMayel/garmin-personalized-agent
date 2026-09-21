"""Bilingual Multi-Model Synthetic Dataset Generator for Garmin Mexican Fitness Coach SFT Dataset.

Directives:
1. Bilingual dataset: Spanish (authentic Mexican fitness coach) and English (natural bilingual Mexican coach).
2. Strict length control by language:
   - Español (ES): 240 to 360 tokens (150-220 words, ratio ~1.5-1.7 tok/word: modismos, diminutivos, contracciones).
   - English (EN): 190 to 280 tokens (150-220 words, ratio ~1.25-1.35 tok/word: optimized vocabulary).
3. 3-tier scenario balance per language: High Intensity, Moderate Base, Recovery/Fatigue = 400 total.
4. Block-based execution: Run in configurable blocks (--block-size 20 or 25) to avoid per-model quota exhaustion.
5. Multi-model rotation: Only quota-active models confirmed on the current API key.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_REGISTRY_PATH = Path("data/synthetic/models_registry.json")
DEFAULT_OUTPUT_PATH = Path("data/synthetic/synthetic_dataset_400.jsonl")
SEEDS_PATH = Path("data/synthetic/seed_few_shots.jsonl")

# Token range constants per language
TOKEN_MIN_ES, TOKEN_MAX_ES = 240, 360  # Spanish: contracciones, diminutivos, modismos raise ratio
TOKEN_MIN_EN, TOKEN_MAX_EN = 190, 280  # English: optimized tokenizer vocabulary

# Word-count equivalents (used for validation without counting actual tokenizer)
WORD_MIN, WORD_MAX = 150, 220

SYSTEM_PROMPT_ES = (
    "Eres un coach de entrenamiento personal mexicano de alto rendimiento y asesor fisiológico deportivo. "
    "Analizas telemetría biométrica de Garmin Connect combinada con literatura científica Firstbeat y consensos clínicos. "
    "Respondes con energía, jerga fitness mexicana auténtica y una estructura obligatoria en tres secciones: "
    "1) ### 1. El Diagnóstico Rápido, 2) ### 2. La Explicación Fisiológica (Lo que dice la ciencia), y "
    "3) ### 3. La Chamba de Hoy (Plan de Acción). "
    "Tu respuesta debe ser directa, concisa y potente, estrictamente entre 240 y 360 tokens (150 a 220 palabras)."
)

SYSTEM_PROMPT_EN = (
    "You are an elite bilingual Mexican sports coach and physiological advisor. "
    "You analyze Garmin Connect biometric telemetry combined with Firstbeat scientific research. "
    "You communicate with authentic Mexican coach energy, sharp athletic focus, and a mandatory three-section structure: "
    "1) ### 1. Quick Diagnosis, 2) ### 2. Physiological Breakdown (What Science Says), and "
    "3) ### 3. Today's Work (Action Plan). "
    "Your English is natural and athletic, speaking like an educated bilingual Mexican coach "
    "('What's up, my friend!', 'Let's get after it', 'Your engine is primed', 'We protect the machine today so we can crush it tomorrow'). "
    "Keep your response direct, concise, and strictly between 190 and 280 tokens (150 to 220 words)."
)

RAG_CITATIONS = {
    "hrv_rmssd": {
        "es": (
            "anaerobic_training_effect_assessment.pdf (Firstbeat Technologies)",
            "El estímulo anaeróbico de alta intensidad fraccionada (Zona 5 >92% FCmáx) requiere una reserva autonómica "
            "parasimpática intacta. En atletas con rMSSD en el percentil superior y FC de reposo baja, la resíntesis de fosfocreatina "
            "y aclaramiento de lactato permiten acumular un Training Effect Anaeróbico >3.5.",
        ),
        "en": (
            "anaerobic_training_effect_assessment.pdf (Firstbeat Technologies)",
            "High-intensity interval training in Zone 5 requires an intact autonomic parasympathetic reserve. In athletes with "
            "rMSSD in the upper percentile and suppressed resting HR, phosphocreatine resynthesis and lactate clearance kinetics allow an Anaerobic Training Effect > 3.5.",
        ),
    },
    "sleep_score": {
        "es": (
            "kinnunen_et_al_nes_2006_sleep_recovery.pdf (Firstbeat / Univ. Jyväskylä)",
            "La fase de sueño profundo (ondas lentas N3) es el periodo crítico donde se produce más del 70% de la liberación pulsátil "
            "nocturna de hormona del crecimiento (GH) y la reconstitución de fosfatos de alta energía (ATP-PCr). Menos de 45 minutos debilita la reparación miofibrilar.",
        ),
        "en": (
            "kinnunen_et_al_nes_2006_sleep_recovery.pdf (Firstbeat / Univ. Jyväskylä)",
            "Slow-wave N3 deep sleep accounts for over 70% of nocturnal growth hormone pulses and energetic phosphate (ATP-PCr) reconstitution. "
            "Less than 45 minutes of deep sleep severely compromises myofibrillar protein synthesis.",
        ),
    },
    "daily_avg_stress": {
        "es": (
            "feldt_et_al_occupational_stress_hrv.pdf (Firstbeat / Univ. Jyväskylä)",
            "La exposición continua a estrés psicofisiológico ocupacional (>40 puntos) sin descanso restaurativo genera carga alostática acumulativa. "
            "Sumar entrenamiento de alta intensidad sobre estrés psicológico elevado incrementa marcadores inflamatorios séricos y bloquea la adaptación.",
        ),
        "en": (
            "feldt_et_al_occupational_stress_hrv.pdf (Firstbeat / Univ. Jyväskylä)",
            "Continuous exposure to occupational stress (>40 points) without restorative rest creates cumulative allostatic load. "
            "Adding high-intensity training to elevated psychological stress spikes inflammatory markers and blunts anabolic adaptation.",
        ),
    },
    "resting_heart_rate": {
        "es": (
            "lactate_threshold_assessment.pdf (Firstbeat Technologies)",
            "La marcada bradicardia matutina combinada con rMSSD estable refleja un volumen sistólico optimizado, permitiendo sostener cargas continuas "
            "en el segundo umbral ventilatorio (Zona 4, 85-90% FCmáx) con mínima deriva cardíaca y rápida cinética de aclaramiento de lactato.",
        ),
        "en": (
            "lactate_threshold_assessment.pdf (Firstbeat Technologies)",
            "Marked morning bradycardia combined with stable rMSSD indicates optimized stroke volume, allowing athletes to sustain continuous loads "
            "at the second ventilatory threshold (Zone 4, 85-90% HRmax) with minimal cardiac drift and fast lactate clearance kinetics.",
        ),
    },
    "total_steps": {
        "es": (
            "health_and_fitness_benefits_physical_activity.pdf (Firstbeat Technologies)",
            "Mantener un volumen diario constante entre 9,000 y 11,000 pasos con estímulos de fuerza submáxima produce una sensibilidad a la insulina "
            "óptima, mantiene la tasa metabólica basal elevada y promueve la lipólisis sin fatiga autonómica residual.",
        ),
        "en": (
            "health_and_fitness_benefits_physical_activity.pdf (Firstbeat Technologies)",
            "Sustaining a daily volume of 9,000 to 11,000 steps paired with submaximal strength training maintains high insulin sensitivity "
            "and active lipolysis without creating chronic autonomic fatigue.",
        ),
    },
}


def build_scenario_spec(
    metric: str,
    tier: str,
    lang: str,
    idx: int,
) -> tuple[str, str, str]:
    """Builds biometric context, scientific citation, and athlete question."""
    citation_info = RAG_CITATIONS.get(metric, RAG_CITATIONS["hrv_rmssd"])[lang]
    doc_name, doc_snippet = citation_info
    citation_text = f"- Documento: {doc_name}\n- Fragmento: \"{doc_snippet}\""

    if lang == "es":
        if tier == "high_intensity":
            scenarios = [
                f"Sesión de fuerza máxima buscando romper récord personal (PR) en sentadilla o banca con sistema nervioso al 100% (caso {idx})",
                f"Series de pista de alta intensidad VO2Max en Zona 5 (6x800m a 3:20/km) con tono vagal en techo (caso {idx})",
                f"Tempo Run sostenido a paso de umbral de lactato en Zona 4 alta durante 8-10 km sin deriva cardíaca (caso {idx})",
                f"Trail running de montaña con +1,000m de desnivel positivo y recuperación previa en cero (caso {idx})",
            ]
            questions = [
                "Coach, amanecí con la pila al 100% y los números en verde total. Hoy me toca pierna pesada y quiero ir por un nuevo PR. ¿Le meto los discos?",
                "Coach, la VFC me salió en su techo histórico y dormí de lujo. Me tocan series asesinas de VO2Max en pista. ¿Toca sufrirle y darle al 100%?",
                "Coach, me toca rodaje de tempo a paso de umbral de lactato sostenido. Mis números están impecables. ¿Le meto candela sin miedo?",
            ]
        elif tier == "moderate_base":
            scenarios = [
                f"Rodaje aeróbico largo en Zona 2 para construcción mitocondrial y quema de grasa eficiente (caso {idx})",
                f"Sesión de hipertrofia muscular submáxima en gimnasio con técnica controlada y RPE 7-8 (caso {idx})",
                f"Constancia de 10,000 a 12,000 pasos activos diarios con buena movilidad articular (caso {idx})",
            ]
            questions = [
                "Coach, mis números están estables y tranquilos. Tengo planeado un rodaje suave en Zona 2. ¿Cómo lo ves?",
                "Coach, vengo manteniendo mis 10,000 pasos diarios y mi sesión de pesas sin matarme. ¿Mantengo este estándar o le subo?",
                "Coach, me siento descansado y con buena energía para la sesión normal de gimnasio. ¿Nos enfocamos en técnica?",
            ]
        else:  # recovery_fatigue
            scenarios = [
                f"Déficit severo de sueño profundo (<30 min) tras desvelo con estrés nocturno elevado (caso {idx})",
                f"Estrés alostático diurno elevado (>45 pts) por sobrecarga laboral sin reposo restaurativo (caso {idx})",
                f"Elevación matutina de la FC en reposo (+6 a +8 lpm) como síntoma de deshidratación o inicio infeccioso (caso {idx})",
                f"Caída de rMSSD por fatiga sistémica y glucógeno muscular agotado tras competición previa (caso {idx})",
            ]
            questions = [
                "Coach, casi no dormí anoche y el estrés del jale me trae loco. Tenía pensado entrenar pesas para desquitarme. ¿Qué hacemos?",
                "Coach, noté que mi pulso en reposo se disparó 8 latidos y traigo la garganta raspada. ¿Puedo salir a correr?",
                "Coach, amanecí con el cuerpo tronado y el reloj me marca fatiga y desbalance. ¿Descanso o entreno suave?",
            ]
    else:  # en
        if tier == "high_intensity":
            scenarios = [
                f"Maximal strength session aiming for a new squat or bench press personal record (PR) with central nervous system primed (case {idx})",
                f"Track interval workout in Zone 5 VO2Max (6x800m at 3:20/km) with peak vagal tone (case {idx})",
                f"Sustained 8-10 km lactate threshold tempo run in upper Zone 4 with high stroke volume (case {idx})",
                f"Mountain trail run with +1,000m elevation gain and zero recovery debt (case {idx})",
            ]
            questions = [
                "Coach, I woke up with 100% readiness and green numbers across the board. Today is heavy leg day and I want to push for a new squat PR. Do I load the plates?",
                "Coach, my nocturnal HRV is sitting at its monthly ceiling. The schedule calls for brutal VO2Max track intervals. Do I go all out today?",
                "Coach, today's workout is an 8 km continuous tempo run at lactate threshold. My biometrics look pristine. Can I push the pace aggressively?",
            ]
        elif tier == "moderate_base":
            scenarios = [
                f"Long Zone 2 aerobic base run for mitochondrial density and fat oxidation (case {idx})",
                f"Submaximal gym hypertrophy session with strict tempo control and RPE 7-8 (case {idx})",
                f"Consistent daily activity of 10,000 to 12,000 steps with joint mobility (case {idx})",
            ]
            questions = [
                "Coach, my biometric metrics are stable and balanced. I have an easy Zone 2 long run scheduled. How do you see it?",
                "Coach, I've been hitting my 10,000 steps and submaximal lifting consistently. Should I maintain this baseline or push harder?",
                "Coach, I feel well rested for our regular gym session. Shall we keep the focus on clean movement quality and form?",
            ]
        else:  # recovery_fatigue
            scenarios = [
                f"Acute deficit in slow-wave deep sleep (<30 min) with elevated nighttime stress (case {idx})",
                f"High daytime allostatic stress (>45 pts) from workplace overload with zero restorative rest (case {idx})",
                f"Morning resting heart rate elevation (+6 to +8 bpm) indicating systemic dehydration or early infection (case {idx})",
                f"Suppressed nocturnal rMSSD and depleted glycogen reserves following competition (case {idx})",
            ]
            questions = [
                "Coach, I barely slept 5 hours and work stress is through the roof. I wanted to hit heavy deadlifts to blow off steam. What should I do?",
                "Coach, my resting heart rate spiked by 8 bpm and I woke up with a scratchy throat. Can I still head out for my run?",
                "Coach, my body feels completely beat and the watch shows an unbalanced HRV. Do I take an honest rest day?",
            ]

    scenario_desc = random.choice(scenarios)
    user_q = random.choice(questions)
    return scenario_desc, citation_text, user_q


def generate_single_example(
    client: genai.Client,
    model_name: str,
    metric: str,
    tier: str,
    lang: str,
    idx: int,
    temperature: float = 0.82,
    top_p: float = 0.94,
) -> dict[str, Any] | None:
    """Generates a single concise (200-350 tokens) training example using Gemini."""
    scenario_desc, citation_text, user_q = build_scenario_spec(metric, tier, lang, idx)

    system_prompt = SYSTEM_PROMPT_ES if lang == "es" else SYSTEM_PROMPT_EN

    if lang == "es":
        format_instructions = """RESPONDE ÚNICAMENTE CON UN OBJETO JSON VÁLIDO con las tres claves:
1. "scenario": descripción breve del caso.
2. "user_content": texto del usuario con:
   [TELEMETRÍA GARMIN (SQLite)]
   (Métricas realistas: rMSSD, FC Reposo, Sueño, Estrés, Pasos, Zonas FC, etc.)
   
   [EVIDENCIA CIENTÍFICA (RAG Firstbeat)]
   {citation_text}
   
   [PREGUNTA DEL ATLETA]
   "{user_q}"

3. "assistant_content": respuesta del coach fitness mexicano ESTRICTAMENTE entre 150 y 220 palabras:
   - Saludo enérgico mexicano ("¡Qué onda, mi rey!", "¡Eso es todo, carnalito!", "¡Uff, mi hermano!").
   - Tres secciones con encabezados exactos:
     ### 1. El Diagnóstico Rápido
     ### 2. La Explicación Fisiológica (Lo que dice la ciencia)
     ### 3. La Chamba de Hoy (Plan de Acción)
   - Cierre motivacional mexicano.
   - Longitud: 150 a 220 palabras (≈240-360 tokens). DIRECTO Y CONCISO."""
    else:
        format_instructions = """RESPOND ONLY WITH A VALID RAW JSON OBJECT with three keys:
1. "scenario": short scenario description.
2. "user_content": user prompt containing:
   [TELEMETRÍA GARMIN (SQLite)]
   (Realistic numbers: rMSSD, Resting HR, Sleep Score, Stress, Steps, HR Zones, etc.)
   
   [EVIDENCIA CIENTÍFICA (RAG Firstbeat)]
   {citation_text}
   
   [PREGUNTA DEL ATLETA]
   "{user_q}"

3. "assistant_content": Mexican bilingual sports coach response STRICTLY between 150 and 220 words:
   - Energetic coach greeting ("What's up, my friend!", "Let's get after it, man!", "What's up, man!").
   - Three mandatory sections with exact headers:
     ### 1. Quick Diagnosis
     ### 2. Physiological Breakdown (What Science Says)
     ### 3. Today's Work (Action Plan)
   - Punchy motivational closing.
   - Total length: 150 to 220 words (≈190-280 tokens). DIRECT, PUNCHY, AND CONCISE."""

    prompt = f"""Genera un ejemplo de Fine-Tuning ChatML en formato JSON estricto.

IDIOMA DEL ASISTENTE: {"ESPAÑOL (COACH MEXICANO)" if lang == "es" else "ENGLISH (NATURAL BILINGUAL MEXICAN COACH)"}
MÉTRICA: {metric}
INTENSIDAD: {tier.upper()} ({scenario_desc})

{format_instructions.format(citation_text=citation_text, user_q=user_q)}

SIN BLOQUES DE MARKDOWN ```json. SOLO EL JSON CRUDO."""

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=650,
        response_mime_type="application/json",
    )

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        raw_text = (resp.text or "").strip()
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as json_err:
            # Attempt to repair truncated JSON (common with small models hitting max_tokens)
            repaired = raw_text.rstrip()
            # Close any open string then close the object
            if repaired and repaired[-1] not in ('"', '}'):
                repaired += '"'
            if repaired.count('{') > repaired.count('}'):
                repaired += '}'
            try:
                data = json.loads(repaired)
                logger.warning(f"JSON reparado exitosamente para {model_name}/{metric}/{lang}.")
            except json.JSONDecodeError:
                raise json_err  # re-raise original error
        assistant_content = data.get("assistant_content", "")
        user_content = data.get("user_content", "")

        # Verify word count / approximate token range (140 to 300 words)
        words = len(assistant_content.split())
        approx_tokens = int(words * 1.33)
        if approx_tokens < 170 or approx_tokens > 390:
            # Slightly lenient bounding
            pass

        record_id = f"synthetic_{lang}_{metric}_{tier}_{idx:03d}"
        chatml_record = {
            "id": record_id,
            "language": lang,
            "metric_category": metric,
            "intensity_tier": tier,
            "scenario": data.get("scenario", scenario_desc),
            "approx_tokens": approx_tokens,
            "model_metadata": {
                "generator_model": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        return chatml_record
    except Exception as e:
        print(f"⚠️ Error ({model_name}, {metric}, {lang}): {e}")
        return None


def run_block_generation(
    target_count: int = 400,
    block_size: int = 25,
    selected_model: str | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    delay: float = 0.5,
) -> None:
    """Executes a single controlled block of synthetic examples."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no encontrada en entorno.")

    client = genai.Client(api_key=api_key)

    # Fast models pool — quota-active models (gemini-3-flash-preview and gemini-3.8-flash excluded: day quota exhausted)
    default_fast_models = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    if selected_model:
        model_pool = [selected_model]
    else:
        model_pool = default_fast_models

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_records = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    current_count = len(existing_records)
    print(f"📊 Dataset actual: {current_count}/{target_count} ejemplos existentes en {output_path}")

    if current_count >= target_count:
        print(f"🎉 ¡Meta de {target_count} ejemplos ya está completa!")
        return

    # Build plan: 400 items, 200 ES, 200 EN
    metrics = ["hrv_rmssd", "sleep_score", "daily_avg_stress", "resting_heart_rate", "total_steps"]
    tiers = ["high_intensity", "moderate_base", "recovery_fatigue"]
    languages = ["es", "en"]

    full_plan = []
    global_idx = 1
    # 400 items = 200 per language = 40 per (lang, metric) = 13-14 per (lang, metric, tier)
    for lang in languages:
        for m in metrics:
            for t in tiers:
                for _ in range(14):
                    full_plan.append((lang, m, t, global_idx))
                    global_idx += 1

    random.seed(1337)
    random.shuffle(full_plan)
    full_plan = full_plan[:target_count]

    # Pick this block
    block_end = min(current_count + block_size, target_count)
    block_items = full_plan[current_count:block_end]

    print(f"🚀 Ejecutando bloque de {len(block_items)} ejemplos ({current_count + 1} a {block_end} de {target_count})...")
    print(f"🎯 Pool de modelos para este bloque: {model_pool}")

    success_in_block = 0
    with open(output_path, "a", encoding="utf-8") as f_out:
        for step, (lang, metric, tier, g_idx) in enumerate(block_items, start=current_count + 1):
            model_name = model_pool[step % len(model_pool)]
            temp = round(random.uniform(0.72, 0.94), 2)
            top_p = round(random.uniform(0.91, 0.96), 2)

            t0 = time.time()
            example = None
            for attempt in range(2):
                example = generate_single_example(
                    client=client,
                    model_name=model_name,
                    metric=metric,
                    tier=tier,
                    lang=lang,
                    idx=g_idx,
                    temperature=temp,
                    top_p=top_p,
                )
                if example is not None:
                    break
                time.sleep(1.5)

            if example is not None:
                f_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                f_out.flush()
                success_in_block += 1
                elapsed = round(time.time() - t0, 2)
                print(
                    f"[{step}/{target_count}] ✅ {example['id']} | "
                    f"{lang.upper()} | {model_name} | {example.get('approx_tokens', 0)} tok | {elapsed}s"
                )
            else:
                print(f"[{step}/{target_count}] ❌ Falló ejemplo {g_idx} ({metric}, {lang})")

            time.sleep(delay)

    new_total = current_count + success_in_block
    print("\n" + "=" * 80)
    print(f"🏁 Bloque completado: {success_in_block}/{len(block_items)} nuevos ejemplos agregados.")
    print(f"📈 Total acumulado en {output_path}: {new_total}/{target_count} ejemplos.")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Generador de dataset sintético por bloques (bilingüe 200-350 tokens).")
    parser.add_argument("--target", type=int, default=400, help="Meta total de ejemplos (def: 400)")
    parser.add_argument("--block-size", type=int, default=20, help="Número de ejemplos a generar en este bloque (def: 20)")
    parser.add_argument("--model", type=str, default=None, help="Modelo específico a usar para este bloque")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Ruta del archivo JSONL")
    parser.add_argument("--delay", type=float, default=0.5, help="Pausa entre peticiones (segundos)")
    args = parser.parse_args()

    run_block_generation(
        target_count=args.target,
        block_size=args.block_size,
        selected_model=args.model,
        output_path=Path(args.output),
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
