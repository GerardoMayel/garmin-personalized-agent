"""Multi-Layer NLP Guardrails and LLM Call Budget Protection.

Shared module for RAG security, text normalization, prompt injection detection,
domain validation, and LLM rate/budget tracking.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from typing import Any

import numpy as np

# ---------------------------------------------------------------------
# 1. NORMALIZADOR ANTI-OFUSCACIÓN (LEETSPEAK & BYPASSES)
# ---------------------------------------------------------------------

LEET_MAP = str.maketrans(
    {
        "@": "a",
        "4": "a",
        "3": "e",
        "1": "i",
        "!": "i",
        "0": "o",
        "5": "s",
        "$": "s",
        "7": "t",
        "+": "t",
    }
)


def normalize_text(text: str) -> str:
    """Desofusca el texto eliminando acentos, caracteres invisibles, leetspeak y repeticiones."""
    # 1. Normalización Unicode (elimina diacríticos/acentos)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.lower()

    # 2. Eliminar caracteres invisibles, nulos o de control
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", "", text)

    # 3. Traducción de Leetspeak común
    text = text.translate(LEET_MAP)

    # 4. Colapsar letras repetidas más de 2 veces (ej: "pleaaase" -> "please", "ignooora" -> "ignora")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 5. Colapsar espaciados de bypass (ej: "i g n o r e" -> "ignore")
    # Une letras individuales consecutivas separadas por espacio
    text = re.sub(r"(?<=\b[a-z])\s+(?=[a-z]\b)", "", text)

    return text.strip()


# ---------------------------------------------------------------------
# 2. REGLAS COMPILADAS DE ATAQUES (HEURÍSTICAS NLP)
# ---------------------------------------------------------------------

ATTACK_PATTERNS: dict[str, re.Pattern[str]] = {
    # Inyecciones de sintaxis y delimitadores estructurales de LLMs
    "SYNTAX_INJECTION": re.compile(
        r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<\/SYS>|###\s*(?:System|Instruction|Assistant):)",
        re.IGNORECASE,
    ),
    # Intento de forzar cambio de instrucciones
    "INSTRUCTION_OVERRIDE": re.compile(
        r"(?:olvida|ignora|omite|desecha|override|ignore|disregard|forget|bypass).*?"
        r"(?:instrucciones|indicaciones|directrices|reglas|prompts?|instructions?|rules?|guidelines?|system\s*prompt)|"
        r"(?:system\s*prompt\s*override|override\s*system\s*prompt|system\s*override)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Adopción de personalidades desreguladas / Jailbreaks
    "ROLEPLAY_JAILBREAK": re.compile(
        r"\b(?:actua|comportate|responde|pretend|act|behave)\s+(?:como|as)\s+(?:un\s+)?(?:.*?)?(?:dan|ia\s+libre|modelo\s+desbloqueado|sin\s+restricciones|sin\s+filtros|unrestricted|root|jailbreak)\b|"
        r"\b(?:developer\s+mode|modo\s+desarrollador|modo\s+dan|dan\s+mode|do\s+anything\s+now|jailbroken|godmode)\b",
        re.IGNORECASE,
    ),
    # Intentos de extraer el System Prompt o información interna
    "PROMPT_LEAK": re.compile(
        r"\b(?:revela|muestra|imprime|repite|dime|show|print|reveal|repeat|dump)\b.*?"
        r"\b(?:system\s*prompt|instrucciones\s+del\s+sistema|texto\s+anterior|tus\s+instrucciones|hidden\s+prompt)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    # Modos hipotéticos para saltarse barreras
    "HYPOTHETICAL_BYPASS": re.compile(
        r"\b(?:en\s+un\s+mundo\s+hipotetico|para\s+una\s+historia\s+ficticia|in\s+a\s+hypothetical\s+universe)\b.*?"
        r"\b(?:sin\s+reglas|no\s+tienes\s+filtros|sin\s+restricciones|without\s+restrictions)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


def scan_prompt_injection(user_raw_query: str) -> tuple[bool, str | None]:
    """Escanea el prompt (tanto en crudo como normalizado) contra las familias de ataques."""
    normalized = normalize_text(user_raw_query)

    for category, pattern in ATTACK_PATTERNS.items():
        if pattern.search(normalized) or pattern.search(user_raw_query):
            return True, category

    return False, None


# ---------------------------------------------------------------------
# 3. VERIFICACIÓN DE DOMINIO (LEXICÓN + COSENO VECTORIAL REUTILIZADO)
# ---------------------------------------------------------------------

# Lista blanca de palabras clave directas (Fast-path léxico sin esperar similitud)
GARMIN_LEXICON = {
    "garmin",
    "rmssd",
    "vfc",
    "hrv",
    "vo2",
    "vo2max",
    "lactato",
    "firstbeat",
    "estres",
    "pulsaciones",
    "frecuencia cardiaca",
    "cadencia",
    "potencia",
    "body battery",
    "tiempo de recuperacion",
    "sueno profundo",
    "sueno",
    "fase rem",
    "entrenamiento",
    "ritmo cardiaco",
    "pulsometro",
    "spo2",
    "epoc",
    "training effect",
    "elevate",
    "running dynamics",
    "umbral anaerobico",
    "recuperacion",
    "descanso",
    "cardiovascular",
    "rhr",
}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calcula similitud coseno entre dos vectores normalizados."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


def check_domain(
    normalized_query: str,
    query_embedding: list[float],
    domain_centroid: np.ndarray,
    threshold: float = 0.52,
) -> tuple[bool, float, str]:
    """Valida el dominio combinando fast-path léxico y similitud de coseno.

    Retorna: (is_in_domain, score, method)
    """
    # 1. Fast-path léxico (O(1))
    for keyword in GARMIN_LEXICON:
        if keyword in normalized_query:
            sim = cosine_similarity(np.array(query_embedding), domain_centroid)
            return True, round(max(sim, 0.70), 4), "lexicon_match"

    # 2. Evaluación semántica continua reutilizando el embedding
    sim = cosine_similarity(np.array(query_embedding), domain_centroid)
    is_valid = sim >= threshold
    method = "semantic_similarity" if is_valid else "out_of_domain"
    return is_valid, round(sim, 4), method


# ---------------------------------------------------------------------
# 4. CONTROLADOR DE PRESUPUESTO LLM (60 / HORA, 100 / DÍA)
# ---------------------------------------------------------------------


class LLMBudgetTracker:
    """Controlador en memoria del presupuesto estricto de llamadas a la API de generación LLM.

    Garantiza que solo las consultas válidas que realmente invocan al LLM descuenten
    del presupuesto, protegiendo las cuotas gratuitas (60/hora, 100/día).
    """

    def __init__(
        self,
        max_per_hour: int | None = None,
        max_per_day: int | None = None,
    ) -> None:
        self.max_per_hour = max_per_hour or int(os.getenv("MAX_RAG_CALLS_PER_HOUR", "60"))
        self.max_per_day = max_per_day or int(os.getenv("MAX_RAG_CALLS_PER_DAY", "100"))
        self._timestamps: list[float] = []

    def check_and_consume(self) -> tuple[bool, str]:
        """Verifica si queda presupuesto disponible y registra la llamada."""
        now = time.time()
        # Depurar registros más antiguos de 24 horas (86400s)
        self._timestamps = [t for t in self._timestamps if now - t < 86400]

        hour_count = sum(1 for t in self._timestamps if now - t < 3600)
        day_count = len(self._timestamps)

        if hour_count >= self.max_per_hour:
            return (
                False,
                f"Límite horario de llamadas al LLM alcanzado ({self.max_per_hour} por hora). Por favor, intenta de nuevo en unos minutos.",
            )

        if day_count >= self.max_per_day:
            return (
                False,
                f"Límite diario total de llamadas al LLM alcanzado ({self.max_per_day} por día). La cuota diaria se renovará en 24 horas.",
            )

        self._timestamps.append(now)
        return True, "ok"

    def get_usage(self) -> dict[str, Any]:
        """Retorna estadísticas actuales de consumo del presupuesto LLM."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 86400]
        hour_count = sum(1 for t in self._timestamps if now - t < 3600)
        day_count = len(self._timestamps)
        return {
            "hour_used": hour_count,
            "hour_limit": self.max_per_hour,
            "day_used": day_count,
            "day_limit": self.max_per_day,
            "remaining_today": max(0, self.max_per_day - day_count),
        }
