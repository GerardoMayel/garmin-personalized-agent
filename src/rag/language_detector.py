"""Hybrid NLP Language Detector for Knowledge Base Chunks.

Uses traditional NLP (langdetect) for sub-millisecond local language classification.
Supports Google Gemini fallback via GEMINI_API_KEY when text is ambiguous or short.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("LanguageDetector")

_CACHE: dict[str, str] = {}


def detect_language(text: str, default_lang: str = "en") -> str:
    """Detecta el código de idioma ISO 639-1 ('en', 'es', etc.) para un texto dado.

    1. Verifica si el resultado está en caché.
    2. Utiliza langdetect como método principal (NLP tradicional sin coste ni latencia).
    3. Si falla o es ambiguo y GEMINI_API_KEY está configurada, utiliza Gemini 1.5 Flash/2.5 Flash como fallback.
    4. En caso de fallo final, retorna default_lang ('en').
    """
    clean_text = text.strip()
    if not clean_text or len(clean_text) < 10:
        return default_lang

    cache_key = clean_text[:120]
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # Paso 1: langdetect tradicional
    try:
        from langdetect import detect

        detected = detect(clean_text)
        if detected:
            iso_code = str(detected).lower()[:2]
            _CACHE[cache_key] = iso_code
            return iso_code
    except Exception as e:
        logger.debug(f"langdetect no pudo determinar idioma: {e}")

    # Paso 2: Fallback con Google Gemini si GEMINI_API_KEY existe
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload: dict[str, Any] = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "Classify the language of the following text into a 2-letter ISO 639-1 code "
                                    "(e.g., 'en', 'es', 'fi', 'fr', 'de'). Return ONLY the 2-letter code in lowercase:\n\n"
                                    f"{clean_text[:500]}"
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 5},
            }
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                raw_code = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                code_str = str(raw_code).strip().lower()[:2]
                if code_str and len(code_str) == 2 and code_str.isalpha():
                    _CACHE[cache_key] = code_str
                    return code_str
        except Exception as e:
            logger.debug(f"Gemini fallback para idioma falló: {e}")

    _CACHE[cache_key] = default_lang
    return default_lang
