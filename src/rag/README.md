# 🧠 Módulo RAG & Base de Conocimiento Biomédica

Este módulo implementa el pipeline completo de Recuperación Aumentada por Generación (RAG) para el asistente personalizado de Garmin: ingestión, segmentación por tokens, gestión de estado mediante ledger, embeddings vectoriales con Google Gemini, almacenamiento vectorial en ChromaDB y síntesis generativa protegida por guardrails multicapa y reranking híbrido.

---

## 📚 Las 3 Fuentes de Conocimiento y sus Roles

El conocimiento se estructura en 3 fuentes ortogonales para garantizar máxima especificidad y prevenir contaminación de contexto:

| Fuente | Chunks | Rol en el Sistema | Cobertura Científica y Técnica |
| :--- | :---: | :--- | :--- |
| **`variables_fisiologia_humana`** | **423** | **Activa en Generación (`/ask`)** | Fisiología deportiva, dinámica del Sistema Nervioso Autónomo (SNA), equilibrio simpático-parasimpático, variabilidad de frecuencia cardíaca (VFC / HRV: métricas rMSSD, SDNN, LF/HF), arquitectura del sueño (fases NREM, REM, sueño profundo de ondas lentas), cinéticas de VO2 Max y lactato, EPOC (exceso de consumo de oxígeno post-ejercicio) y fisiología del sobreentrenamiento y recuperación. |
| **`dispositivos_garmin_sensores`** | **268** | **Activa en Generación (`/ask`)** | Arquitectura de sensores de Garmin: sensores ópticos PPG Elevate v4 y v5 con múltiples canales LED, pulsioximetría PulseOx (SpO2), acelerometría triaxial, altimetría barométrica, GPS multibanda y algoritmos propietarios de Firstbeat Analytics (Body Battery, Sleep Score, Training Readiness, Training Status, Training Effect aeróbico/anaeróbico, Stamina en tiempo real y HRV Status nocturno). |
| **`descripciones_metricas_garmin`** | **11** | **Lookup & Metadatos (Excluida de `/ask`)** | Catálogo técnico de variables de telemetría de Garmin Connect: estructuras JSON (`sleep.json`, `stress.json`, `daily_summary.json`), tipos de datos, unidades y rangos de referencia. Se mantiene en ChromaDB para desambiguación de esquemas y queries de metadatos, pero se excluye deliberadamente del motor de respuesta `/ask` para priorizar la profundidad científica sobre definiciones cortas. |

---

## ⚡ Motor de Reranking Híbrido (Reciprocal Rank Fusion)

Ubicación: [`src/rag/reranker.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/rag/reranker.py) y [`src/rag/guardrails.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/rag/guardrails.py)

Para optimizar la precisión de los fragmentos inyectados al LLM:
1. **Candidate Expansion**: Recupera un grupo amplio de candidatos de ChromaDB (`top_k * 3`, mínimo 10).
2. **Dense Ranking**: Ordena por menor distancia de coseno vectorial.
3. **Lexical BM25 Scoring**: Evalúa la densidad de términos clave, la correspondencia exacta de siglas técnicas (`rMSSD`, `VO2Max`, `Elevate`) y el solapamiento con las etiquetas (`tags`) de los metadatos.
4. **Reciprocal Rank Fusion (RRF $k=60$)**:
   $$\text{RRF}(d) = \frac{0.55}{60 + \text{Rank}_{\text{denso}}(d)} + \frac{0.45}{60 + \text{Rank}_{\text{léxico}}(d)} + \text{Bonus}_{\text{cobertura}}$$
5. **Rendimiento**: Ejecución en memoria en CPU (< 1 ms), sin llamadas a APIs externas ni costes adicionales.

---

## 🌐 Política Estricta de Idiomas

Ubicación: `detect_query_language(text)` en [`src/rag/guardrails.py`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/src/rag/guardrails.py)

- **Español (`es`)**: Idioma base. Toda consulta en español genera una respuesta íntegramente en español.
- **Inglés (`en`)**: Si la consulta se formula en inglés, el motor detecta el idioma e instruye al LLM a responder íntegramente en inglés.
- **Idiomas no permitidos (francés, alemán, italiano, portugués, etc.)**: Interceptados tempranamente por el guardrail y rechazados con `status="unsupported_language"` **sin consumir llamadas a la API de generación**, protegiendo el presupuesto de cuota gratuita.

---

## 🛡️ Guardrails y Protección de Presupuesto

1. **Anti-Ofuscación**: Desofuscación de Leetspeak, eliminación de diacríticos para reglas regex, colapso de letras repetidas y colapso de espaciados de evasión (`i g n o r e`).
2. **Patrones de Ataque**: Detección de `INSTRUCTION_OVERRIDE`, `ROLEPLAY_JAILBREAK`, `PROMPT_LEAK`, `SYNTAX_INJECTION` y `HYPOTHETICAL_BYPASS`.
3. **Clasificador de Dominio**: Reutiliza el vector de embedding de la consulta contra el centroide de referencia de Garmin (umbral `0.52`), con fast-path léxico O(1).
4. **Presupuesto Estricto**: `LLMBudgetTracker` impone topes en memoria de **60 llamadas/hora** y **100 llamadas/día**.
