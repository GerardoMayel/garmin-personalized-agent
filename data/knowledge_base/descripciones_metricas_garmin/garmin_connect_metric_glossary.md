# Glosario Oficial y Explicaciones de Métricas Garmin Connect

Documento de referencia técnica que compila las definiciones formales, fundamentos biológicos, algoritmos embebidos (Firstbeat Analytics) e interpretación de códigos de feedback reportados en la telemetría del usuario de Garmin Connect.

**Fecha de compilación**: 2026-09-18 20:58:44 UTC
**Métricas documentadas**: 10

---

## Estado de Variabilidad de la Frecuencia Cardíaca (HRV) (Heart Rate Variability (HRV) Status)

- **Clave interna**: `hrv_status`
- **Categoría fisiológica**: `recuperacion_autonomica`
- **Unidad de medida**: ms (rMSSD)

### Descripción Técnica y Fundamento Fisiológico
El estado de HRV de Garmin evalúa el equilibrio del Sistema Nervioso Autónomo (SNA) comparando el promedio nocturno de rMSSD (raíz cuadrada de las diferencias medias de intervalos R-R sucesivos) contra una línea base personal calculada durante las últimas 3 a 4 semanas. Se clasifica en cuatro estados: Equilibrado (Balanced), Desequilibrado (Unbalanced), Bajo (Low) y Pobre (Poor). Un estado Equilibrado indica una óptima actividad del tono parasimpático y capacidad de recuperación. Un estado Desequilibrado o Bajo refleja estrés fisiológico acumulado, fatiga simpática, sobreentrenamiento, consumo de alcohol o el inicio de una infección inmunológica.

### Interpretación de Insights y Feedback de Garmin Connect
- **`HRV_BALANCED_7`**: Línea base equilibrada durante los últimos 7 días. El sistema nervioso autónomo mantiene buena resiliencia y adaptabilidad al entrenamiento.
- **`HRV_UNBALANCED_LOW`**: Valores por debajo de la franja normal personal. Indica sobrecarga simpática o recuperación insuficiente; se aconseja priorizar descanso.

---

## Puntuación de Sueño y Arquitectura del Descanso (Sleep Score & Sleep Architecture)

- **Clave interna**: `sleep_score_and_architecture`
- **Categoría fisiológica**: `recuperacion_nocturna`
- **Unidad de medida**: Puntuación (0-100) y segundos por fase

### Descripción Técnica y Fundamento Fisiológico
El Sleep Score de Garmin combina la duración total del sueño con la calidad del descanso evaluada mediante la arquitectura de fases: Sueño Profundo (N3 / Slow-Wave Sleep, 15-25% ideal), Sueño REM (Movimientos Oculares Rápidos, 20-25% ideal), Sueño Ligero y periodos despierto. El algoritmo Firstbeat analiza la variabilidad cardíaca y la frecuencia respiratoria durante la noche para calcular el estrés promedio del sueño (avgSleepStress). Un estrés nocturno < 15 indica sueño profundamente reparador; estrés > 30 señala microdespertares y recuperación autonómica comprometida.

### Interpretación de Insights y Feedback de Garmin Connect
- **`POSITIVE_DEEP`**: Duración adecuada de sueño profundo, promoviendo la liberación de hormona del crecimiento y reparación tisular muscular.
- **`STRESS_POS_EXCELLENT_OR_GOOD_SLEEP_VERY_RESTFUL_DAY`**: Día muy reparador con bajo estrés general, facilitando un sueño de alta calidad y recuperación psicofísica completa.

---

## Nivel de Estrés Diario (All-Day Stress Score)

- **Clave interna**: `all_day_stress`
- **Categoría fisiológica**: `fisiologia_autonomica`
- **Unidad de medida**: Escala 0-100

### Descripción Técnica y Fundamento Fisiológico
Garmin estima el estrés corporal en tiempo real a partir del análisis de los intervalos R-R del sensor óptico PPG Elevate. La escala de 0 a 100 se segmenta en: 0-25 Reposo (predominio parasimpático/recuperación), 26-50 Estrés bajo (actividad mental o física ligera), 51-75 Estrés medio (esfuerzo cognitivo intenso o fatiga) y 76-100 Estrés alto (respuesta simpática de lucha o huida, agotamiento o enfermedad). Los periodos en reposo recargan el Body Battery.

### Interpretación de Insights y Feedback de Garmin Connect
- **`STRESS_REST`**: Nivel de estrés inferior a 25. El organismo se encuentra en homeostasis y reparación biológica.
- **`STRESS_HIGH_SUSTAINED`**: Estrés sostenido superior a 50 durante el día, acelerando el drenaje de reservas biológicas.

---

## Body Battery (Batería Corporal) (Garmin Body Battery)

- **Clave interna**: `body_battery`
- **Categoría fisiológica**: `reserva_energetica`
- **Unidad de medida**: Puntos (1-100)

### Descripción Técnica y Fundamento Fisiológico
Body Battery es una métrica de gestión energética propietaria de Firstbeat Analytics que cuantifica las reservas energéticas del cuerpo en una escala de 1 a 100. Analiza de manera continua la variabilidad cardíaca, el nivel de estrés, la calidad del sueño y la actividad física registrada. El descanso reparador nocturno y las pausas diurnas de bajo estrés recargan la batería (+), mientras que el ejercicio, la digestión pesada, el estrés crónico y el consumo de estimulantes la descargan (-).

### Interpretación de Insights y Feedback de Garmin Connect
- **`BATTERY_CHARGED_HIGH`**: Body Battery por encima de 80 al despertar, indicando óptima preparación para exigencia física o cognitiva.
- **`BATTERY_DRAINED_LOW`**: Body Battery por debajo de 25, sugiriendo moderar la intensidad de entrenamientos para evitar fatiga acumulada.

---

## VO2 Máximo y Condición Cardiorrespiratoria (VO2 Max & Cardiorespiratory Fitness)

- **Clave interna**: `vo2_max`
- **Categoría fisiológica**: `rendimiento_cardiovascular`
- **Unidad de medida**: mL/kg/min

### Descripción Técnica y Fundamento Fisiológico
El VO2 Máximo define el volumen máximo de oxígeno que el cuerpo puede transportar y utilizar durante un esfuerzo de intensidad máxima. Garmin y Firstbeat calculan automáticamente el VO2 Max mediante un algoritmo de aprendizaje automático embebido que correlaciona la velocidad o potencia de carrera con la respuesta de la frecuencia cardíaca submáxima. Filtra automáticamente tramos no estacionarios, desniveles, paradas y deriva cardíaca térmica para proporcionar estimaciones con un margen de error menor al 5% respecto a pruebas de laboratorio con análisis de gases respiratorios.

### Interpretación de Insights y Feedback de Garmin Connect
- **`VO2MAX_EXCELLENT`**: Capacidad cardiorrespiratoria en el percentil superior para tu grupo de edad y sexo.

---

## Efecto de Entrenamiento (Aeróbico y Anaeróbico) y EPOC (Training Effect (Aerobic & Anaerobic) & EPOC)

- **Clave interna**: `training_effect_and_epoc`
- **Categoría fisiológica**: `carga_entrenamiento`
- **Unidad de medida**: Escala 0.0 - 5.0 (y ml/kg EPOC)

### Descripción Técnica y Fundamento Fisiológico
Training Effect mide el impacto de una actividad física sobre la aptitud aeróbica y anaeróbica. El Efecto Aeróbico (0.0 a 5.0) cuantifica la acumulación de EPOC (Exceso de Consumo de Oxígeno Post-Ejercicio), midiendo la perturbación de la homeostasis y el estímulo para mejorar la capacidad mitocondrial y cardiovascular. El Efecto Anaeróbico (0.0 a 5.0) evalúa la capacidad de generar energía sin oxígeno mediante intervalos de alta intensidad (HIIT), esprints o series de fuerza, midiendo la rapidez del incremento de la frecuencia cardíaca y la potencia.

### Interpretación de Insights y Feedback de Garmin Connect
- **`TE_MAINTAINING`**: Puntuación de 2.0 a 2.9: Mantiene la condición cardiovascular actual sin provocar sobrecarga.
- **`TE_IMPROVING`**: Puntuación de 3.0 a 3.9: Provoca adaptaciones fisiológicas progresivas que elevan la aptitud aeróbica.
- **`TE_HIGHLY_IMPROVING`**: Puntuación de 4.0 a 4.9: Estímulo de sobrecarga de alta magnitud; requiere adecuada recuperación posterior.

---

## Frecuencia Cardíaca en Reposo (RHR) (Resting Heart Rate (RHR))

- **Clave interna**: `resting_heart_rate`
- **Categoría fisiológica**: `salud_cardiovascular`
- **Unidad de medida**: ppm (latidos por minuto)

### Descripción Técnica y Fundamento Fisiológico
La frecuencia cardíaca en reposo es el número de contracciones cardíacas por minuto registradas durante los periodos de inactividad más profunda del día o justo antes de despertar. Una elevación sostenida de más de 5-7 ppm sobre el promedio semanal habitual es un marcador temprano de fatiga cardiovascular, deshidratación, desequilibrio en la recuperación o activación del sistema inmune.

### Interpretación de Insights y Feedback de Garmin Connect
- **`RHR_OPTIMAL`**: Frecuencia en reposo estable o en tendencia descendente, indicando adaptaciones positivas de hipertrofia excéntrica ventricular.

---

## Pulsioximetría y Saturación de Oxígeno (SpO2) (Pulse Oximetry & SpO2)

- **Clave interna**: `pulse_ox_spo2`
- **Categoría fisiológica**: `respiratorio_hematologico`
- **Unidad de medida**: Porcentaje (%)

### Descripción Técnica y Fundamento Fisiológico
El sensor óptico de Garmin emite haces combinados de luz roja (660 nm) e infrarroja (940 nm) a través de la piel para medir la proporción de hemoglobina oxigenada versus desoxigenada en los capilares. Los valores normales en personas sanas oscilan entre 95% y 100%. Durante el sueño, lecturas continuas permiten monitorear la hipoxemia nocturna, la aclimatación a la altitud y posibles eventos de apnea del sueño.

### Interpretación de Insights y Feedback de Garmin Connect
- **`SPO2_NORMAL`**: Saturación arterial estimada estable > 95% a nivel de superficie.

---

## Frecuencia Respiratoria (Respiration Rate)

- **Clave interna**: `respiration_rate`
- **Categoría fisiológica**: `respiratorio_pulmonar`
- **Unidad de medida**: brpm (respiraciones por minuto)

### Descripción Técnica y Fundamento Fisiológico
Garmin estima la tasa respiratoria mediante la técnica de arritmia sinusal respiratoria (RSA): al inhalar, el tono vagal se inhibe transitoriamente y la frecuencia cardíaca se acelera ligeramente; al exhalar, el tono vagal aumenta y la frecuencia se ralentiza. El algoritmo analiza estas fluctuaciones micrométricas del intervalo R-R para derivar las respiraciones por minuto durante el día y el sueño. El rango típico de reposo en adultos sanos es de 12 a 20 brpm.

### Interpretación de Insights y Feedback de Garmin Connect
- **`RESPIRATION_RESTING_NORMAL`**: Frecuencia respiratoria nocturna basal de 12-16 brpm, reflejando adecuada ventilación alveolar.

---

## Gasto Calórico: Reposo (BMR) y Calorías Activas (Calorie Expenditure: Resting (BMR) & Active)

- **Clave interna**: `calorie_expenditure_split`
- **Categoría fisiológica**: `metabolismo_energetico`
- **Unidad de medida**: kcal (kilocalorías)

### Descripción Técnica y Fundamento Fisiológico
Garmin desglosa el gasto calórico total en dos componentes termodinámicos esenciales: 1. Calorías en Reposo: Tasa Metabólica Basal (BMR) proyectada matemáticamente mediante la ecuación de Cunningham o Harris-Benedict ajustada con el perfil biométrico (edad, sexo, peso, altura y porcentaje graso). 2. Calorías Activas: Energía consumida por encima del BMR derivada del número de pasos, la intensidad del movimiento medida por acelerómetro y el consumo de oxígeno (VO2) estimado a partir de la frecuencia cardíaca continua.

### Interpretación de Insights y Feedback de Garmin Connect
- **`CALORIES_BALANCED`**: Equilibrio entre gasto activo y reposo acorde a las metas de entrenamiento y mantenimiento del balance de nitrógeno.

---
