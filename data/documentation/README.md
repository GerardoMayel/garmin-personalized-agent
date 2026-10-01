# 🗄️ Documentación de Bases de Datos SQLite (Cloud Storage R2)

Este directorio contiene los esquemas informativos de texto generados automáticamente a partir de las bases de datos SQLite persistidas en **Cloudflare R2** (`garmin-personal-data`).

---

## 📄 Archivos Generados

1. **[`01_catalogo_bases_y_tablas.txt`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/data/documentation/01_catalogo_bases_y_tablas.txt)**:
   - Inventario general de bases de datos, claves remotas en el bucket, tamaños, número de tablas y vistas, total de registros almacenados y claves primarias.
2. **[`02_diagrama_entidad_relacion.txt`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/data/documentation/02_diagrama_entidad_relacion.txt)**:
   - Diagrama Entidad-Relación en formato texto (ASCII / Unicode Box Drawing).
   - Mapea las relaciones longitudinales de telemetría centradas en la clave maestra `calendar_date` (1:1 y 1:N), la tabla de consolidación aplanada (`consolidated_daily_actuals`), las proyecciones (`consolidated_biometric_forecasts`), los promedios bi-semanales (`bi_weekly_average`), la vista temporal unificada (`unified_biometrics_timeline`) y la base de metadatos de modelos de series de tiempo (`weekly_biometric_forecasts.db`).
3. **[`03_mapeo_columnas_tipos_y_muestras.txt`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/data/documentation/03_mapeo_columnas_tipos_y_muestras.txt)**:
   - Diccionario de datos exhaustivo por base y por tabla.
   - Detalla el nombre de columna, la nomenclatura técnica de SQLite (`TEXT`, `INTEGER`, `REAL`, `BLOB`), restricciones (`PRIMARY KEY`, `NOT NULL`, `DEFAULT`) y un **TOP 3 de ejemplos reales** no nulos extraídos directamente de los registros en almacenamiento.
4. **[`00_resumen_esquema_consolidado.txt`](file:///Users/mayelmacbookm4pro/repos/garmin-personalized-agent/data/documentation/00_resumen_esquema_consolidado.txt)**:
   - Documento maestro integrado con todo el catálogo, el diagrama ER y el diccionario de columnas en un único archivo.

---

## 🚀 Cómo Actualizar la Documentación

Para volver a escanear el bucket en Cloudflare R2 y regenerar los documentos de texto, ejecuta desde la raíz del proyecto:

```bash
# Ejecutar inspección remota contra Cloudflare R2:
python data/documentation/map_cloud_databases.py

# O con uv:
uv run python data/documentation/map_cloud_databases.py

# Modo local (inspección de bases SQLite locales en data/ sin llamar a R2):
uv run python data/documentation/map_cloud_databases.py --local

# Modificar número de muestras por columna (ejemplo: TOP 5):
uv run python data/documentation/map_cloud_databases.py --max-samples 5
```
