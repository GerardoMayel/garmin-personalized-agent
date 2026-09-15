# Specification 01: Data Schemas & Contracts

## 1. Overview
Defines typed Pydantic models for raw JSON payloads, normalized parquet schemas, and dense `.fit` records across the pipeline.

---

## 2. Core Entities

### 2.1 Sleep Record (`SleepRecord`)
- `calendar_date`: `date` (YYYY-MM-DD)
- `sleep_start_timestamp`: `datetime`
- `sleep_end_timestamp`: `datetime`
- `total_sleep_seconds`: `int`
- `deep_sleep_seconds`: `int`
- `light_sleep_seconds`: `int`
- `rem_sleep_seconds`: `int`
- `awake_sleep_seconds`: `int`
- `average_spo2`: `Optional[float]`
- `average_respiration_rate`: `Optional[float]`
- `sleep_score`: `Optional[int]` (0-100)

### 2.2 HRV Record (`HRVRecord`)
- `calendar_date`: `date`
- `weekly_avg_rmssd`: `Optional[float]`
- `last_night_avg_rmssd`: `Optional[float]`
- `baseline_low`: `Optional[float]`
- `baseline_balanced_low`: `Optional[float]`
- `baseline_balanced_upper`: `Optional[float]`
- `status`: `str` ("BALANCED", "UNBALANCED", "LOW", "POOR")
- `readings_5min`: `List[Tuple[datetime, float]]`

### 2.3 Activity Telemetry (`FitTelemetryRecord`)
- `activity_id`: `str`
- `timestamp`: `datetime`
- `heart_rate`: `Optional[int]`
- `cadence`: `Optional[int]`
- `power_watts`: `Optional[float]`
- `altitude_meters`: `Optional[float]`
- `speed_mps`: `Optional[float]`
- `position_lat`: `Optional[float]`
- `position_long`: `Optional[float]`
- `temperature_celsius`: `Optional[float]`
