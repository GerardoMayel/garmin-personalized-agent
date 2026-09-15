# Specification 02: Garmin Connect Ingestion Engine

## 1. Overview & Objectives
The **Garmin Ingestion Engine** is responsible for establishing a resilient, authenticated connection with Garmin Connect, downloading daily biometrics and telemetry, persisting raw snapshots in a structured directory hierarchy under `data/raw/YYYY-MM-DD/`, and managing session tokens via Garth to minimize re-authentication prompts and rate limiting.

---

## 2. Authentication & Session Persistence Contract

### 2.1 Credentials
- `GARMIN_EMAIL`: Required in `.env`.
- `GARMIN_PASSWORD`: Required in `.env`.
- `GARMIN_TOKEN_STORE`: Path for caching OAuth tokens (default: `~/.garminconnect` or `~/.garmin_tokens`).

### 2.2 Token Lifecycle
1. **Cache-First Check**: Attempt to restore session using stored tokens from `tokenstore_dir`.
2. **Fallback Authentication**: If tokens are absent, invalid, or expired:
   - Perform full credential login (`client.login()`).
   - Dump session tokens into `tokenstore_dir` using `client.garth.dump(...)`.
3. **Rate Limit Handling**: Catch `GarminConnectTooManyRequestsError`, log structured error, and back off/abort without destroying tokens.

---

## 3. Data Extraction & Storage Contracts

### 3.1 Directory Hierarchy Contract
All raw extracts must strictly follow the partitioned path convention:
```text
data/raw/
└── YYYY-MM-DD/
    ├── sleep.json              # Detailed hypnogram, sleep stages, nocturnal SpO2, respiration
    ├── hrv.json                # Nocturnal HRV metrics, 5-min reading blocks, baseline status
    ├── stress.json             # Minute-level stress and Body Battery values
    ├── daily_summary.json      # Steps, resting HR, active calories, distance, intensity minutes
    ├── max_metrics.json        # VO2 Max (running, cycling) and fitness age
    └── activities/
        ├── activity_<ID>_summary.json  # High-level metadata of recorded activity
        └── activity_<ID>.zip           # Raw .fit binary package (downloaded via ActivityDownloadFormat.FIT)
```

### 3.2 Endpoints & Sync Operations
| Operation | Garmin API Method | Target File |
| :--- | :--- | :--- |
| **Sleep** | `client.get_sleep_data(date_str)` | `sleep.json` |
| **HRV** | `client.get_hrv_data(date_str)` | `hrv.json` |
| **Stress & Body Battery** | `client.get_stress_data(date_str)` | `stress.json` |
| **Daily Summary** | `client.get_user_summary(date_str)` | `daily_summary.json` |
| **VO2 Max / Metrics** | `client.get_max_metrics(date_str)` | `max_metrics.json` |
| **Activity Metadata** | `client.get_activities(0, limit)` | `activity_<ID>_summary.json` |
| **Activity FIT Binary** | `client.download_activity(id, dl_fmt=FIT)` | `activity_<ID>.zip` |

---

## 4. Error Handling & Resilience
- **Individual Endpoint Failures**: An error in one endpoint (e.g. user has no sleep recorded for a given date) must NOT terminate the entire pipeline. The error must be caught, logged at WARNING level, and the pipeline must proceed with the remaining metrics.
- **Idempotency**: Repeated runs on the same date will overwrite or update the JSON payloads without creating duplicate directories. For binary `.fit` files, the ingestor checks for file existence before downloading again to conserve bandwidth and respect rate limits.

---

## 5. Verification & Testing Requirements
- Unit tests (`tests/unit/test_garmin_client.py`) must mock the Garmin API to test:
  1. Successful token-based authentication.
  2. Fallback to credential login when tokens are missing.
  3. Proper payload handling and file creation in `data/raw/YYYY-MM-DD/`.
  4. Error handling when credentials are missing or endpoints raise HTTP errors.
  5. Correct handling of binary `.fit` zip downloads.
