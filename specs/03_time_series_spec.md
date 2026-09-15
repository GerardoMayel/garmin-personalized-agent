# Specification 03: Time Series & Classical ML Specification

## 1. Scope & Mathematical Formulations
Specifies statistical time-series decomposition, baseline calculations, anomaly detection, and Granger causality for recovery vs. load metrics.

## 2. Windows & Baselines
- **Rolling Baselines**: 7-day short-term baseline and 28-day long-term baseline for rMSSD, resting heart rate, and training impulse (TRIMP).
- **Outlier Detection**: Robust Z-scores using median and median absolute deviation (MAD):
  $$\text{Z}_{\text{robust}} = \frac{x_i - \text{median}(X)}{\text{MAD}(X) \times 1.4826}$$
- **Causality Testing**: Granger Causality to test whether acute training load delays impair deep sleep duration or nocturnal HRV 24h-48h post-effort.
