"""Physiological Anomaly Detection for Garmin Biometrics.

Utilizes unsupervised machine learning (Isolation Forest and Local Outlier Factor)
combined with domain heuristics to detect physiological anomalies, fatigue spikes,
and overtraining/illness indicators.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ANOMALY_FEATURES = [
    "resting_heart_rate",
    "hrv_rmssd",
    "daily_avg_stress",
    "sleep_score",
    "total_steps",
]


class AnomalySeverity(StrEnum):
    """Categorical severity of a detected physiological anomaly."""

    NORMAL = "NORMAL"
    MILD_ANOMALY = "MILD_ANOMALY"
    ACUTE_ALERT = "ACUTE_ALERT"


@dataclass
class AnomalyPoint:
    """Individual anomaly evaluation record."""

    calendar_date: str
    is_anomaly: bool
    severity: AnomalySeverity
    isolation_forest_score: float
    lof_score: float
    contributing_factors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly result to dictionary."""
        return {
            "calendar_date": self.calendar_date,
            "is_anomaly": self.is_anomaly,
            "severity": self.severity.value,
            "isolation_forest_score": round(self.isolation_forest_score, 4),
            "lof_score": round(self.lof_score, 4),
            "contributing_factors": self.contributing_factors,
            "metrics": {k: round(v, 2) for k, v in self.metrics.items()},
        }


class PhysiologicalAnomalyDetector:
    """Unsupervised anomaly detector for physiological and biometric time-series."""

    def __init__(
        self,
        features: list[str] | None = None,
        contamination: float = 0.15,
        n_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self.features = features or DEFAULT_ANOMALY_FEATURES
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )
        self.is_fitted = False
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}

    def _prepare_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Validate, extract, and impute available feature columns."""
        available_cols = [c for c in self.features if c in df.columns]
        if not available_cols:
            raise ValueError(f"None of the required features {self.features} found in DataFrame.")

        sub_df = df[available_cols].copy()
        # Clean null values using local median / ffill
        sub_df = sub_df.ffill().bfill().fillna(sub_df.median())
        return sub_df.to_numpy(dtype=float), available_cols

    def fit(self, df: pd.DataFrame) -> PhysiologicalAnomalyDetector:
        """Fit scaler and unsupervised anomaly detection models."""
        X, cols = self._prepare_matrix(df)
        self.features = cols

        # Store baseline population statistics
        for i, col in enumerate(cols):
            self._feature_means[col] = float(np.mean(X[:, i]))
            self._feature_stds[col] = float(np.std(X[:, i])) + 1e-6

        X_scaled = self.scaler.fit_transform(X)

        # Adjust LOF n_neighbors if sample size is small
        effective_neighbors = min(self.n_neighbors, max(2, len(df) - 1))
        self.lof = LocalOutlierFactor(
            n_neighbors=effective_neighbors,
            contamination=self.contamination,
            novelty=True,
        )

        self.iforest.fit(X_scaled)
        self.lof.fit(X_scaled)
        self.is_fitted = True
        logger.info(
            f"PhysiologicalAnomalyDetector trained on {len(df)} samples across {len(cols)} features."
        )
        return self

    def detect(self, df: pd.DataFrame) -> list[AnomalyPoint]:
        """Detect anomalies in the provided DataFrame."""
        if not self.is_fitted:
            self.fit(df)

        X, cols = self._prepare_matrix(df)
        X_scaled = self.scaler.transform(X)

        if_preds = self.iforest.predict(X_scaled)  # -1 for anomaly, 1 for inlier
        if_scores = self.iforest.decision_function(X_scaled)

        lof_preds = self.lof.predict(X_scaled)  # -1 for anomaly, 1 for inlier
        lof_scores = self.lof.decision_function(X_scaled)

        dates = (
            df["calendar_date"].astype(str).tolist()
            if "calendar_date" in df.columns
            else [f"row_{i}" for i in range(len(df))]
        )

        results: list[AnomalyPoint] = []

        for idx in range(len(df)):
            is_if_anomaly = if_preds[idx] == -1
            is_lof_anomaly = lof_preds[idx] == -1

            # Extract row metrics
            row_metrics = {cols[j]: float(X[idx, j]) for j in range(len(cols))}

            # Determine contributing factors based on Z-score deviation
            factors: list[str] = []
            z_scores: dict[str, float] = {}
            for col in cols:
                mean = self._feature_means.get(col, 0.0)
                std = self._feature_stds.get(col, 1.0)
                z = (row_metrics[col] - mean) / std
                z_scores[col] = z

                # Physiological rules
                if col == "hrv_rmssd" and z < -1.5:
                    factors.append(
                        f"HRV significantly suppressed ({row_metrics[col]:.1f} ms, Z={z:.1f})"
                    )
                elif col == "resting_heart_rate" and z > 1.5:
                    factors.append(f"Elevated Resting HR ({row_metrics[col]:.0f} bpm, Z=+{z:.1f})")
                elif col == "daily_avg_stress" and z > 1.5:
                    factors.append(
                        f"High autonomic stress score ({row_metrics[col]:.0f}, Z=+{z:.1f})"
                    )
                elif col == "sleep_score" and z < -1.5:
                    factors.append(
                        f"Poor sleep restoration score ({row_metrics[col]:.0f}, Z={z:.1f})"
                    )
                elif col == "total_steps" and abs(z) > 2.0:
                    status = "Spike" if z > 0 else "Drop"
                    factors.append(
                        f"Extreme activity {status} ({row_metrics[col]:.0f} steps, Z={z:+.1f})"
                    )

            # Determine severity
            if (is_if_anomaly and is_lof_anomaly) or len(factors) >= 2:
                severity = AnomalySeverity.ACUTE_ALERT
                is_anomaly = True
            elif is_if_anomaly or is_lof_anomaly or len(factors) == 1:
                severity = AnomalySeverity.MILD_ANOMALY
                is_anomaly = True
            else:
                severity = AnomalySeverity.NORMAL
                is_anomaly = False

            results.append(
                AnomalyPoint(
                    calendar_date=dates[idx],
                    is_anomaly=is_anomaly,
                    severity=severity,
                    isolation_forest_score=float(if_scores[idx]),
                    lof_score=float(lof_scores[idx]),
                    contributing_factors=factors,
                    metrics=row_metrics,
                )
            )

        return results

    def to_dataframe(self, results: list[AnomalyPoint]) -> pd.DataFrame:
        """Convert list of AnomalyPoints to tabular DataFrame."""
        rows = [r.to_dict() for r in results]
        return pd.DataFrame(rows)


def main() -> None:
    """CLI execution for anomaly detection."""
    parser = argparse.ArgumentParser(description="Garmin Physiological Anomaly Detection.")
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("data/processed/garmin_ml_features.parquet"),
        help="Path to engineered features dataset (parquet or csv).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.15,
        help="Expected proportion of outliers in dataset.",
    )
    args = parser.parse_args()

    file_path: Path = args.features_file
    if not file_path.exists():
        csv_fallback = file_path.with_suffix(".csv")
        if csv_fallback.exists():
            file_path = csv_fallback
        else:
            print(f"Error: Features file not found at {file_path}")
            sys.exit(1)

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    detector = PhysiologicalAnomalyDetector(contamination=args.contamination)
    results = detector.detect(df)
    res_df = detector.to_dataframe(results)

    anomalies = res_df[res_df["is_anomaly"]]
    print("\n" + "=" * 65)
    print("🚨 Garmin Physiological Anomaly Detection Report")
    print("=" * 65)
    print(f"Total dates evaluated: {len(res_df)}")
    print(f"Detected anomalies   : {len(anomalies)}")
    print("-" * 65)
    for _, row in anomalies.iterrows():
        print(f"📅 Date: {row['calendar_date']} | Severity: {row['severity']}")
        print(
            f"   IF Score: {row['isolation_forest_score']:.3f} | LOF Score: {row['lof_score']:.3f}"
        )
        for factor in row["contributing_factors"]:
            print(f"   • {factor}")
        print()
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
