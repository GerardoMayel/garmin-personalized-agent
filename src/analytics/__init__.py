"""Analytics & Machine Learning Package for Garmin Telemetry."""

from src.analytics.anomaly_detection import (
    AnomalyPoint,
    AnomalySeverity,
    PhysiologicalAnomalyDetector,
)
from src.analytics.causality_engine import (
    CausalityTestResult,
    CrossCorrelationResult,
    PhysiologicalCausalityEngine,
)
from src.analytics.dvc_manager import GarminDVCManager
from src.analytics.predictions_manager import (
    BiometricPredictionsManager,
    get_biweekly_target_dates,
)
from src.analytics.time_series_models import (
    EnsembleBiometricForecaster,
    ForecastResult,
    GarminProphetForecaster,
    HoltWintersForecaster,
    SleepRecoveryPredictor,
    compute_physiological_bounds,
)

__all__ = [
    "AnomalyPoint",
    "AnomalySeverity",
    "PhysiologicalAnomalyDetector",
    "ForecastResult",
    "GarminProphetForecaster",
    "HoltWintersForecaster",
    "EnsembleBiometricForecaster",
    "compute_physiological_bounds",
    "SleepRecoveryPredictor",
    "CausalityTestResult",
    "CrossCorrelationResult",
    "PhysiologicalCausalityEngine",
    "GarminDVCManager",
    "BiometricPredictionsManager",
    "get_biweekly_target_dates",
]
