"""Physiological Causality and Temporal Impact Engine.

Evaluates directional influence and time-lagged relationships between daytime
exertion/stress behaviors and subsequent physiological recovery metrics using:
1. Granger Causality Tests (VAR-based F-test).
2. Cross-Correlation Function (CCF) across multi-day lags.
3. Natural language physiological insight generation.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CausalityTestResult:
    """Outcome of a Granger causality hypothesis test."""

    cause_variable: str
    effect_variable: str
    max_lag: int
    optimal_lag: int
    p_value: float
    f_statistic: float
    is_significant: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "cause": self.cause_variable,
            "effect": self.effect_variable,
            "optimal_lag_days": self.optimal_lag,
            "p_value": round(self.p_value, 4),
            "f_statistic": round(self.f_statistic, 3),
            "is_significant": self.is_significant,
            "interpretation": self.interpretation,
        }


@dataclass
class CrossCorrelationResult:
    """Cross-correlation function (CCF) analysis across temporal lags."""

    var1: str
    var2: str
    lags: list[int]
    correlations: list[float]
    peak_lag: int
    peak_correlation: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize CCF result to dictionary."""
        return {
            "var1": self.var1,
            "var2": self.var2,
            "peak_lag": self.peak_lag,
            "peak_correlation": round(self.peak_correlation, 3),
            "lag_curve": {
                lag: round(corr, 3) for lag, corr in zip(self.lags, self.correlations, strict=False)
            },
        }


class PhysiologicalCausalityEngine:
    """Engine for statistical causality testing and lagged interaction analysis."""

    STANDARD_PAIRS: list[tuple[str, str, str]] = [
        ("daily_avg_stress", "hrv_rmssd", "Daytime stress impacting overnight HRV recovery"),
        ("total_steps", "resting_heart_rate", "Daily activity volume driving next-day resting HR"),
        ("daily_avg_stress", "sleep_score", "Daytime stress suppressing sleep score"),
        ("active_kilocalories", "deep_sleep_ratio", "Energy expenditure promoting deep sleep"),
        ("vigorous_minutes", "hrv_rmssd", "High intensity training suppressing next-day HRV"),
    ]

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    @staticmethod
    def test_stationarity(series: pd.Series) -> tuple[bool, float]:
        """Test series stationarity using Augmented Dickey-Fuller (ADF)."""
        clean = series.dropna().to_numpy(dtype=float)
        if len(clean) < 8 or np.all(clean == clean[0]):
            return True, 0.0

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(clean, autolag="AIC")
            p_value = float(result[1])
            is_stationary = p_value < 0.05
            return is_stationary, p_value
        except Exception:
            return True, 0.0

    def compute_cross_correlation(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        max_lags: int = 3,
    ) -> CrossCorrelationResult:
        """Compute normalized cross-correlation across forward and backward lags."""
        if var1 not in df.columns or var2 not in df.columns:
            raise ValueError(f"Variables '{var1}' and/or '{var2}' not found in DataFrame.")

        s1 = df[var1].ffill().bfill().to_numpy(dtype=float)
        s2 = df[var2].ffill().bfill().to_numpy(dtype=float)

        s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-6)
        s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-6)

        lags = list(range(-max_lags, max_lags + 1))
        correlations: list[float] = []

        n = len(s1_norm)
        for lag in lags:
            if lag < 0:
                corr = np.mean(s1_norm[:lag] * s2_norm[-lag:]) if n > abs(lag) else 0.0
            elif lag > 0:
                corr = np.mean(s1_norm[lag:] * s2_norm[:-lag]) if n > lag else 0.0
            else:
                corr = np.mean(s1_norm * s2_norm)
            correlations.append(float(corr))

        peak_idx = int(np.argmax(np.abs(correlations)))
        peak_lag = lags[peak_idx]
        peak_corr = correlations[peak_idx]

        return CrossCorrelationResult(
            var1=var1,
            var2=var2,
            lags=lags,
            correlations=correlations,
            peak_lag=peak_lag,
            peak_correlation=peak_corr,
        )

    def test_granger_causality(
        self,
        df: pd.DataFrame,
        cause_col: str,
        effect_col: str,
        max_lag: int = 2,
    ) -> CausalityTestResult:
        """Evaluate if cause_col Granger-causes effect_col."""
        if cause_col not in df.columns or effect_col not in df.columns:
            raise ValueError(f"Variables '{cause_col}' and '{effect_col}' must be in DataFrame.")

        sub_df = df[[effect_col, cause_col]].ffill().bfill().copy()

        # Granger tests require non-constant data
        if sub_df[cause_col].std() == 0 or sub_df[effect_col].std() == 0:
            return CausalityTestResult(
                cause_variable=cause_col,
                effect_variable=effect_col,
                max_lag=max_lag,
                optimal_lag=1,
                p_value=1.0,
                f_statistic=0.0,
                is_significant=False,
                interpretation="Insufficient variance in variable to perform Granger test.",
            )

        # Enforce maximum lag bound for small sample sets
        effective_lag = min(max_lag, max(1, (len(sub_df) - 2) // 3))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_res = grangercausalitytests(sub_df, maxlag=effective_lag)

            min_p = 1.0
            best_lag = 1
            best_f = 0.0

            for lag in range(1, effective_lag + 1):
                f_test = gc_res[lag][0]["ssr_ftest"]
                f_stat, p_val = float(f_test[0]), float(f_test[1])
                if p_val < min_p:
                    min_p = p_val
                    best_lag = lag
                    best_f = f_stat

            is_sig = min_p < self.significance_level

            # Generate natural language physiological interpretation
            if is_sig:
                interpretation = (
                    f"Statistically significant directional causality detected: past values of "
                    f"'{cause_col}' precede changes in '{effect_col}' with a {best_lag}-day lag "
                    f"(F={best_f:.2f}, p={min_p:.3f})."
                )
            else:
                interpretation = (
                    f"No statistically significant Granger causality found between '{cause_col}' "
                    f"and '{effect_col}' at current sample size (min p={min_p:.3f})."
                )

            return CausalityTestResult(
                cause_variable=cause_col,
                effect_variable=effect_col,
                max_lag=effective_lag,
                optimal_lag=best_lag,
                p_value=min_p,
                f_statistic=best_f,
                is_significant=is_sig,
                interpretation=interpretation,
            )

        except Exception as e:
            logger.warning(f"Granger causality error for {cause_col} -> {effect_col}: {e}")
            return CausalityTestResult(
                cause_variable=cause_col,
                effect_variable=effect_col,
                max_lag=effective_lag,
                optimal_lag=1,
                p_value=1.0,
                f_statistic=0.0,
                is_significant=False,
                interpretation=f"Could not compute test due to mathematical singularity: {e}",
            )

    def run_comprehensive_audit(
        self,
        df: pd.DataFrame,
        max_lag: int = 2,
    ) -> list[CausalityTestResult]:
        """Audit all pre-configured physiological hypothesis pairs."""
        results: list[CausalityTestResult] = []
        for cause, effect, _ in self.STANDARD_PAIRS:
            if cause in df.columns and effect in df.columns:
                res = self.test_granger_causality(df, cause, effect, max_lag=max_lag)
                results.append(res)
        return results


def main() -> None:
    """CLI runner for physiological causality analysis."""
    parser = argparse.ArgumentParser(description="Garmin Physiological Causality & Impact Engine.")
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("data/processed/garmin_ml_features.parquet"),
        help="Path to engineered features dataset.",
    )
    parser.add_argument("--cause", type=str, default=None, help="Specific cause variable.")
    parser.add_argument("--effect", type=str, default=None, help="Specific effect variable.")
    parser.add_argument("--max-lag", type=int, default=2, help="Maximum lag in days.")

    args = parser.parse_args()
    file_path: Path = args.features_file

    if not file_path.exists():
        csv_path = file_path.with_suffix(".csv")
        if csv_path.exists():
            file_path = csv_path
        else:
            print(f"Error: features file not found: {file_path}")
            sys.exit(1)

    df = pd.read_parquet(file_path) if file_path.suffix == ".parquet" else pd.read_csv(file_path)
    engine = PhysiologicalCausalityEngine()

    print("\n" + "=" * 65)
    print("🔬 Garmin Physiological Causality & Temporal Impact Engine")
    print("=" * 65)

    if args.cause and args.effect:
        res = engine.test_granger_causality(df, args.cause, args.effect, max_lag=args.max_lag)
        ccf = engine.compute_cross_correlation(df, args.cause, args.effect, max_lags=3)

        print(f"Hypothesis: Does [{res.cause_variable}] cause [{res.effect_variable}]?")
        print(f"  • P-Value      : {res.p_value:.4f}")
        print(f"  • F-Statistic  : {res.f_statistic:.2f}")
        print(f"  • Optimal Lag  : {res.optimal_lag} day(s)")
        print(f"  • Significant  : {'✅ YES' if res.is_significant else '❌ NO'}")
        print(f"  • Peak CCF     : {ccf.peak_correlation:+.3f} at lag {ccf.peak_lag} day(s)")
        print(f"  • Insight      : {res.interpretation}")
    else:
        results = engine.run_comprehensive_audit(df, max_lag=args.max_lag)
        for res in results:
            ccf = engine.compute_cross_correlation(df, res.cause_variable, res.effect_variable)
            status_icon = "🟢" if res.is_significant else "⚪"
            print(f"\n{status_icon} [{res.cause_variable}] ➔ [{res.effect_variable}]")
            print(
                f"   p={res.p_value:.3f} | F={res.f_statistic:.2f} | Peak Corr={ccf.peak_correlation:+.2f} (Lag {ccf.peak_lag}d)"
            )
            print(f"   {res.interpretation}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
