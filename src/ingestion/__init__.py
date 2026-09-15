"""Ingestion package for Garmin telemetry and health metrics."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "GarminDataIngestor":
        from src.ingestion.garmin_client import GarminDataIngestor

        return GarminDataIngestor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GarminDataIngestor"]
