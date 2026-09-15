"""Centralized Logging Configuration for Garmin Personal Insight Agent.

Provides structured, colorized console logging and asynchronous rotating file
logging using Loguru and standard library logging integration.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_IS_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """Intercept standard library logging messages and redirect to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(
    log_level: Optional[str] = None,
    log_dir: str = "logs",
    log_filename: str = "garmin_agent.log",
    rotation: str = "10 MB",
    retention: str = "14 days",
    force_reconfigure: bool = False,
) -> None:
    """Initialize console and file loggers across the application."""
    global _IS_CONFIGURED
    if _IS_CONFIGURED and not force_reconfigure:
        return

    level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        level = "INFO"

    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    full_log_file = log_path / log_filename

    # Reset loguru handlers
    logger.remove()

    # 1. Console handler
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=level,
        format=console_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 2. Rotating file handler
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    logger.add(
        str(full_log_file),
        level=level,
        format=file_format,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
        encoding="utf-8",
    )

    # Intercept standard library logging (e.g. garminconnect, urllib3, requests)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    _IS_CONFIGURED = True
    logger.debug(f"Logging initialized at level={level}, destination={full_log_file}")


def get_logger(name: Optional[str] = None):
    """Retrieve a configured logger instance bound with module name."""
    if not _IS_CONFIGURED:
        configure_logging()
    if name:
        return logger.bind(module=name)
    return logger


__all__ = ["configure_logging", "get_logger", "logger"]
