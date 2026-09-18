"""Unit tests for centralized logging system."""

from __future__ import annotations

from pathlib import Path

from src.common.logger import configure_logging, get_logger


def test_get_logger_creation(tmp_path: Path):
    """Should return logger instance that writes to log file."""
    log_dir = tmp_path / "logs"
    configure_logging(
        log_level="DEBUG",
        log_dir=str(log_dir),
        log_filename="test.log",
        force_reconfigure=True,
        enqueue=False,
    )

    log = get_logger("unit_test_module")
    log.info("Test message from unit test")

    from loguru import logger

    logger.complete()

    log_file = log_dir / "test.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test message from unit test" in content
