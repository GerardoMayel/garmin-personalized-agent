"""Unit tests for the sync_pipeline CLI entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ingestion.sync_pipeline import parse_args, run_pipeline


class TestSyncPipelineCLI:
    """Test suite for CLI argument parsing and execution in sync_pipeline."""

    def test_parse_args_defaults(self):
        """Should parse default arguments when none are provided."""
        args = parse_args([])
        assert args.days_back == 2
        assert args.date is None
        assert args.no_fit is False
        assert args.raw_dir == "data/raw"
        assert args.db_path is None

    def test_parse_args_custom(self):
        """Should parse custom arguments correctly."""
        args = parse_args(
            [
                "--days-back",
                "5",
                "--date",
                "2026-09-14",
                "--no-fit",
                "--raw-dir",
                "/tmp/raw",
                "--db-path",
                "/tmp/test.db",
            ]
        )
        assert args.days_back == 5
        assert args.date == "2026-09-14"
        assert args.no_fit is True
        assert args.raw_dir == "/tmp/raw"
        assert args.db_path == "/tmp/test.db"

    @patch("src.ingestion.sync_pipeline.GarminDataIngestor")
    @patch("src.ingestion.sync_pipeline.GarminDatabase")
    def test_run_pipeline_window_mode(self, mock_db_cls, mock_ingestor_cls):
        """Should invoke run_sync_window when no specific date is passed."""
        mock_ingestor = MagicMock()
        mock_ingestor_cls.return_value = mock_ingestor

        exit_code = run_pipeline(days_back=2, target_date=None, sync_fit=True)

        assert exit_code == 0
        mock_ingestor.run_sync_window.assert_called_once_with(days_back=2, sync_fit=True)

    @patch("src.ingestion.sync_pipeline.GarminDataIngestor")
    @patch("src.ingestion.sync_pipeline.GarminDatabase")
    def test_run_pipeline_specific_date_mode(self, mock_db_cls, mock_ingestor_cls):
        """Should invoke sync_daily_biometrics with target date and sync_activities."""
        mock_ingestor = MagicMock()
        mock_ingestor_cls.return_value = mock_ingestor

        exit_code = run_pipeline(target_date="2026-09-14", sync_fit=False)

        assert exit_code == 0
        mock_ingestor.sync_daily_biometrics.assert_called_once()
        mock_ingestor.sync_activities.assert_called_once_with(limit=10, download_fit=False)

    @patch("src.ingestion.sync_pipeline.GarminDataIngestor")
    @patch("src.ingestion.sync_pipeline.GarminDatabase")
    def test_run_pipeline_error_handling(self, mock_db_cls, mock_ingestor_cls):
        """Should return exit code 1 if an unhandled exception occurs."""
        mock_ingestor_cls.side_effect = RuntimeError("Fatal connection error")

        exit_code = run_pipeline(days_back=1)

        assert exit_code == 1
