"""Unit tests for Cloudflare R2 storage client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from src.common.r2_storage import R2StorageClient


class TestR2StorageClient:
    """Test suite for Cloudflare R2 storage client."""

    def test_is_configured_false_when_missing_credentials(self, monkeypatch):
        """Should return False when credentials are not configured."""
        monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)

        client = R2StorageClient(access_key_id=None, secret_access_key=None)
        assert client.is_configured() is False

        ok, msg = client.test_connection()
        assert ok is False
        assert "Credenciales incompletas" in msg

    def test_is_configured_true_with_mock_client(self):
        """Should return True when client and credentials are provided."""
        mock_s3 = MagicMock()
        client = R2StorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            endpoint_url="https://mock.r2.cloudflarestorage.com",
            s3_client=mock_s3,
        )
        assert client.is_configured() is True

    def test_test_connection_success(self):
        """Should report success when head_bucket succeeds."""
        mock_s3 = MagicMock()
        client = R2StorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            endpoint_url="https://mock.r2.cloudflarestorage.com",
            s3_client=mock_s3,
        )

        ok, msg = client.test_connection()
        assert ok is True
        assert "Conexión exitosa" in msg
        mock_s3.head_bucket.assert_called_once_with(Bucket="garmin-personal-data")

    def test_test_connection_client_error(self):
        """Should report failure when head_bucket raises ClientError."""
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadBucket",
        )
        client = R2StorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            endpoint_url="https://mock.r2.cloudflarestorage.com",
            s3_client=mock_s3,
        )

        ok, msg = client.test_connection()
        assert ok is False
        assert "403" in msg

    def test_upload_and_download_file(self, tmp_path: Path):
        """Should upload local file and download remote object."""
        mock_s3 = MagicMock()
        client = R2StorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            endpoint_url="https://mock.r2.cloudflarestorage.com",
            s3_client=mock_s3,
        )

        local_file = tmp_path / "test.db"
        local_file.write_bytes(b"SQLITE_TEST_CONTENT")

        # Test upload
        uploaded = client.upload_file(local_file, "processed/test.db")
        assert uploaded is True
        mock_s3.upload_file.assert_called_once_with(
            str(local_file), "garmin-personal-data", "processed/test.db"
        )

        # Test download
        dest_file = tmp_path / "restored.db"
        downloaded = client.download_file("processed/test.db", dest_file)
        assert downloaded is True
        mock_s3.download_file.assert_called_once_with(
            "garmin-personal-data", "processed/test.db", str(dest_file)
        )

    def test_sync_raw_directory(self, tmp_path: Path):
        """Should traverse and upload valid raw partition files."""
        mock_s3 = MagicMock()
        client = R2StorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            endpoint_url="https://mock.r2.cloudflarestorage.com",
            s3_client=mock_s3,
        )

        raw_dir = tmp_path / "raw" / "2026-09-18"
        raw_dir.mkdir(parents=True)
        (raw_dir / "sleep.json").write_text("{}", encoding="utf-8")
        (raw_dir / "hrv.json").write_text("{}", encoding="utf-8")
        (raw_dir / ".gitkeep").write_text("", encoding="utf-8")  # ignored

        stats = client.sync_raw_directory(raw_dir=tmp_path / "raw")
        assert stats["uploaded"] == 2
        assert stats["failed"] == 0
