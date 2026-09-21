"""Cloudflare R2 Object Storage Manager.

Provides S3-compatible cloud persistence for Garmin raw partitions and
processed historical SQLite databases via Cloudflare R2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from dotenv import load_dotenv

from src.common.logger import get_logger

load_dotenv()
logger = get_logger("R2Storage")

DEFAULT_DB_PATH = Path("data/processed/garmin_history.db")
DEFAULT_RAW_DIR = Path("data/raw")


class R2StorageClient:
    """S3-compatible client for Cloudflare R2 bucket operations."""

    def __init__(
        self,
        account_id: str | None = None,
        bucket_name: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self.account_id = account_id or os.getenv("R2_ACCOUNT_ID")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME") or "garmin-personal-data"

        computed_endpoint = (
            f"https://{self.account_id}.r2.cloudflarestorage.com" if self.account_id else None
        )
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL") or computed_endpoint
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")

        if s3_client is not None:
            self._client = s3_client
        else:
            self._client = self._init_s3_client()

    def _init_s3_client(self) -> Any | None:
        """Initializes boto3 S3 client configured for Cloudflare R2."""
        if not self.access_key_id or not self.secret_access_key:
            logger.warning(
                "R2 credentials incomplete: R2_ACCESS_KEY_ID and/or R2_SECRET_ACCESS_KEY missing."
            )
            return None

        if not self.endpoint_url:
            logger.warning("R2 endpoint URL not configured.")
            return None

        cfg = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        )
        return boto3.client(
            service_name="s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
            config=cfg,
        )

    def is_configured(self) -> bool:
        """Check if all required R2 configuration and credentials are provided."""
        return bool(
            self.endpoint_url
            and self.bucket_name
            and self.access_key_id
            and self.secret_access_key
            and self._client is not None
        )

    def test_connection(self) -> tuple[bool, str]:
        """Test authentication and bucket connectivity to Cloudflare R2."""
        if not self.is_configured():
            missing = []
            if not self.access_key_id:
                missing.append("R2_ACCESS_KEY_ID")
            if not self.secret_access_key:
                missing.append("R2_SECRET_ACCESS_KEY")
            return False, f"Credenciales incompletas en .env: faltan {', '.join(missing)}"

        try:
            assert self._client is not None
            self._client.head_bucket(Bucket=self.bucket_name)
            return True, f"Conexión exitosa al bucket R2 '{self.bucket_name}'."
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            logger.error(f"Error de cliente R2 ({error_code}): {error_msg}")
            return False, f"Error de autenticación/permisos en R2: {error_code} - {error_msg}"
        except EndpointConnectionError as e:
            logger.error(f"Error conectando al endpoint R2: {e}")
            return False, f"Error de conexión/SSL al endpoint R2: {e}"
        except Exception as e:
            logger.error(f"Error inesperado probando conexión R2: {e}")
            return False, f"Error inesperado: {e}"

    def upload_file(self, local_path: str | Path, remote_key: str) -> bool:
        """Upload a local file to Cloudflare R2."""
        if not self.is_configured():
            logger.error("No se puede subir archivo: R2 no está configurado.")
            return False

        path_obj = Path(local_path)
        if not path_obj.is_file():
            logger.error(f"Archivo local no existe: {path_obj}")
            return False

        try:
            assert self._client is not None
            logger.info(f"Subiendo {path_obj} -> r2://{self.bucket_name}/{remote_key}")
            self._client.upload_file(str(path_obj), self.bucket_name, remote_key)
            logger.info(f"Subida exitosa: r2://{self.bucket_name}/{remote_key}")
            return True
        except Exception as e:
            logger.error(f"Fallo al subir {path_obj} a R2: {e}")
            return False

    def download_file(self, remote_key: str, local_path: str | Path) -> bool:
        """Download an object from Cloudflare R2 to a local file."""
        if not self.is_configured():
            logger.error("No se puede descargar archivo: R2 no está configurado.")
            return False

        dest_path = Path(local_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            assert self._client is not None
            logger.info(f"Descargando r2://{self.bucket_name}/{remote_key} -> {dest_path}")
            self._client.download_file(self.bucket_name, remote_key, str(dest_path))
            logger.info(f"Descarga completada: {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Fallo al descargar {remote_key} desde R2: {e}")
            return False

    def backup_database(
        self,
        local_db_path: str | Path = DEFAULT_DB_PATH,
        remote_key: str = "processed/garmin_history.db",
    ) -> bool:
        """Upload SQLite database to Cloudflare R2."""
        return self.upload_file(local_db_path, remote_key)

    def restore_database(
        self,
        local_db_path: str | Path = DEFAULT_DB_PATH,
        remote_key: str = "processed/garmin_history.db",
    ) -> bool:
        """Restore SQLite database from Cloudflare R2."""
        return self.download_file(remote_key, local_db_path)

    def sync_raw_directory(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        remote_prefix: str = "raw_data",
    ) -> dict[str, int]:
        """Upload all partitioned JSON and .fit files from data/raw/ to R2."""
        stats = {"uploaded": 0, "failed": 0, "skipped": 0}
        root_path = Path(raw_dir)

        if not root_path.exists():
            logger.warning(f"Directorio raw no existe: {root_path}")
            return stats

        for file_path in root_path.rglob("*"):
            if not file_path.is_file() or file_path.name in {".gitkeep", ".DS_Store"}:
                continue

            rel_path = file_path.relative_to(root_path)
            remote_key = f"{remote_prefix}/{rel_path.as_posix()}"

            if self.upload_file(file_path, remote_key):
                stats["uploaded"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"Sincronización de data/raw/ completada: {stats}")
        return stats

    def restore_raw_directory(
        self,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        remote_prefix: str = "raw_data",
    ) -> dict[str, int]:
        """Download all raw data partitions from Cloudflare R2 into local directory."""
        stats = {"downloaded": 0, "failed": 0, "skipped": 0}
        if not self.is_configured():
            logger.error("R2 no configurado para restaurar raw_data.")
            return stats

        try:
            assert self._client is not None
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=f"{remote_prefix}/")

            dest_root = Path(raw_dir)
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/") or key.endswith(".gitkeep"):
                        continue
                    rel_path = key[len(remote_prefix) + 1 :]
                    local_dest = dest_root / rel_path
                    if local_dest.exists() and local_dest.stat().st_size == obj["Size"]:
                        stats["skipped"] += 1
                        continue
                    if self.download_file(key, local_dest):
                        stats["downloaded"] += 1
                    else:
                        stats["failed"] += 1

            logger.info(f"Restauración de data/raw/ desde R2 completada: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error restaurando data/raw/ desde R2: {e}")
            return stats

    def delete_prefix(self, remote_prefix: str) -> int:
        """Delete all objects matching a remote key prefix in Cloudflare R2."""
        deleted_count = 0
        if not self.is_configured():
            logger.error("R2 no configurado para eliminar objetos.")
            return deleted_count

        try:
            assert self._client is not None
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_prefix)

            for page in pages:
                objects_to_delete = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if objects_to_delete:
                    self._client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={"Objects": objects_to_delete},
                    )
                    deleted_count += len(objects_to_delete)
                    logger.info(f"Eliminados {len(objects_to_delete)} objetos bajo '{remote_prefix}' en R2.")

            return deleted_count
        except Exception as e:
            logger.error(f"Error eliminando prefijo '{remote_prefix}' en R2: {e}")
            return deleted_count


def main() -> None:
    """CLI manager for Cloudflare R2 operations."""
    parser = argparse.ArgumentParser(description="Cloudflare R2 Storage Manager for Garmin Data.")
    parser.add_argument(
        "--test", action="store_true", help="Test connection to Cloudflare R2 bucket"
    )
    parser.add_argument("--backup-db", action="store_true", help="Upload SQLite database to R2")
    parser.add_argument(
        "--restore-db", action="store_true", help="Download SQLite database from R2"
    )
    parser.add_argument("--sync-raw", action="store_true", help="Upload data/raw/ partitions to R2")
    parser.add_argument(
        "--restore-raw",
        action="store_true",
        help="Download raw_data/ partitions from R2 into data/raw/",
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite database"
    )

    args = parser.parse_args()
    client = R2StorageClient()

    if args.test:
        success, msg = client.test_connection()
        print("\n" + "=" * 55)
        print("☁️  Prueba de Conexión a Cloudflare R2")
        print("=" * 55)
        print(f"  • Bucket  : {client.bucket_name}")
        print(f"  • Endpoint: {client.endpoint_url}")
        print(f"  • Estado  : {'✅ ÉXITO' if success else '❌ FALLÓ'}")
        print(f"  • Detalle : {msg}")
        print("=" * 55 + "\n")
        sys.exit(0 if success else 1)

    if args.backup_db:
        ok = client.backup_database(local_db_path=args.db_path)
        sys.exit(0 if ok else 1)

    if args.restore_db:
        ok = client.restore_database(local_db_path=args.db_path)
        sys.exit(0 if ok else 1)

    if args.sync_raw:
        res = client.sync_raw_directory()
        print(f"Resultado de sincronización raw: {res}")
        sys.exit(0 if res["failed"] == 0 else 1)

    if args.restore_raw:
        res = client.restore_raw_directory()
        print(f"Resultado de restauración raw desde R2: {res}")
        sys.exit(0 if res["failed"] == 0 else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
