"""Deployment helper for Hugging Face Spaces.

Uploads the deploy/hf_chroma_space folder to the user's Hugging Face Space
using HF_TOKEN and huggingface_hub API.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def deploy_to_hf_space(space_repo_id: str, token: str | None = None) -> None:
    """Sube el contenido del directorio deploy/hf_chroma_space/ al Space especificado en Hugging Face."""
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN no está configurado en .env ni fue proporcionado como argumento.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub no está instalado. Ejecuta: uv add huggingface_hub")
        sys.exit(1)

    source_dir = Path(__file__).parent.resolve()
    print(f"🚀 Iniciando despliegue de '{source_dir}' hacia el Space: {space_repo_id}...")

    api = HfApi(token=hf_token)

    # Verificar existencia o crear el Space si no existe
    try:
        api.space_info(repo_id=space_repo_id)
        print(f"✅ Space '{space_repo_id}' encontrado.")
    except Exception:
        print(
            f"ℹ️ Space '{space_repo_id}' no encontrado. Creando nuevo Space con SDK Docker (Private)..."
        )
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker",
            private=True,
            exist_ok=True,
        )
        print(f"✅ Space '{space_repo_id}' creado exitosamente.")

    # Subir los archivos del Space (excluyendo deploy_space.py y cachés)
    future = api.upload_folder(
        folder_path=str(source_dir),
        repo_id=space_repo_id,
        repo_type="space",
        ignore_patterns=["deploy_space.py", "__pycache__/*", "*.pyc", ".DS_Store"],
        commit_message="Deploy Garmin ChromaDB FastAPI Backend",
    )
    print(f"🎉 Despliegue completado con éxito: {future}")
    print(f"🔗 URL del Space: https://huggingface.co/spaces/{space_repo_id}")
    print(f"🌐 Endpoint directo de la API: https://{space_repo_id.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Despliega el backend Chroma a un Space de Hugging Face."
    )
    parser.add_argument(
        "--space",
        type=str,
        required=True,
        help="Identificador del Space (ej. 'GerardoMayel/garmin-chroma-backend')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face (opcional, lee HF_TOKEN de .env por defecto)",
    )
    args = parser.parse_args()
    deploy_to_hf_space(space_repo_id=args.space, token=args.token)
