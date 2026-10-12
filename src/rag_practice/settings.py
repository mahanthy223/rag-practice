from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    documents_dir: Path
    vector_store_dir: Path
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    openai_api_key: str | None
    openai_model: str

    @classmethod
    def from_environment(cls) -> Settings:
        root_env = PROJECT_ROOT / ".env"
        legacy_env = PROJECT_ROOT / "Document_Ingestion" / ".env"
        env_file = root_env if root_env.exists() else legacy_env
        if env_file.exists():
            load_dotenv(env_file, override=False)

        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        if chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than zero.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be >= 0 and less than chunk size.")

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        return cls(
            documents_dir=_configured_path("RAG_DOCUMENTS_DIR", "Documents"),
            vector_store_dir=_configured_path(
                "RAG_VECTOR_STORE_DIR", ".data/chroma"
            ),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "pdf_collection"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
