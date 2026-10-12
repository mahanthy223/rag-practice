from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from rag_practice.chunking import split_documents
from rag_practice.embeddings import EmbeddingModel
from rag_practice.settings import Settings
from rag_practice.vector_store import VectorStore

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_documents(input_directory: Path) -> list[Document]:
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {input_directory}")

    source_paths = sorted(
        path
        for path in input_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not source_paths:
        raise FileNotFoundError(
            f"No PDF or TXT files found in {input_directory} (including subfolders)."
        )

    documents: list[Document] = []
    for path in source_paths:
        if path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        else:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return documents


def ingest(settings: Settings) -> int:
    documents = load_documents(settings.documents_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        logger.warning("Source files contained no text to index.")
        return 0

    embeddings = EmbeddingModel(settings.embedding_model).encode(
        [chunk.page_content for chunk in chunks]
    )
    store = VectorStore(settings.vector_store_dir, settings.collection_name)
    stored = store.upsert_documents(chunks, embeddings)
    logger.info(
        "Indexed %d chunks from %d loaded document pages/files.",
        stored,
        len(documents),
    )
    return stored
