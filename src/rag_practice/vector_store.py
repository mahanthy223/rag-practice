from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Indexed source document chunks",
                "hnsw:space": "cosine",
            },
        )
        if (self._collection.metadata or {}).get("hnsw:space") != "cosine":
            raise ValueError(
                f"Collection {collection_name!r} does not use cosine distance. "
                "Choose a new collection name or rebuild the index."
            )

    def upsert_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ) -> int:
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts must match.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not documents:
            return 0

        source_paths = {
            str(document.metadata["source"])
            for document in documents
            if document.metadata.get("source")
        }
        current_ids: set[str] = set()
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            ids = [_document_id(document) for document in batch_docs]
            current_ids.update(ids)
            metadatas = [_chroma_metadata(document) for document in batch_docs]
            self._collection.upsert(
                ids=ids,
                documents=[document.page_content for document in batch_docs],
                metadatas=metadatas,
                embeddings=batch_embeddings.tolist(),
            )

        if source_paths:
            existing = self._collection.get(
                where={"source": {"$in": sorted(source_paths)}},
                include=[],
            )
            stale_ids = [
                doc_id for doc_id in existing["ids"] if doc_id not in current_ids
            ]
            if stale_ids:
                self._collection.delete(ids=stale_ids)
        return len(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1.")
        count = self._collection.count()
        if count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0] or []
        metadatas = result["metadatas"][0] or []
        distances = result["distances"][0] or []
        ids = result["ids"][0] or []

        matches: list[dict[str, Any]] = []
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            score = 1.0 - float(distance)
            if score >= min_score:
                matches.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": score,
                    }
                )
        return matches


def _document_id(document: Document) -> str:
    source = str(document.metadata.get("source", "unknown"))
    page = str(document.metadata.get("page", ""))
    start_index = str(document.metadata.get("start_index", ""))
    digest = hashlib.sha256(
        f"{source}\0{page}\0{start_index}\0{document.page_content}".encode("utf-8")
    ).hexdigest()
    return digest


def _chroma_metadata(document: Document) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in document.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = str(value)
    return metadata
