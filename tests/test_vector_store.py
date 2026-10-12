import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
from langchain_core.documents import Document

from rag_practice.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = chromadb.EphemeralClient()
        self.collection_name = f"test_{uuid.uuid4().hex}"
        self.client_patch = patch(
            "rag_practice.vector_store.chromadb.PersistentClient",
            return_value=self.client,
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.addCleanup(self.client.delete_collection, self.collection_name)

    def test_upsert_is_repeatable_and_search_returns_cosine_score(self) -> None:
        store = VectorStore(Path("."), self.collection_name)
        documents = [
            Document(
                page_content="paid annual leave",
                metadata={"source": "handbook.txt", "page": 1},
            )
        ]
        embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

        self.assertEqual(store.upsert_documents(documents, embeddings), 1)
        self.assertEqual(store.upsert_documents(documents, embeddings), 1)
        results = store.search(embeddings[0], top_k=5, min_score=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "paid annual leave")
        self.assertEqual(results[0]["metadata"]["page"], 1)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)

    def test_rejects_mismatched_embedding_count(self) -> None:
        store = VectorStore(Path("."), self.collection_name)
        with self.assertRaises(ValueError):
            store.upsert_documents(
                [Document(page_content="text", metadata={})],
                np.empty((0, 3)),
            )

    def test_reingestion_removes_old_chunks_for_same_source(self) -> None:
        store = VectorStore(Path("."), self.collection_name)
        store.upsert_documents(
            [Document(page_content="old content", metadata={"source": "guide.txt"})],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        store.upsert_documents(
            [Document(page_content="updated content", metadata={"source": "guide.txt"})],
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

        results = store.search(
            np.asarray([0.0, 1.0], dtype=np.float32),
            top_k=5,
            min_score=-1.0,
        )
        self.assertEqual([result["content"] for result in results], ["updated content"])


if __name__ == "__main__":
    unittest.main()
