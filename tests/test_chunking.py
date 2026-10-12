import unittest

from langchain_core.documents import Document

from rag_practice.chunking import split_documents


class SplitDocumentsTests(unittest.TestCase):
    def test_splits_text_and_preserves_source_metadata(self) -> None:
        source = Document(
            page_content="alpha beta gamma delta epsilon",
            metadata={"source": "handbook.txt"},
        )
        chunks = split_documents([source], chunk_size=12, chunk_overlap=2)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["source"] == "handbook.txt" for chunk in chunks))

    def test_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            split_documents([], chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
