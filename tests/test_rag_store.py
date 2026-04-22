import tempfile
import unittest
import uuid
from pathlib import Path

from src.gui.rag_store import (
    RagStore,
    build_rag_context,
    chunk_text,
    cosine_similarity,
)


class RagStoreTests(unittest.TestCase):
    def _db_path(self) -> Path:
        root = Path(".test_tmp").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root / f"rag_{uuid.uuid4().hex}.db"

    def test_cosine_similarity_scores_related_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_store_search_returns_top_k_chunks(self) -> None:
        db_path = self._db_path()
        store = RagStore(str(db_path))
        store.replace_source_chunks(
            source_type="message",
            source_id="session-1",
            source_label="Long chat",
            chunks=["about embeddings", "about recipes", "about vector memory"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.0, 0.1],
            ],
        )

        results = store.search([1.0, 0.0, 0.0], top_k=2, min_score=0.1)

        self.assertEqual([result.content for result in results], ["about embeddings", "about vector memory"])
        context = build_rag_context(results)
        self.assertIn("[1] Long chat", context)
        self.assertIn("about vector memory", context)

    def test_chunk_text_uses_overlap(self) -> None:
        chunks = chunk_text("alpha beta gamma delta epsilon", max_chars=16, overlap=5)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("alpha beta", chunks[0])

    def test_clear_source_type_removes_indexed_chunks(self) -> None:
        db_path = self._db_path()
        store = RagStore(str(db_path))
        store.replace_source_chunks(
            source_type="attachment",
            source_id="doc-1",
            source_label="doc.txt",
            chunks=["alpha"],
            embeddings=[[1.0, 0.0]],
        )

        store.clear_source_type("attachment")

        results = store.search([1.0, 0.0], top_k=5, min_score=0.0)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
