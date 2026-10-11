import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rag_agent import (
    REFUSAL,
    SQLiteKnowledgeBase,
    answer_question,
    chunk_text,
    cosine_similarity,
)


class RetrievalTests(unittest.TestCase):
    def test_chunk_text_normalizes_and_overlaps(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_text(text, chunk_size=22, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks[index][-5:] in chunks[index + 1] for index in range(len(chunks) - 1)))
        self.assertEqual(chunk_text("  \n  "), [])

    def test_cosine_similarity_handles_zero_and_mismatched_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [2, 0]), 1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)
        self.assertEqual(cosine_similarity([1], [1, 2]), 0.0)

    def test_knowledge_base_replaces_active_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge_base = SQLiteKnowledgeBase(Path(directory) / "knowledge.sqlite3")
            knowledge_base.replace("first.pdf", ["first experience"], [[1.0, 0.0]])
            knowledge_base.replace("second.txt", ["second experience"], [[0.0, 1.0]])

            self.assertEqual(knowledge_base.status(), {"uploaded": True, "chunks": 1, "source": "second.txt"})
            self.assertEqual(knowledge_base.search([0.0, 1.0])[0]["source"], "second.txt")


class AnswerGuardrailTests(unittest.IsolatedAsyncioTestCase):
    async def test_weak_retrieval_refuses_without_calling_generation_model(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge_base = SQLiteKnowledgeBase(Path(directory) / "knowledge.sqlite3")
            knowledge_base.replace("resume.txt", ["Python developer"], [[1.0, 0.0]])
            client = SimpleNamespace(
                embed=AsyncMock(return_value=SimpleNamespace(embeddings=[[0.0, 1.0]])),
                chat=AsyncMock(),
            )

            result = await answer_question(client, knowledge_base, "llama", "embed", "weather tomorrow", 0.5)

        self.assertEqual(result["answer"], REFUSAL)
        self.assertFalse(result["retrieval"]["grounded"])
        client.chat.assert_not_awaited()

    async def test_grounded_answer_returns_tokens_and_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge_base = SQLiteKnowledgeBase(Path(directory) / "knowledge.sqlite3")
            knowledge_base.replace("resume.pdf", ["Built data pipelines"], [[1.0, 0.0]])
            client = SimpleNamespace(
                embed=AsyncMock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]])),
                chat=AsyncMock(
                    return_value=SimpleNamespace(
                        message=SimpleNamespace(content="Built data pipelines [resume.pdf]"),
                        prompt_eval_count=40,
                        eval_count=12,
                        prompt_eval_duration=2_000_000,
                        eval_duration=3_000_000_000,
                    )
                ),
            )

            result = await answer_question(client, knowledge_base, "llama", "embed", "What did I build?", 0.5)

        self.assertTrue(result["retrieval"]["grounded"])
        self.assertEqual(result["usage"], {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52})
        self.assertEqual(result["timings"]["generation_ms"], 3_000.0)
        self.assertEqual(result["timings"]["tokens_per_second"], 4.0)