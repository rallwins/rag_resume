import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ollama
from fastapi.testclient import TestClient

from app import create_app
from rag_agent import SQLiteKnowledgeBase


class AppTests(unittest.TestCase):
    def test_status_reports_langfuse_sdk_tracing_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")
            with patch.dict(
                "os.environ",
                {
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_SECRET_KEY": "secret",
                    "LANGFUSE_TRACING_ENABLED": "false",
                },
            ):
                with TestClient(app) as client:
                    status = client.get("/api/status").json()

        self.assertFalse(status["tracing"])

    def test_architecture_page_is_served(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")) as client:
                response = client.get("/architecture")

        self.assertEqual(response.status_code, 200)
        self.assertIn("How Resume Room works", response.text)
        self.assertIn("Below threshold · refuse", response.text)
        self.assertIn("Metadata-only tracing", response.text)

    def test_chat_requires_an_uploaded_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")) as client:
                response = client.post("/api/chat", json={"prompt": "Where did I work?"})

        self.assertEqual(response.status_code, 409)
        self.assertIn("Upload a resume", response.json()["detail"])

    def test_upload_indexes_text_resume_and_status_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")
            with patch(
                "app.create_embeddings",
                new_callable=AsyncMock,
                return_value=[[1.0, 0.0]],
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/resume",
                        files={"file": ("resume.txt", b"Senior Python engineer", "text/plain")},
                    )
                    status = client.get("/api/status").json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "resume.txt")
        self.assertEqual(status["knowledge_base"], {"uploaded": True, "chunks": 1, "source": "resume.txt"})

    def test_unsupported_resume_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")) as client:
                response = client.post(
                    "/api/resume",
                    files={"file": ("resume.docx", b"content", "application/octet-stream")},
                )

        self.assertEqual(response.status_code, 415)

    def test_missing_embedding_model_returns_pull_command(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")
            with patch(
                "app.create_embeddings",
                new_callable=AsyncMock,
                side_effect=ollama.ResponseError(
                    'model "nomic-embed-text" not found', status_code=404
                ),
            ):
                with TestClient(app) as client:
                    response = client.post(
                        "/api/resume",
                        files={"file": ("resume.txt", b"Python engineer", "text/plain")},
                    )

        self.assertEqual(response.status_code, 503)
        self.assertIn("ollama pull nomic-embed-text", response.json()["detail"])

    def test_chat_returns_retrieval_and_usage_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(knowledge_base_path=Path(directory) / "kb.sqlite3")
            result = {
                "answer": "Senior Python engineer [resume.txt]",
                "retrieval": {
                    "top_similarity": 0.91,
                    "threshold": 0.35,
                    "grounded": True,
                    "matches": [{"source": "resume.txt", "similarity": 0.91}],
                },
                "usage": {"prompt_tokens": 35, "completion_tokens": 8, "total_tokens": 43},
                "timings": {"prompt_eval_ms": 1.2, "generation_ms": 4.5, "tokens_per_second": 1.78},
            }
            with patch("app.answer_question", new_callable=AsyncMock, return_value=result):
                with TestClient(app) as client:
                    client.app.state.knowledge_base.replace("resume.txt", ["Senior Python engineer"], [[1, 0]])
                    response = client.post("/api/chat", json={"prompt": "What role did I have?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["retrieval"]["top_similarity"], 0.91)
        self.assertEqual(response.json()["usage"]["total_tokens"], 43)
        self.assertIn("total_ms", response.json()["timings"])
        self.assertTrue(response.json()["request_id"])