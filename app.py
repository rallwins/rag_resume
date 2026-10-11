from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, nullcontext
import logging
import os
from sysconfig import get_path
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import ollama
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from rag_agent import SQLiteKnowledgeBase, answer_question, chunk_text, create_embeddings

try:
    from langfuse import get_client
except ImportError:  # Langfuse tracing is optional until its credentials are configured.
    get_client = None

logger = logging.getLogger("resume_chat")
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PROMPT_LENGTH = 4_000
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)


def _langfuse_tracing_enabled() -> bool:
    return bool(
        get_client
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() != "false"
    )


def _trace_span(name: str, input_data: dict[str, Any]) -> Any:
    if not _langfuse_tracing_enabled():
        return nullcontext(None)
    try:
        return get_client().start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
        )
    except Exception:
        logger.exception("Could not start Langfuse span")
        return nullcontext(None)


def _extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return content.decode("utf-8-sig").strip()


def create_app(
    model: str = DEFAULT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    knowledge_base_path: str | Path | None = None,
    min_similarity: float | None = None,
) -> FastAPI:
    database_path = Path(
        knowledge_base_path
        or os.getenv("RESUME_KNOWLEDGE_DB", Path(__file__).with_name("data") / "knowledge.sqlite3")
    )
    threshold = min_similarity if min_similarity is not None else float(
        os.getenv("RAG_MIN_SIMILARITY", "0.35")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ollama = ollama.AsyncClient()
        app.state.knowledge_base = SQLiteKnowledgeBase(database_path)
        app.state.request_lock = asyncio.Lock()
        yield

    app = FastAPI(title="Resume Chat", lifespan=lifespan)
    static_directory = Path(__file__).parent / "static"
    if not static_directory.is_dir():
        static_directory = Path(get_path("data")) / "share" / "rag-resume" / "static"
    index_file = static_directory / "index.html"
    architecture_file = static_directory / "architecture.html"

    @app.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/architecture", include_in_schema=False)
    async def architecture() -> FileResponse:
        return FileResponse(architecture_file)

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        return {
            "model": model,
            "embedding_model": embedding_model,
            "knowledge_base": request.app.state.knowledge_base.status(),
            "tracing": _langfuse_tracing_enabled(),
        }

    @app.post("/api/resume")
    async def upload_resume(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        filename = Path(file.filename or "resume").name
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Upload a PDF, TXT, or MD resume.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Resume files must be 10 MB or smaller.")
        try:
            text = _extract_text(filename, content)
        except Exception as error:
            logger.info("resume.parse_failed extension=%s error=%s", extension, error)
            raise HTTPException(status_code=422, detail="Could not read this resume file.") from error
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=422, detail="No readable text was found in the resume.")

        request_id = uuid4().hex[:12]
        started = time.perf_counter()
        async with request.app.state.request_lock:
            try:
                embeddings = await create_embeddings(
                    request.app.state.ollama,
                    embedding_model,
                    chunks,
                )
                request.app.state.knowledge_base.replace(filename, chunks, embeddings)
            except ollama.ResponseError as error:
                logger.exception("resume.embedding_failed request_id=%s", request_id)
                if error.status_code == 404 and "model" in str(error).lower():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Ollama embedding model '{embedding_model}' is not installed. "
                            f"Run `ollama pull {embedding_model}` against the Ollama server "
                            "used by this app, then upload the resume again."
                        ),
                    ) from error
                raise HTTPException(status_code=503, detail=f"Ollama embedding failed: {error}") from error
            except Exception as error:
                logger.exception("resume.index_failed request_id=%s", request_id)
                raise HTTPException(status_code=502, detail="Resume indexing failed. Check the server log.") from error
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "resume.indexed request_id=%s filename=%s chunks=%d characters=%d elapsed_ms=%.2f embedding_model=%s",
            request_id,
            filename,
            len(chunks),
            len(text),
            elapsed_ms,
            embedding_model,
        )
        return {
            "status": "indexed",
            "request_id": request_id,
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
            "elapsed_ms": elapsed_ms,
        }

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
        prompt = body.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="Please enter a message.")
        knowledge_base = request.app.state.knowledge_base
        if not knowledge_base.status()["uploaded"]:
            raise HTTPException(status_code=409, detail="Upload a resume before asking questions.")

        request_id = uuid4().hex[:12]
        started = time.perf_counter()
        async with request.app.state.request_lock:
            try:
                with _trace_span(
                    "resume-chat",
                    {"request_id": request_id, "question_characters": len(prompt), "model": model},
                ) as span:
                    result = await answer_question(
                        request.app.state.ollama,
                        knowledge_base,
                        model,
                        embedding_model,
                        prompt,
                        threshold,
                    )
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                    result["timings"]["total_ms"] = elapsed_ms
                    result["request_id"] = request_id
                    if span is not None:
                        span.update(
                            output={"grounded": result["retrieval"]["grounded"]},
                            metadata={
                                "top_similarity": result["retrieval"]["top_similarity"],
                                "threshold": threshold,
                                **result["usage"],
                                **result["timings"],
                            },
                        )
            except ollama.ResponseError as error:
                logger.exception("chat.ollama_failed request_id=%s model=%s", request_id, model)
                raise HTTPException(status_code=503, detail=f"Ollama request failed: {error}") from error
            except Exception as error:
                logger.exception("chat.failed request_id=%s model=%s", request_id, model)
                raise HTTPException(
                    status_code=502,
                    detail="The chat request failed. Check the server log for its request ID.",
                ) from error

        logger.info(
            "chat.completed request_id=%s model=%s grounded=%s top_similarity=%.4f threshold=%.4f "
            "retrieved_chunks=%d prompt_tokens=%d completion_tokens=%d total_tokens=%d "
            "prompt_eval_ms=%.2f generation_ms=%.2f tokens_per_second=%.2f total_ms=%.2f",
            request_id,
            model,
            result["retrieval"]["grounded"],
            result["retrieval"]["top_similarity"],
            threshold,
            len(result["retrieval"]["matches"]),
            result["usage"]["prompt_tokens"],
            result["usage"]["completion_tokens"],
            result["usage"]["total_tokens"],
            result["timings"]["prompt_eval_ms"],
            result["timings"]["generation_ms"],
            result["timings"]["tokens_per_second"],
            result["timings"]["total_ms"],
        )
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Ollama resume chatbot.")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(args.model, args.embedding_model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()