from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
from agentic_rag.generation.provider import (
    LLMProviderError,
    LLMTimeoutError,
    LLMUnavailableError,
    OllamaProvider,
)
from agentic_rag.retrieval import (
    DeterministicHashEmbedding,
    DeterministicOverlapReranker,
    VectorStore,
    hybrid_search,
)
from agentic_rag.retrieval.qdrant import QdrantClient, QdrantUnavailable
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore

app = FastAPI(title="Production Agentic RAG", version="0.1.0")
store = InMemoryStore()
vector_store = VectorStore(DeterministicHashEmbedding())
provider = OllamaProvider()
service = RetrievalService(store, provider)
reranker = DeterministicOverlapReranker()


class IngestRequest(BaseModel):
    source: str
    filename: str
    mime_type: str = "text/plain"
    content: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    retrieval: str = Field(default="lexical", pattern="^(lexical|vector|hybrid|hybrid\\+reranking)$")
    metadata_filter: dict[str, object] | None = None
    generate: bool = False
    context_tokens: int = Field(default=1800, ge=1, le=12000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": app.version}


@app.get("/v1/retrieval/qdrant-health")
def qdrant_health() -> dict[str, object]:
    try:
        return {"status": "available", "detail": QdrantClient().health()}
    except QdrantUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    document = Document.from_content(request.source, request.filename, request.mime_type, request.content, request.metadata)
    if not store.add_document(document):
        return {"document_id": document.document_id, "duplicate": True, "chunks": 0}
    chunks = chunk_document(document)
    store.add_chunks(chunks)
    vector_store.add_chunks(chunks)
    return {"document_id": document.document_id, "duplicate": False, "chunks": len(chunks)}


def _retrieved(request: QueryRequest):
    if request.retrieval == "lexical":
        return store.search(request.query, request.limit, request.metadata_filter)
    if request.retrieval == "vector":
        return vector_store.search(request.query, request.limit, request.metadata_filter)
    results = hybrid_search(store, vector_store, request.query, request.limit, request.metadata_filter)
    return reranker.rerank(request.query, results, request.limit) if request.retrieval == "hybrid+reranking" else results


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, object]:
    trace_id = uuid4().hex
    results = _retrieved(request)
    if not results:
        return {"status": "INSUFFICIENT_CONTEXT", "answer": "INSUFFICIENT_CONTEXT", "citations": [], "retrieval": {"strategy": request.retrieval, "chunks": []}, "generation": {"status": "not_run"}, "grounding": {"status": "unsupported"}, "latency": {"total_ms": 0}, "trace_id": trace_id}
    if not request.generate:
        return {"status": "OK", "answer": " ".join(r.chunk.content for r in results), "citations": [{"document_id": r.chunk.document_id, "chunk_id": r.chunk.chunk_id, "source": r.chunk.metadata.get("source"), "filename": r.chunk.metadata.get("filename"), "score": r.score} for r in results], "retrieval": {"strategy": request.retrieval, "chunks": [r.chunk.chunk_id for r in results]}, "generation": {"status": "not_run"}, "grounding": {"status": "grounded"}, "latency": {"total_ms": 0}, "trace_id": trace_id, "grounded": True, "retrieved_chunks": [r.chunk.chunk_id for r in results]}
    service.context_tokens = request.context_tokens
    try:
        output = service.generate(request.query, results, request.retrieval)
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail={"code": "LLM_TIMEOUT", "message": str(exc), "trace_id": trace_id}) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail={"code": "LLM_UNAVAILABLE", "message": str(exc), "trace_id": trace_id}) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail={"code": "LLM_PROVIDER_ERROR", "message": str(exc), "trace_id": trace_id}) from exc
    return {"status": output.status, "answer": output.answer, "citations": output.citations, "retrieval": {"strategy": output.retrieval, "chunks": output.retrieved_chunks, "context_tokens": output.context.token_count}, "generation": output.generation, "grounding": output.grounding, "latency": {"total_ms": output.latency_ms}, "trace_id": trace_id, "grounded": output.grounding["status"] == "grounded", "retrieved_chunks": output.retrieved_chunks}
