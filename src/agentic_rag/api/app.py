from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
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
service = RetrievalService(store)
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


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, object]:
    if request.retrieval == "lexical":
        result = service.answer(request.query, request.limit)
        return {"answer": result.answer, "citations": [c.__dict__ for c in result.citations], "retrieved_chunks": result.retrieved_chunks, "grounded": result.grounded, "status": result.status, "model": "deterministic-local", "trace_id": "local"}
    if request.retrieval == "vector":
        results = vector_store.search(request.query, request.limit, request.metadata_filter)
    else:
        results = hybrid_search(store, vector_store, request.query, request.limit, request.metadata_filter)
        if request.retrieval == "hybrid+reranking":
            results = reranker.rerank(request.query, results, request.limit)
    if not results:
        return {"answer": "INSUFFICIENT_CONTEXT", "citations": [], "retrieved_chunks": [], "grounded": False, "status": "INSUFFICIENT_CONTEXT", "retrieval": request.retrieval}
    return {"answer": " ".join(r.chunk.content for r in results), "citations": [{"document_id": r.chunk.document_id, "chunk_id": r.chunk.chunk_id, "score": r.score} for r in results], "retrieved_chunks": [r.chunk.chunk_id for r in results], "grounded": True, "status": "OK", "retrieval": request.retrieval}
