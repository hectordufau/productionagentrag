from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore

app = FastAPI(title="Production Agentic RAG", version="0.1.0")
store = InMemoryStore()
service = RetrievalService(store)


class IngestRequest(BaseModel):
    source: str
    filename: str
    mime_type: str = "text/plain"
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": app.version}


@app.post("/v1/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    document = Document.from_content(
        request.source, request.filename, request.mime_type, request.content
    )
    if not store.add_document(document):
        return {"document_id": document.document_id, "duplicate": True, "chunks": 0}
    chunks = chunk_document(document)
    store.add_chunks(chunks)
    return {"document_id": document.document_id, "duplicate": False, "chunks": len(chunks)}


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, object]:
    result = service.answer(request.query, request.limit)
    return {
        "answer": result.answer,
        "citations": [c.__dict__ for c in result.citations],
        "retrieved_chunks": result.retrieved_chunks,
        "grounded": result.grounded,
        "status": result.status,
        "model": "deterministic-local",
        "trace_id": "local",
    }
