from dataclasses import dataclass

from agentic_rag.storage.models import SearchResult


@dataclass(frozen=True)
class Citation:
    document_id: str
    source: str
    chunk_id: str
    position: int


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[str]
    grounded: bool
    status: str = "OK"


def citations_from(results: list[SearchResult]) -> list[Citation]:
    return [
        Citation(
            r.chunk.document_id,
            r.chunk.metadata.get("source", r.chunk.document_id),
            r.chunk.chunk_id,
            r.chunk.position,
        )
        for r in results
    ]
