from .models import Chunk, Document, SearchResult


class InMemoryStore:
    """Deterministic local store used by the reference implementation and tests."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    def add_document(self, document: Document) -> bool:
        if document.document_id in self.documents:
            return False
        self.documents[document.document_id] = document
        return True

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        terms = set(query.lower().split())
        scored: list[SearchResult] = []
        for chunk in self.chunks.values():
            words = set(chunk.content.lower().split())
            overlap = len(terms & words)
            if overlap:
                scored.append(SearchResult(chunk, overlap / max(len(terms), 1), "keyword"))
        return sorted(scored, key=lambda r: (-r.score, r.chunk.chunk_id))[:limit]
