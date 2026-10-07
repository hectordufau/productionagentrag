from agentic_rag.citations.models import Answer, citations_from
from agentic_rag.storage.store import InMemoryStore


class RetrievalService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def answer(self, query: str, limit: int = 5) -> Answer:
        results = self.store.search(query, limit)
        if not results:
            return Answer("INSUFFICIENT_CONTEXT", [], [], False, "INSUFFICIENT_CONTEXT")
        context = " ".join(result.chunk.content for result in results)
        return Answer(context, citations_from(results), [r.chunk.chunk_id for r in results], True)
