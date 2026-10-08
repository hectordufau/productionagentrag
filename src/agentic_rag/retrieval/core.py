from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from agentic_rag.retrieval.embedding import EmbeddingProvider
from agentic_rag.storage.models import Chunk, SearchResult


class ChunkSearcher(Protocol):
    def search(self, query: str, limit: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]: ...


def _matches(chunk: Chunk, metadata_filter: dict[str, Any] | None) -> bool:
    return not metadata_filter or all(chunk.metadata.get(k) == v for k, v in metadata_filter.items())


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class VectorStore:
    """Local vector index with explicit metadata filtering and deterministic ordering."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._items[chunk.chunk_id] = (chunk, self.embedding_provider.embed(chunk.content))

    def search(self, query: str, limit: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        query_vector = self.embedding_provider.embed(query)
        results = [
            SearchResult(chunk, _cosine(query_vector, vector), "vector")
            for chunk, vector in self._items.values()
            if _matches(chunk, metadata_filter)
        ]
        return sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))[:limit]


def reciprocal_rank_fusion(*rankings: list[SearchResult], limit: int = 5, k: int = 60) -> list[SearchResult]:
    """Fuse ranked lists using explicit RRF score=Σ 1/(k+rank), rank is 1-based."""
    fused: dict[str, tuple[SearchResult, float]] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, 1):
            prior = fused.get(result.chunk.chunk_id)
            score = 1.0 / (k + rank)
            fused[result.chunk.chunk_id] = (result, (prior[1] if prior else 0.0) + score)
    output = [replace(result, score=score, method="hybrid-rrf") for result, score in fused.values()]
    return sorted(output, key=lambda r: (-r.score, r.chunk.chunk_id))[:limit]


def hybrid_search(lexical: ChunkSearcher, vector: Any, query: str, limit: int = 5,
                  metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
    return reciprocal_rank_fusion(
        lexical.search(query, limit, metadata_filter),
        vector.search(query, limit, metadata_filter), limit=limit,
    )
