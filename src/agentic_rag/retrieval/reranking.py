from __future__ import annotations

import re
from abc import ABC, abstractmethod

from agentic_rag.storage.models import SearchResult


class Reranker(ABC):
    name: str
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]: ...


class NoOpReranker(Reranker):
    name = "none"
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        return results[:limit]


class DeterministicOverlapReranker(Reranker):
    """Free local reranker: token coverage with deterministic tie-breaking."""
    name = "deterministic-overlap"
    def rerank(self, query: str, results: list[SearchResult], limit: int) -> list[SearchResult]:
        terms = set(re.findall(r"[\w]+", query.lower()))
        ranked = []
        for result in results:
            words = set(re.findall(r"[\w]+", result.chunk.content.lower()))
            coverage = len(terms & words) / max(len(terms), 1)
            ranked.append((coverage, result.score, result))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].chunk.chunk_id))
        return [item[2] for item in ranked[:limit]]
