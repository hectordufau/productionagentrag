from __future__ import annotations

import uuid
from typing import Any

from agentic_rag.storage.models import Chunk, SearchResult

from .core import _matches
from .embedding import EmbeddingProvider
from .qdrant import QdrantClient


class QdrantVectorStore:
    """VectorStore-compatible adapter backed by a persistent Qdrant collection."""

    def __init__(self, embedding_provider: EmbeddingProvider, client: QdrantClient, collection: str = "rag_chunks") -> None:
        self.embedding_provider = embedding_provider
        self.client = client
        self.collection = collection
        self.client.ensure_collection(collection, embedding_provider.dimension)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        points = []
        for chunk in chunks:
            # Qdrant accepts only unsigned integer or UUID point IDs. Keep the
            # human-readable chunk ID in payload and derive a stable UUID for
            # the storage key so repeated ingestion remains idempotent.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"production-agentic-rag:{chunk.chunk_id}"))
            points.append({"id": point_id, "vector": self.embedding_provider.embed(chunk.content), "payload": {"chunk": chunk.content, "position": chunk.position, "document_id": chunk.document_id, "chunk_id": chunk.chunk_id, **chunk.metadata}})
        if points:
            self.client.upsert_vectors(self.collection, points)

    def search(self, query: str, limit: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[SearchResult]:
        payload_filter = None
        if metadata_filter:
            payload_filter = {"must": [{"key": key, "match": {"value": value}} for key, value in metadata_filter.items()]}
        hits = self.client.search_vectors(self.collection, self.embedding_provider.embed(query), limit, payload_filter)
        output = []
        for hit in hits:
            payload = hit.get("payload", {})
            chunk = Chunk(chunk_id=str(payload.get("chunk_id", hit.get("id"))), document_id=str(payload.get("document_id", "")), content=str(payload.get("chunk", "")), position=int(payload.get("position", 0)), metadata={k: v for k, v in payload.items() if k not in {"chunk", "chunk_id", "document_id", "position"}})
            if _matches(chunk, metadata_filter):
                output.append(SearchResult(chunk, float(hit.get("score", 0.0)), "qdrant"))
        return output
