#!/usr/bin/env python3
"""Run the real Qdrant/SentenceTransformer/Ollama/MCP integration smoke.

The script never substitutes an in-memory fallback. It prints JSON and exits 2
when a required service is unavailable so a blocked smoke cannot look green.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.provider import LLMProviderError, OllamaProvider
from agentic_rag.mcp.client import MCPToolClient
from agentic_rag.mcp.server import create_mcp_server
from agentic_rag.retrieval import QdrantVectorStore, SentenceTransformerEmbedding
from agentic_rag.retrieval.qdrant import QdrantClient, QdrantUnavailable
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore


def main() -> int:
    client = QdrantClient("http://127.0.0.1:6333", timeout=3)
    result: dict[str, Any] = {"services": {"qdrant": "not_run", "sentence_transformers": "not_run", "ollama": "not_run", "mcp": "not_run"}, "checks": {}, "limitations": []}
    try:
        health = client.health()
        result["services"]["qdrant"] = "available"
        result["checks"]["qdrant_health"] = health
    except QdrantUnavailable as exc:
        result["services"]["qdrant"] = "blocked"
        result["limitations"].append(f"Qdrant unavailable at 127.0.0.1:6333: {exc}")
        print(json.dumps(result, indent=2))
        return 2

    embedding = SentenceTransformerEmbedding("all-MiniLM-L6-v2", "cpu")
    result["services"]["sentence_transformers"] = "available"
    result["checks"]["embedding_dimension"] = embedding.dimension
    store = InMemoryStore()
    document = Document.from_content("smoke", "smoke.md", "text/markdown", "Qdrant stores vectors for semantic retrieval.", {"topic": "smoke"})
    store.add_document(document)
    chunks = chunk_document(document)
    store.add_chunks(chunks)
    vector_store = QdrantVectorStore(embedding, client, "rag_smoke_v1")
    vector_store.add_chunks(chunks)
    hits = vector_store.search("where are vectors stored", 1)
    result["checks"]["semantic_retrieval"] = {"count": len(hits), "chunk_ids": [hit.chunk.chunk_id for hit in hits]}

    provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="qwen2.5:3b")
    try:
        response = provider.generate("Answer exactly: Qdrant stores vectors. [chunk_id=smoke]", timeout_s=30)
        result["services"]["ollama"] = "available"
        result["checks"]["ollama"] = {"status": "ok", "model": response.model, "answer": response.answer}
    except LLMProviderError as exc:  # expose provider limitations, never fake success
        result["services"]["ollama"] = "blocked"
        result["limitations"].append(f"Ollama generation failed: {type(exc).__name__}: {exc}")

    metadata = MCPToolClient(create_mcp_server(store)).call("get_document_metadata", {"document_id": document.document_id})
    result["services"]["mcp"] = "available"
    result["checks"]["metadata_lookup"] = metadata
    print(json.dumps(result, indent=2))
    return 0 if result["services"]["ollama"] == "available" else 2


if __name__ == "__main__":
    sys.exit(main())
