"""Run the reproducible M2A retrieval benchmark.
Usage: python scripts/run_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.evaluation import QueryCase, evaluate
from agentic_rag.retrieval import (
    DeterministicHashEmbedding,
    DeterministicOverlapReranker,
    VectorStore,
    hybrid_search,
)
from agentic_rag.storage.models import Chunk
from agentic_rag.storage.store import InMemoryStore


def main() -> None:
    chunks = [
        Chunk("chunk-qdrant", "doc-1", "Qdrant stores vectors for semantic retrieval.", 0, {"topic": "storage"}),
        Chunk("chunk-postgres", "doc-2", "PostgreSQL stores metadata and relational records.", 0, {"topic": "metadata"}),
        Chunk("chunk-checksum", "doc-3", "Content checksums provide deterministic document identity.", 0, {"topic": "identity"}),
    ]
    lexical = InMemoryStore()
    lexical.add_chunks(chunks)
    vector = VectorStore(DeterministicHashEmbedding())
    vector.add_chunks(chunks)
    reranker = DeterministicOverlapReranker()
    cases = [QueryCase("vector storage", ("chunk-qdrant",)), QueryCase("metadata database", ("chunk-postgres",)), QueryCase("document checksum", ("chunk-checksum",))]
    methods = {
        "lexical": lambda q, n: lexical.search(q, n),
        "vector": lambda q, n: vector.search(q, n),
        "hybrid": lambda q, n: hybrid_search(lexical, vector, q, n),
        "hybrid+reranking": lambda q, n: reranker.rerank(q, hybrid_search(lexical, vector, q, n), n),
    }
    report = {"benchmark_version": "m2a-v1", "embedding": "deterministic-hash", "reranker": reranker.name,
              "results": {name: evaluate(cases, fn) for name, fn in methods.items()}}
    output = Path("artifacts/benchmarks/m2a-retrieval-v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
