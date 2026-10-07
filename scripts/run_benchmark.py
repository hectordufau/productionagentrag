"""Run the reproducible, offline M2A retrieval benchmark."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
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

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets/m2a-retrieval-v1.json"
OUTPUT_PATH = ROOT / "artifacts/benchmarks/m2a-retrieval-v1.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    chunks = [
        Chunk(
            "chunk-qdrant",
            "doc-1",
            "Qdrant is a vector database that stores embeddings for semantic retrieval.",
            0,
            {"topic": "storage"},
        ),
        Chunk(
            "chunk-postgres",
            "doc-2",
            "PostgreSQL stores metadata and relational records in tables.",
            0,
            {"topic": "metadata"},
        ),
        Chunk(
            "chunk-checksum",
            "doc-3",
            "Content checksums provide stable fingerprints and deterministic document identity.",
            0,
            {"topic": "identity"},
        ),
        Chunk(
            "chunk-identity",
            "doc-3",
            "Immutable document identity is derived from content and source metadata.",
            1,
            {"topic": "identity"},
        ),
        Chunk(
            "chunk-ingestion",
            "doc-4",
            "Deterministic ingestion chunks documents before indexing and records provenance.",
            0,
            {"topic": "ingestion"},
        ),
        Chunk(
            "chunk-search",
            "doc-5",
            "Vector search retrieves nearest neighbors from an indexed embedding collection.",
            0,
            {"topic": "retrieval"},
        ),
        Chunk(
            "chunk-metadata",
            "doc-2",
            "Metadata filters restrict retrieval to exact payload field matches.",
            1,
            {"topic": "metadata"},
        ),
    ]
    lexical = InMemoryStore()
    lexical.add_chunks(chunks)
    embedding = DeterministicHashEmbedding(128)
    vector = VectorStore(embedding)
    vector.add_chunks(chunks)
    reranker = DeterministicOverlapReranker()
    cases = [
        QueryCase(item["query"], tuple(item["relevant_chunk_ids"])) for item in dataset["cases"]
    ]
    candidate_k, final_k = 10, 5
    methods = {
        "lexical": lambda q, n: lexical.search(q, min(n, candidate_k)),
        "vector": lambda q, n: vector.search(q, min(n, candidate_k)),
        "hybrid": lambda q, n: hybrid_search(lexical, vector, q, min(n, candidate_k)),
        "hybrid+reranking": lambda q, n: reranker.rerank(
            q, hybrid_search(lexical, vector, q, candidate_k), min(n, final_k)
        ),
    }
    report = {
        "benchmark_version": "m2a-retrieval-v1.1",
        "git_commit": _git_commit(),
        "dataset_version": dataset["dataset_version"],
        "dataset_count": len(cases),
        "embedding": {
            "provider": "local",
            "model": embedding.name,
            "version": "1",
            "dimension": embedding.dimension,
        },
        "strategy": {
            "candidate_k": candidate_k,
            "final_k": final_k,
            "fusion": "reciprocal_rank_fusion(k=60)",
            "reranker_provider": "local",
            "reranker_model": reranker.name,
        },
        "configuration": {
            "distance": "cosine",
            "metadata_filter": "exact-match",
            "seed": "sha256-feature-hash",
            "network": False,
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "results": {name: evaluate(cases, fn, final_k) for name, fn in methods.items()},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
