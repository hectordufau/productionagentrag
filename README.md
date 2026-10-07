# M2A Retrieval Engineering

M2A adds retrieval engineering only. M1's lexical `InMemoryStore` and deterministic grounded answer remain the default and remain executable.

## Local retrieval

```bash
python scripts/run_benchmark.py
pytest -q
```

The benchmark writes `artifacts/benchmarks/m2a-retrieval-v1.json` and compares lexical, deterministic local vector, explicit reciprocal-rank-fusion hybrid, and hybrid plus deterministic overlap reranking. The versioned relevance set is `datasets/m2a-retrieval-v1.json`. Metrics are Precision, Recall, HitRate, MRR, NDCG, mean latency and p95 latency.

The local embedding provider is `DeterministicHashEmbedding`: signed feature hashing, fixed dimension, no network/model download. `VectorStore.search(..., metadata_filter={...})` applies exact metadata matches. `reciprocal_rank_fusion` uses `1/(60+rank)` with 1-based ranks.

API query behavior defaults to M1 lexical retrieval. M2A modes are explicit:

```bash
curl -X POST localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"query":"where are vectors stored","retrieval":"hybrid+reranking","limit":5}'
```

The optional Qdrant availability probe is `GET /v1/retrieval/qdrant-health`. It returns HTTP 503 with an explicit error when Qdrant is unavailable; it does not silently fall back. `docker compose up --build` starts the API and pinned Qdrant service.

## Scope boundary

No LangGraph, Agentic RAG, MCP, LLM generation, M2B, tags or release work is included. This repository remains PRE-v1.0.
