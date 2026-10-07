# M2A Final Validation Report

Status: **PRE-v1.0**. M2A retrieval engineering is implemented; no release/tag is created.

| Gate | Status | Evidence |
|---|---|---|
| G5A embedding/vector retrieval | PASS | `DeterministicHashEmbedding`, `VectorStore`, unit tests, benchmark artifact |
| G5B hybrid retrieval | PASS | explicit `reciprocal_rank_fusion`, unit tests, benchmark artifact |
| G5C reranking | PASS | `NoOpReranker` and `DeterministicOverlapReranker`, unit tests |
| G6 evaluation/benchmark | PASS | versioned dataset and `artifacts/benchmarks/m2a-retrieval-v1.json` |
| M1 lexical control | PASS | existing 7 tests plus full suite; API defaults to lexical |

Reproduce with `pytest -q` and `python scripts/run_benchmark.py`. Metrics include Precision, Recall, HitRate, MRR, NDCG, mean latency and p95 latency. Qdrant is provisioned by `docker-compose.yml`; `/v1/retrieval/qdrant-health` reports HTTP 503 explicitly when unavailable.

Limitations: deterministic hash embeddings and overlap reranking are local controls, not trained semantic models. Qdrant health/integration is optional and is not required for the offline benchmark. LangGraph, Agentic RAG, MCP, LLM generation, M2B, tags and releases are out of scope.
