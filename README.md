# Production Agentic RAG

A provider-independent reference implementation for a production-oriented RAG baseline. The repository is deliberately deterministic and offline-friendly: M1 provides ingestion, chunking, lexical retrieval, grounded citations, and fail-closed answers; M2A adds local vector, hybrid, reranking, evaluation, and an optional Qdrant HTTP adapter.

## Quickstart (M1 control)

```bash
python -m pip install -e '.[dev]'
uvicorn agentic_rag.api.app:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/ingest -H 'content-type: application/json' \\
  -d '{"source":"demo","filename":"notes.txt","content":"Qdrant stores vectors."}'
curl -X POST http://localhost:8000/v1/query -H 'content-type: application/json' \\
  -d '{"query":"where are vectors stored"}'
```

The default query mode is M1 lexical retrieval. It returns grounded citations when context is found and `INSUFFICIENT_CONTEXT` otherwise. Document identity is an immutable SHA-256 content checksum; chunking is deterministic.

## Architecture

`POST /v1/ingest` creates a checksum-identified `Document`, chunks it, and indexes chunks in the in-memory lexical and local vector controls. `POST /v1/query` supports `lexical` (default), `vector`, `hybrid`, and `hybrid+reranking`. The API exposes `/health`, `/ready`, `/version`, and an explicit Qdrant health probe.

M2A retrieval components are provider-independent protocols and deterministic implementations:

- `InMemoryStore`: lexical M1 control.
- `DeterministicHashEmbedding` and `VectorStore`: fixed-dimension local vector control with exact metadata filtering.
- `reciprocal_rank_fusion`: explicit `1/(60+rank)` rank fusion.
- `DeterministicOverlapReranker`: transparent local reranking control.
- `QdrantClient`: optional stdlib-only HTTP adapter; it never silently falls back.

## M2A usage and evidence

```bash
make test
make evaluate-retrieval
make benchmark-retrieval
```

The M2A.1 benchmark consumes the frozen 40-case dataset in `datasets/m2a-retrieval-v2.json` and writes `artifacts/benchmarks/m2a1-semantic-retrieval.json` plus `latest.json`. It records dataset checksum, model metadata, category metrics, timing, and resource details. Install `.[semantic]` for the local Apache-2.0 sentence-transformers provider; `--provider hash` remains the offline control.

Qdrant is provisioned by `docker compose`; `/v1/retrieval/qdrant-health` returns HTTP 503 when unavailable. Live lifecycle coverage is isolated in `tests/integration/qdrant` and is run with `make test-integration` against the Qdrant volume.

## Scope

Included: M1 baseline and M2A retrieval engineering. Excluded: M2B, LangGraph, MCP, LLM generation, production persistence claims, tags, releases, and pushes. The repository remains PRE-v1.0.
