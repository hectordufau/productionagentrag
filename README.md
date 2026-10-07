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

The benchmark consumes the versioned eight-case dataset in `datasets/m2a-retrieval-v1.json` and writes `artifacts/benchmarks/m2a-retrieval-v1.json`. It records commit, dataset, embedding, strategy, reranker, configuration, and timestamp metadata plus lexical/vector/hybrid/hybrid+rereanking metrics. `docs/RETRIEVAL-EVALUATION.md` defines the metrics and controls; `docs/RETRIEVAL-BENCHMARK.md` defines reproduction and artifact requirements.

Qdrant is optional for the offline benchmark. `docker compose up --build` provisions it, while `/v1/retrieval/qdrant-health` returns HTTP 503 when it cannot be reached. The adapter's live behavior is covered by mocked HTTP unit tests; no live Qdrant claim is made here.

## Scope

Included: M1 baseline and M2A retrieval engineering. Excluded: M2B, LangGraph, MCP, LLM generation, production persistence claims, tags, releases, and pushes. The repository remains PRE-v1.0.
