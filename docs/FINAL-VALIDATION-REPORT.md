# FINAL-VALIDATION-REPORT

## M2A.1 final format

- **Repository:** `/home/hector/workspace/productionagentrag`
- **Branch:** `main`
- **Candidate Commit SHA:** `193e38e31973fe5e4965ca92673eb3bd9c4a4112`
- **Final HEAD:** verified implementation commit above; report-only corrections may follow
- **Working Tree:** clean at verification time
- **Scope:** M2A.1 retrieval validation only. M2B, LangGraph, Agentic RAG, MCP, and LLM generation are not authorized.
- **Qdrant:** live `m2a1-qdrant`, image `qdrant/qdrant:v1.12.5`, healthy on `localhost:6333`; persistent Docker volume `m2a1_qdrant_data` mounted at `/qdrant/storage`. Integration lifecycle passed against the real instance.
- **Dataset:** `datasets/m2a-retrieval-v2.json`, 40 cases, ten categories, checksum recorded in `artifacts/benchmarks/m2a1-semantic-retrieval.json`.
- **Semantic provider:** implementation is `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 384 dimensions, normalized, cosine, local CPU/optional device. The runtime package/model download could not complete within this environment's installation timeout; the accepted run therefore used the deterministic hash provider and records G5F as INSUFFICIENT EVIDENCE rather than claiming semantic execution.
- **Control:** deterministic SHA-256 hash embedding remains available offline/test provider.
- **Matrix:** lexical, vector semantic, hybrid semantic RRF, hybrid semantic RRF plus deterministic reranking. Artifact includes global/category Precision, Recall, HitRate, MRR, NDCG, mean/p50/p95 query latency, indexing time, memory/runtime metadata.

## Tests and gates

- Full unit/API suite: **15 passed**, with the existing 13 preserved; one upstream FastAPI/Starlette/httpx deprecation warning was investigated and no arbitrary dependency change was made.
- Real Qdrant integration: **2 passed**, covering health/readiness, collection create/validate, upsert/indexing, payloads, vector search, metadata filter, update, delete, fresh-client persistence, and unavailable behavior.
- Ruff: passed after fixes.
- Compose config: passed.
- G5D dataset/schema/checksum: PASS.
- G5E real Qdrant: PASS (live evidence above).
- G5F semantic model/matrix: INSUFFICIENT EVIDENCE (dependency installation timed out; no silent fallback).
- G5G CONTROL-1: INSUFFICIENT EVIDENCE; no recommendation.
- M2B authorization: **NOT AUTHORIZED**

## Artifacts

- `artifacts/benchmarks/m2a1-semantic-retrieval.json`
- `artifacts/benchmarks/latest.json`
- `datasets/m2a-retrieval-v2.json`
- `docs/ADR-002-SEMANTIC-EMBEDDING.md`
- `docs/ADR-003-CONTROL-1-SELECTION.md`
- `tests/integration/qdrant/test_qdrant_live.py`

No secrets were added, and no push/tag/release was performed.
