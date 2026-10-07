# FINAL-VALIDATION-REPORT

## M2A.1 final format

- **Repository:** `/home/hector/workspace/productionagentrag`
- **Branch:** `main`
- **Candidate Commit SHA:** `193e38e31973fe5e4965ca92673eb3bd9c4a4112`
- **Final HEAD:** `a312e61` (local report correction; no push/tag/release)
- **Working Tree:** clean before this retry
- **Scope:** M2A.1 retrieval validation only. M2B, LangGraph, Agentic RAG, MCP, and LLM generation are not authorized.
- **Qdrant:** live `m2a1-qdrant`, image `qdrant/qdrant:v1.12.5`, healthy on `localhost:6333`; persistent Docker volume `m2a1_qdrant_data` mounted at `/qdrant/storage`. Integration lifecycle passed against the real instance.
- **Dataset:** `datasets/m2a-retrieval-v2.json`, 40 cases, ten categories. Recalculated SHA-256: `eec5a26d8285c70daaf228c709f8f041ef693df503abbbca424f0f25dc2491cb`; unchanged.
- **Semantic provider:** implementation is `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 384 dimensions, normalized, cosine, local CPU/optional device. R1 installation retry was stopped after approximately 13 minutes: the existing optional dependency resolved `torch-2.14.1` and then attempted multi-hundred-MB CUDA wheels (including `nvidia_cudnn_cu13`), with no completed install. `sentence-transformers`, `torch`, `transformers`, and `qdrant-client` remain unavailable. No semantic model executed.
- **Control:** deterministic SHA-256 hash embedding remains available offline/test provider.
- **Matrix:** R1 semantic matrix was not run because installation failed; no R1 artifact was created or overwrote. The historical hash-provider artifact remains preserved.

## Tests and gates

- Full unit/API suite: **15 passed**, with the existing 13 preserved; one upstream FastAPI/Starlette/httpx deprecation warning was investigated and no arbitrary dependency change was made.
- Real Qdrant integration: **2 passed**, covering health/readiness, collection create/validate, upsert/indexing, payloads, vector search, metadata filter, update, delete, fresh-client persistence, and unavailable behavior.
- Ruff: passed after fixes.
- Compose config: passed.
- G5D dataset/schema/checksum: PASS.
- G5E real Qdrant: PASS (live evidence above).
- G5F semantic model/matrix: INSUFFICIENT EVIDENCE (optional stack installation did not complete; no silent fallback).
- **G5G CONTROL-1: INSUFFICIENT EVIDENCE; no recommendation.**
- **M2B authorization: NOT AUTHORIZED**

## Artifacts

- `artifacts/benchmarks/m2a1-semantic-retrieval.json`
- `artifacts/benchmarks/latest.json`
- `datasets/m2a-retrieval-v2.json`
- `docs/ADR-002-SEMANTIC-EMBEDDING.md`
- `docs/ADR-003-CONTROL-1-SELECTION.md`
- `tests/integration/qdrant/test_qdrant_live.py`

No secrets were added, and no push/tag/release was performed. The retry was blocked before semantic execution; M2B remains unauthorized.

M2B_NOT_AUTHORIZED
