# ADR-001: M1 lexical control and M2A retrieval

- **Status:** accepted; PRE-v1.0
- **Decision:** retain the in-memory keyword store as the M1 control. Add a dependency-free deterministic hash embedding, local vector index, explicit RRF hybrid fusion, exact metadata filters, and injectable rerankers.
- **Reason:** M2A must be reproducible offline and provide a measurable comparison without paid model services.
- **Qdrant:** Docker Compose provisions Qdrant. The HTTP health probe raises an explicit `QdrantUnavailable` / HTTP 503 rather than silently switching indexes.
- **Evaluation:** `datasets/m2a-retrieval-v1.json` and `scripts/run_benchmark.py` produce machine-readable metrics in `artifacts/benchmarks/m2a-retrieval-v1.json`.
- **Trade-off:** hash embeddings are an engineering control, not a claim of state-of-the-art semantic quality. Qdrant persistence/search wiring is intentionally optional; local retrieval remains available without it.
