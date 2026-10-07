# Retrieval evaluation

## Purpose

M2A evaluation compares stable retrieval controls without network access or a downloaded model. Relevance is explicit by chunk ID, so metrics are auditable and do not depend on an LLM judge.

## Metrics

For each query at final-k: precision is relevant returned chunks divided by returned chunks; recall is relevant returned chunks divided by the relevance set; hit rate records any relevant result; MRR uses the first relevant rank; NDCG uses binary relevance; mean and p95 latency are measured around the search callable. No-answer cases have an empty relevance set and score zero rather than rewarding unrelated output.

## Controls

The four required strategies are lexical M1 control, deterministic local vector retrieval, reciprocal-rank-fusion hybrid, and hybrid followed by deterministic token-overlap reranking. The vector embedding is signed SHA-256 feature hashing with dimension recorded in the artifact. Qdrant is not used by the offline benchmark.

## Gates and interpretation

G1-G4 and G7-G8 remain M1 acceptance gates. M2A maps G5A to vector retrieval, G5B to hybrid retrieval, G5C to reproducible evaluation, G5 to the composite of G5A-G5C, and G6 to reranking. G9-G20 are pending scope, not inferred from local results.

## M2A.1 semantic validation

The frozen `datasets/m2a-retrieval-v2.json` contains 40 explicit cases across ten categories. The matrix is lexical, semantic vector, semantic RRF hybrid, and semantic RRF plus deterministic reranking. `SentenceTransformerEmbedding` uses all-MiniLM-L6-v2 (384, normalized, cosine); `DeterministicHashEmbedding` remains the offline control. Artifacts report global/category Precision, Recall, HitRate, MRR, NDCG, mean/p50/p95 latency, indexing timing, and runtime metadata.

Live Qdrant checks are isolated under `tests/integration/qdrant`; they require the service unless explicitly disabled with `QDRANT_EXPLICITLY_UNAVAILABLE=1`. The FastAPI/httpx warning is an upstream Starlette TestClient deprecation recommending `httpx2`; dependencies were not changed arbitrarily. G5D/G5F are evidence-bound, G5G CONTROL-1 is INSUFFICIENT EVIDENCE, and M2B is not authorized.
