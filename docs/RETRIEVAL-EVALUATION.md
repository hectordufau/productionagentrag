# Retrieval evaluation

## Purpose

M2A evaluation compares stable retrieval controls without network access or a downloaded model. Relevance is explicit by chunk ID, so metrics are auditable and do not depend on an LLM judge.

## Metrics

For each query at final-k: precision is relevant returned chunks divided by returned chunks; recall is relevant returned chunks divided by the relevance set; hit rate records any relevant result; MRR uses the first relevant rank; NDCG uses binary relevance; mean and p95 latency are measured around the search callable. No-answer cases have an empty relevance set and score zero rather than rewarding unrelated output.

## Controls

The four required strategies are lexical M1 control, deterministic local vector retrieval, reciprocal-rank-fusion hybrid, and hybrid followed by deterministic token-overlap reranking. The vector embedding is signed SHA-256 feature hashing with dimension recorded in the artifact. Qdrant is not used by the offline benchmark.

## Gates and interpretation

G1-G4 and G7-G8 remain M1 acceptance gates. M2A maps G5A to vector retrieval, G5B to hybrid retrieval, G5C to reproducible evaluation, G5 to the composite of G5A-G5C, and G6 to reranking. G9-G20 are pending scope, not inferred from local results.

## M2A.1-R1 semantic validation retry

The frozen `datasets/m2a-retrieval-v2.json` contains 40 explicit cases across ten categories and recalculates to SHA-256 `eec5a26d8285c70daaf228c709f8f041ef693df503abbbca424f0f25dc2491cb`. R1 froze candidate-k=10, final-k=5, RRF-k=60, `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, normalized embeddings, cosine distance, and deterministic overlap reranking before execution. The existing optional dependency installation was attempted with a long timeout but stalled downloading the resolved Torch/CUDA stack; the packages were not installed and the model never executed. Therefore the semantic smoke test, R1 matrix, and R1 artifact were not produced. The prior hash-provider artifact is historical evidence only.

Live Qdrant checks are isolated under `tests/integration/qdrant`; they require the service unless explicitly disabled with `QDRANT_EXPLICITLY_UNAVAILABLE=1`. The FastAPI/httpx warning is an upstream Starlette TestClient deprecation recommending `httpx2`; dependencies were not changed arbitrarily. G5D is PASS; G5F and G5G are INSUFFICIENT EVIDENCE. M2B is not authorized.
