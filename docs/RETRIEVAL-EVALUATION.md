# Retrieval evaluation

## Purpose

M2A evaluation compares stable retrieval controls without network access or a downloaded model. Relevance is explicit by chunk ID, so metrics are auditable and do not depend on an LLM judge.

## Metrics

For each query at final-k: precision is relevant returned chunks divided by returned chunks; recall is relevant returned chunks divided by the relevance set; hit rate records any relevant result; MRR uses the first relevant rank; NDCG uses binary relevance; mean and p95 latency are measured around the search callable. No-answer cases have an empty relevance set and score zero rather than rewarding unrelated output.

## Controls

The four required strategies are lexical M1 control, deterministic local vector retrieval, reciprocal-rank-fusion hybrid, and hybrid followed by deterministic token-overlap reranking. The vector embedding is signed SHA-256 feature hashing with dimension recorded in the artifact. Qdrant is not used by the offline benchmark.

## Gates and interpretation

G1-G4 and G7-G8 remain M1 acceptance gates. M2A maps G5A to vector retrieval, G5B to hybrid retrieval, G5C to reproducible evaluation, G5 to the composite of G5A-G5C, and G6 to reranking. G9-G20 are pending scope, not inferred from local results.
