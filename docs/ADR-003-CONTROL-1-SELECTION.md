# ADR-003: CONTROL-1 selection

## Decision

Recommend **HYBRID semantic retrieval without reranking** as the provisional CONTROL-1 retrieval configuration for the next phase, with **MEDIUM confidence**. This is a retrieval recommendation only; M2B remains separately gated and unauthorized until external review.

Configuration:

- lexical retrieval + `sentence-transformers/all-MiniLM-L6-v2` vector retrieval;
- Reciprocal Rank Fusion `k=60`;
- `candidate_k=10`, `final_k=5`;
- cosine distance, normalized 384-dimensional embeddings;
- deterministic overlap reranking disabled by default;
- Qdrant collection dimension 384 with exact metadata filters;
- fail-closed negative-query handling for future `INSUFFICIENT_CONTEXT`.

## Evidence

On the frozen 40-query dataset, vector semantic reached Recall 0.900 and NDCG 0.860 at mean latency 19.80 ms. Hybrid reached Recall 0.892 and NDCG 0.836 at mean latency 15.45 ms, while preserving lexical coverage and avoiding the additional complexity of reranking. Lexical alone was much faster (0.010 ms mean) but had lower Recall 0.704 and NDCG 0.694. Hybrid plus deterministic overlap reranking reached NDCG 0.837, a marginal improvement over hybrid, and was not selected because the reranker is not a trained semantic cross-encoder.

## Caveats

The 40-query dataset supports a provisional, medium-confidence engineering decision, not a universal retrieval claim. Latency is local CPU measurement and must be revalidated with the target corpus and deployment. Historical deterministic-hash results are not semantic evidence. M2B is not authorized by this ADR.
