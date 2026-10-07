# ADR-002: M2A.1 semantic embedding provider

- **Decision:** use `sentence-transformers/all-MiniLM-L6-v2` through `EmbeddingProvider`.
- **License:** Apache-2.0 (model card); dependency is optional and explicitly installed with `.[semantic]`.
- **Revision:** `main` is recorded by the benchmark; pin a model revision before production promotion.
- **Dimension/normalization/distance:** 384 dimensions, L2-normalized embeddings, cosine distance.
- **Runtime/device:** local `sentence-transformers`; CPU by default, `EMBEDDING_DEVICE` may select a supported device. Model download occurs on first use and is cached.
- **Control:** `DeterministicHashEmbedding` remains the dependency-free offline/test implementation and is never silently substituted for semantic mode.
- **Scope:** retrieval only; no generation, LangGraph, MCP, or agentic orchestration.

Resource requirement: first run needs network access and roughly 100 MB model cache; subsequent runs use the local cache. CPU inference is supported; GPU is optional.
