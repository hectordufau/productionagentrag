# ADR-001: Deterministic local baseline

- **Status:** accepted for M1
- **Decision:** start with an in-memory, keyword retrieval baseline and a deterministic local answer constructor.
- **Reason:** it runs without paid services and provides a control for later embedding, reranking and agent experiments.
- **Trade-off:** lexical retrieval is not production-quality semantic retrieval; replacing the store is an explicit roadmap item.
