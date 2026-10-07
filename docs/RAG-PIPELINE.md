# RAG pipeline

1. Ingest computes a SHA-256 document ID and preserves source, filename and metadata.
2. Deterministic chunking writes chunk IDs and document metadata into each chunk.
3. Retrieval selects lexical CONTROL-0, deterministic vector, Hybrid RRF, or optional overlap reranking.
4. Evidence gating checks retrieval results before generation and fails closed with `INSUFFICIENT_CONTEXT`.
5. ContextBuilder deduplicates IDs, preserves rank order and bounds context by a configurable budget.
6. Ollama receives a grounded-generation prompt. Retrieved text is untrusted data and cannot issue instructions.
7. Returned chunk citations are validated against the exact context sent to the provider.
9. Grounding validation runs separately from generation using deterministic evidence overlap.
10. Opt-in agentic mode executes analyze → retrieve → evidence gate → bounded rewrite/retry → baseline generation → citation/grounding validation. It returns a structured decision trace and metrics; it abstains rather than guessing when evidence, citations, or the provider fail.

The latest evaluation artifact (`artifacts/generation-evaluation-v1.json`) contains 4 identical cases for each strategy (2 answerable, 2 unanswerable) and records retrieval/generation/total latency. The real Ollama model completed all cases using the CPU sentence-transformers environment, not hash fallback. Results are a pipeline smoke benchmark only.
