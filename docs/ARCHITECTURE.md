# Architecture

## Baseline RAG + LLM scope

The pipeline is deliberately non-agentic: ingest, chunk, retrieve, evidence gate, bounded context construction, generate, validate citations and validate grounding. Retrieval is provider-independent: lexical CONTROL-0, deterministic vector, Hybrid RRF and optional reranking. Generation is an explicit `LLMProvider`; the shipped real provider is local Ollama.

Documents and retrieved chunks are untrusted data. The grounded prompt prevents document text from becoming instructions. `INSUFFICIENT_CONTEXT` is decided from retrieval evidence before generation. Provider unavailable and timeout failures are explicit and never become deterministic fake answers.

## Data preservation and output

Document identity is SHA-256. Chunks preserve `document_id`, `chunk_id`, `source`, `filename`, and metadata. ContextBuilder deduplicates by chunk ID, preserves retrieval order, and bounds words/tokens. API responses include `status`, `answer`, `citations`, `retrieval`, `generation`, `grounding`, `latency`, and `trace_id`; prompts and raw provider internals are not returned.

## Model and reproducibility

The selected local model is `aratan/Agents-A1-4B-Q4_K_M-GGUF:Q4_K_M`, already available in Ollama. It is a 4.2B Q4 GGUF chosen for local CPU/RAM feasibility and reproducibility. License/redistribution remains subject to the model card; this repository makes no commercial-provider claim. Run `ollama list`, then `uvicorn agentic_rag.api.app:app --reload` and `pytest -q`.

## Explicit non-goals

No LangGraph, MCP, agent/tool routing, memory, observability phase, deployment, commercial API requirement, or M2B work is included. Semantic retrieval remains the existing configured deterministic/local baseline, not a claim of learned embedding quality.
