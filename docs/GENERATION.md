# Baseline RAG + LLM generation

This phase is a non-agentic retrieval-augmented generation baseline. Retrieval and generation are separate: lexical is CONTROL-0, while deterministic hash-vector and Hybrid RRF are local controls. `POST /v1/query` supports `lexical`, `vector`, `hybrid`, and `hybrid+reranking`, and returns status, answer, citations, retrieval, generation, grounding, latency, and trace ID. Set `generate:false` to inspect retrieval without an LLM.

## Provider and reproduction

The real provider is the local Ollama HTTP API (`http://127.0.0.1:11434`) with `aratan/Agents-A1-4B-Q4_K_M-GGUF:Q4_K_M`. Requests use `num_ctx=2048`, `num_predict=256`, and deterministic temperature `0`; the reduced context prevents the model's default 131072-token KV cache from exhausting host memory. No deterministic fallback is used. Generation records provider, model, latency and Ollama token counters when supplied.

```bash
python -m pip install -e '.[dev]'
ollama list
uvicorn agentic_rag.api.app:app --reload
curl -X POST localhost:8000/v1/query -H 'content-type: application/json' -d '{"query":"Where are vectors stored?","retrieval":"hybrid"}'
pytest -q
ruff check src tests
```

The prompt explicitly treats retrieved documents as untrusted data. Context assembly deduplicates chunk IDs, preserves document/chunk/source/filename/metadata, and applies a bounded word/token budget. Generation is skipped when retrieval has no evidence (`INSUFFICIENT_CONTEXT`); citations not present in the sent context are rejected. Grounding uses deterministic lexical evidence checks and reports `grounded`, `partially_grounded`, or `unsupported`.

## Evaluation

`datasets/generation-eval-v1.json` contains answerable and abstention cases and is distinct from `m2a-retrieval-v2.json`. Metrics are deterministic and transparent: Answer Correctness, Groundedness, Citation Correctness and Abstention Correctness, with optional retrieval relevance/completeness. Compare identical cases and configuration with Hybrid+LLM and Vector Semantic+LLM; capture retrieval, generation and total latency. A live Ollama run is required for claimed generation results.

A live run was executed on 2026-10-07 with Ollama model `aratan/Agents-A1-4B-Q4_K_M-GGUF:Q4_K_M`, dataset count 4 (2 answerable, 2 unanswerable), and the semantic CPU environment (`.venv-semantic-cpu`, `all-MiniLM-L6-v2`, CPU). Both strategies completed all four cases with the real provider. Mean metrics were identical: Answer Correctness 0.25, Groundedness 0.50, Citation Correctness 1.00, Abstention Correctness 1.00. Mean total latency was 28,670.68 ms for Hybrid+LLM and 29,598.93 ms for Vector Semantic+LLM. These four cases validate the pipeline, not statistical superiority.

Limitations: the vector provider remains a deterministic hash control rather than a learned semantic encoder in the default API; the evaluation's semantic challenger uses the optional validated sentence-transformers provider only when `.venv-semantic-cpu` is available. Ollama output is model-dependent; lexical grounding is a practical signal, not a proof; no persistence, agent routing, MCP, memory, observability, deployment or commercial provider is included. M2B/agentic work remains future scope.
