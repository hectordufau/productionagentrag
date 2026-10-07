# FINAL-VALIDATION-REPORT

## M2A.1-R2 final format

- **Repository:** `/home/hector/workspace/productionagentrag`
- **Branch:** `main`
- **Implementation Commit:** `f089800f84bd505294e54c6f6468a1a5e9cbb470`
- **Final HEAD:** `97bb3eb` at benchmark verification; documentation/artifact provenance corrections follow
- **Working Tree:** clean before these report corrections
- **Scope:** semantic retrieval validation only; no M2B, LangGraph, agentic generation, MCP, or feature work.
- **Environment:** isolated `.venv-semantic-cpu` (ignored), Python 3.14.7, pip 26.2.1, Ubuntu x86_64.
- **CPU/no-CUDA evidence:** `torch==2.14.1+cpu`; `torch.cuda.is_available()==False`; no `nvidia-*`, `cuda-*`, or `triton` distributions.
- **Dependencies:** sentence-transformers 3.4.1, transformers 4.57.6, qdrant-client 1.19.1.
- **Model:** exactly `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 384 dimensions, L2-normalized, cosine distance, CPU; cache outside Git at `/home/hector/.cache/productionagentrag-hf`.
- **Smoke:** PASS. Related similarity 0.482737; unrelated similarity 0.021246; finite normalized 384-D vectors; provider class `SentenceTransformerEmbedding` proves semantic rather than hash.
- **Dataset:** 40 cases; SHA-256 `eec5a26d8285c70daaf228c709f8f041ef693df503abbbca424f0f25dc2491cb`; verified before benchmark.
- **Frozen configuration:** candidate_k=10, final_k=5, RRF k=60, model/dim/normalization/distance/reranker unchanged.
- **Qdrant:** live `m2a1-qdrant`, `qdrant/qdrant:v1.12.5`, localhost:6333. Separate collection `m2a1_r2_semantic_384`; 384-D semantic upsert/search/filter passed; fresh-client persistence passed (7 upserts, 5 search results, 2 filtered results).
- **Benchmark:** artifact `artifacts/benchmarks/m2a1-r2-semantic-retrieval.json`; `latest.json` updated while historical artifacts were preserved. Matrix includes lexical, vector semantic, hybrid semantic RRF, and hybrid+reranking. Five measured runs plus one warmup; global semantic vector hit-rate 0.900, recall 0.900, MRR 0.8625, nDCG 0.860144. Negative queries produced zero hits for all methods.

## Tests and gates

- Unit/API suite: PASS (run separately).
- Live Qdrant suite: PASS (run separately against localhost:6333).
- Semantic smoke/benchmark: PASS.
- Ruff: PASS.
- Regression suite: PASS.
- G5D dataset/schema/checksum: PASS.
- G5E real Qdrant: PASS.
- G5F semantic model/matrix: PASS.
- G5G CONTROL-1: PASS; provisional recommendation is hybrid semantic RRF without reranking, medium confidence.

## Artifacts and authorization

## CONTROL-1 decision

Provisional recommendation: **hybrid semantic RRF without reranking** (`candidate_k=10`, `final_k=5`, `RRF k=60`, normalized 384-D `all-MiniLM-L6-v2`, cosine, Qdrant). Vector semantic had the highest quality (Recall 0.900, NDCG 0.860), but hybrid was close (Recall 0.892, NDCG 0.836) with lower mean latency (15.45 ms versus 19.80 ms) and lower complexity than adding the deterministic reranker. Lexical remains the fastest control (0.010 ms) but had lower Recall 0.704 and NDCG 0.694. Confidence: **MEDIUM**, because the dataset has 40 cases and local CPU latency is not production capacity evidence.

Modified `scripts/run_benchmark.py`, `docs/RETRIEVAL-BENCHMARK.md`, ADR-003, and this report; generated `artifacts/benchmarks/m2a1-r2-semantic-retrieval.json` and updated `latest.json`. No dataset, model, architecture, or historical artifact was changed. No push/tag/release performed.

M2B_NOT_AUTHORIZED
