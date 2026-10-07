#!/usr/bin/env python3
"""Run the reproducible Hybrid+LLM versus Vector Semantic+LLM evaluation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
from agentic_rag.generation.provider import LLMProviderError, OllamaProvider
from agentic_rag.retrieval import (
    DeterministicHashEmbedding,
    SentenceTransformerEmbedding,
    VectorStore,
    hybrid_search,
)
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/generation-eval-v1.json"
OUTPUT = ROOT / "artifacts/generation-evaluation-v1.json"
MODEL = "aratan/Agents-A1-4B-Q4_K_M-GGUF:Q4_K_M"


def make_corpus() -> tuple[InMemoryStore, list]:
    store = InMemoryStore()
    docs = [
        Document.from_content("kb", "qdrant.md", "text/markdown", "Vectors are stored in Qdrant.", {"topic": "storage"}),
        Document.from_content("kb", "ingestion.md", "text/markdown", "The ingestion pipeline creates a checksum-identified document and chunks it.", {"topic": "ingestion"}),
    ]
    chunks = []
    for document in docs:
        store.add_document(document)
        parts = chunk_document(document)
        store.add_chunks(parts)
        chunks.extend(parts)
    return store, chunks


def score_case(case: dict, output, expected: dict) -> dict[str, float]:
    answer = output.answer.lower()
    expected_answer = (case.get("answer") or "").lower()
    answer_correctness = float(bool(expected_answer) and expected_answer.rstrip(".") in answer)
    groundedness = float(output.grounding.get("status") == "grounded")
    cited = {item["chunk_id"] for item in output.citations}
    relevant = set(case.get("relevant_chunks", []))
    citation_correctness = float(bool(cited) and cited <= relevant) if case["answerable"] else float(not cited)
    abstention_correctness = float((not case["answerable"]) == (output.status == "INSUFFICIENT_CONTEXT"))
    return {"answer_correctness": answer_correctness, "groundedness": groundedness, "citation_correctness": citation_correctness, "abstention_correctness": abstention_correctness}


def run_strategy(name: str, cases: list[dict], store: InMemoryStore, chunks: list, semantic: bool) -> dict:
    provider = OllamaProvider(MODEL)
    vector = VectorStore(SentenceTransformerEmbedding(device="cpu") if semantic else DeterministicHashEmbedding())
    vector.add_chunks(chunks)
    service = RetrievalService(store, provider)
    rows = []
    for case in cases:
        retrieval_started = time.perf_counter()
        results = hybrid_search(store, vector, case["question"], 5) if name == "hybrid+llm" else vector.search(case["question"], 5)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        generation_started = time.perf_counter()
        try:
            output = service.generate(case["question"], results, name)
        except LLMProviderError as exc:
            generation_ms = (time.perf_counter() - generation_started) * 1000
            rows.append({"id": case["id"], "metrics": None, "status": "PROVIDER_ERROR", "error": str(exc), "retrieved_chunks": [result.chunk.chunk_id for result in results], "latency_ms": {"retrieval": retrieval_ms, "generation": generation_ms, "total": retrieval_ms + generation_ms}})
            continue
        generation_ms = (time.perf_counter() - generation_started) * 1000
        metrics = score_case(case, output, {})
        rows.append({"id": case["id"], "metrics": metrics, "status": output.status, "answer": output.answer, "retrieved_chunks": output.retrieved_chunks, "latency_ms": {"retrieval": retrieval_ms, "generation": generation_ms, "total": retrieval_ms + generation_ms}, "provider": output.generation})
    keys = ["answer_correctness", "groundedness", "citation_correctness", "abstention_correctness"]
    complete = [row for row in rows if row["metrics"] is not None]
    metrics = {key: sum(row["metrics"][key] for row in complete) / len(complete) for key in keys} if len(complete) == len(rows) else None
    latency = {key: sum(row["latency_ms"][key] for row in rows) / len(rows) for key in ("retrieval", "generation", "total")}
    return {"strategy": name, "cases": rows, "metrics": metrics, "latency_ms": latency}


def main() -> int:
    cases = json.loads(DATASET.read_text())
    store, chunks = make_corpus()
    result: dict = {"dataset": {"path": str(DATASET.relative_to(ROOT)), "count": len(cases), "answerable": sum(case["answerable"] for case in cases), "unanswerable": sum(not case["answerable"] for case in cases)}, "model": MODEL, "provider": "ollama", "strategies": [], "limitations": []}
    try:
        # Import/model construction is deliberately attempted, never replaced with hash evidence.
        SentenceTransformerEmbedding(device="cpu")
    except (ImportError, OSError, RuntimeError) as exc:
        result["limitations"].append(f"Vector Semantic+LLM blocked: {type(exc).__name__}: {exc}")
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2
    hybrid_result = run_strategy("hybrid+llm", cases, store, chunks, False)
    semantic_result = run_strategy("vector-semantic+llm", cases, store, chunks, True)
    result["strategies"].extend([hybrid_result, semantic_result])
    if any(case["status"] == "PROVIDER_ERROR" for strategy in result["strategies"] for case in strategy["cases"]):
        result["limitations"].append("Ollama generation was attempted for every case but the local provider returned errors; metrics are null and were not fabricated.")
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
