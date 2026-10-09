#!/usr/bin/env python3
"""Reproducible qwen2.5:3b baseline versus bounded agentic evaluation."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from agentic_rag.agentic import AgenticRAGGraph
from agentic_rag.chunking.core import chunk_document
from agentic_rag.generation.baseline import RetrievalService
from agentic_rag.generation.provider import LLMProviderError, OllamaProvider
from agentic_rag.mcp.client import MCPToolClient
from agentic_rag.mcp.server import create_mcp_server
from agentic_rag.retrieval import DeterministicHashEmbedding, VectorStore, hybrid_search
from agentic_rag.storage.models import Document
from agentic_rag.storage.store import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/generation-eval-v2.json"
OUTPUT = ROOT / "artifacts/benchmarks/baseline-vs-agentic-v1.1.json"
BEFORE = ROOT / "artifacts/benchmarks/baseline-vs-agentic-v2.json"
MODEL = "qwen2.5:3b"
GENERATION = {"num_ctx": 2048, "num_predict": 256, "temperature": 0, "think": False}


def load_fixture() -> tuple[dict, InMemoryStore, list, dict[str, str]]:
    fixture = json.loads(DATASET.read_text())
    store = InMemoryStore()
    chunks = []
    ids = {}
    for item in fixture["corpus"]:
        document = Document.from_content(item["source"], item["filename"], item["mime_type"], item["content"], item["metadata"])
        store.add_document(document)
        parts = chunk_document(document)
        store.add_chunks(parts)
        chunks.extend(parts)
        ids[Path(document.filename).stem] = document.document_id
    corpus_blob = json.dumps(fixture["corpus"], sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(corpus_blob).hexdigest()
    if fixture["corpus_sha256"] not in ("TO_BE_FILLED_BY_SCRIPT", actual):
        raise ValueError("dataset corpus_sha256 does not match corpus")
    return fixture, store, chunks, ids


def query_for(case: dict, ids: dict[str, str]) -> str:
    query = case["question"]
    for key, value in ids.items():
        query = query.replace("{document_id:" + key + "}", value)
    return query


def score(case: dict, answer: str, status: str, citations: list[dict], retrieved: list[str]) -> dict[str, object]:
    text = answer.lower()
    facts = case["expected_facts"]
    fact_hits = [fact for fact in facts if fact.lower() in text]
    source_hits = {str(item.get("filename")) for item in citations if item.get("filename")}
    expected_sources = set(case["expected_sources"])
    answerable = bool(case["should_answer"])
    abstained = status in {"INSUFFICIENT_CONTEXT", "ABSTAINED"}
    return {
        "fact_recall": round(len(fact_hits) / len(facts), 6) if facts else float(not answerable),
        "facts_hit": fact_hits,
        "source_precision": round(len(source_hits & expected_sources) / len(source_hits), 6) if source_hits else float(not answerable),
        "source_recall": round(len(source_hits & expected_sources) / len(expected_sources), 6) if expected_sources else float(not answerable),
        "should_answer": answerable,
        "abstention_correct": abstained == (not answerable),
        "quality": round((len(fact_hits) / len(facts) if facts else float(not answerable) + (1.0 if abstained == (not answerable) else 0.0)) / 2, 6) if facts else float(abstained == (not answerable)),
        "grounded": status == "OK" and bool(citations),
        "retrieved_count": len(retrieved),
    }


def run_baseline(cases: list[dict], store: InMemoryStore, chunks: list, ids: dict[str, str]) -> list[dict]:
    vector = VectorStore(DeterministicHashEmbedding())
    vector.add_chunks(chunks)
    service = RetrievalService(store, OllamaProvider(MODEL), context_tokens=GENERATION["num_ctx"])
    rows = []
    for case in cases:
        query = query_for(case, ids)
        started = time.perf_counter()
        retrieved = hybrid_search(store, vector, query, 5)
        retrieval_ms = (time.perf_counter() - started) * 1000
        try:
            output = service.generate(query, retrieved, "hybrid")
            row = {"id": case["id"], "status": output.status, "answer": output.answer, "citations": output.citations, "retrieved_chunks": output.retrieved_chunks, "metrics": score(case, output.answer, output.status, output.citations, output.retrieved_chunks), "trace": {"attempts": 1, "rewrites": 0, "llm_calls": int(output.generation.get("status") == "ok"), "mcp_calls": 0, "strategy": "hybrid"}, "latency_ms": {"retrieval": round(retrieval_ms, 3), "generation": output.generation.get("latency_ms", 0), "total": round((time.perf_counter() - started) * 1000, 3)}}
        except LLMProviderError as exc:
            row = {"id": case["id"], "status": "PROVIDER_ERROR", "answer": "", "citations": [], "retrieved_chunks": [r.chunk.chunk_id for r in retrieved], "metrics": None, "error": f"{type(exc).__name__}: {exc}", "trace": {"attempts": 1, "rewrites": 0, "llm_calls": 1, "mcp_calls": 0, "strategy": "hybrid"}, "latency_ms": {"retrieval": round(retrieval_ms, 3), "generation": round((time.perf_counter() - started) * 1000 - retrieval_ms, 3), "total": round((time.perf_counter() - started) * 1000, 3)}}
        rows.append(row)
    return rows


def run_agentic(cases: list[dict], store: InMemoryStore, chunks: list, ids: dict[str, str]) -> list[dict]:
    vector = VectorStore(DeterministicHashEmbedding())
    vector.add_chunks(chunks)
    rows = []
    for case in cases:
        query = query_for(case, ids)
        server = create_mcp_server(store)
        graph = AgenticRAGGraph(store, vector, OllamaProvider(MODEL), context_tokens=GENERATION["num_ctx"], max_attempts=1, mcp_client=MCPToolClient(server, max_calls=2))
        started = time.perf_counter()
        state = graph.run(query, limit=5)
        metrics = state["metrics"]
        citations = state.get("citations", [])
        answer = state.get("answer", "")
        status = state.get("status", "UNKNOWN")
        rows.append({"id": case["id"], "status": status, "answer": answer, "citations": citations, "retrieved_chunks": [r.chunk.chunk_id for r in state.get("results", [])], "metrics": score(case, answer, status, citations, [r.chunk.chunk_id for r in state.get("results", [])]), "trace": {"attempts": metrics.get("attempts", 0), "rewrites": metrics.get("rewrites", 0), "llm_calls": metrics.get("llm_calls", 0), "mcp_calls": metrics.get("tool_calls", 0), "strategy": state.get("strategy", "unknown"), "route_mode": state.get("route_mode", "unknown")}, "latency_ms": {"retrieval": round(sum(float(x.get("details", {}).get("retrieval_ms", 0)) for x in state.get("decision_trace", [])), 3), "generation": state.get("generation", {}).get("latency_ms", 0), "total": round((time.perf_counter() - started) * 1000, 3)}, "error": state.get("error")})
    return rows


def aggregate(rows: list[dict], cases: list[dict]) -> dict:
    valid = [row for row in rows if row["metrics"] is not None]
    def mean(key: str) -> float:
        return round(sum(float(row["metrics"][key]) for row in valid) / len(valid), 6) if valid else 0.0
    n = len(rows) or 1
    trace_keys = ("attempts", "rewrites", "llm_calls", "mcp_calls")
    trace = {key: sum(int(row["trace"].get(key, 0)) for row in rows) for key in trace_keys}
    categories = defaultdict(list)
    for row, case in zip(rows, cases):
        categories[case["category"]].append(row)
    category_metrics = {name: {"n": len(items), "quality": round(sum((item["metrics"] or {}).get("quality", 0) for item in items) / len(items), 6), "fact_recall": round(sum((item["metrics"] or {}).get("fact_recall", 0) for item in items) / len(items), 6)} for name, items in sorted(categories.items())}
    return {"n": len(rows), "valid": len(valid), "quality": mean("quality"), "fact_recall": mean("fact_recall"), "source_precision": mean("source_precision"), "source_recall": mean("source_recall"), "abstention_accuracy": round(sum(bool((row["metrics"] or {}).get("abstention_correct")) for row in valid) / len(valid), 6) if valid else 0.0, "mean_latency_ms": round(sum(float(row["latency_ms"]["total"]) for row in rows) / n, 3), "attempts": trace["attempts"], "rewrites": trace["rewrites"], "llm_calls": trace["llm_calls"], "mcp_calls": trace["mcp_calls"], "retry_percentage": round(100 * sum(int(row["trace"].get("attempts", 0)) > 1 for row in rows) / n, 3), "rewrite_percentage": round(100 * sum(int(row["trace"].get("rewrites", 0)) > 0 for row in rows) / n, 3), "mcp_percentage": round(100 * sum(int(row["trace"].get("mcp_calls", 0)) > 0 for row in rows) / n, 3), "strategy_distribution": dict(Counter(row["trace"].get("strategy", "unknown") for row in rows)), "categories": category_metrics, "provider_errors": sum(row["status"] == "PROVIDER_ERROR" for row in rows)}


def main() -> int:
    fixture, store, chunks, ids = load_fixture()
    cases = fixture["cases"]
    baseline = run_baseline(cases, store, chunks, ids)
    agentic = run_agentic(cases, store, chunks, ids)
    result = {"version": "baseline-vs-agentic-v1.1", "dataset": {"path": str(DATASET.relative_to(ROOT)), "sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(), "corpus_sha256": hashlib.sha256(json.dumps(fixture["corpus"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "cases": len(cases)}, "provider": {"name": "ollama", "model": MODEL, "timeout_s": 120, "generation": GENERATION}, "strategies": {"baseline": {"cases": baseline, "aggregate": aggregate(baseline, cases)}, "agentic": {"cases": agentic, "aggregate": aggregate(agentic, cases)}}, "before": json.loads(BEFORE.read_text())["strategies"], "regressions": [], "limitations": ["Fact and source scoring is deterministic substring/citation scoring; no sole LLM judge.", "CPU local-model latency is hardware-dependent."]}
    for strategy in ("baseline", "agentic"):
        before = result["before"][strategy]["aggregate"]
        after = result["strategies"][strategy]["aggregate"]
        for metric in ("quality", "fact_recall", "source_precision", "source_recall", "abstention_accuracy"):
            if float(after[metric]) < float(before[metric]):
                result["regressions"].append({"strategy": strategy, "metric": metric, "before": before[metric], "after": after[metric]})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUTPUT.relative_to(ROOT)), "dataset_sha256": result["dataset"]["sha256"], "baseline": result["strategies"]["baseline"]["aggregate"], "agentic": result["strategies"]["agentic"]["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
