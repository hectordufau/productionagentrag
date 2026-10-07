"""A small, explicit state graph for authorized Agentic RAG.

The graph deliberately keeps query analysis and routing rule-based. Only the
answer-writing node calls the configured provider, and it reuses the baseline
context, generation, citation, and grounding implementation.
"""
from __future__ import annotations

import re
import time
from typing import Literal, TypedDict

from agentic_rag.generation.baseline import GenerationResult, RetrievalService
from agentic_rag.generation.provider import LLMProvider, LLMProviderError
from agentic_rag.retrieval import DeterministicOverlapReranker, VectorStore, hybrid_search
from agentic_rag.storage.models import SearchResult
from agentic_rag.storage.store import InMemoryStore

Strategy = Literal["lexical", "vector", "hybrid"]


class AgenticState(TypedDict, total=False):
    query: str
    original_query: str
    rewritten_query: str
    attempt: int
    max_attempts: int
    strategy: Strategy
    results: list[SearchResult]
    evidence_score: float
    answer: str
    citations: list[dict[str, object]]
    status: str
    retrieval: dict[str, object]
    generation: dict[str, object]
    grounding: dict[str, object]
    decision_trace: list[dict[str, object]]
    metrics: dict[str, float | int]
    error: str | None
    metadata_filter: dict[str, object] | None
    limit: int
    context_tokens: int


_INJECTION = re.compile(
    r"\b(ignore|disregard|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|system|retrieved|instructions?)\b"
    r"|\b(reveal|leak|show)\s+(the\s+)?(system prompt|secrets?|hidden instructions?)\b",
    re.IGNORECASE,
)
_PARAPHRASE = re.compile(r"\b(in other words|meaning|refer to|explain|describe|what does .* mean|similar)\b", re.IGNORECASE)


def analyze_query(query: str) -> dict[str, object]:
    """Select a route using stable lexical rules; never call an LLM."""
    normalized = " ".join(query.split())
    if _INJECTION.search(normalized):
        return {"strategy": "lexical", "risk": "prompt_injection", "normalized": normalized}
    if _PARAPHRASE.search(normalized):
        strategy: Strategy = "vector"
    elif len(normalized.split()) >= 9 or re.search(r"\b(compare|versus|both|and)\b", normalized, re.IGNORECASE):
        strategy = "hybrid"
    else:
        strategy = "lexical"
    return {"strategy": strategy, "risk": "normal", "normalized": normalized}


def _rewrite(query: str) -> str:
    """Make one deterministic retrieval-oriented rewrite, without an LLM."""
    rewritten = re.sub(r"\b(what|where|when|who|why|how|is|are|can|could|please)\b", " ", query, flags=re.IGNORECASE)
    rewritten = re.sub(r"[^\w\s-]", " ", rewritten)
    rewritten = " ".join(rewritten.split())
    return rewritten or query


class AgenticRAGGraph:
    """Explicit analyze → retrieve → evaluate → rewrite → generate → validate graph."""

    def __init__(
        self,
        store: InMemoryStore,
        vector_store: VectorStore,
        provider: LLMProvider | None,
        *,
        reranker: DeterministicOverlapReranker | None = None,
        context_tokens: int = 1800,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1 or max_attempts > 2:
            raise ValueError("max_attempts must be between 1 and 2")
        self.store = store
        self.vector_store = vector_store
        self.provider = provider
        self.reranker = reranker
        self.context_tokens = context_tokens
        self.max_attempts = max_attempts
        self.service = RetrievalService(store, provider, context_tokens)

    def _retrieve(self, state: AgenticState) -> list[SearchResult]:
        query = state["query"]
        strategy = state["strategy"]
        metadata_filter = state.get("metadata_filter")
        limit = state["limit"]
        if strategy == "lexical":
            return self.store.search(query, limit, metadata_filter)
        if strategy == "vector":
            return self.vector_store.search(query, limit, metadata_filter)
        results = hybrid_search(self.store, self.vector_store, query, limit, metadata_filter)
        if self.reranker is not None:
            return self.reranker.rerank(query, results, limit)
        return results

    @staticmethod
    def _evidence(results: list[SearchResult]) -> float:
        return round(max((result.score for result in results), default=0.0), 6)

    def _record(self, state: AgenticState, node: str, **details: object) -> None:
        state["decision_trace"].append({"node": node, "attempt": state["attempt"], **details})

    def run(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: dict[str, object] | None = None,
        context_tokens: int | None = None,
    ) -> AgenticState:
        started = time.perf_counter()
        analysis = analyze_query(query)
        state: AgenticState = {
            "query": str(analysis["normalized"]),
            "original_query": query,
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "strategy": analysis["strategy"],  # type: ignore[typeddict-item]
            "results": [],
            "decision_trace": [],
            "metrics": {"attempts": 0, "llm_calls": 0, "rewrites": 0, "latency_ms": 0.0},
            "metadata_filter": metadata_filter,
            "limit": limit,
            "context_tokens": context_tokens or self.context_tokens,
            "status": "RUNNING",
        }
        self._record(state, "analyze", strategy=state["strategy"], risk=analysis["risk"])
        if analysis["risk"] == "prompt_injection":
            state["status"] = "ABSTAINED"
            state["answer"] = "INSUFFICIENT_CONTEXT"
            self._record(state, "abstain", reason="prompt_injection")
            state["metrics"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return state

        for attempt in range(1, self.max_attempts + 1):
            state["attempt"] = attempt
            state["metrics"]["attempts"] = attempt
            results = self._retrieve(state)
            state["results"] = results
            state["evidence_score"] = self._evidence(results)
            self._record(state, "retrieve", strategy=state["strategy"], results=len(results), evidence_score=state["evidence_score"])
            if not results:
                if attempt < self.max_attempts:
                    state["query"] = _rewrite(state["query"])
                    state["metrics"]["rewrites"] += 1
                    self._record(state, "rewrite", query=state["query"])
                    continue
                state["status"] = "ABSTAINED"
                state["answer"] = "INSUFFICIENT_CONTEXT"
                self._record(state, "abstain", reason="no_evidence")
                break

            self.service.context_tokens = state["context_tokens"]
            try:
                output: GenerationResult = self.service.generate(state["query"], results, state["strategy"])
                state["metrics"]["llm_calls"] += 1
            except LLMProviderError as exc:
                state["status"] = "PROVIDER_ERROR"
                state["answer"] = "INSUFFICIENT_CONTEXT"
                state["error"] = str(exc)
                self._record(state, "abstain", reason="provider_failure")
                break
            state.update(
                answer=output.answer,
                citations=output.citations,
                retrieval={"strategy": output.retrieval, "chunks": output.retrieved_chunks, "context_tokens": output.context.token_count},
                generation=output.generation,
                grounding=output.grounding,
            )
            self._record(state, "validate", status=output.status, grounding=output.grounding["status"], citations=len(output.citations))
            if output.status == "OK" and output.grounding["status"] == "grounded":
                state["status"] = "OK"
                break
            state["status"] = "ABSTAINED"
            if attempt < self.max_attempts and output.status != "INSUFFICIENT_CONTEXT":
                state["query"] = _rewrite(state["query"])
                state["metrics"]["rewrites"] += 1
                self._record(state, "rewrite", query=state["query"], reason="invalid_evidence")
                continue
            self._record(state, "abstain", reason="citation_or_grounding_failure")
            break
        state["metrics"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return state

    invoke = run
