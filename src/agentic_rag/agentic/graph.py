"""A small, explicit state graph for authorized Agentic RAG.

The graph deliberately keeps query analysis and routing rule-based. Only the
answer-writing node calls the configured provider, and it reuses the baseline
context, generation, citation, and grounding implementation.
"""
from __future__ import annotations

import re
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag.generation.baseline import GenerationResult, RetrievalService
from agentic_rag.generation.provider import LLMProvider, LLMProviderError
from agentic_rag.mcp.client import MCPToolClient, MCPToolError
from agentic_rag.retrieval import DeterministicOverlapReranker, hybrid_search
from agentic_rag.storage.models import SearchResult
from agentic_rag.storage.store import InMemoryStore

Strategy = Literal["lexical", "vector", "hybrid"]
RouteMode = Literal["retrieval_only", "tool_only", "retrieval_plus_tool"]


class AgenticState(TypedDict, total=False):
    query: str
    original_query: str
    rewritten_query: str
    attempt: int
    max_attempts: int
    strategy: Strategy
    route_mode: RouteMode
    document_id: str
    tool_result: dict[str, object]
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
_METADATA = re.compile(r"\b(metadata|filename|mime(?:[- ]type)?|checksum|document source|created)\b", re.IGNORECASE)
_DOCUMENT_ID = re.compile(r"\b(?:document|doc)[ _:-]+([A-Za-z0-9_.-]+)", re.IGNORECASE)


def _route_mode(query: str) -> tuple[RouteMode, str | None]:
    match = _DOCUMENT_ID.search(query)
    if (not _METADATA.search(query) and not re.search(r"\b(source|cite|citation|according)\b", query, re.IGNORECASE)) or match is None:
        return "retrieval_only", None
    document_id = match.group(1).rstrip("?.!,")
    if re.search(r"\b(source|cite|citation|according)\b", query, re.IGNORECASE):
        return "retrieval_plus_tool", document_id
    return "tool_only", document_id


def analyze_query(query: str) -> dict[str, object]:
    """Select a route using stable lexical rules; never call an LLM."""
    normalized = " ".join(query.split())
    route_mode, document_id = _route_mode(normalized)
    if _INJECTION.search(normalized):
        return {"strategy": "lexical", "risk": "prompt_injection", "normalized": normalized, "route_mode": route_mode, "document_id": document_id}
    if _PARAPHRASE.search(normalized):
        strategy: Strategy = "vector"
    elif len(normalized.split()) >= 9 or re.search(r"\b(compare|versus|both|and)\b", normalized, re.IGNORECASE):
        strategy = "hybrid"
    else:
        strategy = "lexical"
    return {"strategy": strategy, "risk": "normal", "normalized": normalized, "route_mode": route_mode, "document_id": document_id}


_REWRITE_FILLERS = {"what", "does", "where", "when", "who", "why", "how", "is", "are", "can", "could", "please", "refer", "to", "mean", "meaning"}


def rewrite_query(query: str) -> tuple[str, str]:
    """Return a safe deterministic search rewrite or the original query."""
    original = " ".join(query.split())
    tokens = re.findall(r"[\w-]+", original.lower())
    normalized = {"storage": "stores", "stored": "stores", "storing": "stores"}
    rewritten_tokens: list[str] = []
    for token in tokens:
        if token in _REWRITE_FILLERS:
            continue
        token = normalized.get(token, token)
        if token not in rewritten_tokens:
            rewritten_tokens.append(token)
    candidate = " ".join(rewritten_tokens)
    if len(rewritten_tokens) < 2 or candidate == original.lower():
        return original, "original"
    return candidate, "deterministic"


class AgenticRAGGraph:
    """Explicit analyze → retrieve → evaluate → rewrite → generate → validate graph."""

    def __init__(
        self,
        store: InMemoryStore,
        vector_store: Any,
        provider: LLMProvider | None,
        *,
        reranker: DeterministicOverlapReranker | None = None,
        context_tokens: int = 1800,
        max_attempts: int = 2,
        mcp_client: MCPToolClient | None = None,
    ) -> None:
        if max_attempts < 1 or max_attempts > 2:
            raise ValueError("max_attempts must be between 1 and 2")
        self.store = store
        self.vector_store = vector_store
        self.provider = provider
        self.reranker = reranker
        self.context_tokens = context_tokens
        self.max_attempts = max_attempts
        self.mcp_client = mcp_client
        self.service = RetrievalService(store, provider, context_tokens)
        self._compiled = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgenticState)
        graph.add_node("analyze", self._node_analyze)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("tool", self._node_tool)
        graph.add_node("evaluate", self._node_evaluate)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("generate", self._node_generate)
        graph.add_node("validate", self._node_validate)
        graph.add_node("abstain", self._node_abstain)
        graph.add_edge(START, "analyze")
        graph.add_conditional_edges("analyze", self._route_after_analyze, {"retrieve": "retrieve", "tool": "tool", "abstain": "abstain"})
        graph.add_conditional_edges("retrieve", self._route_after_retrieve, {"evaluate": "evaluate", "tool": "tool"})
        graph.add_conditional_edges("tool", self._route_after_tool, {"end": END, "generate": "generate", "abstain": "abstain"})
        graph.add_conditional_edges("evaluate", self._route_evidence, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"})
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", "validate")
        graph.add_conditional_edges("validate", self._route_validation, {"end": END, "rewrite": "rewrite", "abstain": "abstain"})
        graph.add_edge("abstain", END)
        return graph.compile()

    def _node_analyze(self, state: AgenticState) -> AgenticState:
        analysis = analyze_query(state["original_query"])
        state["query"] = str(analysis["normalized"])
        state["strategy"] = analysis["strategy"]  # type: ignore[typeddict-item]
        state["route_mode"] = analysis["route_mode"]  # type: ignore[typeddict-item]
        if analysis.get("document_id"):
            state["document_id"] = str(analysis["document_id"])
        self._record(state, "analyze", strategy=state["strategy"], risk=analysis["risk"], route_mode=state["route_mode"])
        if analysis["risk"] == "prompt_injection":
            state["status"] = "ABSTAINED"
            state["answer"] = "INSUFFICIENT_CONTEXT"
            state["error"] = "prompt_injection"
        return state

    @staticmethod
    def _route_after_analyze(state: AgenticState) -> str:
        if state.get("status") == "ABSTAINED":
            return "abstain"
        return "tool" if state.get("route_mode") == "tool_only" else "retrieve"

    @staticmethod
    def _route_after_retrieve(state: AgenticState) -> str:
        return "tool" if state.get("route_mode") == "retrieval_plus_tool" else "evaluate"

    @staticmethod
    def _route_after_tool(state: AgenticState) -> str:
        if state.get("status") == "MCP_ERROR":
            return "abstain"
        return "end" if state.get("route_mode") == "tool_only" else "generate"

    def _node_tool(self, state: AgenticState) -> AgenticState:
        started = time.perf_counter()
        if self.mcp_client is None or not state.get("document_id"):
            state["status"] = "MCP_ERROR"
            state["error"] = "MCP client unavailable or document_id missing"
            self._record(state, "tool", tool="get_document_metadata", status="error", tool_latency_ms=round((time.perf_counter() - started) * 1000, 3))
            return state
        try:
            state["tool_result"] = self.mcp_client.call("get_document_metadata", {"document_id": state["document_id"]})
        except MCPToolError as exc:
            state["status"] = "MCP_ERROR"
            state["error"] = str(exc)
            self._record(state, "tool", tool="get_document_metadata", status="error", tool_latency_ms=round((time.perf_counter() - started) * 1000, 3))
            return state
        tool_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        state["metrics"]["tool_calls"] += 1
        state["metrics"]["tool_latency_ms"] += tool_latency_ms
        document = state["tool_result"]["document"]
        state["answer"] = f"Metadata for {document['document_id']}: {document['filename']} ({document['mime_type']})."
        state["citations"] = [{"kind": "tool", "tool": "get_document_metadata", "document_id": document["document_id"]}]
        if state.get("route_mode") == "tool_only":
            state["status"] = "OK"
        self._record(state, "tool", tool="get_document_metadata", status="success", tool_latency_ms=tool_latency_ms, document_id=document["document_id"])
        return state

    def _node_retrieve(self, state: AgenticState) -> AgenticState:
        state["attempt"] += 1
        state["metrics"]["attempts"] = state["attempt"]
        state["results"] = self._retrieve(state)
        state["evidence_score"] = self._evidence(state["results"])
        self._record(state, "retrieve", strategy=state["strategy"], results=len(state["results"]), evidence_score=state["evidence_score"])
        return state

    def _node_evaluate(self, state: AgenticState) -> AgenticState:
        if state.get("status") == "ABSTAINED":
            return state
        return state

    def _route_evidence(self, state: AgenticState) -> str:
        if state.get("status") == "ABSTAINED":
            return "abstain"
        if state["results"] and state["evidence_score"] > 0.02:
            return "tool" if state.get("route_mode") == "retrieval_plus_tool" else "generate"
        if state["attempt"] < state["max_attempts"]:
            return "rewrite"
        return "abstain"

    def _node_rewrite(self, state: AgenticState) -> AgenticState:
        state["status"] = "RUNNING"
        original_query = state["query"]
        state["query"], method = rewrite_query(state["query"])
        state["metrics"]["rewrites"] += 1
        if state["evidence_score"] <= 0.1 and state["strategy"] != "hybrid":
            state["strategy"] = "hybrid"
        state["rewritten_query"] = state["query"]
        self._record(state, "rewrite", original_query=original_query, rewritten_query=state["query"], rewrite_method=method, strategy=state["strategy"])
        return state

    def _node_generate(self, state: AgenticState) -> AgenticState:
        self.service.context_tokens = state["context_tokens"]
        try:
            output: GenerationResult = self.service.generate(state["query"], state["results"], state["strategy"])
            state["metrics"]["llm_calls"] += 1
        except LLMProviderError as exc:
            state["status"] = "PROVIDER_ERROR"
            state["answer"] = "INSUFFICIENT_CONTEXT"
            state["error"] = str(exc)
            return state
        state.update(answer=output.answer, citations=output.citations, retrieval={"strategy": output.retrieval, "chunks": output.retrieved_chunks, "context_tokens": output.context.token_count}, generation=output.generation, grounding=output.grounding)
        if state.get("tool_result"):
            document = state["tool_result"]["document"]
            state["citations"].append({"kind": "tool", "tool": "get_document_metadata", "document_id": document["document_id"]})
        return state

    def _node_validate(self, state: AgenticState) -> AgenticState:
        output_status = state.get("status") if state.get("status") == "PROVIDER_ERROR" else ("OK" if state["grounding"]["status"] == "grounded" else "ABSTAINED")
        state["status"] = output_status
        self._record(state, "validate", status=output_status, grounding=state.get("grounding", {}).get("status"), citations=len(state.get("citations", [])))
        return state

    def _route_validation(self, state: AgenticState) -> str:
        if state["status"] == "OK":
            return "end"
        if state["status"] == "PROVIDER_ERROR" or state["attempt"] >= state["max_attempts"]:
            return "abstain"
        return "rewrite"

    def _node_abstain(self, state: AgenticState) -> AgenticState:
        state["status"] = "ABSTAINED" if state.get("status") != "PROVIDER_ERROR" else state["status"]
        state.setdefault("answer", "INSUFFICIENT_CONTEXT")
        self._record(state, "abstain", reason=state.get("error", "insufficient_or_invalid_evidence"))
        return state

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
        state: AgenticState = {
            "query": query,
            "original_query": query,
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "strategy": "lexical",
            "results": [],
            "decision_trace": [],
            "metrics": {"attempts": 0, "llm_calls": 0, "rewrites": 0, "tool_calls": 0, "tool_latency_ms": 0.0, "latency_ms": 0.0},
            "metadata_filter": metadata_filter,
            "limit": limit,
            "context_tokens": context_tokens or self.context_tokens,
            "status": "RUNNING",
        }
        state = self._compiled.invoke(state)
        state["metrics"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return state

    invoke = run
