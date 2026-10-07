from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.citations.models import citations_from
from agentic_rag.generation.grounding import (
    Context,
    build_context,
    grounded_prompt,
    validate_citations,
    validate_grounding,
)
from agentic_rag.generation.provider import LLMProvider, LLMResponse
from agentic_rag.storage.store import InMemoryStore


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations: list[dict[str, object]]
    retrieved_chunks: list[str]
    status: str
    retrieval: str
    generation: dict[str, object]
    grounding: dict[str, object]
    latency_ms: float
    context: Context


class RetrievalService:
    def __init__(self, store: InMemoryStore, provider: LLMProvider | None = None, context_tokens: int = 1800) -> None:
        self.store, self.provider, self.context_tokens = store, provider, context_tokens

    def answer(self, query: str, limit: int = 5):
        """Compatibility lexical control: no generation and deterministic output."""
        from agentic_rag.citations.models import Answer
        results = self.store.search(query, limit)
        if not results:
            return Answer("INSUFFICIENT_CONTEXT", [], [], False, "INSUFFICIENT_CONTEXT")
        return Answer(" ".join(r.chunk.content for r in results), citations_from(results), [r.chunk.chunk_id for r in results], True)

    def generate(self, query: str, results: list, retrieval: str) -> GenerationResult:
        context = build_context(results, self.context_tokens)
        started = __import__("time").perf_counter()
        if not context.chunks:
            return GenerationResult("INSUFFICIENT_CONTEXT", [], [], "INSUFFICIENT_CONTEXT", retrieval, {"status": "not_run"}, {"status": "unsupported"}, 0.0, context)
        if self.provider is None:
            raise RuntimeError("LLM provider is required for generation")
        response: LLMResponse = self.provider.generate(grounded_prompt(query, context))
        valid, invalid = validate_citations(response.answer, context)
        grounding = validate_grounding(response.answer, context, valid, invalid)
        status = "INSUFFICIENT_CONTEXT" if response.answer.strip().upper() == "INSUFFICIENT_CONTEXT" else ("OK" if not invalid else "UNSUPPORTED_CITATION")
        citations = [{"document_id": r.chunk.document_id, "chunk_id": r.chunk.chunk_id, "source": r.chunk.metadata.get("source"), "filename": r.chunk.metadata.get("filename"), "score": r.score} for r in context.chunks if r.chunk.chunk_id in valid]
        total = ( __import__("time").perf_counter() - started) * 1000
        return GenerationResult(response.answer, citations, [r.chunk.chunk_id for r in context.chunks], status, retrieval, {"status": "ok", "provider": response.provider, "model": response.model, "latency_ms": response.latency_ms, "prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens, "total_tokens": response.total_tokens}, {"status": grounding.status, "supported_claims": grounding.supported_claims, "unsupported_claims": grounding.unsupported_claims}, total, context)
