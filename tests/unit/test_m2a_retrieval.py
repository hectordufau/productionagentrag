from agentic_rag.evaluation import QueryCase, evaluate
from agentic_rag.retrieval import (
    DeterministicHashEmbedding,
    DeterministicOverlapReranker,
    VectorStore,
    reciprocal_rank_fusion,
)
from agentic_rag.storage.models import Chunk, SearchResult


def _chunks() -> list[Chunk]:
    return [Chunk("a", "d", "Qdrant stores vectors", 0, {"topic": "vector"}), Chunk("b", "d", "PostgreSQL stores metadata", 1, {"topic": "metadata"})]


def test_vector_retrieval_is_deterministic_and_filters_metadata() -> None:
    index = VectorStore(DeterministicHashEmbedding(32))
    index.add_chunks(_chunks())
    first = index.search("vectors", 2, {"topic": "vector"})
    second = index.search("vectors", 2, {"topic": "vector"})
    assert [r.chunk.chunk_id for r in first] == ["a"]
    assert first == second


def test_rrf_score_is_explicit_and_rank_based() -> None:
    chunks = _chunks()
    one = [SearchResult(chunks[0], 1.0, "x")]
    two = [SearchResult(chunks[0], .5, "y")]
    fused = reciprocal_rank_fusion(one, two)
    assert fused[0].method == "hybrid-rrf"
    assert fused[0].score == 2 / 61


def test_reranker_and_metrics_are_machine_usable() -> None:
    index = VectorStore(DeterministicHashEmbedding())
    index.add_chunks(_chunks())
    results = DeterministicOverlapReranker().rerank("Qdrant vectors", index.search("stores", 2), 2)
    assert results[0].chunk.chunk_id == "a"
    metrics = evaluate([QueryCase("vectors", ("a",))], index.search)
    assert {"precision", "recall", "hit_rate", "mrr", "ndcg", "latency_ms_mean", "latency_ms_p95"} <= metrics.keys()
