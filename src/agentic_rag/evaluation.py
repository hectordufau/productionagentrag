from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_rag.storage.models import SearchResult


@dataclass(frozen=True)
class QueryCase:
    query: str
    relevant_chunk_ids: tuple[str, ...]


def evaluate(
    cases: list[QueryCase], search: Callable[[str, int], list[SearchResult]], limit: int = 5
) -> dict[str, float | int]:
    totals = {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        results = search(case.query, limit)
        latencies.append((time.perf_counter() - start) * 1000)
        relevant = set(case.relevant_chunk_ids)
        found = [result.chunk.chunk_id for result in results]
        hits = [index for index, chunk_id in enumerate(found, 1) if chunk_id in relevant]
        # An empty relevance set is an explicit no-answer case, not a free
        # precision point when the retriever returns unrelated documents.
        totals["precision"] += len(hits) / len(found) if relevant and found else 0.0
        totals["recall"] += len(set(found) & relevant) / len(relevant) if relevant else 0.0
        totals["hit_rate"] += float(bool(hits))
        totals["mrr"] += 1.0 / hits[0] if hits else 0.0
        dcg = sum(1.0 / math.log2(index + 1) for index in hits)
        ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(relevant), limit) + 1))
        totals["ndcg"] += dcg / ideal if ideal else 0.0
    count = max(len(cases), 1)
    return {
        **{key: round(value / count, 6) for key, value in totals.items()},
        "queries": len(cases),
        "latency_ms_mean": round(sum(latencies) / count, 6),
        "latency_ms_p95": round(
            sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0, 6
        ),
    }
