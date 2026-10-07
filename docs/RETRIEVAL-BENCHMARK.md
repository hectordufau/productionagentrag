# Retrieval benchmark

## Reproduction

From the repository root:

```bash
python -m pip install -e '.[dev]'
python scripts/run_benchmark.py
```

The command reads `datasets/m2a-retrieval-v1.json`, uses the checked-in deterministic chunk fixture, and writes `artifacts/benchmarks/m2a-retrieval-v1.json`. `make benchmark-retrieval` is the equivalent target. Run `make evaluate-retrieval` to run the benchmark plus its focused evaluation tests.

## Dataset contract

The dataset is versioned and contains eight cases: exact, paraphrase, ambiguous, multi-term, similar-documents, and no-answer queries. Each case names its relevant chunk IDs. The runner reports the actual case count and does not fabricate relevance or results.

## Artifact contract

The JSON artifact records `git_commit`, dataset version/count, embedding provider/model/version/dimension, strategy configuration (`candidate_k`, `final_k`, fusion), reranker provider/model, explicit configuration, UTC timestamp, and metrics for lexical, vector, hybrid, and hybrid+reranking. Regenerate it after a code or dataset change; the timestamp and commit are intentionally evidence fields.

## Limits

This is an offline control benchmark, not a claim of semantic-model quality or live Qdrant availability. Latency is local process latency and is not a production capacity estimate. A live Qdrant deployment requires the optional compose service and separate integration evidence.
