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

## M2A.1-R1 retry status

The R1 semantic retry is blocked before benchmark execution. The exact existing `.[semantic]` dependency was attempted with a long timeout, but resolution proceeded to a 554.6 MB Torch wheel and then a 553.1 MB CUDA/cuDNN wheel without completing. No versions were changed, no model replacement was used, and no deterministic-hash fallback was accepted as semantic evidence. The frozen dataset checksum remained `eec5a26d8285c70daaf228c709f8f041ef693df503abbbca424f0f25dc2491cb`. The R1 artifact is intentionally absent; the historical artifact is preserved. M2B remains unauthorized.
