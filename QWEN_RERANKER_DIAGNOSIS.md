# Qwen reranker accuracy diagnosis

## What was wrong

The previous VM adapter treated Qwen like a normal CrossEncoder over raw `(query, document)` pairs:

```python
pairs = [(query, document), ...]
model.predict(pairs)
```

That is not the safest path for Qwen3 rerankers. Qwen3 reranker is an instruction-aware yes/no relevance model. The expected input shape includes explicit fields:

```text
<Instruct>: ...
<Query>: ...
<Document>: ...
```

The repo was also pinned to:

```text
sentence-transformers==4.1.0
transformers==4.57.6
```

Sentence Transformers support for the official Qwen3 reranker chat/template path arrived later. With the older CrossEncoder path, Qwen can produce scores that are not calibrated for the intended reranking task. In practice that can demote the relevant retrieved chunk below less relevant chunks, so Qwen looks worse than no reranker.

## Secondary issue

The old dashboard flow allowed separate buttons for retrieval, reranker, and evaluation. That made it easy to evaluate mismatched artifacts:

- no-reranker evaluation from one retrieval run
- Qwen reranked evaluation from a limited or stale `data/reranker_smoke` subset
- old default `limit-artifacts=100`

That comparison is not reliable. Qwen must be compared against the exact same retrieval artifacts and ground-truth rows as the no-reranker baseline.

## Fix applied

- `scripts/wns_vm_adapter_service.py`
  - default Qwen model changed to `tomaarsen/Qwen3-Reranker-4B-seq-cls`
  - Qwen requests are formatted with `<Instruct>`, `<Query>`, and `<Document>` fields by default
  - `QWEN_RERANK_INSTRUCTION` added for WNS domain-specific relevance
  - `QWEN_RERANK_FORMATTED=1` controls the formatted path

- `scripts/run_complete_pipeline.py`
  - one command now runs ingestion, retrieval, baseline evaluation, reranking, reranked evaluation, Qwen-vs-none analysis, and dashboard artifact refresh
  - fresh runs archive old retrieval/reranker/evaluation artifacts first
  - reranker stage uses `--limit-artifacts 0` so the full paired set is evaluated

- `scripts/analyze_reranker_lift.py`
  - compares no-reranker vs Qwen row-by-row on the same query, chunker, embedding, and DB
  - writes exact improved/worse/tied rows under `data/reranker_analysis/`

## How to verify with eyes

Run the complete pipeline from the dashboard or CLI. Then inspect:

```text
data/evaluation/groundtruth_eval_summary.csv
data/evaluation_reranked/groundtruth_eval_summary.csv
data/reranker_analysis/reranker_lift_summary.csv
data/reranker_analysis/reranker_lift_details.csv
data/reranker_analysis/qwen_vs_none_report.json
```

The dashboard `Qwen vs no-reranker diagnosis` table shows the exact queries where Qwen moved the relevant answer lower than the no-reranker baseline.
