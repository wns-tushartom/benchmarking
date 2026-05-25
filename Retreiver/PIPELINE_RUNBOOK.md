# WNS Benchmark Pipeline Runbook

## What is implemented

This repo now has a complete, runnable local/offline benchmark pipeline for the requested matrix:

```text
4 chunking methods × 3 embedding labels × 3 vector DB labels × 1 HNSW index × 1 cosine retrieval × 3 rerankers = 108 configurations
```

It also has a working dashboard served by Python standard library.

The chunking workbook reader includes a Python-standard-library `.xlsx` fallback, so `Run chunk recall` does **not** require `openpyxl` to be installed.

## Important honesty note

The current pipeline uses **local deterministic fallback adapters** for embeddings, vector DBs, and rerankers. This means:

- It validates the full orchestration, matrix generation, recall calculation, CSV outputs, and frontend.
- It gives useful relative signal for chunking and reranking behavior.
- It is **not final production latency** for Jina/OpenAI/Qdrant/PGVector/Weaviate/Amazon/Qwen/BGE services.

Production numbers require replacing the fallback adapters in `source/benchmark_pipeline.py` with real provider clients and DB containers/API services.

## Main files added

```text
source/benchmark_pipeline.py
scripts/run_full_benchmark_matrix.py
scripts/compare_chunking_recall.py
scripts/serve_benchmark_dashboard.py
web/index.html
web/styles.css
web/app.js
tests/test_benchmark_pipeline_core.py
DESIGN.md
```

## Outputs produced

```text
data/chunking_recall_results.csv
data/chunking_recall_summary.csv
data/full_benchmark/benchmark_summary.csv
data/full_benchmark/benchmark_details.csv
data/full_benchmark/benchmark_report.json
```

## Commands

### Run unit tests

```bash
python3 -m unittest tests.test_benchmark_pipeline_core -v
```

### Compare chunking recall

```bash
python3 scripts/compare_chunking_recall.py --limit 50
```

### Run official 135-combination matrix

```bash
python3 scripts/run_full_benchmark_matrix.py --limit 50
```

For a quick smoke test:

```bash
python3 scripts/run_full_benchmark_matrix.py --limit 10 --max-runs 6
```

Include extra candidate chunking methods:

```bash
python3 scripts/run_full_benchmark_matrix.py --limit 50 --include-candidates
```

Include Milvus as extra baseline DB:

```bash
python3 scripts/run_full_benchmark_matrix.py --limit 50 --include-milvus
```

### Start dashboard

```bash
python3 scripts/serve_benchmark_dashboard.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Verified run

Verification completed locally:

- Unit tests: 6/6 passed
- Chunking recall: completed across 6 chunking sheets with 20 query cases
- Full official matrix: completed 108/108 configurations with 10 query cases
- Output rows:
  - `benchmark_summary.csv`: 108 rows
  - `benchmark_details.csv`: 1080 rows
- Errors in summary: 0
- Dashboard API verified at `/api/results`
- Dashboard run endpoint verified at `/api/run/chunking?limit=5`

## Current chunking result from smoke run

For candidate comparison:

- `semantic_split`: Recall@5 = 0.85, Recall@10 = 0.90, 122 chunks
- `fixed_tok1200_ov150`: Recall@5 = 0.70, Recall@10 = 0.80, 56 chunks

Current recommendation from local fallback recall: **semantic_split beats fixed_tok1200_ov150**.

## Next production-hardening steps

1. Add real embedding adapters:
   - Jina v3
   - GTE multilingual base
   - OpenAI text-embedding-3-large
2. Add real vector DB adapters/services:
   - Qdrant HNSW
   - PGVector HNSW
   - Weaviate HNSW
3. Add real rerank adapters:
   - Amazon Rerank v1
   - Qwen3:4B Rerank
   - bge-reranker-base
4. Re-run the same matrix with provider mode enabled.
5. Compare provider results against the local fallback outputs.
