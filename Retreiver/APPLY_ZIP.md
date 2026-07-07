# Apply WNS Retreiver final benchmark dashboard overlay

Apply from the Retreiver repository root, not the parent workspace.

```bash
cd ~/benchmarking/Retreiver
unzip -o /path/to/wns_retreiver_final_product_steps_4_7_20260707.zip
```

## Product defaults

- Product dashboard default: `configs/benchmark.local.json`.
- Full matrix: `5 chunkers × 3 embeddings × 4 vector stores × 1 HNSW × 1 cosine retrieval × 3 rerankers = 180`.
- Secondary no-AWS mode: `configs/benchmark.faiss-noaws.json`, 20 FAISS/OSS-only configs.
- AWS/Bedrock rerank is gated. Do not run Amazon Rerank jobs unless credentials and cost approval are present.

## What this overlay updates

- Adds `tests/test_dashboard_metrics.py` to pin:
  - FAISS appears in pipeline stage winners.
  - Duplicate metric rows collapse to one canonical combo.
  - `read_evaluation()` benchmark-reference rows include FAISS without double-counting archived retries.
  - Broad dashboard evidence lanes can cover the official 180 combos across evaluation, reranked evaluation, direct runs, and nested archive runs.
- Documents the two config modes in `README.md` and `PIPELINE_RUNBOOK.md`.
- Adds `configs/benchmark.faiss-noaws.json` for the 20-combo FAISS/OSS-only mode.
- Updates dashboard one-command start to port `5011`.
- Updates artifact sync to include recursive modular run artifacts plus the dashboard’s evaluation, reranked, reranker-analysis, hallucination, NVIDIA, ingestion, retrieval, and smoke evidence paths.
- Documents modular run retention: keep `latest/`, keep one complete run, archive or delete `faiss_noaws_*` retries.
- Keeps the prior dashboard integrity fixes: FAISS evidence paths, canonical reranker labels, de-dupe by full matrix key, blocked AWS FAISS cells, PGVector `halfvec`, Qdrant `5019/5020`, Bedrock ARN normalization.

## Verify after applying

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate  # if applicable
python -m py_compile scripts/serve_benchmark_dashboard.py scripts/collect_vm_dashboard_artifacts.py scripts/run_retrieval_smoke_from_vm_dbs.py benchmarking/adapters/vector_pgvector.py benchmarking/adapters/remote_rerankers.py
node --check web/app.js
python -m pytest tests/test_dashboard_metrics.py tests/test_modular_benchmark.py -q
python scripts/benchmark_cli.py validate configs/benchmark.local.json
python scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
```

Expected from this build:

```text
29 passed, 1 warning
benchmark.local.json matrix_count = 180
benchmark.faiss-noaws.json matrix_count = 20
```

Dashboard smoke:

```bash
python scripts/serve_benchmark_dashboard.py 5011
# in another shell
curl -s http://127.0.0.1:5011/api/results | python -m json.tool | head -80
```

## Release tag

Do this only after the work laptop has committed the final overlay into the real WNS source repo:

```bash
git tag -a v1.0-benchmark -m "WNS Retreiver benchmark dashboard v1.0"
git push origin v1.0-benchmark
```

Do not tag a dirty VM working tree. That tag would not reproduce the final product.
