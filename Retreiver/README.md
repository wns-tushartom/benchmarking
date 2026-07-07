# WNS Retreiver benchmark dashboard

This is the product entrypoint for the WNS RAG benchmark dashboard and modular benchmark harness.

## Product default

The product default is `configs/benchmark.local.json`.

The dashboard server intentionally loads this file via `scripts/serve_benchmark_dashboard.py` and reports the official full matrix:

```text
5 chunkers × 3 embeddings × 4 vector stores × 1 HNSW index × 1 cosine retrieval × 3 rerankers = 180 configurations
```

The full matrix includes commercial lanes:

- `openai_text-embedding-3-large` requires `OPENAI_API_KEY`.
- `Amazon Rerank v1` requires AWS Bedrock credentials, `AWS_REGION=us-west-2`, and `AMAZON_RERANK_MODEL_ID` as either `amazon.rerank-v1:0` or the full foundation-model ARN.
- Missing Amazon Rerank v1 + FAISS cells are displayed as `blocked` when AWS/Bedrock is not available. They are not a rerun request.

## Secondary mode: FAISS/OSS-only

`configs/benchmark.faiss-noaws.json` is the narrow local/VM validation mode:

```text
5 chunkers × 2 open-source embeddings × 1 FAISS store × 1 HNSW index × 1 cosine retrieval × 2 open-source rerankers = 20 configurations
```

Use it when AWS/Bedrock and OpenAI should stay out of scope:

```bash
python3 scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
python3 scripts/benchmark_cli.py run configs/benchmark.faiss-noaws.json --output-dir data/modular_runs/faiss_noaws_latest
```

## One-command dashboard start

From the Retreiver repo root on the VM or dashboard machine:

```bash
python3 scripts/serve_benchmark_dashboard.py 5011
```

Open:

```text
http://127.0.0.1:5011
```

API smoke:

```bash
curl -s http://127.0.0.1:5011/api/results | python3 -m json.tool | head -80
```

## Verification

```bash
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/run_retrieval_smoke_from_vm_dbs.py benchmarking/adapters/vector_pgvector.py benchmarking/adapters/remote_rerankers.py
node --check web/app.js
python3 -m pytest tests/test_dashboard_metrics.py tests/test_modular_benchmark.py -q
python3 scripts/benchmark_cli.py validate configs/benchmark.local.json
python3 scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
```

Expected config counts:

| Config | Purpose | Matrix count |
|---|---|---:|
| `configs/benchmark.local.json` | Product default, full dashboard matrix | 180 |
| `configs/benchmark.faiss-noaws.json` | FAISS + open-source rerankers only | 20 |

## Artifact retention rule

`data/modular_runs/` is generated runtime evidence, not source code.

Keep:

- `latest/`, the dashboard-selected current matrix run.
- `full/`, if it is the latest complete full run.
- One named complete archive for handoff evidence, if needed.

Sweep or delete:

- `faiss_noaws_*` retry iterations after their summary rows are copied into a complete run or dashboard artifact bundle.
- smoke runs older than the active debugging session.

If you need to preserve noisy FAISS/OSS attempts locally, move them under:

```bash
mkdir -p data/modular_runs/archive/faiss_noaws
mv data/modular_runs/faiss_noaws_* data/modular_runs/archive/faiss_noaws/ 2>/dev/null || true
```

Do not commit generated `data/modular_runs/*` experiment folders.

## Release cut

Only tag after the WNS source repo has committed the final dashboard/test/config/runbook changes:

```bash
git tag -a v1.0-benchmark -m "WNS Retreiver benchmark dashboard v1.0"
git push origin v1.0-benchmark
```

Do not tag a dirty working tree. A tag on uncommitted VM files is fake reproducibility.
