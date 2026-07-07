# WNS Benchmark Pipeline Runbook

## Current product shape

The product dashboard uses `configs/benchmark.local.json` as the default full matrix:

```text
5 chunkers × 3 embeddings × 4 vector stores × 1 HNSW index × 1 cosine retrieval × 3 rerankers = 180 configurations
```

The intentionally gated lane is `Amazon Rerank v1`, which requires AWS Bedrock credentials. Do not rerun OpenAI or launch AWS/Bedrock jobs for the final product unless credentials and cost approval are explicitly available.

## Modes

| Mode | Config | Count | Use |
|---|---|---:|---|
| Full product matrix | `configs/benchmark.local.json` | 180 | Dashboard default and final product view |
| FAISS/OSS-only | `configs/benchmark.faiss-noaws.json` | 20 | Local/VM validation without OpenAI or AWS/Bedrock |

Validate both:

```bash
python3 scripts/benchmark_cli.py validate configs/benchmark.local.json
python3 scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
```

## One-command dashboard start

From the Retreiver repo root:

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

Expected JSON should include:

```text
operational.evaluation.benchmark_reference
operational.known_matrix_count = 180
options.matrix_count = 180
```

## Run selected benchmark combinations

Full product config, selected or bounded run:

```bash
python3 scripts/benchmark_cli.py run configs/benchmark.local.json \
  --output-dir data/modular_runs/latest \
  --max-runs 1 \
  --limit-queries 25
```

FAISS/OSS-only mode:

```bash
python3 scripts/benchmark_cli.py run configs/benchmark.faiss-noaws.json \
  --output-dir data/modular_runs/faiss_noaws_latest \
  --limit-queries 25
```

## VM service defaults

Local service ports used by the current runbooks:

| Service | Port |
|---|---:|
| Dashboard | 5011 |
| Model adapter | 5000 |
| Qdrant HTTP | 5019 |
| Qdrant gRPC | 5020 |
| PGVector | 5003 |
| Weaviate | 5004 |
| NVIDIA rag-server | 5016 |
| NVIDIA ingestor | 5017 |
| NVIDIA frontend | 5018 |

## Verification gate

Run before claiming the dashboard product is fixed:

```bash
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/run_retrieval_smoke_from_vm_dbs.py benchmarking/adapters/vector_pgvector.py benchmarking/adapters/remote_rerankers.py
node --check web/app.js
python3 -m pytest tests/test_dashboard_metrics.py tests/test_modular_benchmark.py -q
python3 scripts/benchmark_cli.py validate configs/benchmark.local.json
python3 scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
```

Expected:

```text
benchmark.local.json matrix_count = 180
benchmark.faiss-noaws.json matrix_count = 20
```

## Modular run retention

`data/modular_runs/` is runtime evidence. Keep `latest/`, keep one complete `full/` or named complete run, and sweep retry folders after their summary rows have been merged into the dashboard evidence.

Recommended local sweep for noisy retry folders:

```bash
mkdir -p data/modular_runs/archive/faiss_noaws
mv data/modular_runs/faiss_noaws_* data/modular_runs/archive/faiss_noaws/ 2>/dev/null || true
```

Do not commit generated experiment folders.
