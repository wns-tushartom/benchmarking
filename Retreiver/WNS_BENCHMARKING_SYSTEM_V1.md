# WNS Benchmarking System v1

## What changed

This repo now contains a modular, config-driven benchmarking harness in addition to the original WNS scripts.

It supports:

- Config-driven experiment matrices.
- Exact official 135-combination WNS matrix:
  - 5 chunkers
  - 3 embeddings
  - 3 vector stores
  - HNSW index
  - Cosine Similarity retrieval
  - 3 rerankers
- Dashboard selectors for running one selected combination or a partial/full matrix.
- Adapter registry for chunkers, embeddings, vector stores, rerankers, evaluators.
- Local/offline fallback adapters for clean-machine execution.
- Richer metrics: Recall@1/3/5/10, MRR, nDCG@10, Precision@5.
- Bootstrap confidence intervals for Recall@5.
- Query category inference and category-level recall.
- Hit/miss examples for winner explanation.
- Run manifest with config hash, dataset hash, git SHA, Python version, provider readiness.
- Pareto frontier analysis.
- Dashboard integration with WNS logo and modular winner rationale.

## Main files

```text
configs/benchmark.local.json
benchmarking/core/config.py
benchmarking/core/registry.py
benchmarking/core/metrics.py
benchmarking/core/schemas.py
benchmarking/core/runner.py
benchmarking/adapters/local.py
scripts/benchmark_cli.py
tests/test_modular_benchmark.py
web/wns-logo.svg
web/index.html
web/app.js
web/styles.css
SELECTABLE_COMBINATIONS.md
WNS_WORK_LAPTOP_SETUP.md
docker-compose.benchmark.yml
requirements-benchmark.txt
.env.example
.env
.gitignore
ENV_SETUP.md
VM_EMBEDDING_ENDPOINTS.md
WINDOWS_ADAPTER_SETUP.md
scripts/check_services.py
scripts/windows/Setup-WnsBenchmark.cmd
scripts/windows/Load-WnsEnv.cmd
scripts/windows/Check-WnsSetup.cmd
scripts/windows/Start-WnsDashboard.cmd
scripts/windows/Run-SelectedBenchmark.cmd
scripts/windows/Setup-WnsBenchmark.ps1
scripts/windows/Load-WnsEnv.ps1
scripts/windows/Check-WnsSetup.ps1
scripts/windows/Start-WnsDashboard.ps1
scripts/windows/Run-SelectedBenchmark.ps1
```

## CLI

Validate config:

```bash
python3 scripts/benchmark_cli.py validate
```

Show generated matrix:

```bash
python3 scripts/benchmark_cli.py matrix
```

Run a smoke benchmark:

```bash
python3 scripts/benchmark_cli.py run --max-runs 12 --limit-queries 40 --output-dir data/modular_runs/latest
```

Run the full modular local benchmark:

```bash
python3 scripts/benchmark_cli.py run --output-dir data/modular_runs/full
```

## Current official selectable matrix

The default modular matrix is now the official WNS set:

- Configurations: **135**
- Dimensions: `chunking_method`, `embedding_model`, `vector_database`, `index_type`, `retrieval_method`, `reranking_model`
- Index: `HNSW`
- Retrieval: `Cosine Similarity`

See:

```text
SELECTABLE_COMBINATIONS.md
WNS_WORK_LAPTOP_SETUP.md
```

## How to add a new model or technique

### Add a config entry

Edit:

```text
configs/benchmark.local.json
```

Example embedding:

```json
"bge_m3": {
  "adapter": "local_hash",
  "dimensions": 1024,
  "provider_ready": false
}
```

Then include it in:

```json
"matrix": {
  "embeddings": ["jina_v3", "bge_m3"]
}
```

### Add a real adapter

Add a new adapter class under:

```text
benchmarking/adapters/
```

Then register it in:

```text
benchmarking/core/registry.py
```

The runner does not need to change if the adapter follows the contract.

## Adapter contracts

Chunker:

```python
def chunk(self) -> list[Chunk]
```

Embedding:

```python
def embed_many(self, texts: Iterable[str]) -> list[list[float]]
```

Vector store:

```python
def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> dict
def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]
```

Reranker:

```python
def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]
```

Evaluator:

```python
def flags(self, query: QueryCase, hits: list[SearchHit]) -> list[bool]
```

## Dashboard

Server:

```bash
python3 scripts/serve_benchmark_dashboard.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

Dashboard now shows:

- WNS logo.
- Original benchmark summary.
- Modular 135-run benchmark summary.
- Selector controls for chunking, embedding, vector database, index, retrieval, rerank.
- Modular manifest details.
- Why-winner explanation.
- Pareto chart.
- Modular leaderboard with MRR, nDCG, CI.
- Existing recall/latency/chunking charts.

## What is still needed for production mode

This v1 is structurally modular. Production-valid results still need real adapters for:

- OpenAI embeddings.
- Jina embeddings.
- Qdrant real HNSW.
- PGVector real HNSW.
- Weaviate real HNSW.
- BGE/Qwen real rerankers.
- Amazon Rerank once AWS region/credentials are ready.

The config/registry structure is now ready for those adapters without changing runner logic.
