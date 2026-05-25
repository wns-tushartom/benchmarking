# WNS Benchmark Adapter Installer

This bundle installs real benchmark adapters into the existing `Retreiver` benchmark repo on the Linux VM.

It is separate from the VM services bundle. Use both:

1. `wns-vm-adapters`: runs embedding service and vector DB services.
2. `wns-benchmark-adapter-installer`: adds client adapters to the benchmark repo so it can call those services.

## What it installs

```text
benchmarking/adapters/remote_embeddings.py
benchmarking/adapters/vector_qdrant.py
benchmarking/adapters/vector_pgvector.py
benchmarking/adapters/vector_weaviate.py
```

It patches:

```text
benchmarking/core/registry.py
configs/benchmark.local.json
requirements-benchmark.txt
```

## Usage on Linux VM

Clone or unzip this installer on the VM, then run:

```bash
cd wns-benchmark-adapter-installer
bash scripts/install_adapters.sh /path/to/Retreiver
```

Example:

```bash
bash scripts/install_adapters.sh ~/main_benchmarking
```

Then in the benchmark repo:

```bash
cd ~/main_benchmarking
source .venv/bin/activate  # if venv exists
python -m pip install -r requirements-benchmark.txt
python scripts/benchmark_cli.py validate
python scripts/check_env.py
python scripts/check_services.py
```

## Required `.env` in benchmark repo

```env
JINA_EMBEDDING_URL=http://VM_HOST:5000/embed/jina
GTE_EMBEDDING_URL=http://VM_HOST:5000/embed/gte
QDRANT_URL=http://VM_HOST:5001
PGVECTOR_DSN=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
DATABASE_URL=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
WEAVIATE_URL=http://VM_HOST:5004
```

## First test run

```bash
python scripts/benchmark_cli.py run \
  --limit-queries 5 \
  --output-dir data/modular_runs/vm_adapter_smoke \
  --chunker entity_heuristic_w6 \
  --embedding jina_v3 \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker bge-reranker-base
```

Then try:

```bash
python scripts/benchmark_cli.py run \
  --limit-queries 5 \
  --output-dir data/modular_runs/vm_adapter_gte_smoke \
  --chunker entity_heuristic_w6 \
  --embedding gte_multilingual_base \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker bge-reranker-base
```
