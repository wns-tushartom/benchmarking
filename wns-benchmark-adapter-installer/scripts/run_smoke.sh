#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_smoke.sh /path/to/Retreiver" >&2
  exit 2
fi

REPO="$1"
cd "$REPO"

python3 scripts/check_env.py
python3 scripts/check_services.py

python3 scripts/benchmark_cli.py run \
  --limit-queries 5 \
  --output-dir data/modular_runs/vm_adapter_jina_qdrant_smoke \
  --chunker entity_heuristic_w6 \
  --embedding jina_v3 \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker bge-reranker-base

python3 scripts/benchmark_cli.py run \
  --limit-queries 5 \
  --output-dir data/modular_runs/vm_adapter_gte_qdrant_smoke \
  --chunker entity_heuristic_w6 \
  --embedding gte_multilingual_base \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker bge-reranker-base
