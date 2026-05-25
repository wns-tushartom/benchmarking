#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/install_adapters.sh /path/to/Retreiver" >&2
  exit 2
fi

REPO="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/scripts/patch_benchmark_repo.py" "$REPO"

cd "$REPO"
python3 -m py_compile \
  benchmarking/adapters/remote_embeddings.py \
  benchmarking/adapters/vector_qdrant.py \
  benchmarking/adapters/vector_pgvector.py \
  benchmarking/adapters/vector_weaviate.py \
  benchmarking/core/registry.py

python3 -m json.tool configs/benchmark.local.json >/tmp/wns_benchmark_config_valid.json
python3 scripts/benchmark_cli.py validate

echo "Adapter install verified."
echo "If dependencies are missing, run: python3 -m pip install -r requirements-benchmark.txt"
