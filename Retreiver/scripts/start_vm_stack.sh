#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-vm/bin/python"
DATA="$ROOT/data"
MANAGER="$ROOT/scripts/vm_adapter_manager.py"

MODEL_ADAPTER_URL="http://127.0.0.1:5000"
JINA_EMBEDDING_URL="http://127.0.0.1:5000/embed/jina"
GTE_EMBEDDING_URL="http://127.0.0.1:5001/embed/gte"
BGE_RERANK_URL="http://127.0.0.1:5002/rerank/bge"
QWEN_RERANK_URL="http://127.0.0.1:5006/rerank/qwen"
NEMOTRON_RERANK_URL="http://127.0.0.1:5007/rerank/nemotron"
GTE_MODERNBERT_RERANK_URL="http://127.0.0.1:5008/rerank/gte-modernbert"
NEMOTRON_3_EMBED_1B_BF16_URL="http://127.0.0.1:5010/v1/embeddings"
NEMOTRON_3_EMBED_1B_NVFP4_URL="http://127.0.0.1:5012/v1/embeddings"
NEMOTRON_3_EMBED_8B_BF16_URL="http://127.0.0.1:5013/v1/embeddings"
QDRANT_URL="http://127.0.0.1:5019"
QDRANT_GRPC_URL="http://127.0.0.1:5015"
WEAVIATE_URL="http://127.0.0.1:5004"
WEAVIATE_GRPC_URL="127.0.0.1:5005"
NVIDIA_RAG_SERVER_URL="http://127.0.0.1:5016"
NVIDIA_INGESTOR_URL="http://127.0.0.1:5017"
NVIDIA_RAG_FRONTEND_URL="http://127.0.0.1:5018"

export MODEL_ADAPTER_URL JINA_EMBEDDING_URL GTE_EMBEDDING_URL
export BGE_RERANK_URL QWEN_RERANK_URL NEMOTRON_RERANK_URL GTE_MODERNBERT_RERANK_URL
export NEMOTRON_3_EMBED_1B_BF16_URL NEMOTRON_3_EMBED_1B_NVFP4_URL NEMOTRON_3_EMBED_8B_BF16_URL
export QDRANT_URL QDRANT_GRPC_URL WEAVIATE_URL WEAVIATE_GRPC_URL
export NVIDIA_RAG_SERVER_URL NVIDIA_INGESTOR_URL NVIDIA_RAG_FRONTEND_URL

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ ! -x "$PY" ]]; then
  echo "Missing $PY. Run VM dependency setup first." >&2
  exit 1
fi
if [[ ! -f "$MANAGER" ]]; then
  echo "Missing $MANAGER" >&2
  exit 1
fi
mkdir -p "$DATA"

write_runtime_endpoints() {
  ROOT="$ROOT" "$PY" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
path = root / ".env.vm.generated"
if path.is_symlink():
    raise SystemExit("Refusing symlinked .env.vm.generated")
keys = (
    "MODEL_ADAPTER_URL",
    "JINA_EMBEDDING_URL",
    "GTE_EMBEDDING_URL",
    "BGE_RERANK_URL",
    "QWEN_RERANK_URL",
    "NEMOTRON_RERANK_URL",
    "GTE_MODERNBERT_RERANK_URL",
    "NEMOTRON_3_EMBED_1B_BF16_URL",
    "NEMOTRON_3_EMBED_1B_NVFP4_URL",
    "NEMOTRON_3_EMBED_8B_BF16_URL",
    "QDRANT_URL",
    "QDRANT_GRPC_URL",
    "WEAVIATE_URL",
    "WEAVIATE_GRPC_URL",
    "NVIDIA_RAG_SERVER_URL",
    "NVIDIA_INGESTOR_URL",
    "NVIDIA_RAG_FRONTEND_URL",
)
values = {key: os.environ[key] for key in keys}
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
lines = [line for line in lines if line.split("=", 1)[0].strip() not in values]
lines.extend(f"{key}={value}" for key, value in values.items())
temporary = path.with_suffix(".tmp")
temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(path)
print(f"Updated endpoint routing: {path}")
PY
}

wait_http() {
  local url="$1"
  local name="$2"
  local attempts="${3:-90}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS --max-time 10 "$url" >/dev/null 2>&1; then
      log "READY $name ($url)"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at $url" >&2
  return 1
}

wait_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  local attempts="${4:-60}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if "$PY" -c 'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),5); s.close()' "$host" "$port" >/dev/null 2>&1; then
      log "READY $name ($host:$port)"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at $host:$port" >&2
  return 1
}

port_open() {
  "$PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(.4); raise SystemExit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1]))) == 0 else 1)' "$1"
}

start_service() {
  local service_id="$1"
  log "Starting allowlisted service: $service_id"
  "$PY" "$MANAGER" start "$service_id" --root "$ROOT"
}

start_dashboard() {
  if curl -fsS --max-time 5 http://127.0.0.1:5011/api/results >/dev/null 2>&1; then
    log "READY dashboard already running"
    return 0
  fi
  if port_open 5011; then
    echo "Port 5011 is occupied by an unknown or unhealthy process; refusing to kill or replace it." >&2
    return 1
  fi
  nohup "$PY" -u "$ROOT/scripts/serve_benchmark_dashboard.py" 5011 0.0.0.0 \
    >"$DATA/dashboard_5011.log" 2>&1 &
  echo $! >"$DATA/dashboard_5011.pid"
  log "STARTED dashboard port=5011 PID=$(<"$DATA/dashboard_5011.pid")"
}

log "Starting WNS VM stack from $ROOT"
write_runtime_endpoints

start_service "pgvector"
start_service "weaviate_http"
start_service "qdrant_http"
start_service "jina_embedding"
start_service "gte_embedding"
start_service "bge_reranker"
start_service "qwen_reranker"
start_dashboard

wait_http "http://127.0.0.1:5000/health" "Jina embedding adapter"
wait_http "http://127.0.0.1:5001/health" "GTE embedding adapter"
wait_http "http://127.0.0.1:5002/health" "BGE reranker adapter"
wait_http "http://127.0.0.1:5006/health" "Qwen reranker adapter"
wait_http "http://127.0.0.1:5019/healthz" "Qdrant HTTP"
wait_tcp 127.0.0.1 5015 "Qdrant gRPC"
wait_tcp 127.0.0.1 5003 "PGVector"
wait_http "http://127.0.0.1:5004/v1/.well-known/ready" "Weaviate HTTP"
wait_tcp 127.0.0.1 5005 "Weaviate gRPC"
wait_http "http://127.0.0.1:5011/api/results" "dashboard/frontend" 120

VM_IP="$(hostname -I 2>/dev/null | cut -d' ' -f1)"
log "ALL CORE SERVICES READY"
echo "Frontend: http://${VM_IP:-127.0.0.1}:5011"
echo "Use the homepage adapter controls for candidate-only services on ports 5007, 5008, 5010, 5012, and 5013."
