#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-vm/bin/python"
DATA="$ROOT/data"
COMPOSE_FILE="$ROOT/docker-compose.benchmark.yml"
MODEL_ADAPTER_URL="http://127.0.0.1:5000"
JINA_EMBEDDING_URL="http://127.0.0.1:5000/embed/jina"
GTE_EMBEDDING_URL="http://127.0.0.1:5000/embed/gte"
BGE_RERANK_URL="http://127.0.0.1:5001/rerank/bge"
QWEN_RERANK_URL="http://127.0.0.1:5002/rerank/qwen"
QDRANT_URL="http://127.0.0.1:5019"
QDRANT_GRPC_URL="http://127.0.0.1:5020"
WEAVIATE_URL="http://127.0.0.1:5004"
WEAVIATE_GRPC_URL="127.0.0.1:5005"
export MODEL_ADAPTER_URL JINA_EMBEDDING_URL GTE_EMBEDDING_URL
export BGE_RERANK_URL QWEN_RERANK_URL QDRANT_URL QDRANT_GRPC_URL
export WEAVIATE_URL WEAVIATE_GRPC_URL

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail_logs() {
  local rc=$?
  echo "VM stack startup failed (exit $rc). Recent logs:" >&2
  for file in "$DATA"/model_adapter_500{0,1,2}.log "$DATA/dashboard_5011.log"; do
    if [[ -f "$file" ]]; then
      echo "--- $file ---" >&2
      tail -n 40 "$file" >&2 || true
    fi
  done
  exit "$rc"
}
trap fail_logs ERR

if [[ ! -x "$PY" ]]; then
  echo "Missing $PY. Run VM dependency setup first." >&2
  exit 1
fi
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Missing $COMPOSE_FILE" >&2
  exit 1
fi
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }

mkdir -p "$DATA"

write_runtime_endpoints() {
  ROOT="$ROOT" "$PY" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
path = root / ".env.vm.generated"
values = {
    key: os.environ[key]
    for key in (
        "MODEL_ADAPTER_URL",
        "JINA_EMBEDDING_URL",
        "GTE_EMBEDDING_URL",
        "BGE_RERANK_URL",
        "QWEN_RERANK_URL",
        "QDRANT_URL",
        "QDRANT_GRPC_URL",
        "WEAVIATE_URL",
        "WEAVIATE_GRPC_URL",
    )
}
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
lines = [line for line in lines if line.split("=", 1)[0].strip() not in values]
lines.extend(f"{key}={value}" for key, value in values.items())
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Updated endpoint routing: {path}")
PY
}

stop_pidfile() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

wait_http() {
  local url="$1"
  local name="$2"
  local attempts="${3:-90}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS --max-time 30 "$url" >/dev/null 2>&1; then
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

start_adapter() {
  local gpu="$1"
  local port="$2"
  local log_file="$DATA/model_adapter_${port}.log"
  local pid_file="$DATA/model_adapter_${port}.pid"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    WNS_MODEL_DEVICE=cuda \
    QWEN_RERANK_BATCH_SIZE=8 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u -m uvicorn scripts.wns_vm_adapter_service:app \
      --app-dir "$ROOT" --host 127.0.0.1 --port "$port" \
      >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  log "STARTED adapter GPU=$gpu port=$port PID=$(cat "$pid_file")"
}

log "Starting WNS VM stack from $ROOT"
write_runtime_endpoints

stop_pidfile "$DATA/model_adapter_5000.pid"
stop_pidfile "$DATA/model_adapter_5001.pid"
stop_pidfile "$DATA/model_adapter_5002.pid"
stop_pidfile "$DATA/dashboard_5011.pid"
if command -v fuser >/dev/null 2>&1; then
  for port in 5000 5001 5002 5011; do
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  done
fi
sleep 2

log "Starting Qdrant, PGVector, and Weaviate"
docker compose -f "$ROOT/docker-compose.benchmark.yml" up -d qdrant postgres-pgvector weaviate

start_adapter 0 5000
start_adapter 1 5001
start_adapter 2 5002

nohup "$PY" -u "$ROOT/scripts/serve_benchmark_dashboard.py" 5011 0.0.0.0 \
  >"$DATA/dashboard_5011.log" 2>&1 &
echo $! >"$DATA/dashboard_5011.pid"
log "STARTED dashboard port=5011 PID=$(cat "$DATA/dashboard_5011.pid")"

wait_http "http://127.0.0.1:5000/health" "embedding adapter"
wait_http "http://127.0.0.1:5001/health" "BGE adapter"
wait_http "http://127.0.0.1:5002/health" "Qwen adapter"
wait_http "http://127.0.0.1:5019/healthz" "Qdrant"
wait_tcp 127.0.0.1 5020 "Qdrant gRPC"
wait_tcp 127.0.0.1 5003 "PGVector"
wait_http "http://127.0.0.1:5004/v1/.well-known/ready" "Weaviate"
wait_tcp 127.0.0.1 5005 "Weaviate gRPC"
wait_http "http://127.0.0.1:5011/api/results" "dashboard/frontend" 120

VM_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
log "ALL SERVICES READY"
echo "Frontend: http://${VM_IP:-127.0.0.1}:5011"
echo "Logs: $DATA/model_adapter_5000.log, $DATA/model_adapter_5001.log, $DATA/model_adapter_5002.log, $DATA/dashboard_5011.log"
