#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVER_DIR="$ROOT/Retreiver"
VM_HOST=""
SKIP_PIP=0
SKIP_DOCKER=0
SKIP_MODEL_SERVICE=0
SKIP_DASHBOARD=0
SKIP_MODEL_SMOKE=0
PREPARE_DATA=0
CHUNK_MODE=chunk-only
INCLUDE_IMAGE_MARKERS=0
WITH_NVIDIA_RAG=0
NVIDIA_RAG_ARGS=()
DRY_RUN=0
FORCE_ENV=0
MODEL_PORT=5000
DASHBOARD_PORT=5009
QDRANT_HTTP_PORT=5001
QDRANT_GRPC_PORT=5002
PGVECTOR_PORT=5003
WEAVIATE_HTTP_PORT=5004
WEAVIATE_GRPC_PORT=5005
MINERU_PORT=5010
WNS_POSTGRES_PASSWORD="${WNS_POSTGRES_PASSWORD:-wns_password}"

usage() {
  cat <<'USAGE'
Usage:
  bash setup_all_on_vm.sh [--host VM_HOST_OR_IP] [options]

Default behavior:
  1. Detect VM host/IP if --host is omitted
  2. Write Retreiver/.env.vm.generated
  3. Create Retreiver/.env if missing, or preserve existing .env
  4. Create Python venv and install model + benchmark adapter dependencies
  5. Start Qdrant, PGVector, and Weaviate through Docker Compose
  6. Start unified FastAPI model adapter service on port 5000
  7. Run health checks and lightweight smoke tests
  8. Write VM_SETUP_RESULT.txt

Options:
  --host HOST              Public/reachable VM host or IP used in generated .env URLs
  --force-env              Replace Retreiver/.env with generated VM env
  --skip-pip               Do not create venv or install Python packages
  --skip-docker            Do not start vector DB Docker Compose services
  --skip-model-service     Do not start model adapter FastAPI service
  --skip-dashboard         Do not start dashboard on port 5009
  --skip-model-smoke       Do not call model endpoints, only health URLs
  --prepare-data           Rebuild benchmark/chunking data after services start
  --chunk-mode MODE        Data prep mode for run_chunking_pipeline.py: all, missing, chunk-only. Default: chunk-only
  --include-image-markers  Pass image/layout markers into PDF extraction when --prepare-data uses all/missing
  --with-nvidia-rag        Also run setup_nvidia_rag_pipeline_on_vm.sh after base services
  --nvidia-rag-root PATH   Forward NVIDIA rag-main source path to NVIDIA setup
  --nvidia-rag-zip ZIP     Forward NVIDIA rag-main.zip path to NVIDIA setup
  --dry-run                Print actions and write generated files without starting services
  -h, --help               Show this help

Host-facing ports:
  5000  unified model adapter FastAPI: /embed/jina, /embed/gte, /rerank/qwen, /rerank/bge
  5001  Qdrant HTTP
  5002  Qdrant gRPC
  5003  PGVector/Postgres
  5004  Weaviate HTTP
  5005  Weaviate gRPC
  5009  WNS benchmark dashboard
  5010  MinerU/layout extraction API if configured separately
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) VM_HOST="${2:-}"; shift 2 ;;
    --force-env) FORCE_ENV=1; shift ;;
    --skip-pip) SKIP_PIP=1; shift ;;
    --skip-docker) SKIP_DOCKER=1; shift ;;
    --skip-model-service) SKIP_MODEL_SERVICE=1; shift ;;
    --skip-dashboard) SKIP_DASHBOARD=1; shift ;;
    --skip-model-smoke) SKIP_MODEL_SMOKE=1; shift ;;
    --prepare-data) PREPARE_DATA=1; shift ;;
    --chunk-mode) PREPARE_DATA=1; CHUNK_MODE="${2:-}"; shift 2 ;;
    --include-image-markers) INCLUDE_IMAGE_MARKERS=1; shift ;;
    --with-nvidia-rag) WITH_NVIDIA_RAG=1; shift ;;
    --nvidia-rag-root) WITH_NVIDIA_RAG=1; NVIDIA_RAG_ARGS+=(--rag-root "${2:-}"); shift 2 ;;
    --nvidia-rag-zip) WITH_NVIDIA_RAG=1; NVIDIA_RAG_ARGS+=(--rag-zip "${2:-}"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$CHUNK_MODE" in
  all|missing|chunk-only) ;;
  *) echo "Invalid --chunk-mode: $CHUNK_MODE" >&2; exit 2 ;;
esac

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MISSING command: $1" >&2
    return 1
  fi
}

detect_host() {
  if [[ -n "$VM_HOST" ]]; then
    echo "$VM_HOST"
    return
  fi
  local detected=""
  detected="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -z "$detected" ]]; then
    detected="127.0.0.1"
  fi
  echo "$detected"
}

existing_env_value() {
  local key="$1"
  local file="$RETRIEVER_DIR/.env"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); gsub(/^\"|\"$/, ""); gsub(/^\047|\047$/, ""); print; exit}' "$file"
}

write_env() {
  local host="$1"
  mkdir -p "$RETRIEVER_DIR/logs" "$RETRIEVER_DIR/run"
  local openai_api_key aws_access_key_id aws_secret_access_key aws_region aws_default_region amazon_rerank_model_id hf_token qwen_api_key
  openai_api_key="${OPENAI_API_KEY:-$(existing_env_value OPENAI_API_KEY)}"
  aws_access_key_id="${AWS_ACCESS_KEY_ID:-$(existing_env_value AWS_ACCESS_KEY_ID)}"
  aws_secret_access_key="${AWS_SECRET_ACCESS_KEY:-$(existing_env_value AWS_SECRET_ACCESS_KEY)}"
  aws_region="${AWS_REGION:-$(existing_env_value AWS_REGION)}"
  aws_default_region="${AWS_DEFAULT_REGION:-$(existing_env_value AWS_DEFAULT_REGION)}"
  amazon_rerank_model_id="${AMAZON_RERANK_MODEL_ID:-$(existing_env_value AMAZON_RERANK_MODEL_ID)}"
  hf_token="${HF_TOKEN:-$(existing_env_value HF_TOKEN)}"
  qwen_api_key="${QWEN_API_KEY:-$(existing_env_value QWEN_API_KEY)}"
  aws_region="${aws_region:-us-east-1}"
  aws_default_region="${aws_default_region:-$aws_region}"
  amazon_rerank_model_id="${amazon_rerank_model_id:-amazon.rerank-v1:0}"
  cat > "$RETRIEVER_DIR/.env.vm.generated" <<EOF
# Generated by setup_all_on_vm.sh
WNS_BENCHMARK_MODE=vm_remote_required

# Unified VM model adapter service, one HTTP port for model adapters
MODEL_ADAPTER_URL=http://$host:$MODEL_PORT
JINA_EMBEDDING_URL=http://$host:$MODEL_PORT/embed/jina
GTE_EMBEDDING_URL=http://$host:$MODEL_PORT/embed/gte
QWEN_RERANK_URL=http://$host:$MODEL_PORT/rerank/qwen
BGE_RERANK_URL=http://$host:$MODEL_PORT/rerank/bge

# Commercial APIs, fill only if using these commercial adapters
OPENAI_API_KEY=$openai_api_key
AWS_ACCESS_KEY_ID=$aws_access_key_id
AWS_SECRET_ACCESS_KEY=$aws_secret_access_key
AWS_REGION=$aws_region
AWS_DEFAULT_REGION=$aws_default_region
AMAZON_RERANK_MODEL_ID=$amazon_rerank_model_id

# Vector DBs on VM host-facing ports
QDRANT_URL=http://$host:$QDRANT_HTTP_PORT
QDRANT_GRPC_URL=http://$host:$QDRANT_GRPC_PORT
PGVECTOR_DSN=postgresql://wns:${WNS_POSTGRES_PASSWORD}@$host:$PGVECTOR_PORT/wns_benchmark
DATABASE_URL=postgresql://wns:${WNS_POSTGRES_PASSWORD}@$host:$PGVECTOR_PORT/wns_benchmark
WEAVIATE_URL=http://$host:$WEAVIATE_HTTP_PORT
WEAVIATE_GRPC_URL=$host:$WEAVIATE_GRPC_PORT
WNS_POSTGRES_PASSWORD=$WNS_POSTGRES_PASSWORD

# MinerU/layout extraction service, optional but recommended for final extraction runs
MINERU_API_URL=http://$host:$MINERU_PORT
MINERU_REQUIRE_LAYOUT=1

# Model names and device
JINA_EMBEDDING_MODEL=jinaai/jina-embeddings-v3
GTE_EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
HF_TOKEN=$hf_token
QWEN_RERANK_MODEL=tomaarsen/Qwen3-Reranker-4B-seq-cls
QWEN_API_KEY=$qwen_api_key
BGE_RERANKER_MODEL=BAAI/bge-reranker-base
WNS_MODEL_DEVICE=cuda
EOF

  if [[ "$FORCE_ENV" == "1" || ! -f "$RETRIEVER_DIR/.env" ]]; then
    run cp "$RETRIEVER_DIR/.env.vm.generated" "$RETRIEVER_DIR/.env"
    echo "Wrote $RETRIEVER_DIR/.env"
  else
    echo "Kept existing $RETRIEVER_DIR/.env. New generated values are in .env.vm.generated."
    echo "Use --force-env if you want this script to replace .env."
  fi
}

install_python_deps() {
  if [[ "$SKIP_PIP" == "1" ]]; then
    echo "Skipping Python package install."
    return
  fi
  need_cmd python3
  run python3 -m venv "$RETRIEVER_DIR/.venv-vm"
  # shellcheck source=/dev/null
  source "$RETRIEVER_DIR/.venv-vm/bin/activate"
  run python -m pip install --upgrade pip wheel
  run python -m pip install 'setuptools<82'
  if [[ -f "$RETRIEVER_DIR/requirements-benchmark.txt" ]]; then
    run python -m pip install -r "$RETRIEVER_DIR/requirements-benchmark.txt"
  fi
  if [[ "$PREPARE_DATA" == "1" && "$CHUNK_MODE" != "chunk-only" && -f "$RETRIEVER_DIR/requirements-mineru.txt" ]]; then
    run python -m pip install -r "$RETRIEVER_DIR/requirements-mineru.txt"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # WNS A10G VM reports driver CUDA 12.8. Avoid PyTorch cu130, which fails with driver-too-old.
    run python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
  else
    run python -m pip install torch torchvision torchaudio
  fi
}

start_vector_dbs() {
  if [[ "$SKIP_DOCKER" == "1" ]]; then
    echo "Skipping Docker vector DB startup."
    return
  fi
  need_cmd docker
  run env WNS_POSTGRES_PASSWORD="$WNS_POSTGRES_PASSWORD" docker compose -f "$RETRIEVER_DIR/docker-compose.benchmark.yml" up -d qdrant postgres-pgvector weaviate
}

start_model_service() {
  if [[ "$SKIP_MODEL_SERVICE" == "1" ]]; then
    echo "Skipping model adapter service startup."
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ start unified model adapter service on port $MODEL_PORT"
    return
  fi
  if [[ ! -x "$RETRIEVER_DIR/.venv-vm/bin/python" ]]; then
    echo "Missing VM venv. Run without --skip-pip first." >&2
    exit 1
  fi
  if [[ -f "$RETRIEVER_DIR/run/wns_vm_adapter_service.pid" ]]; then
    old_pid="$(cat "$RETRIEVER_DIR/run/wns_vm_adapter_service.pid" || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      kill "$old_pid" || true
      sleep 2
    fi
  fi
  # shellcheck source=/dev/null
  source "$RETRIEVER_DIR/.venv-vm/bin/activate"
  set -a
  # shellcheck source=/dev/null
  source "$RETRIEVER_DIR/.env.vm.generated"
  set +a
  nohup "$RETRIEVER_DIR/.venv-vm/bin/python" -m uvicorn scripts.wns_vm_adapter_service:app --app-dir "$RETRIEVER_DIR" --host 0.0.0.0 --port "$MODEL_PORT" > "$RETRIEVER_DIR/logs/wns_vm_adapter_service.log" 2>&1 &
  echo $! > "$RETRIEVER_DIR/run/wns_vm_adapter_service.pid"
  echo "Started model adapter service: PID $(cat "$RETRIEVER_DIR/run/wns_vm_adapter_service.pid")"
}

start_dashboard() {
  if [[ "$SKIP_DASHBOARD" == "1" ]]; then
    echo "Skipping dashboard startup."
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ start WNS dashboard on port $DASHBOARD_PORT"
    return
  fi
  if [[ ! -x "$RETRIEVER_DIR/.venv-vm/bin/python" ]]; then
    echo "Missing VM venv. Run without --skip-pip first." >&2
    exit 1
  fi
  if [[ -f "$RETRIEVER_DIR/run/wns_dashboard.pid" ]]; then
    old_pid="$(cat "$RETRIEVER_DIR/run/wns_dashboard.pid" || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      kill "$old_pid" || true
      sleep 2
    fi
  fi
  nohup "$RETRIEVER_DIR/.venv-vm/bin/python" "$RETRIEVER_DIR/scripts/serve_benchmark_dashboard.py" "$DASHBOARD_PORT" 0.0.0.0 > "$RETRIEVER_DIR/logs/wns_dashboard.log" 2>&1 &
  echo $! > "$RETRIEVER_DIR/run/wns_dashboard.pid"
  echo "Started dashboard: PID $(cat "$RETRIEVER_DIR/run/wns_dashboard.pid")"
}

http_wait() {
  local url="$1"
  local name="$2"
  local attempts="${3:-30}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ check $name: $url"
    return 0
  fi
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      echo "OK: $name"
      return 0
    fi
    sleep 2
  done
  echo "CHECK FAILED: $name at $url" >&2
  return 1
}

tcp_wait() {
  local host="$1"
  local port="$2"
  local name="$3"
  local attempts="${4:-30}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ check $name: $host:$port"
    return 0
  fi
  for _ in $(seq 1 "$attempts"); do
    if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=5):
    pass
PY
    then
      echo "OK: $name"
      return 0
    fi
    sleep 2
  done
  echo "CHECK FAILED: $name at $host:$port" >&2
  return 1
}

post_smoke() {
  local url="$1"
  local name="$2"
  local payload="$3"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ smoke $name: $url"
    return 0
  fi
  curl -fsS --max-time 180 -X POST "$url" -H 'Content-Type: application/json' -d "$payload" >/tmp/wns_smoke_response.json
  python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/wns_smoke_response.json')
data = json.loads(p.read_text())
if not (data.get('embeddings') or data.get('data') or data.get('scores') or data.get('results')):
    raise SystemExit(f'Unexpected smoke response keys: {sorted(data.keys())}')
print('OK smoke response keys:', ','.join(sorted(data.keys())))
PY
  echo "OK: $name smoke"
}

show_model_logs() {
  local log="$RETRIEVER_DIR/logs/wns_vm_adapter_service.log"
  if [[ -f "$log" ]]; then
    echo "--- last 120 lines of model adapter log ---" >&2
    tail -n 120 "$log" >&2 || true
    echo "--- end model adapter log ---" >&2
  fi
}

post_smoke_required() {
  local url="$1"
  local name="$2"
  local payload="$3"
  if ! post_smoke "$url" "$name" "$payload"; then
    echo "REQUIRED smoke failed: $name" >&2
    show_model_logs
    return 1
  fi
}

pgvector_auth_check() {
  local py_bin="python3"
  if [[ -x "$RETRIEVER_DIR/.venv-vm/bin/python" ]]; then
    py_bin="$RETRIEVER_DIR/.venv-vm/bin/python"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ check PGVector DSN auth"
    return 0
  fi
  local dsn="postgresql://wns:$WNS_POSTGRES_PASSWORD@127.0.0.1:$PGVECTOR_PORT/wns_benchmark"
  "$py_bin" - "$dsn" <<'PY'
import sys
try:
    import psycopg
except Exception as exc:
    raise SystemExit(f"psycopg missing; cannot verify PGVector DSN auth: {exc}")
try:
    with psycopg.connect(sys.argv[1], connect_timeout=5) as conn:
        conn.execute("SELECT 1").fetchone()
except Exception as exc:
    raise SystemExit(f"PGVector DSN auth failed: {exc}")
print("OK: PGVector DSN auth")
PY
}

prepare_data() {
  if [[ "$PREPARE_DATA" != "1" ]]; then
    echo "Skipping benchmark data prep. Use --prepare-data when chunking workbook needs rebuild."
    return
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ prepare benchmark data: run_chunking_pipeline.py --mode $CHUNK_MODE"
    return
  fi
  if [[ ! -x "$RETRIEVER_DIR/.venv-vm/bin/python" ]]; then
    echo "Missing VM venv. Run without --skip-pip first." >&2
    exit 1
  fi
  local cmd=("$RETRIEVER_DIR/.venv-vm/bin/python" "$RETRIEVER_DIR/scripts/run_chunking_pipeline.py" --mode "$CHUNK_MODE")
  if [[ "$INCLUDE_IMAGE_MARKERS" == "1" ]]; then
    cmd+=(--include-image-markers)
  fi
  echo "+ ${cmd[*]}"
  (cd "$RETRIEVER_DIR" && "${cmd[@]}")
}

run_checks() {
  local host="$1"
  echo "Running readiness checks..."
  http_wait "http://127.0.0.1:$MODEL_PORT/health" "model adapter health" 45
  http_wait "http://127.0.0.1:$QDRANT_HTTP_PORT/healthz" "Qdrant HTTP" 20
  tcp_wait "127.0.0.1" "$PGVECTOR_PORT" "PGVector/Postgres" 20
  pgvector_auth_check
  http_wait "http://127.0.0.1:$WEAVIATE_HTTP_PORT/v1/meta" "Weaviate HTTP" 20
  if [[ "$SKIP_DASHBOARD" != "1" ]]; then
    http_wait "http://127.0.0.1:$DASHBOARD_PORT" "WNS dashboard" 20
  fi

  local py_bin="python3"
  if [[ -x "$RETRIEVER_DIR/.venv-vm/bin/python" ]]; then
    py_bin="$RETRIEVER_DIR/.venv-vm/bin/python"
  fi
  if [[ "$DRY_RUN" != "1" ]] && command -v "$py_bin" >/dev/null 2>&1; then
    "$py_bin" -m py_compile "$RETRIEVER_DIR/scripts/wns_vm_adapter_service.py" "$RETRIEVER_DIR/benchmarking/core/runner.py" "$RETRIEVER_DIR/benchmarking/core/registry.py"
    (cd "$RETRIEVER_DIR" && "$py_bin" scripts/benchmark_cli.py validate)
  fi

  if [[ "$SKIP_MODEL_SMOKE" == "1" || "$SKIP_MODEL_SERVICE" == "1" ]]; then
    echo "Skipping model smoke tests."
  else
    post_smoke_required "http://127.0.0.1:$MODEL_PORT/embed/jina" "Jina embedding" '{"texts":["refund policy","flight change"]}'
    post_smoke_required "http://127.0.0.1:$MODEL_PORT/embed/gte" "GTE embedding" '{"texts":["refund policy","flight change"]}'
    post_smoke_required "http://127.0.0.1:$MODEL_PORT/rerank/bge" "BGE rerank" '{"query":"refund policy","documents":["refund policy details","seat selection rules"],"top_k":2}'
    if [[ "${WNS_ENABLE_QWEN_SMOKE:-0}" == "1" ]]; then
      post_smoke_required "http://127.0.0.1:$MODEL_PORT/rerank/qwen" "Qwen rerank" '{"query":"refund policy","documents":["refund policy details","seat selection rules"],"top_k":2}'
    else
      echo "Qwen smoke skipped by default because Qwen3:4B download/load is heavy. Set WNS_ENABLE_QWEN_SMOKE=1 to preload/test it."
    fi
  fi
}

write_result() {
  local host="$1"
  cat > "$ROOT/VM_SETUP_RESULT.txt" <<EOF
WNS VM setup completed/generated.

Health URLs:
  Model adapter: http://$host:$MODEL_PORT/health
  WNS dashboard: http://$host:$DASHBOARD_PORT
  Jina embedding: http://$host:$MODEL_PORT/embed/jina
  GTE embedding: http://$host:$MODEL_PORT/embed/gte
  Qwen rerank: http://$host:$MODEL_PORT/rerank/qwen
  BGE rerank: http://$host:$MODEL_PORT/rerank/bge
  Qdrant: http://$host:$QDRANT_HTTP_PORT
  PGVector: postgresql://wns:wns_password@$host:$PGVECTOR_PORT/wns_benchmark
  Weaviate: http://$host:$WEAVIATE_HTTP_PORT/v1/meta

Generated env:
  $RETRIEVER_DIR/.env.vm.generated

If Retreiver/.env already existed, it was preserved unless --force-env was used.
Add OPENAI_API_KEY and AWS credentials in Retreiver/.env only if running commercial OpenAI/Amazon adapters.
EOF
  echo "Wrote $ROOT/VM_SETUP_RESULT.txt"
}

run_nvidia_rag_setup() {
  if [[ "$WITH_NVIDIA_RAG" != "1" ]]; then
    return
  fi
  if [[ ! -x "$ROOT/setup_nvidia_rag_pipeline_on_vm.sh" ]]; then
    echo "Missing setup_nvidia_rag_pipeline_on_vm.sh" >&2
    exit 1
  fi
  local args=(--host "$VM_HOST" --force-env)
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run --skip-start)
  fi
  args+=("${NVIDIA_RAG_ARGS[@]}")
  run bash "$ROOT/setup_nvidia_rag_pipeline_on_vm.sh" "${args[@]}"
}

main() {
  VM_HOST="$(detect_host)"
  echo "WNS VM setup root: $ROOT"
  echo "Using VM host: $VM_HOST"
  write_env "$VM_HOST"
  install_python_deps
  start_vector_dbs
  start_model_service
  start_dashboard
  prepare_data
  run_checks "$VM_HOST"
  run_nvidia_rag_setup
  write_result "$VM_HOST"
  echo "Done."
}

main "$@"
