#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVER_DIR="$ROOT/Retreiver"
RAG_ROOT="${NVIDIA_RAG_ROOT:-}"
RAG_ZIP="${NVIDIA_RAG_ZIP:-}"
VM_HOST=""
DRY_RUN=0
SKIP_START=0
SKIP_DOCKER_LOGIN=0
FORCE_ENV=0
NVIDIA_RAG_SERVER_PORT=5006
NVIDIA_INGESTOR_PORT=5007
NVIDIA_RAG_FRONTEND_PORT=5008

usage() {
  cat <<'USAGE'
Usage:
  bash setup_nvidia_rag_pipeline_on_vm.sh [--rag-root PATH | --rag-zip rag-main.zip] [--host HOST] [options]

Purpose:
  Set up NVIDIA RAG Blueprint for Project Smiley Integrated RAG Pipeline.

What it does:
  1. Finds or extracts NVIDIA rag-main source.
  2. Writes Retreiver/.env.project-smiley-nvidia and merges values into .env if requested.
  3. Writes rag-main/deploy/compose/.env.project-smiley for NVIDIA-hosted model mode.
  4. Patches host-facing ports so Project Smiley uses 5006, 5007, 5008 externally.
  5. Optionally logs into nvcr.io and starts vector DB, ingestor, RAG server, and NVIDIA frontend.
  6. Runs Project Smiley health checks and writes Retreiver/data/nvidia_rag/health.json.

Required for real startup:
  export NGC_API_KEY=nvapi-...

Host-facing ports:
  5006  NVIDIA rag-server /v1
  5007  NVIDIA ingestor-server /v1
  5008  NVIDIA reference frontend

Options:
  --rag-root PATH          Path to extracted NVIDIA rag-main directory
  --rag-zip ZIP           Path to rag-main.zip to extract under this package
  --host HOST             Public/reachable VM host/IP for generated URLs
  --force-env             Append generated NVIDIA env vars into Retreiver/.env
  --skip-start            Write/patch files only, do not start Docker services
  --skip-docker-login     Do not run docker login nvcr.io
  --dry-run               Print actions without starting services
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rag-root) RAG_ROOT="${2:-}"; shift 2 ;;
    --rag-zip) RAG_ZIP="${2:-}"; shift 2 ;;
    --host) VM_HOST="${2:-}"; shift 2 ;;
    --force-env) FORCE_ENV=1; shift ;;
    --skip-start) SKIP_START=1; shift ;;
    --skip-docker-login) SKIP_DOCKER_LOGIN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  echo "+ $*" >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MISSING command: $1" >&2
    exit 1
  fi
}

detect_host() {
  if [[ -n "$VM_HOST" ]]; then
    echo "$VM_HOST"
  else
    hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1"
  fi
}

find_rag_root() {
  if [[ -n "$RAG_ROOT" && -d "$RAG_ROOT" ]]; then
    echo "$(cd "$RAG_ROOT" && pwd)"
    return
  fi
  for candidate in "$ROOT/rag-main" "$ROOT/../rag-main" "$HOME/rag-main"; do
    if [[ -d "$candidate/deploy/compose" ]]; then
      echo "$(cd "$candidate" && pwd)"
      return
    fi
  done
  if [[ -n "$RAG_ZIP" && -f "$RAG_ZIP" ]]; then
    need_cmd unzip
    run unzip -q -o "$RAG_ZIP" -d "$ROOT"
    if [[ -d "$ROOT/rag-main/deploy/compose" ]]; then
      echo "$ROOT/rag-main"
      return
    fi
  fi
  for zip_candidate in "$ROOT/rag-main.zip" "$ROOT/../rag-main.zip" "$HOME/rag-main.zip"; do
    if [[ -f "$zip_candidate" ]]; then
      need_cmd unzip
      run unzip -q -o "$zip_candidate" -d "$ROOT"
      if [[ -d "$ROOT/rag-main/deploy/compose" ]]; then
        echo "$ROOT/rag-main"
        return
      fi
    fi
  done
  echo "Could not find NVIDIA rag-main. Pass --rag-root PATH or --rag-zip rag-main.zip." >&2
  exit 1
}

patch_file_once() {
  local file="$1"
  local old="$2"
  local new="$3"
  python3 - "$file" "$old" "$new" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = sys.argv[2]
new = sys.argv[3]
text = path.read_text(encoding='utf-8')
if new in text:
    raise SystemExit(0)
if old not in text:
    print(f"ERROR: required port patch pattern not found in {path}: {old}", file=sys.stderr)
    raise SystemExit(2)
path.write_text(text.replace(old, new), encoding='utf-8')
PY
}

patch_ports() {
  local rag="$1"
  echo "Patching NVIDIA compose host ports for Project Smiley..."
  patch_file_once "$rag/deploy/compose/docker-compose-rag-server.yaml" '"8081:8081"' '"${NVIDIA_RAG_SERVER_HOST:-0.0.0.0}:${NVIDIA_RAG_SERVER_PORT:-5006}:8081"'
  patch_file_once "$rag/deploy/compose/docker-compose-rag-server.yaml" '"8090:3000"' '"${NVIDIA_RAG_FRONTEND_HOST:-0.0.0.0}:${NVIDIA_RAG_FRONTEND_PORT:-5008}:3000"'
  patch_file_once "$rag/deploy/compose/docker-compose-ingestor-server.yaml" '"8082:8082"' '"${NVIDIA_INGESTOR_HOST:-0.0.0.0}:${NVIDIA_INGESTOR_PORT:-5007}:8082"'
  patch_file_once "$rag/deploy/compose/docker-compose-ingestor-server.yaml" '"6379:6379"' '"127.0.0.1:6379:6379"'
  patch_file_once "$rag/deploy/compose/docker-compose-ingestor-server.yaml" '"7670:7670"' '"127.0.0.1:7670:7670"'
  patch_file_once "$rag/deploy/compose/docker-compose-ingestor-server.yaml" '"7671:7671"' '"127.0.0.1:7671:7671"'
  patch_file_once "$rag/deploy/compose/docker-compose-ingestor-server.yaml" '"8265:8265"' '"127.0.0.1:8265:8265"'
  patch_file_once "$rag/deploy/compose/vectordb.yaml" '"9011:9011"' '"127.0.0.1:9011:9011"'
  patch_file_once "$rag/deploy/compose/vectordb.yaml" '"9010:9010"' '"127.0.0.1:9010:9010"'
  patch_file_once "$rag/deploy/compose/vectordb.yaml" '9200:9200' '127.0.0.1:9200:9200'
}

write_envs() {
  local rag="$1"
  local host="$2"
  mkdir -p "$RETRIEVER_DIR/data/nvidia_rag" "$RETRIEVER_DIR/logs"
  cat > "$RETRIEVER_DIR/.env.project-smiley-nvidia" <<EOF
# Generated by setup_nvidia_rag_pipeline_on_vm.sh
NVIDIA_RAG_BLUEPRINT_ROOT=$rag
NVIDIA_RAG_SERVER_URL=http://$host:$NVIDIA_RAG_SERVER_PORT
NVIDIA_INGESTOR_URL=http://$host:$NVIDIA_INGESTOR_PORT
NVIDIA_RAG_FRONTEND_URL=http://$host:$NVIDIA_RAG_FRONTEND_PORT
NVIDIA_RAG_COLLECTION=multimodal_data
NVIDIA_RAG_EMBEDDING_DIMENSIONS=2048
EOF
  if [[ "$FORCE_ENV" == "1" ]]; then
    if [[ -f "$RETRIEVER_DIR/.env" ]]; then
      cp "$RETRIEVER_DIR/.env" "$RETRIEVER_DIR/.env.before-nvidia-rag.$(date +%Y%m%d-%H%M%S)"
    fi
    grep -v '^NVIDIA_RAG_' "$RETRIEVER_DIR/.env" 2>/dev/null > "$RETRIEVER_DIR/.env.tmp" || true
    cat "$RETRIEVER_DIR/.env.project-smiley-nvidia" >> "$RETRIEVER_DIR/.env.tmp"
    mv "$RETRIEVER_DIR/.env.tmp" "$RETRIEVER_DIR/.env"
  fi

  cat > "$rag/deploy/compose/.env.project-smiley" <<EOF
# Project Smiley NVIDIA RAG Blueprint env
# Real startup requires NGC_API_KEY in shell or this file.
NGC_API_KEY=${NGC_API_KEY:-}
NVIDIA_API_KEY=${NGC_API_KEY:-}
NVIDIA_RAG_SERVER_PORT=$NVIDIA_RAG_SERVER_PORT
NVIDIA_INGESTOR_PORT=$NVIDIA_INGESTOR_PORT
NVIDIA_RAG_FRONTEND_PORT=$NVIDIA_RAG_FRONTEND_PORT
TAG=${TAG:-2.6.0}
PROMPT_CONFIG_FILE=$rag/deploy/compose/nemotron3-super-prompt.yaml
APP_VECTORSTORE_NAME=elasticsearch
APP_VECTORSTORE_URL=http://elasticsearch:9200
COLLECTION_NAME=multimodal_data
APP_VECTORSTORE_SEARCHTYPE=dense
VECTOR_DB_TOPK=100
APP_RETRIEVER_TOPK=10
ENABLE_RERANKER=True
APP_EMBEDDINGS_SERVERURL=https://integrate.api.nvidia.com/v1
APP_EMBEDDINGS_MODELNAME=nvidia/llama-nemotron-embed-vl-1b-v2
APP_EMBEDDINGS_DIMENSIONS=2048
APP_RANKING_SERVERURL=
APP_RANKING_MODELNAME=nvidia/llama-nemotron-rerank-1b-v2
APP_LLM_SERVERURL=
APP_LLM_MODELNAME=nvidia/nemotron-3-super-120b-a12b
APP_QUERYREWRITER_SERVERURL=
APP_FILTEREXPRESSIONGENERATOR_SERVERURL=
SUMMARY_LLM_SERVERURL=
ENABLE_AGENTIC_RAG=true
ENABLE_QUERYREWRITER=False
ENABLE_FILTER_GENERATOR=False
APP_NVINGEST_PDFEXTRACTMETHOD=nemotron_parse
APP_NVINGEST_EXTRACTTEXT=True
APP_NVINGEST_EXTRACTTABLES=True
APP_NVINGEST_EXTRACTCHARTS=True
APP_NVINGEST_EXTRACTIMAGES=False
APP_NVINGEST_TEXTDEPTH=page
NEMOTRON_PARSE_HTTP_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
NEMOTRON_PARSE_INFER_PROTOCOL=http
OCR_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1
OCR_INFER_PROTOCOL=http
YOLOX_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v1
YOLOX_INFER_PROTOCOL=http
YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1
YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL=http
YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT=https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1
YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL=http
VITE_API_CHAT_URL=http://rag-server:8081/v1
VITE_API_VDB_URL=http://ingestor-server:8082/v1
EOF
  echo "Wrote $RETRIEVER_DIR/.env.project-smiley-nvidia"
  echo "Wrote $rag/deploy/compose/.env.project-smiley"
}

start_services() {
  local rag="$1"
  if [[ "$SKIP_START" == "1" ]]; then
    echo "Skipping Docker startup."
    return
  fi
  need_cmd docker
  if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "NGC_API_KEY is required to pull/use NVIDIA RAG Blueprint images." >&2
    exit 1
  fi
  if [[ "$SKIP_DOCKER_LOGIN" != "1" ]]; then
    echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
  fi
  (cd "$rag" && run docker compose --env-file deploy/compose/.env.project-smiley -f deploy/compose/vectordb.yaml --profile elasticsearch up -d seaweedfs elasticsearch)
  (cd "$rag" && run docker compose --env-file deploy/compose/.env.project-smiley -f deploy/compose/docker-compose-ingestor-server.yaml up -d redis nv-ingest-ms-runtime ingestor-server)
  (cd "$rag" && run docker compose --env-file deploy/compose/.env.project-smiley -f deploy/compose/docker-compose-rag-server.yaml up -d rag-server rag-frontend)
}

main() {
  need_cmd python3
  local rag
  rag="$(find_rag_root)"
  local host
  host="$(detect_host)"
  patch_ports "$rag"
  write_envs "$rag" "$host"
  start_services "$rag"
  if [[ "$DRY_RUN" != "1" && "$SKIP_START" != "1" ]]; then
    python3 "$RETRIEVER_DIR/scripts/check_nvidia_rag_pipeline.py" --env-file "$RETRIEVER_DIR/.env.project-smiley-nvidia" --out "$RETRIEVER_DIR/data/nvidia_rag/health.json"
  elif [[ "$SKIP_START" == "1" ]]; then
    echo "NVIDIA services configured only; health check skipped because --skip-start was set."
  fi
  cat > "$ROOT/NVIDIA_RAG_SETUP_RESULT.txt" <<EOF
Project Smiley NVIDIA RAG setup
rag_root=$rag
rag_server=http://$host:$NVIDIA_RAG_SERVER_PORT/v1
ingestor=http://$host:$NVIDIA_INGESTOR_PORT/v1
frontend=http://$host:$NVIDIA_RAG_FRONTEND_PORT
health_json=Retreiver/data/nvidia_rag/health.json
EOF
  echo "Wrote $ROOT/NVIDIA_RAG_SETUP_RESULT.txt"
}

main "$@"
