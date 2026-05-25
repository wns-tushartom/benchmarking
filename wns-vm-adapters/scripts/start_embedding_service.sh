#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then set -a; source .env; set +a; fi
source .venv/bin/activate
HOST="${WNS_VM_HOST:-0.0.0.0}"
PORT="${WNS_EMBEDDING_PORT:-5000}"
echo "Starting WNS embedding service on ${HOST}:${PORT}"
exec uvicorn app.embedding_service:app --host "$HOST" --port "$PORT"
