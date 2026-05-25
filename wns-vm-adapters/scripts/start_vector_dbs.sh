#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then set -a; source .env; set +a; fi
docker compose -f docker-compose.vector-dbs.yml up -d
echo "Vector DBs starting: Qdrant http://127.0.0.1:5001, PGVector 127.0.0.1:5003, Weaviate http://127.0.0.1:5004"
