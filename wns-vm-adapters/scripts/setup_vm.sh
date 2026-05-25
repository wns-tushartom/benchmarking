#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ ! -f .env ]; then cp .env.example .env; echo "Created .env"; fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
try:
 import torch
 print('torch:', torch.__version__)
 print('cuda_available:', torch.cuda.is_available())
 if torch.cuda.is_available(): print('cuda_device:', torch.cuda.get_device_name(0))
except Exception as exc:
 print('torch check failed:', exc)
PY
echo "Setup complete. Ports used: 5000 embedding, 5001 Qdrant, 5003 PGVector, 5004 Weaviate."
