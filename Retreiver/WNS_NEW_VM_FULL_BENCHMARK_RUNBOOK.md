# WNS new VM full benchmark runbook

Use this on a fresh VM after the repo/package is copied or pulled.

## 0. Required inputs

- VM host/IP reachable from browser and from inside the VM.
- Docker access for the VM user.
- NVIDIA/CUDA GPU strongly recommended for Jina, GTE, BGE, Qwen.
- Optional for commercial full 135 matrix:
  - `OPENAI_API_KEY`
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`
  - `AMAZON_RERANK_MODEL_ID`, preferably full Bedrock foundation-model ARN if the account requires ARN.
- Optional for NVIDIA RAG lane:
  - `NGC_API_KEY`
  - local `rag-main/` clone or `rag-main.zip`

## 1. Base setup, WNS services

```bash
cd ~/benchmarking
VM_IP=$(hostname -I | awk '{print $1}')

# Optional, set commercial keys before setup so .env preserves them.
# export OPENAI_API_KEY='...'
# export AWS_ACCESS_KEY_ID='...'
# export AWS_SECRET_ACCESS_KEY='...'
# export AWS_REGION='us-east-1'
# export AWS_DEFAULT_REGION='us-east-1'
# export AMAZON_RERANK_MODEL_ID='arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0'

bash setup_all_on_vm.sh \
  --host "$VM_IP" \
  --force-env \
  --prepare-data \
  --chunk-mode chunk-only
```

Expected services:

- `5000`: WNS model adapter
- `5001`: Qdrant HTTP
- `5002`: Qdrant gRPC
- `5003`: PGVector/Postgres
- `5004`: Weaviate HTTP
- `5005`: Weaviate gRPC
- `5009`: dashboard

## 2. Service verification

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate

python3 scripts/check_services.py
python3 scripts/wns_setup_doctor.py
python3 scripts/benchmark_cli.py validate
```

Expected benchmark validation:

```json
{
  "ok": true,
  "experiment": "wns-rag-benchmark-selectable",
  "matrix_count": 135
}
```

## 3. Model endpoint smoke

```bash
curl -sS http://127.0.0.1:5000/health && echo

curl -sS http://127.0.0.1:5000/embed/gte \
  -H 'Content-Type: application/json' \
  -d '{"texts":["refund old ticket and issue new ticket"]}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model"), d.get("dimensions"), len(d.get("embeddings", [])))'

curl -sS http://127.0.0.1:5000/embed/jina \
  -H 'Content-Type: application/json' \
  -d '{"texts":["refund old ticket and issue new ticket"]}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model"), d.get("dimensions"), len(d.get("embeddings", [])))'

curl -sS http://127.0.0.1:5000/rerank/bge \
  -H 'Content-Type: application/json' \
  -d '{"query":"refund policy","documents":["refund policy details","seat selection rules"],"top_k":2}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model"), d.get("results", [])[0])'
```

Only run Qwen smoke after BGE works, because first load can be heavy:

```bash
curl -sS http://127.0.0.1:5000/rerank/qwen \
  -H 'Content-Type: application/json' \
  -d '{"query":"refund policy","documents":["refund policy details","seat selection rules"],"top_k":2}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model"), d.get("results", [])[0])'
```

## 4. Full real open-source/provider-VM pipeline

This is the current dashboard pipeline. It runs every supported live VM combination:

- 5 chunkers
- 2 VM embeddings, GTE and Jina
- 3 vector DBs
- no-reranker baseline evaluation
- 2 VM rerankers, BGE and Qwen

That produces 30 base retrieval combinations and 60 reranked combinations.

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate

python3 scripts/run_complete_pipeline.py \
  --sheets all \
  --embeddings all \
  --stores all \
  --rerankers all \
  --top-k 10 \
  --fresh-run
```

If extraction audit is not available on the fresh VM and you need a dry run only, add `--allow-partial-extraction`. Do not use that for final client numbers.

## 5. Full configured 135 matrix, commercial included

This runs the config-driven 135 matrix:

- 5 chunkers
- 3 embeddings, Jina, GTE, OpenAI
- 3 vector DBs
- 3 rerankers, Amazon, Qwen, BGE

Do this only after OpenAI and AWS/Bedrock credentials are valid.

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate

python3 scripts/benchmark_cli.py run \
  configs/benchmark.local.json \
  --output-dir data/modular_runs/full_135_real \
  --limit-queries 0
```

Progress files:

```bash
watch -n 10 'cat data/modular_runs/full_135_real/status.json 2>/dev/null || true'
```

Final files:

- `data/modular_runs/full_135_real/modular_summary.csv`
- `data/modular_runs/full_135_real/modular_details.csv`
- `data/modular_runs/full_135_real/analysis.json`
- `data/modular_runs/full_135_real/MODULAR_REPORT.md`

## 6. Dashboard refresh

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate
python3 scripts/collect_vm_dashboard_artifacts.py
```

Dashboard:

```text
http://VM_IP:5009
```

## 7. If NVIDIA RAG is required too

```bash
cd ~/benchmarking
export NGC_API_KEY='...'

bash setup_all_on_vm.sh \
  --host "$VM_IP" \
  --force-env \
  --with-nvidia-rag \
  --nvidia-rag-root "$PWD/rag-main"
```

Then verify:

```bash
cd ~/benchmarking/Retreiver
source .venv-vm/bin/activate
python3 scripts/check_nvidia_rag_pipeline.py --env-file .env.project-smiley-nvidia
```
