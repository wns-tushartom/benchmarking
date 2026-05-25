# WNS benchmark work-laptop setup

This guide sets up the WNS benchmark harness on a work laptop.

## What you can run immediately

The local fallback benchmark works without external services or API keys.

It validates:

- matrix orchestration
- selectable combinations
- dashboard controls
- recall/MRR/nDCG/precision metrics
- reports and CSV outputs

It does not validate real provider latency/cost until provider adapters and services are wired.

## Windows quick setup

Use Command Prompt, not PowerShell, because your work laptop blocks PowerShell by group policy.

From project root:

```cmd
cd "C:\Users\U481019\OneDrive - WNS\Documents\main_benchmarking"
scripts\windows\Setup-WnsBenchmark.cmd
```

If you are currently inside `scripts\windows`, run:

```cmd
cd ..\..
scripts\windows\Setup-WnsBenchmark.cmd
```

Then check anytime with:

```cmd
scripts\windows\Check-WnsSetup.cmd
```

Start dashboard:

```cmd
scripts\windows\Start-WnsDashboard.cmd
```

Read:

```text
WINDOWS_ADAPTER_SETUP.md
VM_EMBEDDING_ENDPOINTS.md
```

## Minimum setup, local fallback mode

### 1. Install prerequisites

Required:

- Python 3.10+
- Git
- A terminal

Recommended:

- Docker Desktop, needed for Qdrant/PGVector/Weaviate real services
- VS Code

### 2. Copy or unzip the project

Unzip the delivered package or clone/copy the `Retreiver` folder.

Expected project root:

```text
Retreiver/
  configs/
  benchmarking/
  scripts/
  source/
  data/
  web/
```

### 3. Open terminal in project root

```bash
cd Retreiver
```

### 4. Optional virtual environment

Linux/macOS/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows CMD:

```cmd
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

### 5. Run validation

```bash
python3 scripts/benchmark_cli.py validate
```

On Windows CMD, if `python3` is missing, use:

```cmd
py scripts/benchmark_cli.py validate
```

Expected matrix count:

```text
135
```

### 6. Run one selected smoke benchmark

```bash
python3 scripts/benchmark_cli.py run \
  --limit-queries 20 \
  --output-dir data/modular_runs/latest \
  --chunker entity_heuristic_w6 \
  --embedding jina_v3 \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker "bge-reranker-base"
```

CMD version:

```cmd
scripts\windows\Run-SelectedBenchmark.cmd entity_heuristic_w6 jina_v3 Qdrant bge-reranker-base 20
```

### 7. Start dashboard

```bash
python3 scripts/serve_benchmark_dashboard.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

Use selectors and click:

```text
Run selected combo
```

## Full local fallback benchmark

Run all official 135 combinations:

```bash
python3 scripts/benchmark_cli.py run --limit-queries 0 --output-dir data/modular_runs/latest
```

## Optional Python packages

Local fallback mode is stdlib-first. For real providers and convenience tools:

```bash
python3 -m pip install -r requirements-benchmark.txt
```

If your work laptop blocks installs, local fallback still runs without this file.

## Optional Docker services for real provider mode

Start vector DB services:

```bash
docker compose -f docker-compose.benchmark.yml up -d qdrant postgres-pgvector weaviate
```

Set environment variables:

Linux/macOS/WSL:

```bash
export QDRANT_URL=http://127.0.0.1:5001
export PGVECTOR_DSN=postgresql://wns:wns_password@127.0.0.1:5003/wns_benchmark
export WEAVIATE_URL=http://127.0.0.1:5004
```

Windows CMD:

```cmd
call scripts\windows\Load-WnsEnv.cmd
```

## API keys and `.env`

Use the central env file:

```bash
cp .env.example .env
```

Then fill `.env` with:

```text
OPENAI_API_KEY, only for OpenAI embedding
JINA_EMBEDDING_URL, VM/remote endpoint for self-hosted Jina v3
GTE_EMBEDDING_URL, VM/remote endpoint for self-hosted GTE multilingual
JINA_API_KEY, optional only if hosted/authenticated Jina endpoint
HF_TOKEN, optional only if the VM/HF endpoint requires auth
QDRANT_URL
PGVECTOR_DSN or DATABASE_URL
WEAVIATE_URL
AWS_ACCESS_KEY_ID, only for Amazon Rerank v1
AWS_SECRET_ACCESS_KEY, only for Amazon Rerank v1
AWS_REGION or AWS_DEFAULT_REGION, only for Amazon Rerank v1
QWEN_RERANK_URL, VM/remote endpoint only
```

Load it on Linux/macOS/WSL:

```bash
set -a
source .env
set +a
```

Read:

```text
ENV_SETUP.md
VM_EMBEDDING_ENDPOINTS.md
```

Open-source rerankers/embeddings:

- run `jina_v3` and `gte_multilingual_base` on the WNS-provided VM, not on laptop CPU
- keep `JINA_EMBEDDING_URL` and `GTE_EMBEDDING_URL` pointed at VM embedding endpoints
- install `sentence-transformers` / model weights on the VM side, not necessarily on the laptop
- BGE reranker may run locally only for tiny smoke tests; prefer VM if latency/CPU is poor
- run Qwen3:4B reranker from the VM/remote endpoint, not on the work laptop

## What is yet to be set up

### Done

Exact official 135-combination WNS matrix
- dashboard selectors
- selected combo runner
- config-driven dimensions
- local fallback adapters
- metrics and reports
- WNS logo dashboard
- Docker compose scaffold for vector DB services

### Still needed for production-real results

1. Real OpenAI embedding adapter
2. Real Jina embedding adapter
3. Real Qdrant adapter using HNSW and cosine
4. Real PGVector adapter using HNSW and cosine
5. Real Weaviate adapter using HNSW and cosine
6. Real BGE reranker adapter
7. Real Qwen3:4B reranker adapter
8. Amazon Rerank v1 adapter through AWS/Bedrock
9. Embedding/rerank cache to avoid repeated paid calls
10. Per-stage timing and cost tracking
11. Resume failed benchmark runs
12. Work-laptop credential setup and network allow-listing

## Recommended first production slice

Do not start with all 135 real combinations.

Start with this small slice:

```text
entity_heuristic_w6 + openai_text-embedding-3-large + Qdrant + HNSW + Cosine Similarity + bge-reranker-base
entity_heuristic_w6 + jina_v3 + Qdrant + HNSW + Cosine Similarity + bge-reranker-base
Heading_sections_l2 + openai_text-embedding-3-large + Qdrant + HNSW + Cosine Similarity + bge-reranker-base
Heading_sections_l2 + jina_v3 + Qdrant + HNSW + Cosine Similarity + bge-reranker-base
```

Once Qdrant is trusted, add PGVector and Weaviate.

Once local/open-source reranking is trusted, add Amazon Rerank.
