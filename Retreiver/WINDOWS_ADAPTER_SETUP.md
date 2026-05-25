# Windows setup scripts for WNS benchmark adapters

These scripts are for your Windows work laptop.

Because your work laptop blocks PowerShell by group policy, use the `.cmd` scripts. They run through normal Command Prompt.

Run everything from the project root:

```cmd
cd "C:\Users\U481019\OneDrive - WNS\Documents\main_benchmarking"
```

If you are already inside `scripts\windows`, go back to project root first:

```cmd
cd ..\..
```

## One-command setup, CMD only

```cmd
scripts\windows\Setup-WnsBenchmark.cmd
```

What it does:

1. Creates `.env` from `.env.example` if missing.
2. Creates `.venv`.
3. Installs `requirements-benchmark.txt`.
4. Starts Docker services:
   - Qdrant
   - PGVector/Postgres
   - Weaviate
5. Loads `.env` into the CMD process.
6. Runs env check.
7. Runs service readiness check.
8. Validates the 135-combo benchmark config.
9. Runs a 5-query smoke benchmark.

## If Docker Desktop is not running yet

Start Docker Desktop manually, then run:

```cmd
scripts\windows\Setup-WnsBenchmark.cmd
```

If you want to skip Docker during first setup:

```cmd
scripts\windows\Setup-WnsBenchmark.cmd --skip-docker
```

## Load `.env`

For the current CMD process:

```cmd
call scripts\windows\Load-WnsEnv.cmd
```

Use `call` so variables remain in the current CMD session.

## Check setup

```cmd
scripts\windows\Check-WnsSetup.cmd
```

This runs:

```text
scripts/check_env.py
scripts/check_services.py
scripts/benchmark_cli.py validate
```

## Start dashboard

```cmd
scripts\windows\Start-WnsDashboard.cmd
```

Open:

```text
http://127.0.0.1:8765
```

## Run one selected combination

```cmd
scripts\windows\Run-SelectedBenchmark.cmd fixed_tok1200_ov150 jina_v3 Qdrant bge-reranker-base 20
```

Available chunkers:

```text
entity_heuristic_w6
entity_heuristic_w5
entity_heuristic_w4
Heading_sections_l2
fixed_tok1200_ov150
```

Available embeddings:

```text
jina_v3
gte_multilingual_base
openai_text-embedding-3-large
```

Available vector DBs:

```text
Qdrant
PGVector
Weaviate
```

Available rerankers:

```text
Amazon Rerank v1
Qwen3:4B Rerank
bge-reranker-base
```

## Embedding VM note

Do not run `jina_v3` or `gte_multilingual_base` on the WNS laptop CPU. They are open-source, but open-source does not mean they should run locally on an underpowered office laptop.

Use the WNS-provided VM as the embedding host:

```env
JINA_EMBEDDING_MODE=vm_remote
JINA_EMBEDDING_URL=http://YOUR_VM_HOST:PORT/embed/jina
JINA_API_KEY=
JINA_EMBEDDING_MODEL=jina-embeddings-v3

GTE_EMBEDDING_MODE=vm_remote
GTE_EMBEDDING_URL=http://YOUR_VM_HOST:PORT/embed/gte
HF_TOKEN=
GTE_EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
```

`JINA_API_KEY` is optional and should stay blank for a self-hosted VM endpoint unless the endpoint itself requires auth.

## Qwen note

Do not run Qwen3:4B reranker locally on the work laptop.

Put the VM endpoint in `.env`:

```env
QWEN_RERANK_URL=http://YOUR_VM_HOST:PORT/...
QWEN_API_KEY=
QWEN_RERANK_LOCATION=vm_remote
```

The future Qwen adapter will call this endpoint.

## Current adapter status

These Windows scripts prepare the laptop environment and services. They do not magically implement provider adapters.

Still to code:

```text
OpenAIEmbeddingAdapter
JinaEmbeddingAdapter
GTEEmbeddingAdapter
QdrantVectorAdapter
PGVectorVectorAdapter
WeaviateVectorAdapter
BGERerankerAdapter
QwenRemoteRerankerAdapter
AmazonBedrockRerankAdapter
```

Recommended first implementation order:

1. Qdrant real adapter
2. OpenAI embedding adapter
3. Jina embedding adapter
4. BGE local reranker adapter
5. PGVector adapter
6. Weaviate adapter
7. Qwen remote reranker adapter
8. Amazon Bedrock reranker adapter
