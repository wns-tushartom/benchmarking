# WNS benchmark environment variables

Use this with:

```bash
cp .env.example .env
```

Then fill real values in `.env`.

Do not share `.env`. It is ignored by `.gitignore`.

## Load env

Linux/macOS/WSL:

```bash
set -a
source .env
set +a
```

Windows CMD:

```cmd
call scripts\windows\Load-WnsEnv.cmd
```

Windows PowerShell is blocked on the WNS laptop by group policy, so prefer CMD scripts.

If using `python-dotenv` later, the benchmark can auto-load `.env`, but currently the safest path is shell env.

## Required for first real provider slice

Fill these first:

```text
JINA_EMBEDDING_URL, VM/remote endpoint for self-hosted Jina v3
GTE_EMBEDDING_URL, VM/remote endpoint for self-hosted GTE multilingual
QDRANT_URL
```

Only fill `OPENAI_API_KEY` if the selected slice uses `openai_text-embedding-3-large`.
`JINA_API_KEY` is optional and only needed for a hosted/authenticated Jina endpoint, not for self-hosted VM Jina.

## Open-source embedding VM note

Do not run `jina_v3` or `gte_multilingual_base` on the WNS laptop CPU.

Use the WNS-provided VM:

```text
JINA_EMBEDDING_MODE=vm_remote
JINA_EMBEDDING_URL=http://YOUR_VM_HOST:PORT/embed/jina
JINA_API_KEY=
JINA_EMBEDDING_MODEL=jina-embeddings-v3

GTE_EMBEDDING_MODE=vm_remote
GTE_EMBEDDING_URL=http://YOUR_VM_HOST:PORT/embed/gte
HF_TOKEN=only_if_endpoint_requires_auth
GTE_EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
```

If using BGE reranker locally:

```text
BGE_RERANKER_MODEL=BAAI/bge-reranker-base
BGE_RERANKER_DEVICE=cpu
```

## Required for full vector DB comparison

```text
PGVECTOR_DSN
DATABASE_URL, optional alias
WEAVIATE_URL
WEAVIATE_API_KEY, only if auth enabled
QDRANT_API_KEY, only if Qdrant Cloud/auth
```

## Required for Amazon reranking

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION or AWS_DEFAULT_REGION
AMAZON_RERANK_MODEL_ID, confirm exact Bedrock model ID
```

Also required in AWS:

- Bedrock enabled
- rerank model access approved
- IAM permission to call the rerank/invoke endpoint

## Qwen reranker note

Do not run Qwen3:4B reranker locally on the work laptop unless it has enough RAM/VRAM.

Set it as a VM/remote endpoint:

```text
QWEN_RERANK_URL=http://YOUR_VM_HOST:PORT/...
QWEN_API_KEY=only_if_endpoint_requires_auth
QWEN_RERANK_LOCATION=vm_remote
```

The adapter should call the VM endpoint rather than loading the Qwen model locally.

## What still needs code adapters

The env file centralizes config, but real provider execution still needs these adapters:

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
