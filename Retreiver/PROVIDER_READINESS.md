# WNS Production Benchmark Readiness

## Current status

Local/offline benchmark is complete. Real provider benchmark is partially blocked by runtime availability.

## Checked environment

### Available

- `OPENAI_API_KEY`: present
- `JINA_API_KEY`: present
- Docker: available
- Qdrant local service: reachable at `127.0.0.1:6333`
- Qdrant version: `1.17.1`

### Missing / blocked

- AWS Amazon Rerank:
  - `AWS_ACCESS_KEY_ID`: missing
  - `AWS_DEFAULT_REGION` / `AWS_REGION`: missing
  - cannot run Amazon Rerank production benchmark yet
- PGVector:
  - Postgres port `5432`: closed
  - `psql`: missing
  - `PGVECTOR_DSN` / `DATABASE_URL`: missing
- Weaviate:
  - `127.0.0.1:8080` readiness endpoint returned 404 / not ready
  - `WEAVIATE_URL`: missing
- Qdrant API URL env:
  - `QDRANT_URL`: missing, but local service is reachable directly at `http://127.0.0.1:6333`

## What this means

The current 162-run matrix is valid for:

- benchmark orchestration
- recall calculation
- result files
- dashboard visualization
- chunking/embedding/reranking relative signal under deterministic fallback adapters

It is not valid for:

- production vector DB latency ranking
- actual provider cost
- real embedding latency
- real reranker latency

## Recommended provider-mode order

Do not start with all 135/162 real combinations. Start with a small high-signal slice:

1. Qdrant + OpenAI embedding + available local/API reranker
2. Qdrant + Jina/GTE embedding + available local/API reranker
3. Compare `semantic_split` vs `entity_heuristic_w6`
4. Add PGVector only after Postgres/pgvector is configured
5. Add Weaviate only after service readiness is fixed
6. Add Amazon Rerank only after AWS keys and region are configured

## Minimum provider slice to run first

```text
semantic_split + openai_text-embedding-3-large + Qdrant HNSW
semantic_split + gte_multilingual_base + Qdrant HNSW
entity_heuristic_w6 + openai_text-embedding-3-large + Qdrant HNSW
entity_heuristic_w6 + gte_multilingual_base + Qdrant HNSW
```

Use the same 506 query cases and compare against the local fallback output.

## Setup needed for full provider matrix

### PGVector

Need:

```text
Postgres running on 5432
pgvector extension installed
PGVECTOR_DSN or DATABASE_URL configured
psql installed for verification
```

### Weaviate

Need:

```text
Weaviate service running and ready
WEAVIATE_URL configured
WEAVIATE_API_KEY if auth enabled
```

### Amazon Rerank

Need:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION or AWS_DEFAULT_REGION
Access to Amazon Rerank v1 / Bedrock rerank endpoint
```

### Qdrant

Current usable local endpoint:

```text
http://127.0.0.1:6333
```

Optional env to standardize scripts:

```bash
export QDRANT_URL=http://127.0.0.1:6333
```
