# WNS VM embedding endpoints

Use the WNS-provided VM for open-source embedding models. Do not run these on the WNS laptop CPU for real benchmark runs.

## Required `.env`

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

`JINA_API_KEY` and `HF_TOKEN` are optional. Leave blank unless the VM endpoint requires auth.

## Suggested endpoint contract

The future adapter should call both endpoints with the same shape.

Request:

```json
{
  "model": "jina-embeddings-v3",
  "texts": ["first text", "second text"]
}
```

Response:

```json
{
  "model": "jina-embeddings-v3",
  "dimensions": 1024,
  "embeddings": [[0.01, 0.02], [0.03, 0.04]]
}
```

For GTE:

```json
{
  "model": "Alibaba-NLP/gte-multilingual-base",
  "texts": ["first text", "second text"]
}
```

## Health check

If possible, expose:

```text
GET /health
```

Response:

```json
{"ok": true}
```

If the endpoint does not expose `/health`, `scripts/check_services.py` can still check the exact embedding URL but may show `CHECK` for non-GET endpoints. That is acceptable until the adapter is implemented.

## Laptop responsibility

The laptop should only do:

- chunking orchestration
- sending text batches to VM embedding endpoints
- writing vectors to Qdrant/PGVector/Weaviate
- dashboard/results viewing

The VM should do:

- loading `jina-embeddings-v3`
- loading `Alibaba-NLP/gte-multilingual-base`
- optionally loading Qwen3 reranker
