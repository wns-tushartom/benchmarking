# WNS VM adapter setup guide

Goal: run heavy open-source models and vector databases on the WNS VM, then let the benchmark runner call them through adapters.

## 1. Target architecture

```text
Benchmark runner
  -> Embedding adapter
      -> Jina VM endpoint
      -> GTE VM endpoint
      -> OpenAI API, only if selected
  -> Vector DB adapter
      -> Qdrant on VM
      -> PGVector/Postgres on VM
      -> Weaviate on VM
  -> Reranker adapter
      -> BGE/Qwen VM endpoint or Amazon Bedrock
```

Do not load `jina_v3`, `gte_multilingual_base`, or Qwen inside the benchmark process. Treat them as services.

## 2. VM services to run

On the VM, run:

```text
1. embedding service, FastAPI
2. Qdrant
3. PGVector/Postgres
4. Weaviate
5. optional Qwen/BGE rerank service
```

## 3. Install VM Python environment

```bash
mkdir -p ~/wns-benchmark-vm
cd ~/wns-benchmark-vm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn sentence-transformers torch transformers numpy
```

If VM has GPU, install the correct CUDA PyTorch build as per WNS VM CUDA version.

## 4. Create embedding service on VM

Create `embedding_service.py`:

```python
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="WNS Embedding Service")

JINA_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
GTE_MODEL_NAME = os.getenv("GTE_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

models = {}

class EmbedRequest(BaseModel):
    model: str | None = None
    texts: List[str]

@app.get("/health")
def health():
    return {"ok": True, "loaded_models": list(models.keys())}

def get_model(key: str):
    if key not in models:
        if key == "jina":
            models[key] = SentenceTransformer(JINA_MODEL_NAME, trust_remote_code=True, device=DEVICE)
        elif key == "gte":
            models[key] = SentenceTransformer(GTE_MODEL_NAME, trust_remote_code=True, device=DEVICE)
        else:
            raise ValueError(f"unknown model key: {key}")
    return models[key]

@app.post("/embed/jina")
def embed_jina(req: EmbedRequest):
    model = get_model("jina")
    vectors = model.encode(req.texts, normalize_embeddings=True, convert_to_numpy=True)
    return {
        "model": JINA_MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "embeddings": vectors.tolist(),
    }

@app.post("/embed/gte")
def embed_gte(req: EmbedRequest):
    model = get_model("gte")
    vectors = model.encode(req.texts, normalize_embeddings=True, convert_to_numpy=True)
    return {
        "model": GTE_MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "embeddings": vectors.tolist(),
    }
```

Start it:

```bash
source .venv/bin/activate
export EMBEDDING_DEVICE=cuda
uvicorn embedding_service:app --host 0.0.0.0 --port 5000
```

For CPU-only fallback:

```bash
export EMBEDDING_DEVICE=cpu
uvicorn embedding_service:app --host 0.0.0.0 --port 5000
```

## 5. Verify embedding endpoints from any machine that can reach VM

```bash
curl http://VM_HOST:5000/health
```

```bash
curl -X POST http://VM_HOST:5000/embed/jina \
  -H "Content-Type: application/json" \
  -d '{"texts":["refund policy", "flight change"]}'
```

```bash
curl -X POST http://VM_HOST:5000/embed/gte \
  -H "Content-Type: application/json" \
  -d '{"texts":["refund policy", "flight change"]}'
```

Expected response has:

```text
model
dimensions
embeddings
```

## 6. Run vector databases on VM

Create `docker-compose.vector-dbs.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "5001:5001"
      - "5002:5002"
    volumes:
      - qdrant_storage:/qdrant/storage

  postgres-pgvector:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: wns
      POSTGRES_PASSWORD: wns_password
      POSTGRES_DB: wns_benchmark
    ports:
      - "5003:5003"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "5004:5004"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      DEFAULT_VECTORIZER_MODULE: none
      CLUSTER_HOSTNAME: node1
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  qdrant_storage:
  pgvector_data:
  weaviate_data:
```

Start:

```bash
docker compose -f docker-compose.vector-dbs.yml up -d
```

Verify:

```bash
curl http://VM_HOST:5001/
curl http://VM_HOST:5004/v1/meta
psql "postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark" -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 7. Benchmark `.env` values

Use VM URLs:

```env
JINA_EMBEDDING_MODE=vm_remote
JINA_EMBEDDING_URL=http://VM_HOST:5000/embed/jina
JINA_API_KEY=

GTE_EMBEDDING_MODE=vm_remote
GTE_EMBEDDING_URL=http://VM_HOST:5000/embed/gte
HF_TOKEN=

QDRANT_URL=http://VM_HOST:5001
PGVECTOR_DSN=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
DATABASE_URL=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
WEAVIATE_URL=http://VM_HOST:5004
```

## 8. Adapter classes needed in benchmark repo

Create these real adapters:

```text
benchmarking/adapters/remote_embeddings.py
  RemoteHTTPEmbeddingAdapter

benchmarking/adapters/vector_qdrant.py
  QdrantVectorStoreAdapter

benchmarking/adapters/vector_pgvector.py
  PGVectorStoreAdapter

benchmarking/adapters/vector_weaviate.py
  WeaviateVectorStoreAdapter
```

Then register them in:

```text
benchmarking/core/registry.py
```

Recommended registry names:

```text
embedding.remote_http
vector_store.qdrant
vector_store.pgvector
vector_store.weaviate
```

Update config so real runs use these adapters instead of `local_hash` and `local_vector`.

## 9. Remote embedding adapter shape

The benchmark runner expects:

```python
embed_many(texts: Iterable[str]) -> list[list[float]]
```

Minimal adapter behavior:

```text
- pick endpoint from config/env
- batch texts
- POST {"texts": [...]} to VM endpoint
- return response["embeddings"]
- retry on transient failures
- timeout after 60 seconds
- cache later if needed
```

## 10. Vector adapter shape

The benchmark runner expects vector stores to support:

```python
upsert(chunks, vectors) -> dict
search(query_vector, top_k) -> list[SearchHit]
```

Each vector DB adapter must:

```text
- create/reset a per-run collection/table/class
- store chunk id, pdf_name, paragraph, metadata, vector
- use HNSW where supported
- use cosine distance/similarity
- return SearchHit(chunk, score)
```

## 11. First test slice

Do not run the full matrix first.

Start with:

```text
entity_heuristic_w6 + jina_v3 + Qdrant + HNSW + cosine + bge-reranker-base
entity_heuristic_w6 + gte_multilingual_base + Qdrant + HNSW + cosine + bge-reranker-base
```

Once this works, add:

```text
PGVector
Weaviate
Qwen reranker
Amazon Rerank
```

## 12. Definition of done

Adapters are ready when:

```text
1. /health works on VM embedding service
2. /embed/jina returns vectors
3. /embed/gte returns vectors
4. Qdrant upsert and search work
5. PGVector upsert and search work
6. Weaviate upsert and search work
7. benchmark selected run produces summary CSV and detail CSV
8. metrics include recall@5, recall@10, MRR, NDCG, latency
```
