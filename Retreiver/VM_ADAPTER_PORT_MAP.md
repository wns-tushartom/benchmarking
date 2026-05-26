# WNS VM adapter port map

Use only host-facing ports `5000-5010`.

## Ports

- `5000`: unified FastAPI model adapter service
  - `GET /health`
  - `POST /embed/jina`
  - `POST /embed/gte`
  - `POST /rerank/qwen`
  - `POST /rerank/bge`
- `5001`: Qdrant HTTP, container `6333`
- `5002`: Qdrant gRPC, container `6334`
- `5003`: PGVector/Postgres, container `5432`
- `5004`: Weaviate HTTP, container `8080`
- `5005`: Weaviate gRPC, container `50051`

## Can all adapters use one port?

For HTTP model adapters, yes. That is why Jina, GTE, Qwen rerank, and BGE rerank all sit behind one FastAPI service on port `5000` with different paths.

For vector databases, no, not directly. Qdrant HTTP, Qdrant gRPC, Postgres/PGVector, and Weaviate are separate network protocols/services. They need separate host ports unless we add a real reverse proxy or gateway. Postgres is not HTTP, so a path-based proxy cannot share it with FastAPI.

Recommended setup: keep model adapters on one port, keep databases on separate ports. It is simpler and easier to debug.

## `.env` values

```env
JINA_EMBEDDING_URL=http://VM_HOST:5000/embed/jina
GTE_EMBEDDING_URL=http://VM_HOST:5000/embed/gte
QWEN_RERANK_URL=http://VM_HOST:5000/rerank/qwen
BGE_RERANK_URL=http://VM_HOST:5000/rerank/bge

QDRANT_URL=http://VM_HOST:5001
QDRANT_GRPC_URL=http://VM_HOST:5002
PGVECTOR_DSN=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
DATABASE_URL=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
WEAVIATE_URL=http://VM_HOST:5004
WEAVIATE_GRPC_URL=VM_HOST:5005
```
