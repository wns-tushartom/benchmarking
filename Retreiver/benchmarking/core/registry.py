from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def register(self, kind: str, name: str, adapter: Any) -> None:
        if not kind or not name:
            raise ValueError("kind and name are required")
        self._items[kind][name] = adapter

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._items[kind][name]
        except KeyError as exc:
            raise KeyError(f"Adapter not registered: {kind}.{name}") from exc

    def list(self, kind: str) -> List[str]:
        return sorted(self._items.get(kind, {}).keys())


def default_registry() -> Registry:
    from benchmarking.adapters.local import LocalChunkWorkbookAdapter, LocalHashEmbeddingAdapter, LocalVectorStoreAdapter, WeightedOverlapReranker, OverlapEvaluator
    from benchmarking.adapters.remote_embeddings import OpenAIEmbeddingAdapter, RemoteHTTPEmbeddingAdapter
    from benchmarking.adapters.remote_rerankers import AmazonBedrockRerankerAdapter, RemoteHTTPRerankerAdapter
    from benchmarking.adapters.vector_qdrant import QdrantVectorStoreAdapter
    from benchmarking.adapters.vector_pgvector import PGVectorStoreAdapter
    from benchmarking.adapters.vector_weaviate import WeaviateVectorStoreAdapter

    r = Registry()
    r.register("chunker", "local_workbook", LocalChunkWorkbookAdapter)
    r.register("embedding", "local_hash", LocalHashEmbeddingAdapter)
    r.register("embedding", "openai", OpenAIEmbeddingAdapter)
    r.register("embedding", "remote_http", RemoteHTTPEmbeddingAdapter)
    r.register("vector_store", "local_vector", LocalVectorStoreAdapter)
    r.register("vector_store", "qdrant", QdrantVectorStoreAdapter)
    r.register("vector_store", "pgvector", PGVectorStoreAdapter)
    r.register("vector_store", "weaviate", WeaviateVectorStoreAdapter)
    r.register("reranker", "weighted_overlap", WeightedOverlapReranker)
    r.register("reranker", "amazon_bedrock", AmazonBedrockRerankerAdapter)
    r.register("reranker", "remote_http", RemoteHTTPRerankerAdapter)
    r.register("evaluator", "overlap_relevance", OverlapEvaluator)
    return r
