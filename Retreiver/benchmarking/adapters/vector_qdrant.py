from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from benchmarking.core.schemas import Chunk, SearchHit


class QdrantVectorStoreAdapter:
    def __init__(self, name: str, url_env: str = "QDRANT_URL", api_key_env: str = "QDRANT_API_KEY", collection_prefix: str = "wns_benchmark", **_: Any):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except Exception as exc:
            raise RuntimeError("qdrant-client is required for QdrantVectorStoreAdapter. Install requirements-benchmark.txt.") from exc
        self.models = models
        self.name = name
        self.url = os.environ.get(url_env, "http://127.0.0.1:5001")
        self.api_key = os.environ.get(api_key_env) or None
        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=300)
        self.collection = f"{collection_prefix}_{os.getpid()}_{int(time.time())}"
        self.chunks: Dict[str, Chunk] = {}

    def reset_collection(self, schema: Any = None) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        start = time.perf_counter()
        dim = len(vectors[0])
        self.reset_collection()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self.models.VectorParams(size=dim, distance=self.models.Distance.COSINE),
            hnsw_config=self.models.HnswConfigDiff(m=16, ef_construct=100),
        )
        points = []
        self.chunks = {}
        for i, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
            self.chunks[str(i)] = chunk
            points.append(self.models.PointStruct(id=i, vector=vector, payload={"chunk_key": str(i), "chunk_id": chunk.id, "pdf_name": chunk.pdf_name, "paragraph": chunk.paragraph}))
        for batch_start in range(0, len(points), 256):
            self.client.upsert(collection_name=self.collection, points=points[batch_start:batch_start + 256], wait=True)
        return {"upsert_latency_s": time.perf_counter() - start, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        if hasattr(self.client, "search"):
            results = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=top_k, with_payload=True)
        else:
            response = self.client.query_points(collection_name=self.collection, query=query_vector, limit=top_k, with_payload=True)
            results = getattr(response, "points", response)
        hits: List[SearchHit] = []
        for item in results:
            payload = item.payload or {}
            chunk = self.chunks.get(str(payload.get("chunk_key")))
            if chunk:
                hits.append(SearchHit(chunk=chunk, score=float(item.score)))
        return hits
