from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List

from benchmarking.core.schemas import Chunk, SearchHit


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return cleaned or "wns_benchmark"


class QdrantVectorStoreAdapter:
    def __init__(self, name: str, url: str = "", collection_prefix: str = "wns_benchmark", **_: Any):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except Exception as exc:
            raise RuntimeError("qdrant-client is required. Run: python -m pip install qdrant-client") from exc
        self.QdrantClient = QdrantClient
        self.Distance = Distance
        self.VectorParams = VectorParams
        self.name = name
        self.url = url or os.environ.get("QDRANT_URL", "http://127.0.0.1:5001")
        self.api_key = os.environ.get("QDRANT_API_KEY") or None
        self.collection_name = f"{safe_name(collection_prefix)}_{safe_name(name)}_{os.getpid()}"
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.chunks_by_id: Dict[int, Chunk] = {}

    def reset_collection(self, schema: Any = None) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        from qdrant_client.models import PointStruct

        started = time.perf_counter()
        dim = len(vectors[0])
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.VectorParams(size=dim, distance=self.Distance.COSINE),
        )
        points = []
        self.chunks_by_id = {}
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
            point_id = int(idx)
            self.chunks_by_id[point_id] = chunk
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "pdf_name": chunk.pdf_name,
                        "paragraph": chunk.paragraph,
                        "parent_id": chunk.parent_id,
                        "metadata": chunk.metadata or {},
                    },
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return {"upsert_latency_s": time.perf_counter() - started, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        # qdrant-client versions differ. query_points is newer, search is older.
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(collection_name=self.collection_name, query=query_vector, limit=top_k)
            points = getattr(result, "points", result)
        else:
            points = self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=top_k)
        hits: List[SearchHit] = []
        for point in points:
            point_id = int(point.id)
            chunk = self.chunks_by_id.get(point_id)
            if chunk is None:
                payload = point.payload or {}
                chunk = Chunk(
                    id=int(payload.get("chunk_id", point_id)),
                    pdf_name=str(payload.get("pdf_name", "")),
                    paragraph=str(payload.get("paragraph", "")),
                    parent_id=str(payload.get("parent_id", "")),
                    metadata=payload.get("metadata") or {},
                )
            hits.append(SearchHit(chunk=chunk, score=float(point.score)))
        return hits
