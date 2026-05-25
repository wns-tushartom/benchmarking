from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from benchmarking.core.schemas import Chunk, SearchHit


def class_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", name).title().replace(" ", "")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "WnsBenchmark" + cleaned
    return cleaned[:200]


class WeaviateVectorStoreAdapter:
    def __init__(self, name: str, url: str = "", class_prefix: str = "WnsBenchmark", **_: Any):
        self.name = name
        self.url = (url or os.environ.get("WEAVIATE_URL", "http://127.0.0.1:5004")).rstrip("/")
        self.api_key = os.environ.get("WEAVIATE_API_KEY", "")
        self.class_name = class_name(f"{class_prefix}_{name}_{os.getpid()}")
        self.chunks_by_id: Dict[int, Chunk] = {}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, payload: Any = None, timeout: float = 30.0) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 404 and method == "DELETE":
                return {}
            raise RuntimeError(f"Weaviate HTTP {exc.code} {method} {path}: {raw[:500]}") from exc

    def reset_collection(self, schema: Any = None) -> None:
        self._request("DELETE", f"/v1/schema/{self.class_name}")

    def _create_class(self, dim: int) -> None:
        self._request("DELETE", f"/v1/schema/{self.class_name}")
        payload = {
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [
                {"name": "idx", "dataType": ["int"]},
                {"name": "chunk_id", "dataType": ["int"]},
                {"name": "pdf_name", "dataType": ["text"]},
                {"name": "paragraph", "dataType": ["text"]},
                {"name": "parent_id", "dataType": ["text"]},
                {"name": "metadata_json", "dataType": ["text"]},
            ],
        }
        self._request("POST", "/v1/schema", payload)

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        started = time.perf_counter()
        self._create_class(len(vectors[0]))
        self.chunks_by_id = {}
        objects = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
            self.chunks_by_id[idx] = chunk
            objects.append(
                {
                    "class": self.class_name,
                    "properties": {
                        "idx": idx,
                        "chunk_id": int(chunk.id),
                        "pdf_name": chunk.pdf_name,
                        "paragraph": chunk.paragraph,
                        "parent_id": chunk.parent_id,
                        "metadata_json": json.dumps(chunk.metadata or {}),
                    },
                    "vector": vector,
                }
            )
        # Batch endpoint is stable across Weaviate v1.
        self._request("POST", "/v1/batch/objects", {"objects": objects}, timeout=120.0)
        return {"upsert_latency_s": time.perf_counter() - started, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        vector_json = json.dumps([float(v) for v in query_vector])
        query = {
            "query": f'''
            {{
              Get {{
                {self.class_name}(
                  nearVector: {{vector: {vector_json}}}
                  limit: {int(top_k)}
                ) {{
                  idx
                  chunk_id
                  pdf_name
                  paragraph
                  parent_id
                  metadata_json
                  _additional {{ distance }}
                }}
              }}
            }}
            '''
        }
        body = self._request("POST", "/v1/graphql", query)
        rows = (((body.get("data") or {}).get("Get") or {}).get(self.class_name) or [])
        hits: List[SearchHit] = []
        for row in rows:
            idx = int(row.get("idx", 0))
            chunk = self.chunks_by_id.get(idx)
            if chunk is None:
                try:
                    metadata = json.loads(row.get("metadata_json") or "{}")
                except Exception:
                    metadata = {}
                chunk = Chunk(
                    id=int(row.get("chunk_id", idx)),
                    pdf_name=str(row.get("pdf_name", "")),
                    paragraph=str(row.get("paragraph", "")),
                    parent_id=str(row.get("parent_id", "")),
                    metadata=metadata,
                )
            distance = float((row.get("_additional") or {}).get("distance", 1.0))
            hits.append(SearchHit(chunk=chunk, score=1.0 - distance))
        return hits
