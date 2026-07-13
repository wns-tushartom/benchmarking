from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from benchmarking.core.schemas import Chunk, SearchHit
from benchmarking.adapters.vector_namespace import safe_weaviate_namespace


def _request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            return {}
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


class WeaviateVectorStoreAdapter:
    def __init__(
        self,
        name: str,
        url_env: str = "WEAVIATE_URL",
        class_prefix: str = "WnsBenchmark",
        namespace: str | None = None,
        **_: Any,
    ):
        self.name = name
        self.url = os.environ.get(url_env, "http://127.0.0.1:5004").rstrip("/")
        self.class_name = (
            safe_weaviate_namespace(namespace)
            if namespace is not None
            else f"{class_prefix}{os.getpid()}{int(time.time())}"
        )
        self.physical_namespace = self.class_name

    def reset_collection(self, schema: Any = None) -> None:
        try:
            _request("DELETE", f"{self.url}/v1/schema/{self.class_name}")
        except Exception:
            pass

    def drop_namespace(self) -> None:
        """Idempotently remove this adapter's physical class."""
        self.reset_collection()

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        start = time.perf_counter()
        self.reset_collection()
        _request("POST", f"{self.url}/v1/schema", {
            "class": self.class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "chunk_id", "dataType": ["text"]},
                {"name": "pdf_name", "dataType": ["text"]},
                {"name": "paragraph", "dataType": ["text"]},
                {"name": "page_number", "dataType": ["text"]},
                {"name": "source_type", "dataType": ["text"]},
                {"name": "parser_method", "dataType": ["text"]},
            ],
        })
        objects = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
            metadata = chunk.metadata or {}
            objects.append({
                "class": self.class_name,
                "id": f"00000000-0000-0000-0000-{i:012d}",
                "properties": {
                    "chunk_id": str(chunk.id),
                    "pdf_name": chunk.pdf_name,
                    "paragraph": chunk.paragraph,
                    "page_number": str(metadata.get("page_number", "")),
                    "source_type": str(metadata.get("source_type", "")),
                    "parser_method": str(metadata.get("parser_method", "")),
                },
                "vector": vector,
            })
        for i in range(0, len(objects), 100):
            _request("POST", f"{self.url}/v1/batch/objects", {"objects": objects[i:i+100]})
        return {"upsert_latency_s": time.perf_counter() - start, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        gql = {
            "query": "{ Get { %s(nearVector:{vector:%s} limit:%d) { chunk_id pdf_name paragraph page_number source_type parser_method _additional { distance certainty } } } }" % (
                self.class_name,
                json.dumps([float(v) for v in query_vector]),
                int(top_k),
            )
        }
        data = _request("POST", f"{self.url}/v1/graphql", gql)
        rows = data.get("data", {}).get("Get", {}).get(self.class_name, [])
        hits: List[SearchHit] = []
        for i, row in enumerate(rows, 1):
            add = row.get("_additional", {})
            score = add.get("certainty")
            if score is None and add.get("distance") is not None:
                score = 1.0 - float(add["distance"])
            hits.append(SearchHit(Chunk(id=int(row.get("chunk_id") or i), pdf_name=row.get("pdf_name", ""), paragraph=row.get("paragraph", ""), parent_id=str(row.get("chunk_id", i)), metadata={"store": "Weaviate", "page_number": row.get("page_number", ""), "source_type": row.get("source_type", ""), "parser_method": row.get("parser_method", "")}), float(score or 0.0)))
        return hits
