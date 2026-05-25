from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, List


class RemoteHTTPEmbeddingAdapter:
    def __init__(
        self,
        model_name: str,
        endpoint_env: str = "",
        endpoint_url: str = "",
        api_key_env: str = "",
        timeout_seconds: int = 120,
        batch_size: int = 32,
        **_: Any,
    ):
        self.name = model_name
        self.model_name = model_name
        self.endpoint_env = endpoint_env
        self.endpoint_url = endpoint_url or (os.environ.get(endpoint_env, "") if endpoint_env else "")
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.timeout_seconds = int(timeout_seconds)
        self.batch_size = int(batch_size)
        self.cost_per_1k_tokens = None
        if not self.endpoint_url:
            raise RuntimeError(f"Missing embedding endpoint for {model_name}. Set {endpoint_env} or endpoint_url in config.")

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(t or "") for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(items), self.batch_size):
            out.extend(self._post_batch(items[i : i + self.batch_size]))
        return out

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]

    def _post_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "texts": texts}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint_url, data=data, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"Embedding endpoint HTTP {exc.code} for {self.endpoint_url}: {body}") from exc
        except Exception as exc:
            raise RuntimeError(f"Embedding endpoint failed for {self.endpoint_url}: {exc}") from exc
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError(
                f"Bad embedding response from {self.endpoint_url}: expected {len(texts)} vectors, got {type(vectors).__name__}"
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.last_latency_ms = latency_ms
        return vectors
