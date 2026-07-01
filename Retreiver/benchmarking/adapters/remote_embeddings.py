from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable, List
OPENAI_MODEL_ALIASES = {
    "openai_text-embedding-3-large": "text-embedding-3-large",
    "openai_text_embedding_3_large": "text-embedding-3-large",
}


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def _extract_embeddings(response: dict[str, Any]) -> List[List[float]]:
    if isinstance(response.get("data"), list):
        vectors = []
        for item in response["data"]:
            if isinstance(item, dict) and "embedding" in item:
                vectors.append([float(x) for x in item["embedding"]])
        if vectors:
            return vectors
    for key in ("embeddings", "vectors"):
        if isinstance(response.get(key), list):
            return [[float(x) for x in vec] for vec in response[key]]
    if isinstance(response.get("embedding"), list):
        return [[float(x) for x in response["embedding"]]]
    raise RuntimeError(f"Embedding endpoint response did not contain embeddings. Keys: {sorted(response.keys())}")


class OpenAIEmbeddingAdapter:
    def __init__(self, model_name: str, dimensions: int = 1536, batch_size: int = 32, api_key_env: str = "OPENAI_API_KEY", url: str = "https://api.openai.com/v1/embeddings", **_: Any):
        self.name = model_name
        self.model_name = model_name
        self.api_model_name = OPENAI_MODEL_ALIASES.get(model_name, model_name)
        self.dimensions = int(dimensions)
        self.batch_size = int(batch_size)
        self.url = url
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} is required for {model_name}. Put it in .env or the process environment.")
    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(t or "") for t in texts]
        out: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            payload = {
                "model": self.api_model_name,
                "input": batch,
                "dimensions": self.dimensions,
            }
            response = _post_json(self.url, payload, headers=headers)
            vectors = _extract_embeddings(response)
            if len(vectors) != len(batch):
                raise RuntimeError(f"OpenAI returned {len(vectors)} embeddings for {len(batch)} inputs")
            out.extend(vectors)
        if out:
            actual_dim = len(out[0])
            if actual_dim != self.dimensions:
                raise RuntimeError(f"OpenAI returned dimension {actual_dim}, expected {self.dimensions}")
        return out
    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]


class RemoteHTTPEmbeddingAdapter:
    def __init__(self, model_name: str, endpoint_env: str, dimensions: int = 768, batch_size: int = 32, api_key_env: str | None = None, **_: Any):
        self.name = model_name
        self.model_name = model_name
        self.api_model_name = OPENAI_MODEL_ALIASES.get(model_name, model_name)
        self.dimensions = int(dimensions)
        self.batch_size = int(batch_size)
        self.url = os.environ.get(endpoint_env, "").strip()
        if not self.url:
            raise RuntimeError(f"{endpoint_env} is required for {model_name}. Open-source embeddings must run from the VM endpoint, not local fallback.")
        self.api_key = os.environ.get(api_key_env or "", "").strip() if api_key_env else ""

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(t or "") for t in texts]
        out: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            payloads = [
                {"model": self.model_name, "input": batch},
                {"inputs": batch, "model": self.model_name},
                {"texts": batch, "model": self.model_name},
            ]
            last_error: Exception | None = None
            for payload in payloads:
                try:
                    vectors = _extract_embeddings(_post_json(self.url, payload, headers=headers))
                    if len(vectors) == len(batch):
                        out.extend(vectors)
                        break
                    last_error = RuntimeError(f"endpoint returned {len(vectors)} embeddings for {len(batch)} inputs")
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"Embedding endpoint failed for {self.model_name}: {last_error}")
        if out:
            self.dimensions = len(out[0])
        return out

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]
