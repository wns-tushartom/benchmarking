from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, List


class RetryableHTTPError(RuntimeError):
    """HTTP response status for which a request may be retried safely."""


def _positive_int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be a positive integer; got {raw!r}"
        ) from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer; got {raw!r}")
    return value


def _finite_nonnegative_float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be a finite, non-negative number; got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(
            f"{name} must be a finite, non-negative number; got {raw!r}"
        )
    return value


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body or "{}")
    except urllib.error.HTTPError as exc:
        error_type = (
            RetryableHTTPError
            if exc.code in {408, 429} or 500 <= exc.code <= 599
            else RuntimeError
        )
        try:
            raw_body = exc.read()
        except http.client.IncompleteRead as body_error:
            raw_body = body_error.partial
        except Exception:
            raw_body = b""
        body = (
            raw_body.decode("utf-8", errors="replace")
            if isinstance(raw_body, bytes)
            else str(raw_body)
        )
        message = f"HTTP {exc.code} from {url}: {body[:1000]}"
        raise error_type(message) from exc


def _extract_embeddings(response: dict[str, Any]) -> List[List[float]]:
    def vector(raw: Any) -> List[float]:
        if not isinstance(raw, list) or not raw:
            raise RuntimeError("embedding endpoint returned an empty or malformed vector")
        values: List[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError("embedding endpoint vectors must contain finite numeric values")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError("embedding endpoint vectors must contain finite numeric values")
            values.append(numeric)
        return values

    if isinstance(response.get("data"), list):
        vectors = []
        for item in response["data"]:
            if not isinstance(item, dict) or "embedding" not in item:
                raise RuntimeError("embedding endpoint returned a malformed data item")
            vectors.append(vector(item["embedding"]))
        if vectors:
            return vectors
    for key in ("embeddings", "vectors"):
        if isinstance(response.get(key), list):
            return [vector(item) for item in response[key]]
    if isinstance(response.get("embedding"), list):
        return [vector(response["embedding"])]
    raise RuntimeError(f"Embedding endpoint response did not contain embeddings. Keys: {sorted(response.keys())}")


def _response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("request_id", "requestId", "id", "model"):
        value = response.get(key)
        if isinstance(value, (str, int)) and str(value):
            metadata[key] = str(value)[:512]
    provider = response.get("ResponseMetadata")
    if isinstance(provider, dict):
        request_id = provider.get("RequestId")
        if isinstance(request_id, (str, int)) and str(request_id):
            metadata["request_id"] = str(request_id)[:512]
    return metadata


def _require_expected_model(response: dict[str, Any], expected: str, required: bool) -> None:
    if not required:
        return
    actual = response.get("model")
    if not isinstance(actual, str) or not actual.strip():
        raise RuntimeError(f"embedding endpoint did not report model; expected {expected!r}")
    if actual.strip() != expected:
        raise RuntimeError(
            f"embedding endpoint model mismatch: expected {expected!r}, got {actual.strip()!r}"
        )


def _input_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("input_tokens", usage.get("prompt_tokens"))
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


class OpenAIEmbeddingAdapter:
    def __init__(self, model_name: str, dimensions: int = 3072, batch_size: int = 32, api_key_env: str = "OPENAI_API_KEY", url: str = "https://api.openai.com/v1/embeddings", **_: Any):
        self.name = model_name
        self.model_name = model_name
        self.dimensions = int(dimensions)
        self.batch_size = int(batch_size)
        self.url = url
        self.last_response_metadata: dict[str, Any] = {}
        self.input_tokens = 0
        self.input_tokens_complete = True
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} is required for {model_name}. Put it in .env or the process environment.")
        self.timeout_seconds = _positive_int_env(
            "OPENAI_EMBEDDING_TIMEOUT_SECONDS", "300"
        )
        self.retry_delay_seconds = _finite_nonnegative_float_env(
            "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS", "2"
        )

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(t or "") for t in texts]
        out: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            response: dict[str, Any] | None = None
            for attempt in range(6):
                try:
                    response = _post_json(
                        self.url,
                        {"model": self.model_name, "input": batch},
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                    break
                except (
                    RetryableHTTPError,
                    TimeoutError,
                    urllib.error.URLError,
                    http.client.IncompleteRead,
                    OSError,
                ) as exc:
                    self.input_tokens_complete = False
                    if attempt == 5:
                        raise
                    delay = self.retry_delay_seconds * (2**attempt)
                    print(
                        "OPENAI_EMBEDDING_RETRY "
                        f"attempt={attempt + 1} delay_s={delay} error={exc!r}",
                        flush=True,
                    )
                    time.sleep(delay)
            if response is None:
                raise RuntimeError(
                    "OpenAI embedding request exhausted retries without a response"
                )
            metadata = _response_metadata(response)
            measured = _input_tokens(response)
            if measured is None:
                self.input_tokens_complete = False
            elif self.input_tokens_complete:
                self.input_tokens += measured
            if self.input_tokens_complete:
                metadata["embedding_input_tokens"] = self.input_tokens
            self.last_response_metadata = metadata
            vectors = _extract_embeddings(response)
            if len(vectors) != len(batch):
                raise RuntimeError(f"OpenAI returned {len(vectors)} embeddings for {len(batch)} inputs")
            out.extend(vectors)
        if out:
            self.dimensions = len(out[0])
        return out

    def embed(self, text: str) -> List[float]:
        return self.embed_many([text])[0]


class RemoteHTTPEmbeddingAdapter:
    def __init__(
        self,
        model_name: str,
        endpoint_env: str,
        dimensions: int = 768,
        batch_size: int = 32,
        api_key_env: str | None = None,
        require_response_model: bool = False,
        expected_response_model: str | None = None,
        request_model: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        serving_contract: str = "",
        max_input_tokens: int | None = None,
        **_: Any,
    ):
        self.name = model_name
        self.model_name = model_name
        self.request_model = request_model or model_name
        self.query_prefix = str(query_prefix)
        self.document_prefix = str(document_prefix)
        self.serving_contract = str(serving_contract)
        self.max_input_tokens = int(max_input_tokens) if max_input_tokens is not None else None
        self.require_response_model = bool(require_response_model)
        self.expected_response_model = expected_response_model or model_name
        self.dimensions = int(dimensions)
        self.batch_size = int(batch_size)
        self.url = os.environ.get(endpoint_env, "").strip()
        self.last_response_metadata: dict[str, Any] = {}
        self.response_metadata_history: list[dict[str, Any]] = []
        if not self.url:
            raise RuntimeError(f"{endpoint_env} is required for {model_name}. Open-source embeddings must run from the VM endpoint, not local fallback.")
        self.api_key = os.environ.get(api_key_env or "", "").strip() if api_key_env else ""

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return self._embed_many(texts)

    def embed_many_with_role(self, texts: Iterable[str], role: str) -> List[List[float]]:
        prefixes = {"query": self.query_prefix, "document": self.document_prefix}
        if role not in prefixes:
            raise ValueError(f"unsupported embedding role: {role!r}")
        prefix = prefixes[role]
        return self._embed_many(f"{prefix}{str(text or '')}" for text in texts)

    def _embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        items = [str(t or "") for t in texts]
        out: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            payloads = [
                {"model": self.request_model, "input": batch},
                {"inputs": batch, "model": self.request_model},
                {"texts": batch, "model": self.request_model},
            ]
            if self.serving_contract == "vllm_openai_compatible":
                if self.max_input_tokens is None or self.max_input_tokens <= 0:
                    raise RuntimeError("vLLM embedding adapters require a positive max_input_tokens value")
                payloads[0]["truncate_prompt_tokens"] = self.max_input_tokens
            last_error: Exception | None = None
            for payload in payloads:
                try:
                    response = _post_json(self.url, payload, headers=headers)
                    _require_expected_model(response, self.expected_response_model, self.require_response_model)
                    self.last_response_metadata = _response_metadata(response)
                    self.response_metadata_history.append(dict(self.last_response_metadata))
                    vectors = _extract_embeddings(response)
                    if len(vectors) == len(batch):
                        for vector in vectors:
                            if len(vector) != self.dimensions:
                                raise RuntimeError(
                                    f"embedding dimension mismatch for {self.model_name}: "
                                    f"expected {self.dimensions}, got {len(vector)}"
                                )
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
