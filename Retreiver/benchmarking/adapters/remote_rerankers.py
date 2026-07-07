from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, List

from benchmarking.core.schemas import SearchHit


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def _scores_from_response(response: dict[str, Any], count: int) -> List[float]:
    if isinstance(response.get("scores"), list):
        return [float(x) for x in response["scores"]]
    if isinstance(response.get("results"), list):
        scores = [0.0] * count
        direct = []
        for i, item in enumerate(response["results"]):
            if not isinstance(item, dict):
                continue
            score = item.get("score", item.get("relevance_score", item.get("relevanceScore")))
            if score is None:
                continue
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < count:
                scores[index] = float(score)
            else:
                direct.append(float(score))
        return scores if any(scores) or not direct else direct
    if isinstance(response.get("data"), list):
        return [float(item.get("score", item.get("relevance_score", 0.0))) for item in response["data"] if isinstance(item, dict)]
    raise RuntimeError(f"Rerank endpoint response did not contain scores. Keys: {sorted(response.keys())}")


class RemoteHTTPRerankerAdapter:
    def __init__(self, name: str, endpoint_env: str, api_key_env: str | None = None, model: str | None = None, **_: Any):
        self.name = name
        self.model = model or name
        self.url = os.environ.get(endpoint_env, "").strip()
        if not self.url:
            raise RuntimeError(f"{endpoint_env} is required for reranker {name}. Open-source rerankers must run from the VM endpoint, not local fallback.")
        self.api_key = os.environ.get(api_key_env or "", "").strip() if api_key_env else ""

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        documents = [h.chunk.paragraph for h in hits]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payloads = [
            {"model": self.model, "query": query, "documents": documents, "top_k": len(documents)},
            {"model": self.model, "query": query, "texts": documents, "top_k": len(documents)},
            {"query": query, "passages": documents, "model": self.model, "top_k": len(documents)},
        ]
        last_error: Exception | None = None
        for payload in payloads:
            try:
                scores = _scores_from_response(_post_json(self.url, payload, headers=headers), len(hits))
                if len(scores) >= len(hits):
                    rescored = [SearchHit(hit.chunk, float(scores[i])) for i, hit in enumerate(hits)]
                    rescored.sort(key=lambda h: h.score, reverse=True)
                    return rescored[:top_k]
                last_error = RuntimeError(f"endpoint returned {len(scores)} scores for {len(hits)} documents")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Reranker endpoint failed for {self.name}: {last_error}")


class AmazonBedrockRerankerAdapter:
    def __init__(self, name: str, model_id_env: str = "AMAZON_RERANK_MODEL_ID", region_env: str = "AWS_REGION", **_: Any):
        self.name = name
        self.region = os.environ.get(region_env) or os.environ.get("AWS_DEFAULT_REGION")
        if not self.region:
            raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION is required for Amazon Rerank v1")
        raw_model_id = os.environ.get(model_id_env, "amazon.rerank-v1:0").strip()
        self.model_id = self._model_arn(raw_model_id)
        try:
            import boto3  # type: ignore
        except Exception as exc:
            raise RuntimeError("boto3 is required for Amazon Rerank v1. Install boto3 in the benchmark venv.") from exc
        self.client = boto3.client("bedrock-agent-runtime", region_name=self.region)

    def _model_arn(self, model_id: str) -> str:
        if model_id.startswith("arn:aws:bedrock:"):
            return model_id
        return f"arn:aws:bedrock:{self.region}::foundation-model/{model_id}"

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        sources = [{"type": "INLINE", "inlineDocumentSource": {"textDocument": {"text": h.chunk.paragraph}}} for h in hits]
        response = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=sources,
            rerankingConfiguration={"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": {"modelConfiguration": {"modelArn": self.model_id}, "numberOfResults": min(len(hits), max(top_k, 1))}},
        )
        results = response.get("results", [])
        rescored: List[SearchHit] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevanceScore", item.get("score", 0.0))
            if isinstance(idx, int) and 0 <= idx < len(hits):
                rescored.append(SearchHit(hits[idx].chunk, float(score)))
        if not rescored:
            raise RuntimeError("Amazon Rerank v1 returned no rerank results")
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]
