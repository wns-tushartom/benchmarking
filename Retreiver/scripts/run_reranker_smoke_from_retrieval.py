#!/usr/bin/env python3
"""Apply real reranker endpoints to retrieval smoke artifacts.

Reads data/retrieval_smoke/*.json, calls /rerank/bge and/or /rerank/qwen on the
model adapter service, and writes reranked artifacts to data/reranker_smoke/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.wns_env import load_env_files

ROOT = Path(__file__).resolve().parents[1]
RERANKERS = {
    "bge-reranker-base": ("BGE_RERANK_URL", "http://127.0.0.1:5000/rerank/bge"),
    "qwen3_4b_rerank": ("QWEN_RERANK_URL", "http://127.0.0.1:5000/rerank/qwen"),
    "Qwen3:4B Rerank": ("QWEN_RERANK_URL", "http://127.0.0.1:5000/rerank/qwen"),
    "Amazon Rerank v1": ("", ""),
    "amazon_bedrock": ("", ""),
}


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")[:96]


def canonical_reranker(value: str) -> str:
    raw = str(value or "").strip()
    normalized = "_".join(part for part in raw.lower().replace(":", " ").replace("-", " ").split() if part)
    if normalized in {"qwen", "qwen3", "qwen3_4b", "qwen3_4b_rerank", "qwen3_4b_reranker"}:
        return "qwen3_4b_rerank"
    if normalized in {"amazon", "amazon_rerank", "amazon_rerank_v1", "amazon_bedrock", "amazon_bedrock_rerank"} or ("amazon" in normalized and "rerank" in normalized):
        return "Amazon Rerank v1"
    if normalized in {"bge", "bge_reranker", "bge_reranker_base"}:
        return "bge-reranker-base"
    return raw


def load_env(root: Path) -> None:
    load_env_files(root)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def amazon_model_arn(model_id: str, region: str) -> str:
    if model_id.startswith("arn:aws:bedrock:"):
        return model_id
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"


def amazon_rerank(query: str, documents: list[str], top_k: int) -> dict[str, Any]:
    try:
        import boto3  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("boto3 is required for Amazon Rerank v1") from exc
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION is required for Amazon Rerank v1")
    model_id = amazon_model_arn(os.environ.get("AMAZON_RERANK_MODEL_ID", "amazon.rerank-v1:0").strip(), region)
    sources = [
        {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}}}
        for doc in documents
    ]
    client = boto3.client("bedrock-agent-runtime", region_name=region)
    return client.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": model_id},
                "numberOfResults": min(len(documents), max(top_k, 1)),
            },
        },
    )


def load_retrieval_artifacts(path: Path, limit: int) -> list[Path]:
    files = [p for p in path.glob("*.json") if p.name != "summary.json"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit] if limit else files


def rerank_one(payload: dict[str, Any], reranker: str, top_k: int) -> dict[str, Any]:
    reranker = canonical_reranker(reranker)
    hits = payload.get("hits") or []
    docs = [str(h.get("paragraph") or "") for h in hits]
    start = time.perf_counter()
    if reranker == "Amazon Rerank v1":
        response = amazon_rerank(payload.get("query", ""), docs, top_k)
    else:
        env, default_url = RERANKERS[reranker]
        url = os.environ.get(env, default_url)
        response = post_json(url, {"query": payload.get("query", ""), "documents": docs, "top_k": top_k})
    elapsed = time.perf_counter() - start
    ranked_hits = []
    for i, item in enumerate(response.get("results", []), 1):
        idx = int(item.get("index", 0))
        if idx < 0 or idx >= len(hits):
            continue
        h = dict(hits[idx])
        h["original_rank"] = h.get("rank", idx + 1)
        h["rank"] = i
        h["rerank_score"] = float(item.get("score", item.get("relevance_score", item.get("relevanceScore", 0))))
        ranked_hits.append(h)
    return {
        **payload,
        "created_at": datetime.now().isoformat(),
        "reranker": reranker,
        "rerank_seconds": round(elapsed, 6),
        "retrieval_seconds": payload.get("retrieval_seconds", 0),
        "total_seconds": round(float(payload.get("retrieval_seconds") or 0) + elapsed, 6),
        "retrieved_count": len(ranked_hits),
        "top_k": top_k,
        "hits": ranked_hits,
        "base_artifact": payload.get("artifact", ""),
        "reranker_model": response.get("model", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-dir", default="data/retrieval_smoke")
    parser.add_argument("--out-dir", default="data/reranker_smoke")
    parser.add_argument("--rerankers", nargs="*", default=["bge-reranker-base", "qwen3_4b_rerank"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-artifacts", type=int, default=0, help="0 = all retrieval artifacts")
    args = parser.parse_args()

    os.chdir(ROOT)
    load_env(ROOT)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_retrieval_artifacts(ROOT / args.retrieval_dir, args.limit_artifacts)
    results, errors = [], []
    if not artifacts:
        errors.append({
            "error": "no_retrieval_artifacts",
            "retrieval_dir": args.retrieval_dir,
            "hint": "Run scripts/run_retrieval_smoke_from_vm_dbs.py for the selected current run before reranking.",
        })
        summary = {"created_at": datetime.now().isoformat(), "result_count": 0, "error_count": len(errors), "errors": errors[:100]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 1
    for path in artifacts:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["artifact"] = str(path.relative_to(ROOT))
            for requested_reranker in args.rerankers:
                reranker = canonical_reranker(requested_reranker)
                if reranker not in RERANKERS:
                    raise ValueError(f"Unknown reranker {requested_reranker}; known={sorted(RERANKERS)}")
                result = rerank_one(payload, reranker, args.top_k)
                name = f"{safe_name(payload.get('sheet',''))}_{safe_name(payload.get('embedding',''))}_{safe_name(payload.get('store',''))}_{safe_name(reranker)}_{safe_name(payload.get('query',''))}.json"
                (out_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"OK reranker={reranker} sheet={payload.get('sheet')} embedding={payload.get('embedding')} store={payload.get('store')} query={payload.get('query')!r} seconds={result['rerank_seconds']}", flush=True)
                results.append(result)
        except Exception as exc:
            err = {"artifact": str(path), "error": repr(exc)}
            errors.append(err)
            print(f"ERROR {path}: {exc!r}", flush=True)
    summary = {"created_at": datetime.now().isoformat(), "result_count": len(results), "error_count": len(errors), "errors": errors[:100]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
