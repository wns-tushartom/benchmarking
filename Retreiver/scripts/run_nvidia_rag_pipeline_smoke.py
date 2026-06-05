#!/usr/bin/env python3
"""Run a Project Smiley smoke query through NVIDIA RAG Blueprint.

Uses NVIDIA rag-server /v1/search by default because it is the best benchmark-facing
proof: retrieval results, reranker behavior, latency, and citations can be inspected
without relying on a long generation call.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "nvidia_rag" / "smoke_latest.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.time()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                body: Any = json.loads(text) if text else {}
            except json.JSONDecodeError:
                body = {"raw": text[:4000]}
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "latency_seconds": round(time.time() - started, 6), "body": body}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "latency_seconds": round(time.time() - started, 6), "error": exc.read().decode("utf-8", errors="replace")[:4000]}
    except Exception as exc:
        return {"ok": False, "status": None, "latency_seconds": round(time.time() - started, 6), "error": str(exc)}


def search_payload(query: str, collection: str, top_k: int, reranker_top_k: int, reranker: bool) -> dict[str, Any]:
    return {
        "query": query,
        "reranker_top_k": reranker_top_k,
        "vdb_top_k": top_k,
        "collection_names": [collection],
        "messages": [],
        "enable_query_rewriting": False,
        "enable_reranker": reranker,
    }


def generate_payload(query: str, collection: str, top_k: int, reranker_top_k: int, reranker: bool, agentic: bool) -> dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": query}],
        "use_knowledge_base": True,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 768,
        "reranker_top_k": reranker_top_k,
        "vdb_top_k": top_k,
        "collection_names": [collection],
        "enable_query_rewriting": False,
        "enable_reranker": reranker,
        "enable_citations": True,
        "stop": [],
        "filter_expr": "",
    }
    if agentic:
        payload["agentic"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--url", default="", help="Base NVIDIA rag-server URL. Defaults to NVIDIA_RAG_SERVER_URL or http://127.0.0.1:5006")
    parser.add_argument("--query", default="refund old ticket and issue new ticket")
    parser.add_argument("--collection", default=os.getenv("NVIDIA_RAG_COLLECTION", "multimodal_data"))
    parser.add_argument("--mode", choices=["search", "generate"], default="search")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-top-k", type=int, default=5)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    load_env_file(ROOT / ".env.vm.generated")
    base = (args.url or os.getenv("NVIDIA_RAG_SERVER_URL", "http://127.0.0.1:5006")).rstrip("/")
    url = f"{base}/v1/{args.mode}"
    body = generate_payload(args.query, args.collection, args.top_k, args.reranker_top_k, not args.disable_reranker, args.agentic) if args.mode == "generate" else search_payload(args.query, args.collection, args.top_k, args.reranker_top_k, not args.disable_reranker)
    result = post_json(url, body, args.timeout)
    payload = {
        "ok": result["ok"],
        "mode": args.mode,
        "url": url,
        "query": args.query,
        "collection": args.collection,
        "request": body,
        "response": result,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "mode": args.mode, "latency_seconds": result["latency_seconds"], "status": result.get("status"), "out": str(out.relative_to(ROOT)), "error": result.get("error", "")[:300]}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
