#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def safe_name(value):
    return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_")[:96]
def clean_text(value):
    text = " ".join(str(value or "").split())
    max_chars = int(os.environ.get("RERANK_MAX_CHARS", "2500"))
    return text[:max_chars]
def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1200]}") from exc
def load_artifacts(path, limit):
    files = [p for p in Path(path).glob("*.json") if p.name != "summary.json"]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit] if limit else files
def rerank_file(path, out_dir, url, top_k):
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits = payload.get("hits") or []
    docs = [clean_text(h.get("paragraph") or "") for h in hits]
    start = time.perf_counter()
    response = post_json(url, {
        "query": payload.get("query", ""),
        "documents": docs,
        "top_k": top_k,
    })
    elapsed = time.perf_counter() - start
    ranked = []
    for rank, item in enumerate(response.get("results", []), 1):
        idx = int(item.get("index", 0))
        if idx < 0 or idx >= len(hits):
            continue
        h = dict(hits[idx])
        h["paragraph"] = clean_text(h.get("paragraph") or "")
        h["original_rank"] = h.get("rank", idx + 1)
        h["rank"] = rank
        h["rerank_score"] = float(item.get("score", 0))
        ranked.append(h)
    result = dict(payload)
    result.update({
        "created_at": datetime.now().isoformat(),
        "reranker": "bge-reranker-base",
        "reranker_model": response.get("model", ""),
        "rerank_seconds": round(elapsed, 6),
        "retrieval_seconds": payload.get("retrieval_seconds", 0),
        "total_seconds": round(float(payload.get("retrieval_seconds") or 0) + elapsed, 6),
        "retrieved_count": len(ranked),
        "top_k": top_k,
        "hits": ranked,
        "base_artifact": str(path),
        "rerank_max_chars": int(os.environ.get("RERANK_MAX_CHARS", "2500")),
    })
    name = "{}_{}_{}_bge-reranker-base_{}.json".format(
        safe_name(payload.get("sheet", "")),
        safe_name(payload.get("embedding", "")),
        safe_name(payload.get("store", "")),
safe_name(payload.get("query", "")),
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-dir", default="data/retrieval_smoke")
    ap.add_argument("--out-dir", default="data/reranker_smoke")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit-artifacts", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    url = os.environ.get("BGE_RERANK_URL", "http://127.0.0.1:5000/rerank/bge")
    artifacts = load_artifacts(args.retrieval_dir, args.limit_artifacts)
    print(f"artifacts={len(artifacts)} url={url} top_k={args.top_k} max_chars={os.environ.get('RERANK_MAX_CHARS', '2500')}", flush=True)
    results = []
    errors = []
    for i, path in enumerate(artifacts, 1):
        try:
            result = rerank_file(path, args.out_dir, url, args.top_k)
            print(f"OK {i}/{len(artifacts)} sheet={result.get('sheet')} embedding={result.get('embedding')} store={result.get('store')} seconds={result.get('rerank_seconds')} query={result.get('query')!r}", flush=True)
            results.append(result)
        except Exception as exc:
            err = {"artifact": str(path), "error": repr(exc)}
            errors.append(err)
            print(f"ERROR {i}/{len(artifacts)} artifact={path.name}: {exc!r}", flush=True)
    summary = {
        "created_at": datetime.now().isoformat(),
        "reranker": "bge-reranker-base",
        "result_count": len(results),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if results else 1
if __name__ == "__main__":
    raise SystemExit(main())
