#!/usr/bin/env python3
"""Benchmark NVIDIA RAG Blueprint retrieval against Project Smiley ground truth.

This is a real-service benchmark lane, separate from the modular Vector DB matrix.
It reads the current WNS ground-truth CSV/XLSX, queries NVIDIA rag-server /v1/search,
normalizes returned chunks/citations, and writes deterministic lexical relevance metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "nvidia_rag"
DEFAULT_GT = ROOT / "data" / "groundtruth" / "groundtruth_500.csv"

QUERY_COLS = ["query", "question", "user_query", "prompt"]
PDF_COLS = ["pdf_name", "expected_pdf", "relevant_pdf", "document", "file", "filename"]
TEXT_COLS = ["paragraph", "expected_text", "context", "ground_truth", "ground truth", "answer", "relevant_text"]
ID_COLS = ["id", "query_id", "qid"]


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


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(s)))


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def first(row: dict[str, Any], names: list[str]) -> str:
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_xlsx(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl
    except Exception as exc:  # pragma: no cover
        raise SystemExit("openpyxl is required for XLSX ground truth. Install with: pip install openpyxl") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        return []
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h or "").strip() for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:] if any(v is not None and str(v).strip() for v in row)]


def load_groundtruth(path: Path, limit: int) -> list[dict[str, str]]:
    raw = load_xlsx(path) if path.suffix.lower() in {".xlsx", ".xlsm"} else load_csv(path)
    cases: list[dict[str, str]] = []
    for idx, row in enumerate(raw, 1):
        query = first(row, QUERY_COLS)
        if not query:
            continue
        cases.append({
            "id": first(row, ID_COLS) or str(idx),
            "query": query,
            "expected_pdf": first(row, PDF_COLS),
            "expected_text": first(row, TEXT_COLS),
        })
        if limit > 0 and len(cases) >= limit:
            break
    return cases


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


def nested_get(obj: Any, path: list[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def find_items(body: Any) -> list[Any]:
    candidates = [
        body.get("results") if isinstance(body, dict) else None,
        body.get("documents") if isinstance(body, dict) else None,
        body.get("chunks") if isinstance(body, dict) else None,
        body.get("passages") if isinstance(body, dict) else None,
        body.get("citations") if isinstance(body, dict) else None,
        body.get("retrieved_documents") if isinstance(body, dict) else None,
        nested_get(body, ["data", "results"]),
        nested_get(body, ["data", "documents"]),
        nested_get(body, ["data", "chunks"]),
        nested_get(body, ["response", "results"]),
        nested_get(body, ["response", "documents"]),
    ]
    for value in candidates:
        if isinstance(value, list):
            return value
    return []


def hit_source(hit: Any) -> str:
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    doc = hit.get("document") if isinstance(hit.get("document"), dict) else {}
    for value in [
        hit.get("pdf_name"), hit.get("document_name"), hit.get("source"), hit.get("filename"), hit.get("title"), hit.get("id"),
        meta.get("pdf_name"), meta.get("source"), meta.get("filename"), meta.get("document_name"),
        doc.get("pdf_name"), doc.get("source"), doc.get("filename"), doc.get("document_name"),
    ]:
        if str(value or "").strip():
            return str(value).strip()
    return ""


def hit_text(hit: Any) -> str:
    if isinstance(hit, str):
        return re.sub(r"\s+", " ", hit).strip()
    if not isinstance(hit, dict):
        return ""
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    doc = hit.get("document") if isinstance(hit.get("document"), dict) else {}
    for value in [
        hit.get("paragraph"), hit.get("text"), hit.get("content"), hit.get("chunk"), hit.get("page_content"), hit.get("passage"),
        doc.get("text"), doc.get("content"), doc.get("page_content"),
        meta.get("text"), meta.get("content"), meta.get("paragraph"),
    ]:
        if str(value or "").strip():
            return re.sub(r"\s+", " ", str(value)).strip()
    try:
        return json.dumps(hit, ensure_ascii=False)[:1200]
    except Exception:
        return ""


def hit_score(hit: Any) -> Any:
    if not isinstance(hit, dict):
        return ""
    for key in ["score", "relevance_score", "similarity", "distance", "rank_score"]:
        if key in hit:
            return hit[key]
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    for key in ["score", "relevance_score", "similarity", "distance"]:
        if key in meta:
            return meta[key]
    return ""


def normalize_hits(body: Any, top_k: int) -> list[dict[str, Any]]:
    hits = []
    for idx, item in enumerate(find_items(body)[:top_k], 1):
        hits.append({
            "rank": idx,
            "pdf_name": hit_source(item),
            "paragraph": hit_text(item),
            "score": hit_score(item),
        })
    return hits


def relevance_flags(case: dict[str, str], hits: list[dict[str, Any]], threshold: float) -> tuple[list[bool], list[float]]:
    expected_pdf = norm(case.get("expected_pdf"))
    expected_text = case.get("expected_text", "")
    flags: list[bool] = []
    scores: list[float] = []
    for hit in hits:
        source = norm(hit.get("pdf_name"))
        pdf_ok = bool(expected_pdf and (expected_pdf == source or expected_pdf in source or source in expected_pdf))
        text_score = overlap(expected_text, hit.get("paragraph", "")) if expected_text else 0.0
        scores.append(round(text_score, 6))
        flags.append(bool(pdf_ok or text_score >= threshold))
    return flags, scores


def dcg(flags: list[bool], k: int) -> float:
    return sum((1.0 if flag else 0.0) / math.log2(i + 2) for i, flag in enumerate(flags[:k]))


def ndcg(flags: list[bool], k: int) -> float:
    actual = dcg(flags, k)
    ideal = dcg(sorted(flags[:k], reverse=True), k)
    return 0.0 if ideal <= 0 else actual / ideal


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def nvidia_pipeline_label() -> str:
    return os.getenv("NVIDIA_RAG_PIPELINE_LABEL", "NVIDIA RAG service lane")


def nvidia_embedding_label() -> str:
    return os.getenv("NVIDIA_RAG_EMBEDDING_MODEL") or "not reported by NVIDIA service"


def nvidia_splitter_label() -> str:
    return os.getenv("NVIDIA_RAG_SPLITTER") or "not reported by NVIDIA service"


def nvidia_reranker_label(enabled: bool) -> str:
    if not enabled:
        return "none"
    return os.getenv("NVIDIA_RAG_RERANKER_MODEL") or "service reranker enabled"


def nvidia_success_flags(details: list[dict[str, Any]]) -> tuple[bool, bool, int]:
    successful_count = sum(1 for r in details if int(r.get("ok", 0)) == 1)
    all_queries_ok = bool(details) and successful_count == len(details)
    partial_ok = 0 < successful_count < len(details)
    return all_queries_ok, partial_ok, successful_count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def evaluate_case(case: dict[str, str], response: dict[str, Any], top_k: int, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hits = normalize_hits(response.get("body") or {}, top_k)
    flags, text_scores = relevance_flags(case, hits, threshold)
    first_rank = next((i + 1 for i, ok in enumerate(flags) if ok), 0)
    detail = {
        "id": case["id"],
        "query": case["query"],
        "pipeline": nvidia_pipeline_label(),
        "sheet": nvidia_splitter_label(),
        "embedding": nvidia_embedding_label(),
        "store": os.getenv("NVIDIA_RAG_COLLECTION", "multimodal_data"),
        "reranker": nvidia_reranker_label(bool(response.get("reranker_enabled"))),
        "top_k": top_k,
        "retrieved_count": len(hits),
        "latency_seconds": response.get("latency_seconds", ""),
        "status": response.get("status", ""),
        "ok": int(bool(response.get("ok"))),
        "expected_pdf": case.get("expected_pdf", ""),
        "top_pdf": hits[0].get("pdf_name", "") if hits else "",
        "first_relevant_rank": first_rank,
        "hit_at_1": int(any(flags[:1])),
        "hit_at_3": int(any(flags[:3])),
        "hit_at_5": int(any(flags[:5])),
        "hit_at_10": int(any(flags[:10])),
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "precision_at_5": round(sum(1 for x in flags[:5] if x) / max(1, min(5, len(hits))), 6),
        "ndcg_at_5": round(ndcg(flags, 5), 6),
        "best_text_overlap": max(text_scores or [0.0]),
        "top_snippet": (hits[0].get("paragraph", "")[:500] if hits else response.get("error", "")),
        "error": response.get("error", ""),
    }
    return detail, hits


def summarize(rows: list[dict[str, Any]], collection: str, reranker_enabled: bool) -> list[dict[str, Any]]:
    if not rows:
        return []
    hit_ranks = [int(r["first_relevant_rank"]) for r in rows if int(r["first_relevant_rank"]) > 0]
    recall5 = avg([float(r["hit_at_5"]) for r in rows])
    mrr_score = avg([float(r["mrr"]) for r in rows])
    ndcg5 = avg([float(r["ndcg_at_5"]) for r in rows])
    latency = avg([float(r["latency_seconds"] or 0) for r in rows])
    winner_score = round((0.45 * recall5) + (0.30 * mrr_score) + (0.20 * ndcg5) + (0.05 * (1 / (1 + latency))), 6)
    return [{
        "pipeline": nvidia_pipeline_label(),
        "sheet": nvidia_splitter_label(),
        "embedding": nvidia_embedding_label(),
        "store": collection,
        "reranker": nvidia_reranker_label(reranker_enabled),
        "evaluated_queries": len(rows),
        "successful_queries": sum(1 for r in rows if int(r.get("ok", 0)) == 1),
        "recall_at_1": avg([float(r["hit_at_1"]) for r in rows]),
        "recall_at_3": avg([float(r["hit_at_3"]) for r in rows]),
        "recall_at_5": recall5,
        "recall_at_10": avg([float(r["hit_at_10"]) for r in rows]),
        "mrr": mrr_score,
        "precision_at_5": avg([float(r["precision_at_5"]) for r in rows]),
        "ndcg_at_5": ndcg5,
        "avg_first_relevant_rank": avg([float(x) for x in hit_ranks]),
        "no_hit_queries": sum(1 for r in rows if int(r["first_relevant_rank"]) == 0),
        "avg_latency_seconds": latency,
        "winner_score": winner_score,
    }]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--url", default="", help="Base NVIDIA rag-server URL. Defaults to NVIDIA_RAG_SERVER_URL or http://127.0.0.1:5006")
    parser.add_argument("--groundtruth", default=str(DEFAULT_GT))
    parser.add_argument("--collection", default=os.getenv("NVIDIA_RAG_COLLECTION", "multimodal_data"))
    parser.add_argument("--limit", type=int, default=25, help="0 means all ground-truth rows")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-top-k", type=int, default=5)
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--text-threshold", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    load_env_file(ROOT / ".env.project-smiley-nvidia")
    load_env_file(ROOT / ".env.vm.generated")
    collection = args.collection
    base = (args.url or os.getenv("NVIDIA_RAG_SERVER_URL", "http://127.0.0.1:5006")).rstrip("/")
    url = f"{base}/v1/search"
    gt_path = Path(args.groundtruth)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path
    cases = load_groundtruth(gt_path, args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    details: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    first_error = ""
    reranker_enabled = not args.disable_reranker
    started = time.time()
    for index, case in enumerate(cases, 1):
        payload = search_payload(case["query"], collection, args.top_k, args.reranker_top_k, reranker_enabled)
        response = post_json(url, payload, args.timeout)
        response["reranker_enabled"] = reranker_enabled
        detail, hits = evaluate_case(case, response, args.top_k, args.text_threshold)
        details.append(detail)
        if hits and len(previews) < 20:
            previews.append({"query": case["query"], "hits": hits[:3], "latency_seconds": response.get("latency_seconds")})
        if not response.get("ok") and not first_error:
            first_error = response.get("error", "") or json.dumps(response.get("body", {}), ensure_ascii=False)[:600]
        print(f"{index}/{len(cases)} status={response.get('status')} hits={len(hits)} query={case['query'][:80]!r}", flush=True)

    all_queries_ok, partial_ok, successful_count = nvidia_success_flags(details)
    summary = summarize(details, collection, reranker_enabled) if successful_count else []
    mode = "reranked" if reranker_enabled else "baseline"
    details_path = OUT_DIR / f"benchmark_{mode}_details.csv"
    summary_path = OUT_DIR / f"benchmark_{mode}_summary.csv"
    report_path = OUT_DIR / f"benchmark_{mode}_report.json"
    latest_path = OUT_DIR / f"benchmark_{mode}_latest.json"
    write_csv(details_path, details)
    write_csv(summary_path, summary)
    # Compatibility/latest files are overwritten intentionally, while baseline/reranked files remain separate.
    write_csv(OUT_DIR / "benchmark_details.csv", details)
    write_csv(OUT_DIR / "benchmark_summary.csv", summary)
    report = {
        "ok": all_queries_ok,
        "partial_ok": partial_ok,
        "created_at": datetime.now().isoformat(),
        "url": url,
        "collection": collection,
        "groundtruth": str(gt_path.relative_to(ROOT) if gt_path.is_relative_to(ROOT) else gt_path),
        "groundtruth_rows_loaded": len(cases),
        "evaluated_rows": len(details),
        "successful_queries": successful_count,
        "failed_queries": len(details) - successful_count,
        "top_k": args.top_k,
        "reranker_enabled": reranker_enabled,
        "mode": mode,
        "text_threshold": args.text_threshold,
        "total_seconds": round(time.time() - started, 6),
        "best": summary[0] if summary else None,
        "first_error": first_error,
        "preview": previews,
        "methodology": "Deterministic benchmark over NVIDIA RAG /v1/search. Relevance uses expected PDF match when present, otherwise lexical overlap against ground-truth context/answer. ok=true only when every evaluated query succeeds.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_payload = {"report": report, "summary": summary, "details_preview": details[:20]}
    latest_path.write_text(json.dumps(latest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "benchmark_latest.json").write_text(json.dumps(latest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "ok": report["ok"],
        "groundtruth_rows": len(cases),
        "evaluated_rows": len(details),
        "successful_queries": report["successful_queries"],
        "summary": str(summary_path.relative_to(ROOT)),
        "details": str(details_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "error": first_error[:300],
    }, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
