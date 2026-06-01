#!/usr/bin/env python3
"""Evaluate retrieval smoke artifacts against a ground-truth CSV/XLSX.

Expected columns are flexible. At minimum provide a query column:
- query/question
Optional relevance columns:
- pdf_name/expected_pdf/relevant_pdf/document
- paragraph/expected_text/context/ground_truth/answer

A hit is relevant when expected PDF matches a retrieved PDF, or expected text overlaps
with the retrieved paragraph above --text-threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

QUERY_COLS = ["query", "question", "user_query", "prompt"]
PDF_COLS = ["pdf_name", "expected_pdf", "relevant_pdf", "document", "file", "filename"]
TEXT_COLS = ["paragraph", "expected_text", "context", "ground_truth", "ground truth", "answer", "relevant_text"]
ID_COLS = ["id", "query_id", "qid"]


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
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for n in names:
        if n in lowered and str(lowered[n] or "").strip():
            return str(lowered[n]).strip()
    return ""


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_xlsx(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl
    except Exception as exc:  # pragma: no cover
        raise SystemExit("openpyxl is required for XLSX ground truth. Install with: pip install openpyxl") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h or "").strip() for h in rows[0]]
    return [dict(zip(headers, r)) for r in rows[1:] if any(v is not None and str(v).strip() for v in r)]


def load_groundtruth(path: Path) -> list[dict[str, str]]:
    raw = load_xlsx(path) if path.suffix.lower() in {".xlsx", ".xlsm"} else load_csv(path)
    out = []
    for i, row in enumerate(raw, 1):
        query = first(row, QUERY_COLS)
        if not query:
            continue
        out.append({
            "id": first(row, ID_COLS) or str(i),
            "query": query,
            "expected_pdf": first(row, PDF_COLS),
            "expected_text": first(row, TEXT_COLS),
        })
    return out


def load_smokes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(path.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            obj["artifact"] = str(p.relative_to(ROOT))
            rows.append(obj)
        except Exception:
            pass
    return rows


def relevance_flags(case: dict[str, str], hits: list[dict[str, Any]], threshold: float) -> list[bool]:
    expected_pdf = norm(case.get("expected_pdf"))
    expected_text = case.get("expected_text", "")
    flags = []
    for h in hits:
        pdf_ok = bool(expected_pdf and expected_pdf == norm(h.get("pdf_name")))
        text_score = overlap(expected_text, h.get("paragraph", "")) if expected_text else 0.0
        flags.append(bool(pdf_ok or text_score >= threshold))
    return flags


def dcg(flags: list[bool], k: int) -> float:
    return sum((1.0 if flag else 0.0) / math.log2(i + 2) for i, flag in enumerate(flags[:k]))


def evaluate_case(case: dict[str, str], smoke: dict[str, Any], threshold: float) -> dict[str, Any]:
    hits = smoke.get("hits") or []
    flags = relevance_flags(case, hits, threshold)
    first_rank = next((i + 1 for i, ok in enumerate(flags) if ok), 0)
    return {
        "id": case["id"],
        "query": case["query"],
        "sheet": smoke.get("sheet", ""),
        "embedding": smoke.get("embedding", ""),
        "store": smoke.get("store", ""),
        "top_k": smoke.get("top_k", len(hits)),
        "retrieved_count": len(hits),
        "latency_seconds": smoke.get("retrieval_seconds", ""),
        "expected_pdf": case.get("expected_pdf", ""),
        "top_pdf": (hits[0] or {}).get("pdf_name", "") if hits else "",
        "first_relevant_rank": first_rank,
        "hit_at_1": int(any(flags[:1])),
        "hit_at_3": int(any(flags[:3])),
        "hit_at_5": int(any(flags[:5])),
        "hit_at_10": int(any(flags[:10])),
        "mrr": round(1.0 / first_rank, 6) if first_rank else 0.0,
        "ndcg_at_5": round(dcg(flags, 5), 6),
        "artifact": smoke.get("artifact", ""),
    }


def avg(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, help="CSV/XLSX with query + expected pdf/text columns")
    parser.add_argument("--smoke-dir", default="data/retrieval_smoke")
    parser.add_argument("--out-dir", default="data/evaluation")
    parser.add_argument("--text-threshold", type=float, default=0.35)
    args = parser.parse_args()

    gt = load_groundtruth(Path(args.groundtruth))
    smokes = load_smokes(ROOT / args.smoke_dir)
    smoke_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in smokes:
        smoke_by_query[norm(s.get("query"))].append(s)

    detail = []
    missing = []
    for case in gt:
        matched = smoke_by_query.get(norm(case["query"]), [])
        if not matched:
            missing.append(case)
            continue
        for smoke in matched:
            detail.append(evaluate_case(case, smoke, args.text_threshold))

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in detail:
        groups[(r["sheet"], r["embedding"], r["store"])].append(r)
    summary = []
    for (sheet, embedding, store), rows in sorted(groups.items()):
        summary.append({
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "evaluated_queries": len(rows),
            "recall_at_1": avg([float(r["hit_at_1"]) for r in rows]),
            "recall_at_3": avg([float(r["hit_at_3"]) for r in rows]),
            "recall_at_5": avg([float(r["hit_at_5"]) for r in rows]),
            "recall_at_10": avg([float(r["hit_at_10"]) for r in rows]),
            "mrr": avg([float(r["mrr"]) for r in rows]),
            "ndcg_at_5": avg([float(r["ndcg_at_5"]) for r in rows]),
            "avg_latency_seconds": avg([float(r["latency_seconds"] or 0) for r in rows]),
        })
    summary.sort(key=lambda r: (-float(r["recall_at_5"]), -float(r["mrr"]), float(r["avg_latency_seconds"])))

    out = ROOT / args.out_dir
    write_csv(out / "groundtruth_eval_details.csv", detail)
    write_csv(out / "groundtruth_eval_summary.csv", summary)
    (out / "groundtruth_eval_report.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(),
        "groundtruth_rows": len(gt),
        "retrieval_smoke_artifacts": len(smokes),
        "evaluated_rows": len(detail),
        "missing_query_count": len(missing),
        "missing_queries": missing[:50],
        "summary_rows": len(summary),
        "best": summary[0] if summary else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"groundtruth_rows={len(gt)} retrieval_smoke_artifacts={len(smokes)} evaluated_rows={len(detail)} missing_queries={len(missing)}")
    print(f"summary={out / 'groundtruth_eval_summary.csv'}")
    print(f"details={out / 'groundtruth_eval_details.csv'}")
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
