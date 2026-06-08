#!/usr/bin/env python3
"""Compare reranked ground-truth evaluation against the no-reranker baseline.

Outputs:
- data/reranker_analysis/reranker_lift_summary.csv
- data/reranker_analysis/reranker_lift_details.csv
- data/reranker_analysis/qwen_vs_none_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def n(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def rank_value(value: Any) -> int:
    r = int(n(value))
    return r if r > 0 else 999


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("sheet", "")),
        str(row.get("embedding", "")),
        str(row.get("store", "")),
        str(row.get("query", "")).strip().lower(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data/evaluation")
    parser.add_argument("--reranked-dir", default="data/evaluation_reranked")
    parser.add_argument("--out-dir", default="data/reranker_analysis")
    parser.add_argument("--reranker", default="qwen3_4b_rerank")
    args = parser.parse_args()

    base_dir = ROOT / args.base_dir
    reranked_dir = ROOT / args.reranked_dir
    out_dir = ROOT / args.out_dir
    base_details = read_csv(base_dir / "groundtruth_eval_details.csv")
    reranked_details = read_csv(reranked_dir / "groundtruth_eval_details.csv")
    reranker_name = args.reranker

    base_by_key = {key(r): r for r in base_details if str(r.get("reranker", "none") or "none") == "none"}
    reranked_rows = [r for r in reranked_details if str(r.get("reranker", "")) == reranker_name]

    details: list[dict[str, Any]] = []
    missing_base = 0
    for row in reranked_rows:
        base = base_by_key.get(key(row))
        if not base:
            missing_base += 1
            continue
        base_rank = rank_value(base.get("first_relevant_rank"))
        rerank_rank = rank_value(row.get("first_relevant_rank"))
        delta_mrr = n(row.get("mrr")) - n(base.get("mrr"))
        delta_hit5 = n(row.get("hit_at_5")) - n(base.get("hit_at_5"))
        if rerank_rank < base_rank:
            verdict = "improved"
        elif rerank_rank > base_rank:
            verdict = "worse"
        else:
            verdict = "same"
        details.append({
            "sheet": row.get("sheet", ""),
            "embedding": row.get("embedding", ""),
            "store": row.get("store", ""),
            "query": row.get("query", ""),
            "reranker": reranker_name,
            "base_rank": 0 if base_rank == 999 else base_rank,
            "reranked_rank": 0 if rerank_rank == 999 else rerank_rank,
            "verdict": verdict,
            "delta_mrr": round(delta_mrr, 6),
            "delta_hit_at_5": round(delta_hit5, 6),
            "base_top_pdf": base.get("top_pdf", ""),
            "reranked_top_pdf": row.get("top_pdf", ""),
            "expected_pdf": row.get("expected_pdf", ""),
            "base_artifact": base.get("artifact", ""),
            "reranked_artifact": row.get("artifact", ""),
        })

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(row["sheet"], row["embedding"], row["store"], row["reranker"])].append(row)

    summary: list[dict[str, Any]] = []
    for (sheet, embedding, store, reranker), rows in grouped.items():
        summary.append({
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "reranker": reranker,
            "paired_queries": len(rows),
            "improved_queries": sum(1 for r in rows if r["verdict"] == "improved"),
            "worse_queries": sum(1 for r in rows if r["verdict"] == "worse"),
            "same_queries": sum(1 for r in rows if r["verdict"] == "same"),
            "avg_delta_mrr": avg([float(r["delta_mrr"]) for r in rows]),
            "avg_delta_hit_at_5": avg([float(r["delta_hit_at_5"]) for r in rows]),
        })
    summary.sort(key=lambda r: (float(r["avg_delta_mrr"]), float(r["avg_delta_hit_at_5"])))

    causes = []
    if missing_base:
        causes.append(f"{missing_base} reranked rows had no matching no-reranker baseline row. That means the comparison set is stale or mismatched.")
    total_worse = sum(1 for r in details if r["verdict"] == "worse")
    total_paired = len(details)
    if total_paired and total_worse / total_paired > 0.25:
        causes.append("Qwen is demoting relevant hits for a meaningful share of paired queries; inspect reranker_lift_details.csv for the exact queries and PDFs.")
    if not details:
        causes.append("No paired baseline-versus-Qwen rows were found. Run retrieval, Qwen reranking, and both evaluations on the same selected pipeline first.")
    causes.append("Most common setup issue: reranker evaluation used a limited/stale artifact subset while no-reranker evaluation used a fuller retrieval set. Run the complete pipeline with artifact cleanup and --limit-artifacts 0.")
    causes.append("Model issue to verify: Qwen3 reranker is instruction-aware and yes/no-logit based. If the VM uses an old sentence-transformers/transformers stack or a generic CrossEncoder path without the Qwen reranker template, Qwen scores can be poorly calibrated.")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "reranker_lift_summary.csv", summary)
    write_csv(out_dir / "reranker_lift_details.csv", details)
    report = {
        "created_at": datetime.now().isoformat(),
        "reranker": reranker_name,
        "base_detail_rows": len(base_details),
        "reranked_detail_rows": len(reranked_details),
        "paired_rows": len(details),
        "missing_base_rows": missing_base,
        "summary_rows": len(summary),
        "worst_pipeline": summary[0] if summary else None,
        "likely_causes": causes,
        "outputs": {
            "summary": str((out_dir / "reranker_lift_summary.csv").relative_to(ROOT)),
            "details": str((out_dir / "reranker_lift_details.csv").relative_to(ROOT)),
        },
    }
    (out_dir / "qwen_vs_none_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if details else 1


if __name__ == "__main__":
    raise SystemExit(main())
