#!/usr/bin/env python3
"""Evaluate hallucination/grounding risk for retrieval artifacts.

This audit does not generate chatbot answers. It checks whether the retrieved context is
sufficient to support the expected answer/text from a ground-truth file. If --use-llm is
provided and OPENAI_API_KEY is available, it uses an LLM judge. Otherwise it uses a
transparent lexical support score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUERY_COLS = ["query", "question", "user_query", "prompt"]
PDF_COLS = ["pdf_name", "expected_pdf", "relevant_pdf", "document", "file", "filename"]
TEXT_COLS = ["answer", "ground_truth", "ground truth", "expected_answer", "expected_text", "context", "paragraph", "relevant_text"]
ID_COLS = ["id", "query_id", "qid"]


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(s).lower()))


def support_score(expected: str, context: str) -> float:
    expected_tokens = tokens(expected)
    context_tokens = tokens(context)
    if not expected_tokens or not context_tokens:
        return 0.0
    return round(len(expected_tokens & context_tokens) / max(1, len(expected_tokens)), 6)


def first(row: dict[str, Any], names: list[str]) -> str:
    lowered = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        if name in lowered and str(lowered[name] or "").strip():
            return str(lowered[name]).strip()
    return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_xlsx(path: Path) -> list[dict[str, Any]]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h or "").strip() for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:] if any(v is not None and str(v).strip() for v in row)]


def load_groundtruth(path: Path) -> list[dict[str, str]]:
    raw = read_xlsx(path) if path.suffix.lower() in {".xlsx", ".xlsm"} else read_csv(path)
    rows = []
    for i, row in enumerate(raw, 1):
        query = first(row, QUERY_COLS)
        answer = first(row, TEXT_COLS)
        if not query:
            continue
        rows.append({
            "id": first(row, ID_COLS) or str(i),
            "query": query,
            "expected_pdf": first(row, PDF_COLS),
            "expected_answer": answer,
        })
    return rows


def load_artifacts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(path.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        obj["artifact"] = str(p.relative_to(ROOT))
        rows.append(obj)
    return rows


def llm_judge(model: str, query: str, expected: str, context: str) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    prompt = f"""You are auditing RAG grounding for an enterprise support assistant.
Return strict JSON only with keys: verdict, score, reason.
verdict must be one of: grounded, partial, unsupported, no_expected_answer.
score is 0 to 1.
Question: {query}
Expected answer or ground truth: {expected or '[missing]'}
Retrieved context:
{context[:6000]}
"""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Be strict about unsupported claims."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(text)
    except Exception:
        return {"verdict": "partial", "score": 0.5, "reason": text[:500]}


def deterministic_judge(expected: str, context: str) -> dict[str, Any]:
    if not expected.strip():
        return {"verdict": "no_expected_answer", "score": 0.0, "reason": "No expected answer/text was provided in ground truth."}
    score = support_score(expected, context)
    if score >= 0.72:
        verdict = "grounded"
    elif score >= 0.38:
        verdict = "partial"
    else:
        verdict = "unsupported"
    return {"verdict": verdict, "score": score, "reason": f"Lexical support score={score:.3f}. Use --use-llm for semantic judging."}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--artifact-dir", default="data/reranker_smoke")
    parser.add_argument("--out-dir", default="data/hallucination")
    parser.add_argument("--top-k-context", type=int, default=5)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", default=os.getenv("HALLUCINATION_JUDGE_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    gt = load_groundtruth(Path(args.groundtruth))
    artifacts = load_artifacts(ROOT / args.artifact_dir)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        by_query[norm(artifact.get("query", "")).lower()].append(artifact)

    rows = []
    evaluated = 0
    for case in gt:
        if evaluated >= args.limit:
            break
        matches = by_query.get(norm(case["query"]).lower(), [])
        if not matches:
            rows.append({
                "id": case["id"], "query": case["query"], "sheet": "", "embedding": "", "store": "", "reranker": "", "verdict": "missing_retrieval", "support_score": 0.0, "risk": "high", "reason": "No retrieval/reranker artifact found for this query.", "artifact": "",
            })
            evaluated += 1
            continue
        for artifact in matches:
            if evaluated >= args.limit:
                break
            hits = artifact.get("hits") or []
            context = "\n\n".join(str(h.get("paragraph", "")) for h in hits[:args.top_k_context])
            try:
                judge = llm_judge(args.model, case["query"], case.get("expected_answer", ""), context) if args.use_llm else deterministic_judge(case.get("expected_answer", ""), context)
            except Exception as exc:
                judge = {"verdict": "judge_error", "score": 0.0, "reason": str(exc)}
            score = float(judge.get("score") or 0)
            verdict = str(judge.get("verdict") or "unsupported")
            risk = "low" if verdict == "grounded" and score >= 0.72 else "medium" if verdict in {"grounded", "partial"} else "high"
            rows.append({
                "id": case["id"],
                "query": case["query"],
                "sheet": artifact.get("sheet", ""),
                "embedding": artifact.get("embedding", ""),
                "store": artifact.get("store", ""),
                "reranker": artifact.get("reranker", "none"),
                "verdict": verdict,
                "support_score": round(score, 6),
                "risk": risk,
                "reason": str(judge.get("reason", ""))[:700],
                "artifact": artifact.get("artifact", ""),
            })
            evaluated += 1
            if args.use_llm:
                time.sleep(0.2)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sheet"], row["embedding"], row["store"], row["reranker"])].append(row)
    summary = []
    for (sheet, embedding, store, reranker), group in grouped.items():
        total = len(group)
        high = sum(1 for r in group if r["risk"] == "high")
        grounded = sum(1 for r in group if r["verdict"] == "grounded")
        avg_score = round(sum(float(r["support_score"] or 0) for r in group) / max(1, total), 6)
        summary.append({
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "reranker": reranker,
            "evaluated_queries": total,
            "grounded_rate": round(grounded / max(1, total), 6),
            "high_risk_count": high,
            "hallucination_risk_rate": round(high / max(1, total), 6),
            "avg_support_score": avg_score,
        })
    summary.sort(key=lambda r: (float(r["hallucination_risk_rate"]), -float(r["grounded_rate"]), -float(r["avg_support_score"])))

    out = ROOT / args.out_dir
    write_csv(out / "hallucination_details.csv", rows)
    write_csv(out / "hallucination_summary.csv", summary)
    report = {
        "created_at": datetime.now().isoformat(),
        "judge": "llm" if args.use_llm else "lexical_support",
        "model": args.model if args.use_llm else None,
        "groundtruth_rows": len(gt),
        "artifact_rows": len(artifacts),
        "evaluated_rows": len(rows),
        "summary_rows": len(summary),
        "best": summary[0] if summary else None,
    }
    (out / "hallucination_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"hallucination_evaluated_rows={len(rows)} summary_rows={len(summary)} judge={report['judge']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
