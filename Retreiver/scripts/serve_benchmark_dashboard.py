#!/usr/bin/env python3
"""Serve the WNS benchmark dashboard using only the Python standard library."""

from __future__ import annotations

import csv
import cgi
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.core.config import generate_matrix, load_benchmark_config

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FULL_DIR = ROOT / "data" / "full_benchmark"
MODULAR_DIR = ROOT / "data" / "modular_runs" / "latest"
INGESTION_DIR = ROOT / "data" / "db_ingestion_runs"
RERANKER_DIR = ROOT / "data" / "reranker_smoke"
RETRIEVAL_DIR = ROOT / "data" / "retrieval_smoke"
SNAPSHOT_PATH = ROOT / "data" / "vm_dashboard_snapshot.json"
EVAL_DIR = ROOT / "data" / "evaluation"
HALLUCINATION_DIR = ROOT / "data" / "hallucination"
NVIDIA_RAG_DIR = ROOT / "data" / "nvidia_rag"
GROUNDTRUTH_DIR = ROOT / "data" / "groundtruth"
PDF_AUDIT_PATH = ROOT / "data" / "pdf_extraction_audit.csv"
CONFIG_PATH = ROOT / "configs" / "benchmark.local.json"


def benchmark_options() -> dict:
    cfg = load_benchmark_config(CONFIG_PATH)
    matrix = cfg.get("matrix", {})
    return {
        "chunkers": matrix.get("chunkers", []),
        "embeddings": matrix.get("embeddings", []),
        "vector_stores": matrix.get("vector_stores", []),
        "index_types": matrix.get("index_types", ["HNSW"]),
        "retrieval_methods": matrix.get("retrieval_methods") or matrix.get("retrievers", ["Cosine Similarity"]),
        "rerankers": matrix.get("rerankers", []),
        "matrix_count": len(generate_matrix(cfg)),
        "mode": cfg.get("experiment", {}).get("mode", "vm_remote_required"),
    }


def selected_cli_args(qs: dict[str, list[str]]) -> list[str]:
    arg_map = {
        "chunker": "--chunker",
        "embedding": "--embedding",
        "vector_store": "--vector-store",
        "index_type": "--index-type",
        "retrieval_method": "--retrieval-method",
        "reranker": "--reranker",
    }
    args: list[str] = []
    for key, flag in arg_map.items():
        value = qs.get(key, ["all"])[0]
        if value and value != "all":
            args += [flag, value]
    return args


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def read_ingestion_summaries() -> list[dict]:
    rows: list[dict] = []
    if not INGESTION_DIR.exists():
        return rows
    for path in sorted(INGESTION_DIR.glob("*/summary.csv")):
        run_id = path.parent.name
        for row in read_csv(path):
            row["run_id"] = run_id
            rows.append(row)
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows


def summarize_ingestion(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get("status") == "ok"]
    failures = [r for r in rows if r.get("status") and r.get("status") != "ok"]
    embeddings = sorted({r.get("embedding", "") for r in completed if r.get("embedding")})
    stores = sorted({r.get("store", "") for r in completed if r.get("store")})
    sheets = sorted({r.get("sheet", "") for r in completed if r.get("sheet")})
    combos = {(r.get("sheet"), r.get("embedding"), r.get("store")) for r in completed}
    return {
        "rows": rows,
        "completed_rows": len(completed),
        "failure_rows": len(failures),
        "embeddings": embeddings,
        "stores": stores,
        "sheets": sheets,
        "combo_count": len(combos),
        "latest_run_id": rows[0].get("run_id") if rows else None,
        "latest_created_at": rows[0].get("created_at") if rows else None,
    }


def read_reranker_smokes(limit: int = 250) -> list[dict]:
    smokes: list[dict] = []
    if not RERANKER_DIR.exists():
        return smokes
    paths = [p for p in RERANKER_DIR.glob("*.json") if p.name != "summary.json"]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"error": str(exc)}
        payload["artifact"] = str(path.relative_to(ROOT))
        payload["hits"] = (payload.get("hits") or [])[:3]
        smokes.append(payload)
    smokes.sort(key=lambda r: (r.get("created_at", ""), r.get("rerank_seconds", 999999)), reverse=True)
    return smokes


def read_retrieval_smokes(limit: int = 250) -> list[dict[str, Any]]:
    smokes: list[dict[str, Any]] = []
    if not RETRIEVAL_DIR.exists():
        return smokes
    paths = [p for p in RETRIEVAL_DIR.glob("*.json") if p.name != "summary.json"]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"error": str(exc)}
        payload["artifact"] = str(path.relative_to(ROOT))
        payload["hits"] = (payload.get("hits") or [])[:3]
        smokes.append(payload)
    smokes.sort(key=lambda r: (r.get("created_at", ""), r.get("retrieval_seconds", 999999)), reverse=True)
    return smokes


def retrieval_smoke_count() -> int:
    if not RETRIEVAL_DIR.exists():
        return 0
    return sum(1 for p in RETRIEVAL_DIR.glob("*.json") if p.name != "summary.json")


def reranker_smoke_count() -> int:
    if not RERANKER_DIR.exists():
        return 0
    return sum(1 for p in RERANKER_DIR.glob("*.json") if p.name != "summary.json")


def read_snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        return {}
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path.relative_to(ROOT))}


def read_nvidia_rag() -> dict:
    health = read_json_file(NVIDIA_RAG_DIR / "health.json")
    smoke = read_json_file(NVIDIA_RAG_DIR / "smoke_latest.json")
    ingestion = read_json_file(NVIDIA_RAG_DIR / "ingestion_latest.json")
    files = []
    if NVIDIA_RAG_DIR.exists():
        files = [str(p.relative_to(ROOT)) for p in sorted(NVIDIA_RAG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
    return {
        "health": health,
        "smoke": smoke,
        "ingestion": ingestion,
        "files": files,
        "configured": bool(health or smoke or ingestion or os.getenv("NVIDIA_RAG_SERVER_URL") or os.getenv("NVIDIA_INGESTOR_URL")),
    }


def read_evaluation_dir(path: Path) -> dict:
    summary = sort_summary(read_csv(path / "groundtruth_eval_summary.csv")) if path.exists() else []
    details = read_csv(path / "groundtruth_eval_details.csv") if path.exists() else []
    report_path = path / "groundtruth_eval_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report = {"error": str(exc)}
    return {"summary": summary, "details": details[:500], "report": report}


def read_evaluation() -> dict:
    base = read_evaluation_dir(EVAL_DIR)
    base["reranked"] = read_evaluation_dir(ROOT / "data" / "evaluation_reranked")
    gt_files = sorted([str(p.relative_to(ROOT)) for p in GROUNDTRUTH_DIR.glob("*")]) if GROUNDTRUTH_DIR.exists() else []
    base["groundtruth_files"] = gt_files
    return base


def read_hallucination() -> dict:
    summary = sort_summary(read_csv(HALLUCINATION_DIR / "hallucination_summary.csv")) if HALLUCINATION_DIR.exists() else []
    details = read_csv(HALLUCINATION_DIR / "hallucination_details.csv") if HALLUCINATION_DIR.exists() else []
    report_path = HALLUCINATION_DIR / "hallucination_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report = {"error": str(exc)}
    return {"summary": summary, "details": details[:500], "report": report}


def read_pdf_audit() -> dict:
    rows = read_csv(PDF_AUDIT_PATH)
    needs = [r for r in rows if str(r.get("needs_ocr_review", "")).lower() in {"1", "true", "yes"} or r.get("status") in {"needs_ocr", "partial_ocr_review", "failed"}]
    ok = [r for r in rows if r not in needs]
    return {
        "rows": rows[:500],
        "total": len(rows),
        "ok_count": len(ok),
        "needs_ocr_count": len(needs),
        "needs_ocr": needs[:100],
    }


def add_multi(cmd: list[str], flag: str, values: list[str]) -> None:
    vals = [v for v in values if v and v != "all"]
    if vals:
        cmd.append(flag)
        cmd.extend(vals)

def sort_summary(rows: list[dict]) -> list[dict]:
    def key(r: dict) -> tuple:
        try:
            winner = float(r.get("winner_score") or 0)
        except ValueError:
            winner = 0.0
        try:
            recall = float(r.get("recall_at_5") or 0)
        except ValueError:
            recall = 0.0
        try:
            mrr = float(r.get("mrr") or 0)
        except ValueError:
            mrr = 0.0
        try:
            latency = float(r.get("avg_query_latency_ms") or r.get("avg_latency_ms") or r.get("avg_latency_seconds") or 999999)
        except ValueError:
            latency = 999999.0
        return (-winner, -recall, -mrr, latency)
    return sorted(rows, key=key)


def safe_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip(".-_")[:80] or "dataset"


def safe_extract_zip(zip_path: Path, target_dir: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                continue
            suffix = name.suffix.lower()
            if suffix not in {".pdf", ".csv", ".xlsx", ".txt", ".md"}:
                continue
            dest = target_dir / "extracted" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(str(dest.relative_to(ROOT)))
    return extracted


def upload_next_steps(saved_path: Path, dataset_dir: Path, extracted: list[str]) -> list[str]:
    rel_dir = str(dataset_dir.relative_to(ROOT))
    steps = [
        f"Review uploaded files under {rel_dir}",
        "If this is a new corpus, copy PDFs into data/pdfs or update the extraction script to read this upload folder.",
        "Prepare/attach a ground-truth CSV with id, query, expected_pdf or expected text span before claiming quality metrics.",
        "Run ingestion and retrieval only after ground truth and chunking inputs are ready.",
    ]
    if saved_path.suffix.lower() == ".zip":
        steps.insert(1, f"ZIP extracted {len(extracted)} supported files under {rel_dir}/extracted")
    return steps


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/options":
            self.send_json(benchmark_options())
            return
        if parsed.path == "/api/results":
            summary = sort_summary(read_csv(FULL_DIR / "benchmark_summary.csv"))
            chunking = sort_summary(read_csv(ROOT / "data" / "chunking_recall_summary.csv"))
            best = summary[0].get("recall_at_5") if summary else None
            query_count = summary[0].get("query_count") if summary else None
            files = [
                "data/full_benchmark/benchmark_summary.csv",
                "data/full_benchmark/benchmark_details.csv",
                "data/full_benchmark/benchmark_report.json",
                "data/chunking_recall_summary.csv",
                "data/chunking_recall_results.csv",
                "data/chunking_methods_output_v2.xlsx",
                "data/modular_runs/latest/modular_summary.csv",
                "data/modular_runs/latest/modular_details.csv",
                "data/modular_runs/latest/analysis.json",
                "data/modular_runs/latest/manifest.json",
                "data/modular_runs/latest/MODULAR_REPORT.md",
                "data/vm_dashboard_snapshot.json",
                "data/nvidia_rag/health.json",
                "data/nvidia_rag/smoke_latest.json",
                "data/nvidia_rag/ingestion_latest.json",
                "data/retrieval_smoke/summary.json",
                "data/evaluation/groundtruth_eval_summary.csv",
                "data/evaluation/groundtruth_eval_details.csv",
                "data/evaluation/groundtruth_eval_report.json",
                "WNS_VM_PROGRESS_20260528.md",
                "JINA_TIMINGS_20260528.md",
                "RERANKER_SMOKE_20260528.md",
                "WNS_PARALLEL_EXECUTION_PLAN.md",
                "WNS_BENCHMARK_FINAL_REPORT.md",
                "MODULAR_BENCHMARKING_ROADMAP.md",
            ]
            modular_summary = sort_summary(read_csv(MODULAR_DIR / "modular_summary.csv"))
            modular_analysis = {}
            modular_manifest = {}
            for name, target in [("analysis", MODULAR_DIR / "analysis.json"), ("manifest", MODULAR_DIR / "manifest.json")]:
                if target.exists():
                    try:
                        if name == "analysis":
                            modular_analysis = json.loads(target.read_text(encoding="utf-8"))
                        else:
                            modular_manifest = json.loads(target.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            ingestion_rows = read_ingestion_summaries()
            retrieval_limit = int(parse_qs(parsed.query).get("retrieval_limit", ["120"])[0])
            reranker_limit = int(parse_qs(parsed.query).get("reranker_limit", ["120"])[0])
            retrieval_smokes = read_retrieval_smokes(limit=retrieval_limit)
            reranker_smokes = read_reranker_smokes(limit=reranker_limit)
            retrieval_total = retrieval_smoke_count()
            reranker_total = reranker_smoke_count()
            evaluation = read_evaluation()
            hallucination = read_hallucination()
            pdf_audit = read_pdf_audit()
            nvidia_rag = read_nvidia_rag()
            snapshot = read_snapshot()
            self.send_json({
                "summary_count": len(summary),
                "query_count": query_count,
                "best_recall_at_5": best,
                "summary": summary,
                "chunking": chunking,
                "modular": {
                    "summary": modular_summary,
                    "analysis": modular_analysis,
                    "manifest": modular_manifest,
                },
                "operational": {
                    "ingestion": summarize_ingestion(ingestion_rows),
                    "reranker_smokes": reranker_smokes,
                    "reranker_smoke_total": reranker_total,
                    "reranker_smoke_loaded": len(reranker_smokes),
                    "retrieval_smokes": retrieval_smokes,
                    "retrieval_smoke_total": retrieval_total,
                    "retrieval_smoke_loaded": len(retrieval_smokes),
                    "evaluation": evaluation,
                    "hallucination": hallucination,
                    "nvidia_rag": nvidia_rag,
                    "vm_snapshot": snapshot,
                    "service_health": snapshot.get("health", []),
                    "known_pdf_count": pdf_audit.get("total") or 225,
                    "extracted_pdf_count": pdf_audit.get("ok_count") or 222,
                    "failed_pdf_count": pdf_audit.get("needs_ocr_count") or 3,
                    "pdf_audit": pdf_audit,
                    "known_matrix_count": benchmark_options().get("matrix_count"),
                    "metrics_status": "Paused, no query ground-truth CSV requested yet",
                    "openai_status": "Pending OPENAI_API_KEY and cost approval",
                    "amazon_status": "Pending AWS/Bedrock credentials",
                },
                "files": [f for f in files if (ROOT / f).exists()],
                "options": benchmark_options(),
            })
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload-dataset":
            try:
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
                file_item = form["file"] if "file" in form else None
                if file_item is None or not getattr(file_item, "filename", ""):
                    self.send_json({"error": "No file uploaded"}, 400)
                    return
                original = Path(file_item.filename).name
                suffix = Path(original).suffix.lower()
                if suffix not in {".pdf", ".zip", ".csv", ".xlsx"}:
                    self.send_json({"error": "Supported uploads: .pdf, .zip, .csv, .xlsx"}, 400)
                    return
                label_field = form.getfirst("label", Path(original).stem)
                label = safe_label(str(label_field))
                dataset_dir = ROOT / "data" / "uploads" / label
                dataset_dir.mkdir(parents=True, exist_ok=True)
                saved_path = dataset_dir / original
                with saved_path.open("wb") as out:
                    shutil.copyfileobj(file_item.file, out)
                extracted = safe_extract_zip(saved_path, dataset_dir) if suffix == ".zip" else []
                self.send_json({
                    "ok": True,
                    "dataset": label,
                    "saved_path": str(saved_path.relative_to(ROOT)),
                    "bytes": saved_path.stat().st_size,
                    "extracted_files": extracted[:200],
                    "extracted_count": len(extracted),
                    "next_steps": upload_next_steps(saved_path, dataset_dir, extracted),
                })
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        qs = parse_qs(parsed.query)
        limit = qs.get("limit", ["50"])[0]
        max_runs = qs.get("max_runs", ["0"])[0]
        try:
            if parsed.path == "/api/run/matrix":
                cmd = [sys.executable, "scripts/run_full_benchmark_matrix.py", "--limit", limit, "--max-runs", max_runs]
                if qs.get("include_candidates", ["0"])[0] == "1":
                    cmd.append("--include-candidates")
                if qs.get("include_milvus", ["0"])[0] == "1":
                    cmd.append("--include-milvus")
            elif parsed.path == "/api/run/chunking":
                cmd = [sys.executable, "scripts/compare_chunking_recall.py", "--limit", limit]
            elif parsed.path in {"/api/run/modular", "/api/run/selected"}:
                cmd = [sys.executable, "scripts/benchmark_cli.py", "run", "--limit-queries", limit, "--max-runs", max_runs, "--output-dir", "data/modular_runs/latest"] + selected_cli_args(qs)
            elif parsed.path == "/api/run/nvidia-health":
                cmd = [sys.executable, "scripts/check_nvidia_rag_pipeline.py", "--out", "data/nvidia_rag/health.json"]
            elif parsed.path == "/api/run/nvidia-smoke":
                cmd = [sys.executable, "scripts/run_nvidia_rag_pipeline_smoke.py", "--mode", qs.get("mode", ["search"])[0], "--collection", qs.get("collection", ["multimodal_data"])[0], "--top-k", qs.get("top_k", ["10"])[0], "--reranker-top-k", qs.get("reranker_top_k", ["5"])[0]]
                queries = [q.strip() for raw in qs.get("query", []) for q in raw.split("\n") if q.strip()]
                if queries:
                    cmd += ["--query", queries[0]]
                if qs.get("disable_reranker", ["0"])[0] == "1":
                    cmd.append("--disable-reranker")
                if qs.get("agentic", ["0"])[0] == "1":
                    cmd.append("--agentic")
            elif parsed.path == "/api/run/nvidia-ingest":
                cmd = [sys.executable, "scripts/ingest_nvidia_rag_documents.py", "--collection", qs.get("collection", ["multimodal_data"])[0], "--limit", limit or "5", "--batch-size", qs.get("batch_size", ["2"])[0]]
                if qs.get("create_collection", ["0"])[0] == "1":
                    cmd.append("--create-collection")
                if qs.get("poll", ["0"])[0] == "1":
                    cmd.append("--poll")
            elif parsed.path == "/api/run/parse-pdfs-mineru":
                cmd = [sys.executable, "scripts/prepare_benchmark_input_mineru.py"]
                if qs.get("include_image_markers", ["0"])[0] == "1":
                    cmd.append("--include-image-markers")
            elif parsed.path == "/api/run/ingest-selected":
                cmd = [sys.executable, "scripts/run_long_db_ingestion.py", "--skip-existing-store-success"]
                add_multi(cmd, "--sheets", qs.get("sheet", []))
                add_multi(cmd, "--embeddings", qs.get("embedding", []))
                add_multi(cmd, "--stores", qs.get("store", []))
                if limit and limit != "0":
                    cmd += ["--limit", limit]
            elif parsed.path == "/api/run/retrieval-smoke":
                cmd = [sys.executable, "scripts/run_retrieval_smoke_from_vm_dbs.py", "--top-k", qs.get("top_k", ["5"])[0]]
                add_multi(cmd, "--sheets", qs.get("sheet", []))
                add_multi(cmd, "--embeddings", qs.get("embedding", []))
                add_multi(cmd, "--stores", qs.get("store", []))
                queries = [q.strip() for raw in qs.get("query", []) for q in raw.split("\n") if q.strip()]
                if queries:
                    cmd.append("--queries")
                    cmd.extend(queries)
                elif qs.get("groundtruth", [""])[0]:
                    cmd += ["--queries-file", qs.get("groundtruth", [""])[0]]
                if max_runs and max_runs != "0":
                    cmd += ["--max-combos", max_runs]
            elif parsed.path == "/api/run/full-gt-retrieval":
                gt = qs.get("groundtruth", [""])[0] or os.getenv("WNS_GROUNDTRUTH_PATH", "")
                if not gt:
                    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
                    gt = str(candidates[-1]) if candidates else ""
                if not gt:
                    self.send_json({"error": "No groundtruth file found. Put CSV/XLSX under data/groundtruth/."}, 400)
                    return
                cmd = [sys.executable, "scripts/run_retrieval_smoke_from_vm_dbs.py", "--queries-file", gt, "--top-k", qs.get("top_k", ["10"])[0]]
                add_multi(cmd, "--sheets", benchmark_options().get("chunkers", []))
                add_multi(cmd, "--embeddings", [e for e in benchmark_options().get("embeddings", []) if e in {"gte_multilingual_base", "jina_v3"}])
                add_multi(cmd, "--stores", benchmark_options().get("vector_stores", []))
                cmd += ["--max-combos", qs.get("max_runs", ["36"])[0] or "36"]
            elif parsed.path == "/api/run/reranker-smoke":
                cmd = [sys.executable, "scripts/run_reranker_smoke_from_retrieval.py", "--top-k", qs.get("top_k", ["10"])[0]]
                rerankers = qs.get("reranker", []) or benchmark_options().get("rerankers", [])
                rerankers = [r for r in rerankers if r not in {"Amazon Rerank v1", "amazon_bedrock"}]
                add_multi(cmd, "--rerankers", rerankers or ["bge-reranker-base", "qwen3_4b_rerank"])
                if limit and limit != "0":
                    cmd += ["--limit-artifacts", limit]
            elif parsed.path == "/api/run/evaluate-reranked-groundtruth":
                gt = qs.get("groundtruth", [""])[0] or os.getenv("WNS_GROUNDTRUTH_PATH", "")
                if not gt:
                    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
                    gt = str(candidates[-1]) if candidates else ""
                if not gt:
                    self.send_json({"error": "No groundtruth file found. Put CSV/XLSX under data/groundtruth/."}, 400)
                    return
                cmd = [sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt, "--smoke-dir", "data/reranker_smoke", "--out-dir", "data/evaluation_reranked"]
            elif parsed.path == "/api/run/evaluate-groundtruth":
                gt = qs.get("groundtruth", [""])[0] or os.getenv("WNS_GROUNDTRUTH_PATH", "")
                if not gt:
                    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
                    gt = str(candidates[-1]) if candidates else ""
                if not gt:
                    self.send_json({"error": "No groundtruth file found. Put CSV/XLSX under data/groundtruth/ or pass ?groundtruth=/path/file.csv"}, 400)
                    return
                cmd = [sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt]
            elif parsed.path == "/api/run/evaluate-hallucination":
                gt = qs.get("groundtruth", [""])[0] or os.getenv("WNS_GROUNDTRUTH_PATH", "")
                if not gt:
                    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
                    gt = str(candidates[-1]) if candidates else ""
                if not gt:
                    self.send_json({"error": "No groundtruth file found. Put CSV/XLSX under data/groundtruth/ or pass ?groundtruth=/path/file.csv"}, 400)
                    return
                artifact_dir = "data/reranker_smoke" if RERANKER_DIR.exists() else "data/retrieval_smoke"
                cmd = [sys.executable, "scripts/evaluate_hallucination_grounding.py", "--groundtruth", gt, "--artifact-dir", artifact_dir, "--limit", qs.get("limit", ["120"])[0] or "120"]
                if qs.get("use_llm", ["0"])[0] == "1":
                    cmd.append("--use-llm")
            else:
                self.send_json({"error": "unknown endpoint"}, 404)
                return
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=3600)
            self.send_json({"exit_code": proc.returncode, "output": proc.stdout + proc.stderr})
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    server = ThreadingHTTPServer((host, port), Handler)
    shown_host = socket.gethostbyname(socket.gethostname()) if host == "0.0.0.0" else host
    print(f"WNS benchmark dashboard: http://{shown_host}:{port} (bind={host})")
    server.serve_forever()


if __name__ == "__main__":
    main()
