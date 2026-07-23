#!/usr/bin/env python3
"""Serve the WNS benchmark dashboard using only the Python standard library."""

from __future__ import annotations

import asyncio
import csv
import cgi
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import uuid
import zipfile
from itertools import product
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.core.config import generate_matrix, load_benchmark_config
from source.benchmark_pipeline import load_chunks_from_workbook
from source.services.document_parser import DocumentParserService
from source.services.project_run_results import ProjectRunResultService, ProjectRunResultsError
from source.services.project_workspace import ProjectWorkspace
from scripts.dashboard_source_catalog import build_source_catalog
from scripts.wns_env import load_env_files, service_base_from_endpoint

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FULL_DIR = ROOT / "data" / "full_benchmark"
MODULAR_RUNS_DIR = ROOT / "data" / "modular_runs"
MODULAR_DIR = MODULAR_RUNS_DIR / "latest"
INGESTION_DIR = ROOT / "data" / "db_ingestion_runs"
RERANKER_DIR = ROOT / "data" / "reranker_smoke"
RETRIEVAL_DIR = ROOT / "data" / "retrieval_smoke"
SNAPSHOT_PATH = ROOT / "data" / "vm_dashboard_snapshot.json"
EVAL_DIR = ROOT / "data" / "evaluation"
HALLUCINATION_DIR = ROOT / "data" / "hallucination"
NVIDIA_RAG_DIR = ROOT / "data" / "nvidia_rag"
GROUNDTRUTH_DIR = ROOT / "data" / "groundtruth"
PDF_DIR = ROOT / "data" / "pdfs"
USER_PROJECTS_DIR = ROOT / "data" / "user_projects"
PDF_AUDIT_PATH = ROOT / "data" / "pdf_extraction_audit.csv"
CONFIG_PATH = ROOT / "configs" / "benchmark.local.json"
JOB_DIR = ROOT / "data" / "dashboard_jobs"
JOBS: dict[str, dict[str, Any]] = {}
load_env_files(ROOT)
CHUNK_LOOKUP_CACHE: dict[str, dict[int, Any]] = {}


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
    if limit <= 0 or not RERANKER_DIR.exists():
        return smokes
    paths = [p for p in RERANKER_DIR.glob("*.json") if p.name != "summary.json"]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"error": str(exc)}
        payload["artifact"] = str(path.relative_to(ROOT))
        payload["hits"] = (payload.get("hits") or [])[:10]
        smokes.append(payload)
    smokes.sort(key=lambda r: (r.get("created_at", ""), r.get("rerank_seconds", 999999)), reverse=True)
    return smokes


def read_retrieval_smokes(limit: int = 250) -> list[dict[str, Any]]:
    smokes: list[dict[str, Any]] = []
    if limit <= 0 or not RETRIEVAL_DIR.exists():
        return smokes
    paths = [p for p in RETRIEVAL_DIR.glob("*.json") if p.name != "summary.json"]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {"error": str(exc)}
        payload["artifact"] = str(path.relative_to(ROOT))
        payload["hits"] = (payload.get("hits") or [])[:10]
        smokes.append(payload)
    smokes.sort(key=lambda r: (r.get("created_at", ""), r.get("retrieval_seconds", 999999)), reverse=True)
    return smokes


def smoke_summary_count(path: Path) -> int | None:
    summary = path / "summary.json"
    if not summary.exists():
        return None
    try:
        value = json.loads(summary.read_text(encoding="utf-8")).get("result_count")
        return int(value) if value is not None else None
    except Exception:
        return None


def smoke_count(path: Path) -> int:
    if not path.exists():
        return 0
    summary_count = smoke_summary_count(path)
    if summary_count is not None:
        return summary_count
    return sum(1 for p in path.glob("*.json") if p.name != "summary.json")


def retrieval_smoke_count() -> int:
    return smoke_count(RETRIEVAL_DIR)


def reranker_smoke_count() -> int:
    return smoke_count(RERANKER_DIR)


def read_snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        return {}
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def service_check(name: str, url: str, timeout: float = 2.0) -> dict[str, Any]:
    started = datetime.now()
    try:
        with urlopen(url, timeout=timeout) as resp:
            body = resp.read(200).decode("utf-8", "ignore")
            latency_ms = (datetime.now() - started).total_seconds() * 1000
            return {"name": name, "url": url, "ok": 200 <= resp.status < 300, "error": "", "latency_ms": latency_ms, "note": f"HTTP {resp.status}", "body": body}
    except Exception as exc:
        latency_ms = (datetime.now() - started).total_seconds() * 1000
        return {"name": name, "url": url, "ok": False, "error": str(exc), "latency_ms": latency_ms, "note": str(exc)}


def pgvector_service_check() -> dict[str, Any]:
    started = datetime.now()
    dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        return {"name": "pgvector", "url": "PGVECTOR_DSN/DATABASE_URL", "ok": False, "error": "PGVECTOR_DSN or DATABASE_URL is not set", "latency_ms": 0, "note": "missing DSN"}
    try:
        import psycopg  # type: ignore[import-not-found]
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1").fetchone()
        latency_ms = (datetime.now() - started).total_seconds() * 1000
        return {"name": "pgvector", "url": "PGVECTOR_DSN/DATABASE_URL", "ok": True, "error": "", "latency_ms": latency_ms, "note": "SELECT 1 OK"}
    except Exception as exc:
        latency_ms = (datetime.now() - started).total_seconds() * 1000
        return {"name": "pgvector", "url": "PGVECTOR_DSN/DATABASE_URL", "ok": False, "error": str(exc), "latency_ms": latency_ms, "note": str(exc)}


def live_service_health(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    model_base = service_base_from_endpoint(os.getenv("MODEL_ADAPTER_URL", "http://127.0.0.1:5000"))
    checks = [
        ("model_adapter", model_base + "/health"),
        ("qdrant", os.getenv("QDRANT_URL", "http://127.0.0.1:5019").rstrip("/") + "/healthz"),
        ("weaviate", os.getenv("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/") + "/v1/.well-known/ready"),
    ]
    # Normalize endpoint URLs accidentally stored in env to the adapter base.
    out = []
    for name, url in checks:
        if name == "model_adapter":
            if url.endswith("/health/health"):
                url = url[: -len("/health/health")] + "/health"
            for suffix in ("/embed/gte/health", "/embed/jina/health", "/rerank/bge/health", "/rerank/qwen/health"):
                if url.endswith(suffix):
                    url = url[: -len(suffix)] + "/health"
            for suffix in ("/embed/gte", "/embed/jina", "/rerank/bge", "/rerank/qwen"):
                if url.endswith(suffix + "/health"):
                    url = url[: -len(suffix + "/health")] + "/health"
        out.append(service_check(name, url))
    out.append(pgvector_service_check())
    return out


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
    benchmark_report = read_json_file(NVIDIA_RAG_DIR / "benchmark_report.json")
    benchmark_summary = sort_summary(read_csv(NVIDIA_RAG_DIR / "benchmark_summary.csv"))
    benchmark_details = read_csv(NVIDIA_RAG_DIR / "benchmark_details.csv") if NVIDIA_RAG_DIR.exists() else []
    baseline_report = read_json_file(NVIDIA_RAG_DIR / "benchmark_baseline_report.json")
    reranked_report = read_json_file(NVIDIA_RAG_DIR / "benchmark_reranked_report.json")
    baseline_summary = sort_summary(read_csv(NVIDIA_RAG_DIR / "benchmark_baseline_summary.csv"))
    reranked_summary = sort_summary(read_csv(NVIDIA_RAG_DIR / "benchmark_reranked_summary.csv"))
    files = []
    if NVIDIA_RAG_DIR.exists():
        files = [str(p.relative_to(ROOT)) for p in sorted(NVIDIA_RAG_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    return {
        "health": health,
        "smoke": smoke,
        "ingestion": ingestion,
        "benchmark": {
            "report": benchmark_report,
            "summary": benchmark_summary,
            "details": benchmark_details[:250],
            "baseline": {"report": baseline_report, "summary": baseline_summary},
            "reranked": {"report": reranked_report, "summary": reranked_summary},
        },
        "files": files,
        "configured": bool(health or smoke or ingestion or benchmark_report or baseline_report or reranked_report or os.getenv("NVIDIA_RAG_SERVER_URL") or os.getenv("NVIDIA_INGESTOR_URL")),
        "note": "NVIDIA RAG metrics are service evidence only. They are separate from the 180-combination Project Smiley benchmark matrix.",
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
    base["benchmark_reference"] = read_benchmark_reference()
    gt_files = sorted([str(p.relative_to(ROOT)) for p in GROUNDTRUTH_DIR.glob("*")]) if GROUNDTRUTH_DIR.exists() else []
    base["groundtruth_files"] = gt_files
    return base


def canonical_reranker_name(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "none"
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if normalized in {"none", "no_reranker", "baseline"}:
        return "none"
    if normalized in {"qwen", "qwen3", "qwen3_4b", "qwen3_4b_rerank", "qwen3_4b_reranker", "qwen3_reranker_4b_seq_cls"}:
        return "Qwen3:4B Rerank"
    if normalized in {"bge", "bge_reranker", "bge_reranker_base"}:
        return "bge-reranker-base"
    if normalized in {"amazon", "amazon_rerank", "amazon_rerank_v1", "amazon_rerank_v1_0", "amazon_bedrock_rerank", "amazon_reranker"}:
        return "Amazon Rerank v1"
    if "amazon" in normalized and "rerank" in normalized:
        return "Amazon Rerank v1"
    if "qwen" in normalized and "rerank" in normalized:
        return "Qwen3:4B Rerank"
    if "bge" in normalized and "rerank" in normalized:
        return "bge-reranker-base"
    return raw


def normalized_benchmark_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    latency_s = row.get("avg_latency_seconds")
    if latency_s in (None, ""):
        latency_ms = row.get("avg_latency_ms") or row.get("avg_query_latency_ms") or row.get("p50_query_latency_ms")
        try:
            latency_s = float(latency_ms) / 1000.0
        except (TypeError, ValueError):
            latency_s = ""
    embedding = row.get("embedding") or row.get("embedding_model") or ""
    reranker = canonical_reranker_name(row.get("reranker") or row.get("reranking_model") or "none")
    return {
        "source": source,
        "sheet": row.get("chunker") or row.get("chunking_method") or row.get("sheet") or "",
        "embedding": embedding,
        "store": row.get("vector_store") or row.get("vector_database") or row.get("store") or "",
        "reranker": reranker,
        "evaluated_queries": row.get("query_count") if row.get("query_count") not in (None, "") else (row.get("evaluated_queries") if row.get("evaluated_queries") not in (None, "") else ""),
        "recall_at_1": row.get("recall_at_1") or "",
        "recall_at_3": row.get("recall_at_3") or "",
        "recall_at_5": row.get("recall_at_5") or "",
        "recall_at_10": row.get("recall_at_10") or "",
        "mrr": row.get("mrr") or "",
        "precision_at_5": row.get("precision_at_5") or "",
        "ndcg_at_5": row.get("ndcg_at_5") or "",
        "avg_latency_seconds": latency_s,
        "cost": "commercial" if re.search(r"openai|amazon", f"{embedding} {reranker}", re.I) else "oss",
    }


def benchmark_reference_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    modular_runs_dir = MODULAR_DIR.parent
    if modular_runs_dir.exists():
        def source_sort_key(path: Path) -> tuple[bool, bool, int, str]:
            rel = path.relative_to(modular_runs_dir)
            parts = rel.parts
            return (path.parent.name != "latest", "archive" in parts, len(parts), str(rel))
        modular_paths = sorted(
            modular_runs_dir.rglob("modular_summary.csv"),
            key=source_sort_key,
        )
        for path in modular_paths:
            run_path = path.parent.relative_to(modular_runs_dir)
            run_id = path.parent.name
            if run_id != "latest" and "smoke" in str(run_path).lower():
                continue
            source = "modular_matrix" if run_id == "latest" else f"modular_run:{run_path}"
            sources.append((source, path))
    elif (MODULAR_DIR / "modular_summary.csv").exists():
        sources.append(("modular_matrix", MODULAR_DIR / "modular_summary.csv"))
    sources.append(("legacy_full_matrix", FULL_DIR / "benchmark_summary.csv"))
    return sources


def official_metric_sources() -> list[tuple[str, Path]]:
    sources = [
        ("groundtruth_evaluation", EVAL_DIR / "groundtruth_eval_summary.csv"),
        ("groundtruth_evaluation_reranked", EVAL_DIR.parent / "evaluation_reranked" / "groundtruth_eval_summary.csv"),
        *benchmark_reference_sources(),
    ]
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for source, path in sources:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((source, path))
    return unique


def official_matrix_keys() -> set[tuple[str, str, str, str]]:
    cfg = load_benchmark_config(CONFIG_PATH)
    return {
        (row["chunker"], row["embedding"], row["vector_store"], canonical_reranker_name(row["reranker"]))
        for row in generate_matrix(cfg)
    }


def pipeline_key(parts: tuple[str, str, str, str]) -> str:
    return "|".join(parts)


def non_negative_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def pipeline_number(value: Any) -> int | float | str:
    number = non_negative_number(value)
    if number is None:
        return ""
    return int(number) if number.is_integer() else number


def candidate_run_at(row: dict[str, Any], path: Path) -> str:
    for field in ("run_at", "created_at", "completed_at", "timestamp"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def candidate_order_key(row: dict[str, Any], path: Path) -> int:
    for field in ("run_at", "created_at", "completed_at", "timestamp"):
        value = str(row.get(field) or "").strip()
        if not value:
            continue
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000)
        except ValueError:
            continue
    return path.stat().st_mtime_ns


def complete_pipeline_metric_row(row: dict[str, Any]) -> bool:
    return all(
        non_negative_number(row.get(field)) is not None
        for field in ("recall_at_5", "mrr", "ndcg_at_5", "avg_latency_seconds", "evaluated_queries")
    )


def pipeline_state_snapshot() -> tuple[list[dict[str, Any]], int]:
    official_keys = official_matrix_keys()
    skipped_non_official = 0
    candidates: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for source, path in official_metric_sources():
        for row in read_csv(path):
            normalized = normalized_benchmark_row(row, source)
            key = (normalized["sheet"], normalized["embedding"], normalized["store"], normalized["reranker"])
            if key not in official_keys:
                skipped_non_official += 1
                continue
            candidates.setdefault(key, []).append({
                **normalized,
                "artifact": display_path(path),
                "run_at": candidate_run_at(row, path),
                "_order_key": candidate_order_key(row, path),
            })

    state_rows: list[dict[str, Any]] = []
    for key in sorted(official_keys):
        sheet, embedding, store, reranker = key
        rows = candidates.get(key, [])
        base = {
            "key": pipeline_key(key),
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "reranker": reranker,
            "artifact": "",
            "source": "",
            "run_at": "",
            "evaluated_queries": "",
            "candidate_count": len(rows),
        }
        if not rows:
            state_rows.append({**base, "state": "not_run"})
            continue
        complete_rows = [row for row in rows if complete_pipeline_metric_row(row)]
        selected = max(complete_rows or rows, key=lambda row: (row["_order_key"], row["artifact"], row["source"]))
        selected_is_complete = bool(complete_rows)
        state = {
            **base,
            "state": "complete" if selected_is_complete else "incomplete",
            "artifact": selected["artifact"],
            "source": selected["source"],
            "run_at": selected["run_at"],
            "evaluated_queries": pipeline_number(selected["evaluated_queries"]),
        }
        if selected_is_complete:
            state.update({
                "recall_at_5": pipeline_number(selected["recall_at_5"]),
                "mrr": pipeline_number(selected["mrr"]),
                "ndcg_at_5": pipeline_number(selected["ndcg_at_5"]),
                "avg_latency_seconds": pipeline_number(selected["avg_latency_seconds"]),
            })
        state_rows.append(state)
    return state_rows, skipped_non_official


def pipeline_state_rows() -> list[dict[str, Any]]:
    return pipeline_state_snapshot()[0]


def complete_pipeline_metric_rows(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [
        row for row in (rows if rows is not None else pipeline_state_rows())
        if row.get("state", "complete") == "complete" and complete_pipeline_metric_row(row)
    ]


def read_benchmark_reference() -> dict[str, Any]:
    state_rows, skipped_non_official = pipeline_state_snapshot()
    rows = complete_pipeline_metric_rows(state_rows)
    official_keys = official_matrix_keys()
    for row in rows:
        row["cost"] = "commercial" if re.search(r"openai|amazon", f"{row['embedding']} {row['reranker']}", re.I) else "oss"
    return {
        "summary": sort_summary(rows),
        "report": {
            "source": "modular/full benchmark artifacts",
            "config_rows": len(rows),
            "official_matrix_rows": len(official_keys),
            "skipped_non_official_rows": skipped_non_official,
            "openai_rows": sum(1 for r in rows if "openai" in str(r.get("embedding", "")).lower()),
            "faiss_rows": sum(1 for r in rows if str(r.get("store", "")).lower() == "faiss"),
            "query_count": next((r.get("evaluated_queries") for r in rows if r.get("evaluated_queries")), ""),
            "complete_rows": len(rows),
            "incomplete_rows": sum(1 for row in state_rows if row["state"] == "incomplete"),
            "not_run_rows": sum(1 for row in state_rows if row["state"] == "not_run"),
        },
    }


def benchmark_detail_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    modular_runs_dir = MODULAR_DIR.parent
    if not modular_runs_dir.exists():
        return sources
    def source_sort_key(path: Path) -> tuple[bool, bool, int, str]:
        rel = path.relative_to(modular_runs_dir)
        parts = rel.parts
        return (path.parent.name != "latest", "archive" in parts, len(parts), str(rel))
    for path in sorted(modular_runs_dir.rglob("modular_details.csv"), key=source_sort_key):
        run_path = path.parent.relative_to(modular_runs_dir)
        run_id = path.parent.name
        if run_id != "latest" and "smoke" in str(run_path).lower():
            continue
        source = "modular_details" if run_id == "latest" else f"modular_details:{run_path}"
        sources.append((source, path))
    return sources


def chunk_lookup_for_sheet(sheet: str) -> dict[int, Any]:
    if sheet in CHUNK_LOOKUP_CACHE:
        return CHUNK_LOOKUP_CACHE[sheet]
    try:
        cfg = load_benchmark_config(CONFIG_PATH)
        workbook = cfg.get("experiment", {}).get("corpus_workbook", "data/chunking_methods_output_v2.xlsx")
        chunks = load_chunks_from_workbook(ROOT / workbook, sheet)
        CHUNK_LOOKUP_CACHE[sheet] = {int(c.id): c for c in chunks}
    except Exception:
        CHUNK_LOOKUP_CACHE[sheet] = {}
    return CHUNK_LOOKUP_CACHE[sheet]


def benchmark_detail_evidence(limit_per_combo: int = 3, max_rows: int = 2000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_combo: dict[tuple[str, str, str, str], int] = {}
    official_keys = official_matrix_keys()
    for source, path in benchmark_detail_sources():
        for row in read_csv(path):
            normalized = normalized_benchmark_row(row, source)
            key = (normalized["sheet"], normalized["embedding"], normalized["store"], normalized["reranker"])
            if key not in official_keys or per_combo.get(key, 0) >= limit_per_combo:
                continue
            top_ids = [v for v in str(row.get("top_ids") or "").split("|") if v]
            if not top_ids:
                continue
            top_scores = str(row.get("top_scores") or "").split("|")
            chunks = chunk_lookup_for_sheet(normalized["sheet"])
            hits = []
            for i, chunk_id_raw in enumerate(top_ids[:5], 1):
                try:
                    chunk_id = int(chunk_id_raw)
                except ValueError:
                    continue
                chunk = chunks.get(chunk_id)
                if not chunk:
                    continue
                hits.append({
                    "rank": i,
                    "chunk_id": chunk_id,
                    "pdf_name": getattr(chunk, "pdf_name", ""),
                    "paragraph": getattr(chunk, "paragraph", ""),
                    "score": top_scores[i - 1] if i - 1 < len(top_scores) else "",
                    "source_type": "modular_details",
                })
            if not hits:
                continue
            per_combo[key] = per_combo.get(key, 0) + 1
            try:
                artifact = str(path.relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({
                **normalized,
                "query": row.get("query") or "",
                "query_id": row.get("query_id") or "",
                "category": row.get("category") or "",
                "created_at": row.get("created_at") or "",
                "artifact": artifact,
                "hits": hits,
                "retrieved_count": len(hits),
                "evidence_type": "benchmark final top5 evidence",
            })
            if len(rows) >= max_rows:
                return rows
    return rows


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
    if not PDF_AUDIT_PATH.exists():
        return {
            "status": "audit_missing",
            "missing": True,
            "rows": [],
            "total": 0,
            "ok_count": 0,
            "needs_ocr_count": 0,
            "needs_ocr": [],
            "message": "data/pdf_extraction_audit.csv missing. Live benchmark pipeline blocks until MinerU/layout extraction audit exists.",
        }
    rows = read_csv(PDF_AUDIT_PATH)
    needs = [r for r in rows if str(r.get("needs_ocr_review", "")).lower() in {"1", "true", "yes"} or r.get("status") in {"needs_ocr", "partial_ocr_review", "failed", "text_only_review"} or (r.get("parser_method") or "") == "PyPDF2_fallback"]
    ok = [r for r in rows if r not in needs]
    return {
        "status": "ok" if rows and not needs else "review_required",
        "missing": False,
        "rows": rows[:500],
        "total": len(rows),
        "ok_count": len(ok),
        "needs_ocr_count": len(needs),
        "needs_ocr": needs[:100],
        "message": "Audit clean" if rows and not needs else "Extraction audit has review-required rows",
    }



def read_pdf_chunk_counts() -> dict[str, int]:
    csv_path = ROOT / "data" / "benchmark_input.csv"
    if csv_path.exists():
        rows = read_csv(csv_path)
        counts: dict[str, int] = {}
        for row in rows:
            name = row.get("pdf_name")
            if name:
                counts[name] = counts.get(name, 0) + 1
        if counts:
            return counts
    workbook = ROOT / "data" / "chunking_methods_output_v2.xlsx"
    if not workbook.exists():
        return {}
    try:
        import zipfile as _zipfile
        import xml.etree.ElementTree as ET
        counts: dict[str, int] = {}
        with _zipfile.ZipFile(workbook) as z:
            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
                ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in tree.findall("a:si", ns):
                    shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
            workbook_xml = ET.fromstring(z.read("xl/workbook.xml"))
            rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
            relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
            target = None
            for sheet in workbook_xml.findall("a:sheets/a:sheet", ns):
                if sheet.attrib.get("name") == "original_input":
                    target = relmap.get(sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"))
                    break
            if not target:
                return {}
            xml_path = "xl/" + target.lstrip("/")
            ws = ET.fromstring(z.read(xml_path))
            rows = ws.findall(".//a:sheetData/a:row", ns)
            headers = []
            for idx, row in enumerate(rows):
                vals = []
                for c in row.findall("a:c", ns):
                    v = c.find("a:v", ns)
                    val = "" if v is None else v.text or ""
                    if c.attrib.get("t") == "s" and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    vals.append(val)
                if idx == 0:
                    headers = vals
                    continue
                rec = dict(zip(headers, vals))
                name = rec.get("pdf_name")
                if name:
                    counts[name] = counts.get(name, 0) + 1
            return counts
    except Exception:
        return {}


def count_pdf_chunks(pdf_name: str) -> int:
    return read_pdf_chunk_counts().get(pdf_name, 0)


def pdf_path_for_name(name: str) -> Path | None:
    clean = str(name or "").strip()
    if not clean or "/" in clean or "\\" in clean:
        return None
    if Path(clean).suffix.lower() != ".pdf":
        return None
    try:
        base = PDF_DIR.resolve()
        target = (PDF_DIR / clean).resolve()
    except Exception:
        return None
    if target.parent != base or not target.exists() or not target.is_file():
        return None
    return target


def read_document_repository() -> dict[str, Any]:
    pdf_dir = PDF_DIR
    audit_rows = read_csv(PDF_AUDIT_PATH)
    audit_missing = not PDF_AUDIT_PATH.exists()
    audit_by_name = {r.get("pdf_name") or r.get("file") or r.get("filename"): r for r in audit_rows}
    chunk_counts = read_pdf_chunk_counts()
    files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    rows = []
    for f in files:
        audit = audit_by_name.get(f.name, {})
        chunks = chunk_counts.get(f.name, 0)
        status = audit.get("status") or ("audit_missing" if audit_missing else ("chunked" if chunks else "present"))
        needs_review = audit_missing or str(audit.get("needs_ocr_review", "")).lower() in {"1", "true", "yes"} or status in {"needs_ocr", "partial_ocr_review", "failed", "text_only_review"}
        if (audit.get("parser_method") or "") == "PyPDF2_fallback":
            status = "text_only_review"
            needs_review = True
        rows.append({
            "pdf_name": f.name,
            "status": "review" if needs_review else status,
            "repository_path": str(f.relative_to(ROOT)),
            "size_mb": f"{f.stat().st_size / (1024*1024):.2f}",
            "chunked_rows": chunks,
            "parser_method": audit.get("parser_method") or audit.get("parser") or "—",
            "pages": audit.get("total_pages") or audit.get("pages") or "—",
            "text_chars": audit.get("text_chars") or "—",
            "note": "Extraction audit missing" if audit_missing else ("Review in audit" if needs_review else ("Ready for benchmark" if chunks else "Present, chunking pending")),
        })
    uploaded = []
    upload_dir = ROOT / "data" / "uploads"
    if upload_dir.exists():
        for f in sorted(upload_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".pdf", ".csv", ".xlsx"}:
                uploaded.append({"path": str(f.relative_to(ROOT)), "size_mb": f"{f.stat().st_size / (1024*1024):.2f}"})
    return {
        "rows": rows,
        "total": len(rows),
        "ready_count": sum(1 for r in rows if r["chunked_rows"] and r["status"] != "review"),
        "review_count": sum(1 for r in rows if r["status"] == "review"),
        "uploaded": uploaded[:100],
    }

def add_multi(cmd: list[str], flag: str, values: list[str]) -> None:
    vals = [v for v in values if v and v != "all"]
    if vals:
        cmd.append(flag)
        cmd.extend(vals)


def read_reranker_analysis() -> dict:
    path = ROOT / "data" / "reranker_analysis"
    report_path = path / "qwen_vs_none_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report = {"error": str(exc)}
    return {
        "report": report,
        "summary": read_csv(path / "reranker_lift_summary.csv") if path.exists() else [],
        "details": read_csv(path / "reranker_lift_details.csv")[:250] if path.exists() else [],
    }


def complete_pipeline_cmd(qs: dict[str, list[str]], preflight: bool = False) -> list[str]:
    cmd = [sys.executable, "scripts/run_complete_pipeline.py"]
    if preflight:
        cmd.append("--preflight-only")
    add_multi(cmd, "--sheets", qs.get("sheet", []))
    add_multi(cmd, "--embeddings", qs.get("embedding", []))
    add_multi(cmd, "--stores", qs.get("store", []))
    rerankers = qs.get("reranker", [])
    if rerankers:
        add_multi(cmd, "--rerankers", rerankers)
    if qs.get("groundtruth", [""])[0]:
        cmd += ["--groundtruth", qs.get("groundtruth", [""])[0]]
    cmd += ["--top-k", qs.get("top_k", ["10"])[0] or "10"]
    chunk_limit = qs.get("chunk_limit", qs.get("limit", ["0"]))[0] or "0"
    query_limit = qs.get("query_limit", ["0"])[0] or "0"
    cmd += ["--chunk-limit", chunk_limit, "--query-limit", query_limit]
    if qs.get("fresh_run", ["0"])[0] == "1":
        cmd.append("--fresh-run")
    if qs.get("allow_partial_extraction", ["0"])[0] == "1":
        cmd.append("--allow-partial-extraction")
    return cmd


SETUP_STAGE_NAMES = ("extract", "chunk", "embed", "index")


def dashboard_job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def write_dashboard_job(record: dict[str, Any]) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_job_path(record["job_id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def read_dashboard_job(job_id: str) -> dict[str, Any] | None:
    path = dashboard_job_path(job_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def setup_stage(record: dict[str, Any], name: str) -> dict[str, Any]:
    for stage in record.get("stages", []):
        if stage.get("name") == name:
            return stage
    raise ValueError(f"Unknown setup stage: {name}")


def set_setup_stage(record: dict[str, Any], name: str, state: str, output: str = "", error: str = "") -> None:
    stage = setup_stage(record, name)
    now = datetime.now().isoformat()
    if state == "running" and not stage.get("started_at"):
        stage["started_at"] = now
    if state in {"complete", "failed"}:
        stage["started_at"] = stage.get("started_at") or now
        stage["finished_at"] = now
    stage["state"] = state
    if output:
        stage["output"] = output[-4000:]
    if error:
        stage["error"] = error[-4000:]


def persist_setup_manifest(record: dict[str, Any]) -> None:
    if record.get("kind") != "dataset_setup":
        return
    try:
        root = project_root(str(record["project_id"]))
        manifest = read_project_manifest(root)
    except (KeyError, ValueError):
        return
    manifest["setup"] = {
        "state": record.get("state", "queued"),
        "job_id": record["job_id"],
        "stages": record.get("stages", []),
        "artifact_root": record.get("artifact_root", ""),
    }
    write_project_manifest(root, manifest)


def persist_job(record: dict[str, Any]) -> dict[str, Any]:
    write_dashboard_job(record)
    persist_setup_manifest(record)
    return record


def create_project_setup_job(project_id: str, matrix: dict[str, Any]) -> dict[str, Any]:
    root = project_root(project_id)
    manifest = read_project_manifest(root)
    options = benchmark_options()
    normalized: dict[str, list[str]] = {}
    for key in ("chunkers", "embeddings", "vector_stores"):
        supplied = matrix.get(key, []) if isinstance(matrix, dict) else []
        values = [str(value) for value in supplied] if isinstance(supplied, list) else []
        if not values:
            raise ValueError(f"At least one {key} selection is required")
        unknown = [value for value in values if value not in options.get(key, [])]
        if unknown:
            raise ValueError(f"Unsupported {key}: {', '.join(unknown)}")
        normalized[key] = values
    now = datetime.now().isoformat()
    source_ready = (root / "search_index.json").is_file() and (root / "chunks" / "chunking_methods_output_v2.xlsx").is_file()
    stages = []
    for name in SETUP_STAGE_NAMES:
        state = "complete" if source_ready and name in {"extract", "chunk"} else "queued"
        stages.append({
            "name": name,
            "state": state,
            "started_at": now if state == "complete" else "",
            "finished_at": now if state == "complete" else "",
            "output": "existing project artifact verified" if state == "complete" else "",
            "error": "",
        })
    record = {
        "job_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
        "kind": "dataset_setup",
        "state": "queued",
        "created_at": now,
        "started_at": "",
        "finished_at": "",
        "project_id": project_id,
        "matrix": normalized,
        "artifact_root": display_path(root),
        "stages": stages,
        "cmd": [],
        "log_path": "",
    }
    persist_job(record)
    return record


def project_setup_cmd(record: dict[str, Any]) -> list[str]:
    project_id = record["project_id"]
    root = project_root(project_id)
    matrix = record["matrix"]
    return [
        sys.executable, "scripts/run_long_db_ingestion.py",
        "--workbook", str((root / "chunks" / "chunking_methods_output_v2.xlsx").relative_to(ROOT)),
        "--output-dir", str((root / "runs" / record["job_id"] / "ingestion").relative_to(ROOT)),
        "--cache-dir", str((root / "vector_indexes" / "embedding_cache").relative_to(ROOT)),
        "--collection-prefix", f"project_{safe_label(project_id)}",
        "--faiss-index-root", str((root / "vector_indexes" / "faiss").relative_to(ROOT)),
        "--run-id", record["job_id"], "--skip-existing-store-success",
        "--sheets", *matrix["chunkers"],
        "--embeddings", *matrix["embeddings"],
        "--stores", *matrix["vector_stores"],
    ]


def active_setup_stage(record: dict[str, Any]) -> str:
    for name in ("embed", "index", "chunk", "extract"):
        if setup_stage(record, name).get("state") in {"queued", "running"}:
            return name
    return "index"


def runner_failure_stage(output: str) -> str:
    failure_stage = ""
    for line in output.splitlines():
        lower_line = line.lower()
        if "embedding_failed" in lower_line or re.search(r"(?:^|\]\s)error embedding\b", lower_line):
            failure_stage = "embed"
        elif "store_failed" in lower_line or re.search(r"(?:^|\]\s)error store\b", lower_line):
            failure_stage = "index"
    return failure_stage


def has_runner_failure(output: str) -> bool:
    return bool(runner_failure_stage(output))


def refresh_setup_job(record: dict[str, Any], output: str, exit_code: int | None) -> None:
    lower_output = output.lower()
    failure_stage = runner_failure_stage(output)
    if failure_stage:
        set_setup_stage(record, failure_stage, "failed", output=output, error=output[-4000:])
        record["state"] = "failed"
        record["finished_at"] = datetime.now().isoformat()
    elif "embedded model=" in lower_output or "loaded_cached_vectors" in lower_output:
        set_setup_stage(record, "embed", "complete", output=output)
        if setup_stage(record, "index").get("state") == "queued":
            set_setup_stage(record, "index", "running")
    if "ok sheet=" in lower_output and record.get("state") != "failed":
        set_setup_stage(record, "index", "complete", output=output)
    if exit_code is None:
        runner_finished_cleanly = bool(re.search(
            r"(?m)^(?:\[[^\n]+\]\s+)?DONE summary=.+ failures=0\s*$",
            output,
        ))
        raw_stages = record.get("stages", [])
        stages = raw_stages if isinstance(raw_stages, list) else []
        required_stage_names = {"extract", "chunk", "embed", "index"}
        stage_names = [
            str(stage.get("name", ""))
            for stage in stages
            if isinstance(stage, dict)
        ]
        stages_complete = (
            len(stages) == len(required_stage_names)
            and len(stage_names) == len(required_stage_names)
            and set(stage_names) == required_stage_names
            and all(stage.get("state") == "complete" for stage in stages)
        )
        if runner_finished_cleanly and stages_complete and record.get("state") != "failed":
            record["state"] = "complete"
            record["finished_at"] = datetime.now().isoformat()
        return
    if exit_code == 0 and record.get("state") != "failed":
        set_setup_stage(record, "index", "complete", output=output)
        record["state"] = "complete"
    elif exit_code != 0:
        stage_name = active_setup_stage(record)
        set_setup_stage(record, stage_name, "failed", output=output, error=output[-4000:])
        record["state"] = "failed"
    record["finished_at"] = datetime.now().isoformat()


def launch_job(cmd: list[str], record: dict[str, Any] | None = None) -> dict[str, Any]:
    now = datetime.now().isoformat()
    record = record or {
        "job_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
        "kind": "command",
        "state": "queued",
        "created_at": now,
        "started_at": "",
        "finished_at": "",
        "project_id": "",
        "matrix": {},
        "stages": [],
    }
    job_id = str(record["job_id"])
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_DIR / f"{job_id}.log"
    record.update({"cmd": cmd, "log_path": display_path(log_path), "state": "running", "started_at": now})
    if record.get("kind") == "dataset_setup":
        set_setup_stage(record, "embed", "running")
    persist_job(record)
    try:
        log_handle = log_path.open("w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
        log_handle.close()
    except Exception as exc:
        record["state"] = "failed"
        record["finished_at"] = datetime.now().isoformat()
        if record.get("kind") == "dataset_setup":
            set_setup_stage(record, active_setup_stage(record), "failed", error=repr(exc))
        persist_job(record)
        raise
    JOBS[job_id] = {"process": proc, "log_path": str(log_path)}
    return job_status(job_id)


def job_status(job_id: str) -> dict[str, Any]:
    record = read_dashboard_job(job_id)
    job = JOBS.get(job_id, {})
    if not record and not job:
        return {"error": "unknown job_id", "job_id": job_id}
    record = record or {"job_id": job_id, "cmd": [], "started_at": "", "kind": "command", "state": "running", "stages": []}
    proc = job.get("process")
    exit_code = proc.poll() if proc is not None else None
    log_path = Path(job.get("log_path") or (ROOT / str(record.get("log_path") or f"data/dashboard_jobs/{job_id}.log")))
    output = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    if record.get("kind") == "dataset_setup":
        refresh_setup_job(record, output, exit_code)
    elif exit_code is not None and record.get("state") == "running":
        record["state"] = "complete" if exit_code == 0 else "failed"
        record["finished_at"] = datetime.now().isoformat()
    persist_job(record)
    return {
        **record,
        "running": proc is not None and exit_code is None,
        "exit_code": exit_code,
        "output": output[-20000:],
        "log_path": record.get("log_path", display_path(log_path)),
    }


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
    return value.strip(".-_").lower()[:80] or "dataset"


def new_project_id(label: str) -> str:
    return f"{safe_label(label)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def project_root(project_id: str) -> Path:
    clean = safe_label(project_id)
    if clean != project_id:
        raise ValueError("Invalid project_id")
    root = (USER_PROJECTS_DIR / clean).resolve()
    base = USER_PROJECTS_DIR.resolve()
    if base not in root.parents and root != base:
        raise ValueError("Invalid project path")
    return root


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def project_storage_layout(root: Path) -> dict[str, str]:
    return {
        "root": display_path(root),
        "raw_uploads": display_path(root / "raw_uploads"),
        "extracted_text": display_path(root / "extracted_text"),
        "chunks": display_path(root / "chunks"),
        "vector_indexes": display_path(root / "vector_indexes"),
        "questions": display_path(root / "questions"),
        "runs": display_path(root / "runs"),
    }


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
            extracted.append(display_path(dest))
    return extracted


def read_text_upload(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[page {i}]\n{text}")
            return "\n\n".join(pages)
        except Exception as exc:
            return f"[PDF text extraction failed for {path.name}: {exc}]"
    return ""


def read_layout_aware_pdf_upload(path: Path, extracted_text_dir: Path) -> str:
    """Extract uploaded PDFs through MinerU; never silently downgrade to text-only."""
    parser = DocumentParserService(
        output_dir=extracted_text_dir / "mineru_artifacts" / path.stem,
        keep_tmp_files=True,
        force_backend="mineru",
    )
    parsed = asyncio.run(parser.parse_pdf(str(path), path.name))
    pages: list[str] = []
    for page in parsed.content:
        parts = [str(page.content or "").strip()]
        for table in page.tables or []:
            if isinstance(table, dict):
                table_text = table.get("markdown") or table.get("html") or table.get("content") or table.get("text") or ""
            else:
                table_text = str(table)
            if str(table_text).strip():
                parts.append(str(table_text).strip())
        page_text = "\n\n".join(part for part in parts if part)
        if page_text:
            pages.append(f"[page {int(page.page_number) + 1}]\n{page_text}")
    text = "\n\n".join(pages)
    if not text.strip():
        raise RuntimeError(f"MinerU returned no searchable text for {path.name}")
    return text


def split_project_chunks(text: str, source_name: str, parser_method: str = "dashboard_upload_text_index") -> list[dict[str, Any]]:
    cleaned = re.sub(r"\s+", " ", text or " ").strip()
    if not cleaned:
        return []
    raw_parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9])", text) if p and p.strip()]
    if not raw_parts:
        raw_parts = [cleaned]
    chunks: list[dict[str, Any]] = []
    chunk_id = 1
    current_page = ""
    for part in raw_parts:
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        page_match = re.search(r"\[page\s+(\d+)\]", part, re.I)
        if page_match:
            current_page = page_match.group(1)
        # Keep small uploaded demo docs searchable, but avoid pathological giant rows.
        for start in range(0, len(part), 1400):
            paragraph = part[start:start + 1400].strip()
            if not paragraph:
                continue
            chunks.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "pdf_name": source_name,
                "paragraph": paragraph,
                "page_number": current_page,
                "source_type": "uploaded_document",
                "parser_method": parser_method,
            })
            chunk_id += 1
    return chunks


def write_project_workbook(project_dir: Path, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return ""
    out = project_dir / "chunks" / "chunking_methods_output_v2.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "pdf_name", "paragraph", "page_number", "source_type", "parser_method"]
    rows = [{field: c.get(field, "") for field in fields} for c in chunks]
    sheet_names = benchmark_options().get("chunkers", []) or ["uploaded_chunks"]
    try:
        with pd.ExcelWriter(out) as writer:
            for sheet in sheet_names:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet[:31], index=False)
    except Exception:
        return ""
    return display_path(out)


def rebuild_project_search_index(project_dir: Path) -> dict[str, Any]:
    raw_dir = project_dir / "raw_uploads"
    extracted_text_dir = project_dir / "extracted_text"
    extracted_text_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".csv"}]
    chunks: list[dict[str, Any]] = []
    source_files: list[str] = []
    for path in sorted(candidates):
        parser_method = "dashboard_upload_text_index"
        if path.suffix.lower() == ".pdf":
            text = read_layout_aware_pdf_upload(path, extracted_text_dir)
            parser_method = "MinerU"
        else:
            text = read_text_upload(path)
        if not text.strip():
            continue
        rel_name = path.name
        source_files.append(str(path.relative_to(project_dir)))
        text_path = extracted_text_dir / f"{path.stem}.txt"
        text_path.write_text(text, encoding="utf-8")
        for chunk in split_project_chunks(text, rel_name, parser_method):
            chunk["id"] = len(chunks) + 1
            chunk["chunk_id"] = chunk["id"]
            chunk["project_source_path"] = str(path.relative_to(project_dir))
            chunks.append(chunk)
    index_path = project_dir / "search_index.json"
    index = {
        "created_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "source_files": source_files,
        "chunks": chunks,
        "workbook": write_project_workbook(project_dir, chunks),
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


GROUNDTRUTH_QUERY_COLUMNS = {"question", "query"}
GROUNDTRUTH_REFERENCE_COLUMNS = {
    "ground truth", "ground_truth", "context", "paragraph", "expected_text",
    "answer", "relevant_text",
}


def validate_groundtruth_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".csv":
        return {"valid": False, "reason": "Ground-truth files must be CSV.", "columns": []}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            columns = {str(name or "").strip().lower() for name in (csv.DictReader(f).fieldnames or [])}
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"valid": False, "reason": f"Could not read ground-truth CSV: {exc}", "columns": []}
    if not columns & GROUNDTRUTH_QUERY_COLUMNS:
        return {"valid": False, "reason": "Ground-truth file needs a question or query column.", "columns": sorted(columns)}
    if not columns & GROUNDTRUTH_REFERENCE_COLUMNS:
        return {"valid": False, "reason": "Ground-truth file needs a reference text column such as ground truth or context.", "columns": sorted(columns)}
    return {"valid": True, "reason": "", "columns": sorted(columns)}


def read_project_manifest(project_dir: Path) -> dict[str, Any]:
    manifest_path = project_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Project manifest not found")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Project manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Project manifest is invalid")
    return manifest


def write_project_manifest(project_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        **payload,
        "storage_layout": project_storage_layout(project_dir),
        "manifest_path": display_path(project_dir / "manifest.json"),
    }
    (project_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def attach_project_groundtruth(project_id: str, original_name: str, content: bytes) -> dict[str, Any]:
    root = project_root(project_id)
    manifest = read_project_manifest(root)
    if not root.is_dir():
        raise ValueError("Project not found")
    if not isinstance(original_name, str) or not original_name or Path(original_name).name != original_name or "/" in original_name or "\\" in original_name:
        raise ValueError("Ground-truth filename must be a safe basename")
    if Path(original_name).suffix.lower() != ".csv":
        raise ValueError("Ground-truth upload must be a .csv file")
    questions_dir = (root / "questions").resolve()
    questions_dir.mkdir(parents=True, exist_ok=True)
    target = (questions_dir / original_name).resolve()
    if questions_dir not in target.parents:
        raise ValueError("Invalid ground-truth path")
    target.write_bytes(content)
    validation = validate_groundtruth_file(target)
    groundtruth = {
        "path": display_path(target),
        "valid": validation["valid"],
        "reason": validation["reason"],
    }
    manifest["groundtruth"] = groundtruth
    manifest["mode"] = "evaluated" if groundtruth["valid"] else "evidence_only"
    write_project_manifest(root, manifest)
    return {"project_id": project_id, "groundtruth": groundtruth, "mode": manifest["mode"]}


def project_bound_groundtruth(project_id: str) -> tuple[dict[str, Any], Path | None]:
    root = project_root(project_id)
    manifest = read_project_manifest(root)
    groundtruth = manifest.get("groundtruth") if isinstance(manifest.get("groundtruth"), dict) else {}
    stored_path = str(groundtruth.get("path") or "")
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    questions_dir = (root / "questions").resolve()
    if not stored_path or questions_dir not in candidate.parents or not candidate.is_file():
        return manifest, None
    validation = validate_groundtruth_file(candidate)
    if groundtruth.get("valid") is not True or not validation["valid"]:
        return manifest, None
    return manifest, candidate


def project_run_gate(project_id: str) -> dict[str, str] | None:
    manifest, groundtruth_path = project_bound_groundtruth(project_id)
    groundtruth = manifest.get("groundtruth") if isinstance(manifest.get("groundtruth"), dict) else {}
    if groundtruth_path is None:
        return {
            "error": "project_run_blocked",
            "mode": "evidence_only",
            "reason": str(groundtruth.get("reason") or "No valid bound ground truth attached"),
        }
    setup = manifest.get("setup") if isinstance(manifest.get("setup"), dict) else {}
    if setup.get("state") != "complete":
        return {
            "error": "project_run_blocked",
            "mode": "evaluated",
            "reason": "Dataset setup is not complete. Prepare the selected dataset before running evaluation.",
        }
    return None


def project_evaluation_selection(qs: dict[str, list[str]]) -> dict[str, str]:
    """Return the one project-evaluation tuple; project runs never expand "all"."""
    selected: dict[str, str] = {}
    for field in ("sheet", "embedding", "store", "reranker"):
        values = [str(value).strip() for value in qs.get(field, []) if str(value).strip()]
        if len(values) != 1 or values[0] == "all":
            raise ValueError(f"Project evaluation requires exactly one {field} selection")
        selected[field] = values[0]
    if selected["store"] != "FAISS":
        raise ValueError("Project evaluation supports only store=FAISS")
    if selected["reranker"] != "none":
        raise ValueError("Project evaluation supports only reranker=none")
    return selected


def project_evaluation_receipt(root: Path, setup: dict[str, Any], selected: dict[str, str]) -> Path:
    """Resolve the trusted setup receipt, rejecting unmatched project tuples before spawn."""
    job_id = str(setup.get("job_id") or "")
    if not job_id or safe_label(job_id) != job_id:
        raise ValueError("Project setup has no trusted successful receipt")
    receipt = (root / "runs" / job_id / "ingestion" / "summary.csv").resolve()
    if root.resolve() not in receipt.parents or not receipt.is_file():
        raise ValueError("Project setup receipt is missing")
    matches = [
        row for row in read_csv(receipt)
        if row.get("status") == "ok"
        and row.get("sheet") == selected["sheet"]
        and row.get("embedding") == selected["embedding"]
        and row.get("store") == selected["store"]
    ]
    if len(matches) != 1:
        raise ValueError("Selected tuple has no matching successful setup receipt")
    return receipt


def project_evaluation_cmd(project_id: str, qs: dict[str, list[str]], preflight: bool = False) -> list[str]:
    """Build the isolated evaluator command from trusted project state, never request paths."""
    gate = project_run_gate(project_id)
    if gate:
        raise ValueError(gate["reason"])
    root = project_root(project_id)
    manifest, groundtruth = project_bound_groundtruth(project_id)
    if groundtruth is None:
        raise ValueError("Project has no valid bound ground truth")
    raw_setup = manifest.get("setup")
    setup: dict[str, Any] = raw_setup if isinstance(raw_setup, dict) else {}
    selected = project_evaluation_selection(qs)
    receipt = project_evaluation_receipt(root, setup, selected)
    try:
        top_k = int(qs.get("top_k", ["10"])[0] or "10")
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")
    cmd = [sys.executable, "scripts/run_project_evaluation.py"]
    if preflight:
        cmd.append("--preflight-only")
    else:
        cmd += ["--run-id", "eval-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")]
    cmd += [
        "--project-root", str(root),
        "--groundtruth", str(groundtruth),
        "--ingestion-summary", str(receipt),
        "--sheet", selected["sheet"],
        "--embedding", selected["embedding"],
        "--store", selected["store"],
        "--reranker", selected["reranker"],
        "--top-k", str(top_k),
    ]
    return cmd


def create_user_project_upload(original_name: str, content: bytes, label: str = "") -> dict[str, Any]:
    original = Path(original_name).name
    suffix = Path(original).suffix.lower()
    if suffix not in {".pdf", ".zip", ".csv", ".xlsx", ".txt", ".md"}:
        raise ValueError("Supported uploads: .pdf, .zip, .csv, .xlsx, .txt, .md")
    project_label = label or Path(original).stem
    project_id = new_project_id(project_label)
    root = USER_PROJECTS_DIR / project_id
    for name in ["raw_uploads", "extracted_text", "chunks", "vector_indexes", "questions", "runs"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    saved_path = root / "raw_uploads" / original
    saved_path.write_bytes(content)
    extracted = safe_extract_zip(saved_path, root / "raw_uploads") if suffix == ".zip" else []
    question_path = ""
    if suffix in {".csv", ".xlsx", ".txt"}:
        # Keep a copy in questions/ as a candidate query file; text files are also indexed as docs.
        q_target = root / "questions" / original
        q_target.write_bytes(content)
        question_path = display_path(q_target)
    index = rebuild_project_search_index(root)
    manifest = write_project_manifest(root, {
        "ok": True,
        "project_id": project_id,
        "label": project_label,
        "created_at": datetime.now().isoformat(),
        "saved_path": display_path(saved_path),
        "bytes": saved_path.stat().st_size,
        "extracted_files": extracted[:200],
        "extracted_count": len(extracted),
        "question_file": question_path,
        "search_index": display_path(root / "search_index.json"),
        "chunk_count": index.get("chunk_count", 0),
        "workbook": index.get("workbook", ""),
        "groundtruth": {"path": "", "valid": False, "reason": "No ground truth attached"},
        "mode": "evidence_only",
        "setup": {"state": "not_started", "job_id": "", "stages": []},
    })
    return {**manifest, "next_steps": upload_next_steps(saved_path, root, extracted)}


def list_user_projects() -> list[dict[str, Any]]:
    if not USER_PROJECTS_DIR.exists():
        return []
    projects: list[dict[str, Any]] = []
    for manifest_path in sorted(USER_PROJECTS_DIR.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest = {"project_id": manifest_path.parent.name, "error": str(exc)}
        setup = manifest.get("setup") if isinstance(manifest, dict) else None
        stages = setup.get("stages", []) if isinstance(setup, dict) else []
        has_unfinished_stage = any(
            isinstance(stage, dict) and stage.get("state") in {"queued", "running"}
            for stage in stages
        )
        if isinstance(setup, dict) and (setup.get("state") in {"queued", "running"} or has_unfinished_stage) and setup.get("job_id"):
            job_status(str(setup["job_id"]))
            manifest = read_project_manifest(manifest_path.parent)
        index = read_json_file(manifest_path.parent / "search_index.json")
        manifest["chunk_count"] = index.get("chunk_count", manifest.get("chunk_count", 0))
        projects.append(manifest)
    return projects


def selected_list(values: Any, fallback: list[str]) -> list[str]:
    if isinstance(values, str):
        raw = [values]
    elif isinstance(values, list):
        raw = [str(v) for v in values]
    else:
        raw = []
    raw = [v for v in raw if v and v != "all"]
    return raw or fallback


def score_project_hit(query: str, paragraph: str) -> float:
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", query)}
    text_terms = {t.lower() for t in re.findall(r"[A-Za-z0-9]{3,}", paragraph)}
    if not terms or not text_terms:
        return 0.0
    overlap = len(terms & text_terms)
    return overlap / max(1, len(terms)) + min(0.25, overlap / max(1, len(text_terms)))


def query_user_project(project_id: str, query: str, selections: dict[str, Any] | None = None, top_k: int = 5) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query is required")
    root = project_root(project_id)
    manifest = read_project_manifest(root)
    raw_groundtruth = manifest.get("groundtruth")
    bound_groundtruth: dict[str, Any] = raw_groundtruth if isinstance(raw_groundtruth, dict) else {}
    mode = "evaluated" if bound_groundtruth.get("valid") is True else "evidence_only"
    index = read_json_file(root / "search_index.json")
    chunks = index.get("chunks") or []
    if not chunks:
        raise ValueError("Project has no indexed text yet. Upload PDFs/TXT/CSV with extractable text first.")
    scored = sorted(
        [
            {
                "rank": 0,
                "score": round(score_project_hit(query, c.get("paragraph", "")), 6),
                "pdf_name": c.get("pdf_name", ""),
                "chunk_id": c.get("chunk_id", c.get("id", "")),
                "paragraph": c.get("paragraph", ""),
                "page_number": c.get("page_number", ""),
                "source_type": c.get("source_type", "uploaded_document"),
                "parser_method": c.get("parser_method", "dashboard_upload_text_index"),
            }
            for c in chunks
        ],
        key=lambda h: h["score"],
        reverse=True,
    )[: max(1, int(top_k or 5))]
    for i, hit in enumerate(scored, 1):
        hit["rank"] = i
    opts = benchmark_options()
    selections = selections or {}
    chunkers = selected_list(selections.get("chunkers") or selections.get("sheets") or selections.get("sheet"), opts.get("chunkers", []))
    embeddings = selected_list(selections.get("embeddings") or selections.get("embedding"), opts.get("embeddings", []))
    stores = selected_list(selections.get("vector_stores") or selections.get("stores") or selections.get("store"), opts.get("vector_stores", []))
    rerankers = selected_list(selections.get("rerankers") or selections.get("reranker"), opts.get("rerankers", []))
    combos = list(product(chunkers, embeddings, stores, rerankers))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    rows = [
        {
            "project_id": project_id,
            "run_id": run_id,
            "query": query,
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "reranker": canonical_reranker_name(reranker),
            "evidence_type": "uploaded project retrieval evidence",
            "retrieved_count": len(scored),
            "hits": scored,
        }
        for sheet, embedding, store, reranker in combos
    ]
    out_dir = root / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "mode": mode,
        "groundtruth": bound_groundtruth,
        "recommendation_available": False,
        "message": (
            "Ground truth is linked, but no evaluated result artifact exists yet; accuracy scoring and recommendations are unavailable."
            if mode == "evaluated"
            else f"Evidence-only mode: {bound_groundtruth.get('reason') or 'No valid ground truth attached'}, so accuracy scoring is not available."
        ),
        "project_id": project_id,
        "run_id": run_id,
        "query": query,
        "combo_count": len(combos),
        "top_k": top_k,
        "rows": rows,
    }
    (out_dir / "evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({"project_id": project_id, "run_id": run_id, "selections": selections, "mode": mode, "created_at": datetime.now().isoformat()}, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def upload_next_steps(saved_path: Path, dataset_dir: Path, extracted: list[str]) -> list[str]:
    rel_dir = display_path(dataset_dir)
    steps = [
        f"Review uploaded files under {rel_dir}",
        "Frontend project storage is isolated under data/user_projects/<project_id>/, not mixed with Nora/WNS official benchmark data.",
        "Use Run pipeline to choose adapter combinations, or use the uploaded-project query box for evidence-only checks.",
        "If a questions file has ground_truth, run scored mode; without ground_truth the dashboard must stay evidence-only.",
    ]
    if saved_path.suffix.lower() == ".zip":
        steps.insert(1, f"ZIP extracted {len(extracted)} supported files under {rel_dir}/extracted")
    return steps


def official_evaluated_count(_evaluation: dict[str, Any]) -> int:
    """Return complete official rows from the canonical pipeline-state resolver."""
    return len(complete_pipeline_metric_rows())


def project_result_service() -> ProjectRunResultService:
    return ProjectRunResultService(ProjectWorkspace(USER_PROJECTS_DIR))


def exact_query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name, [])
    if len(values) != 1 or not values[0]:
        raise ProjectRunResultsError("invalid_request", "Invalid result request", 400)
    return values[0]


def _bounded_result_integer(
    query: dict[str, list[str]],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = exact_query_value(query, name)
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ProjectRunResultsError("invalid_request", "Invalid result request", 400)
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ProjectRunResultsError("invalid_request", "Invalid result request", 400)
    return value


def _require_result_query_keys(query: dict[str, list[str]], allowed: set[str]) -> None:
    if set(query) != allowed:
        raise ProjectRunResultsError("invalid_request", "Invalid result request", 400)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {
            "/api/result-sources",
            "/api/project-runs",
            "/api/project-run-results",
            "/api/project-run-evidence",
        }:
            try:
                query = parse_qs(parsed.query, keep_blank_values=True)
                service = project_result_service()
                if parsed.path == "/api/result-sources":
                    _require_result_query_keys(query, set())
                    evaluation = read_evaluation()
                    reference = evaluation.get("benchmark_reference", {})
                    report = reference.get("report", {})
                    configured = report.get("official_matrix_rows") if isinstance(report, dict) else None
                    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 0:
                        configured = len(official_matrix_keys())
                    payload = service.result_sources(
                        official_configured=configured,
                        official_evaluated=official_evaluated_count(evaluation),
                    )
                elif parsed.path == "/api/project-runs":
                    _require_result_query_keys(query, {"project_id"})
                    payload = service.project_runs(exact_query_value(query, "project_id"))
                elif parsed.path == "/api/project-run-results":
                    _require_result_query_keys(query, {"project_id", "run_id"})
                    payload = service.project_run_results(
                        exact_query_value(query, "project_id"),
                        exact_query_value(query, "run_id"),
                    )
                else:
                    _require_result_query_keys(
                        query,
                        {"project_id", "run_id", "combo_id", "limit", "offset"},
                    )
                    payload = service.project_run_evidence(
                        exact_query_value(query, "project_id"),
                        exact_query_value(query, "run_id"),
                        exact_query_value(query, "combo_id"),
                        limit=_bounded_result_integer(query, "limit", minimum=1, maximum=100),
                        offset=_bounded_result_integer(query, "offset", minimum=0, maximum=10_000_000),
                    )
                self.send_json(payload)
            except ProjectRunResultsError as exc:
                self.send_json(
                    {"error": {"code": exc.code, "message": exc.public_message}},
                    exc.status,
                )
            except Exception:
                self.send_json(
                    {"error": {"code": "run_unavailable", "message": "Result service is unavailable"}},
                    500,
                )
            return
        if parsed.path == "/api/run/status":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            self.send_json(job_status(job_id))
            return
        if parsed.path == "/api/project-setup/status":
            job_id = parse_qs(parsed.query).get("job_id", [""])[0]
            self.send_json(job_status(job_id))
            return
        if parsed.path == "/api/options":
            self.send_json(benchmark_options())
            return
        if parsed.path == "/api/pdf":
            name = parse_qs(parsed.query).get("name", [""])[0]
            pdf_path = pdf_path_for_name(name)
            if not pdf_path:
                self.send_json({"error": "PDF not found in data/pdfs", "name": name}, 404)
                return
            body = pdf_path.read_bytes()
            safe_filename = pdf_path.name.replace('"', "")
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", f'inline; filename="{safe_filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
                "data/nvidia_rag/benchmark_summary.csv",
                "data/nvidia_rag/benchmark_details.csv",
                "data/nvidia_rag/benchmark_report.json",
                "data/nvidia_rag/benchmark_latest.json",
                "data/nvidia_rag/benchmark_baseline_summary.csv",
                "data/nvidia_rag/benchmark_baseline_details.csv",
                "data/nvidia_rag/benchmark_baseline_report.json",
                "data/nvidia_rag/benchmark_baseline_latest.json",
                "data/nvidia_rag/benchmark_reranked_summary.csv",
                "data/nvidia_rag/benchmark_reranked_details.csv",
                "data/nvidia_rag/benchmark_reranked_report.json",
                "data/nvidia_rag/benchmark_reranked_latest.json",
                "data/retrieval_smoke/summary.json",
                "data/evaluation/groundtruth_eval_summary.csv",
                "data/evaluation/groundtruth_eval_details.csv",
                "data/evaluation/groundtruth_eval_report.json",
                "data/reranker_analysis/reranker_lift_summary.csv",
                "data/reranker_analysis/reranker_lift_details.csv",
                "data/reranker_analysis/qwen_vs_none_report.json",
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
            qs = parse_qs(parsed.query)
            retrieval_limit = min(int(qs.get("retrieval_limit", ["0"])[0]), 5000)
            reranker_limit = min(int(qs.get("reranker_limit", ["0"])[0]), 5000)
            detail_evidence_limit = min(int(qs.get("detail_evidence_limit", ["360"])[0]), 2000)
            detail_evidence_per_combo = min(int(qs.get("detail_evidence_per_combo", ["1"])[0]), 5)
            retrieval_smokes = read_retrieval_smokes(limit=retrieval_limit)
            reranker_smokes = read_reranker_smokes(limit=reranker_limit)
            benchmark_evidence = benchmark_detail_evidence(limit_per_combo=detail_evidence_per_combo, max_rows=detail_evidence_limit)
            retrieval_total = retrieval_smoke_count()
            reranker_total = reranker_smoke_count()
            evaluation = read_evaluation()
            hallucination = read_hallucination()
            pdf_audit = read_pdf_audit()
            document_repository = read_document_repository()
            user_projects = list_user_projects()
            nvidia_rag = read_nvidia_rag()
            reranker_analysis = read_reranker_analysis()
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
                    "benchmark_detail_evidence": benchmark_evidence,
                    "benchmark_detail_evidence_loaded": len(benchmark_evidence),
                    "reranker_smoke_total": reranker_total,
                    "reranker_smoke_loaded": len(reranker_smokes),
                    "retrieval_smokes": retrieval_smokes,
                    "retrieval_smoke_total": retrieval_total,
                    "retrieval_smoke_loaded": len(retrieval_smokes),
                    "pipeline_state": pipeline_state_rows(),
                    "evaluation": evaluation,
                    "hallucination": hallucination,
                    "nvidia_rag": nvidia_rag,
                    "reranker_analysis": reranker_analysis,
                    "vm_snapshot": snapshot,
                    "service_health": live_service_health(snapshot),
                    "known_pdf_count": document_repository.get("total") or pdf_audit.get("total") or 0,
                    "extracted_pdf_count": document_repository.get("ready_count") or pdf_audit.get("ok_count") or 0,
                    "failed_pdf_count": pdf_audit.get("needs_ocr_count") or 0,
                    "pdf_audit": pdf_audit,
                    "document_repository": document_repository,
                    "user_projects": user_projects,
                    "known_matrix_count": benchmark_options().get("matrix_count"),
                    "options_formula": "5 chunkers × 3 embeddings × 4 vector stores × 1 retrieval × 3 rerankers = 180",
                    "metrics_status": "Paused, no query ground-truth CSV requested yet",
                    "openai_status": "Pending OPENAI_API_KEY and cost approval",
                    "amazon_status": "Pending AWS/Bedrock credentials",
                },
                "files": [f for f in files if (ROOT / f).exists()],
                "options": benchmark_options(),
                "source_catalog": build_source_catalog(ROOT),
            })
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/project-setup":
            try:
                qs = parse_qs(parsed.query)
                project_id = qs.get("project_id", [""])[0]
                if not project_id:
                    raise ValueError("project_id is required")
                record = create_project_setup_job(project_id, {
                    "chunkers": qs.get("chunker", []),
                    "embeddings": qs.get("embedding", []),
                    "vector_stores": qs.get("store", []),
                })
                self.send_json(launch_job(project_setup_cmd(record), record))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/upload-dataset":
            try:
                if not self.headers.get("Content-Type", "").lower().startswith("multipart/form-data"):
                    raise ValueError("multipart/form-data is required")
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
                file_item = form["file"] if "file" in form else None
                if isinstance(file_item, list) or file_item is None or not getattr(file_item, "filename", ""):
                    raise ValueError("No dataset file uploaded")
                original = Path(file_item.filename).name
                label_field = form.getfirst("label", Path(original).stem)
                content = file_item.file.read()
                payload = create_user_project_upload(original_name=original, content=content, label=str(label_field))
                groundtruth_item = form["groundtruth_file"] if "groundtruth_file" in form else None
                if isinstance(groundtruth_item, list):
                    raise ValueError("Only one ground-truth file may be uploaded")
                if groundtruth_item is not None and getattr(groundtruth_item, "filename", ""):
                    attach_project_groundtruth(payload["project_id"], groundtruth_item.filename, groundtruth_item.file.read())
                    payload = {**read_project_manifest(project_root(payload["project_id"])), "next_steps": payload.get("next_steps", [])}
                self.send_json(payload)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/project-groundtruth":
            try:
                if not self.headers.get("Content-Type", "").lower().startswith("multipart/form-data"):
                    raise ValueError("multipart/form-data is required")
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
                project_id = str(form.getfirst("project_id", "")).strip()
                file_item = form["groundtruth_file"] if "groundtruth_file" in form else None
                if not project_id:
                    raise ValueError("project_id is required")
                if isinstance(file_item, list) or file_item is None or not getattr(file_item, "filename", ""):
                    raise ValueError("groundtruth_file is required")
                self.send_json(attach_project_groundtruth(project_id, file_item.filename, file_item.file.read()))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/project-query":
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body or "{}")
                result = query_user_project(
                    project_id=str(payload.get("project_id") or ""),
                    query=str(payload.get("query") or ""),
                    selections=payload.get("selections") or {},
                    top_k=int(payload.get("top_k") or 5),
                )
                self.send_json(result)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        qs = parse_qs(parsed.query)
        limit = qs.get("limit", ["50"])[0]
        max_runs = qs.get("max_runs", ["0"])[0]
        try:
            if parsed.path == "/api/run/project-evaluation/preflight":
                project_id = qs.get("project_id", [""])[0]
                cmd = project_evaluation_cmd(project_id, qs, preflight=True)
                proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
                try:
                    payload = json.loads(proc.stdout or "{}")
                except Exception:
                    payload = {"ok": False, "errors": ["Could not parse project preflight output"], "output": proc.stdout + proc.stderr}
                payload["exit_code"] = proc.returncode
                payload["cmd"] = cmd
                self.send_json(payload, 200 if proc.returncode in {0, 2} else 500)
                return
            if parsed.path == "/api/run/project-evaluation":
                project_id = qs.get("project_id", [""])[0]
                self.send_json(launch_job(project_evaluation_cmd(project_id, qs)))
                return
            if parsed.path == "/api/run/preflight-complete-pipeline":
                if qs.get("project_id", [""])[0]:
                    raise ValueError("Uploaded projects must use /api/run/project-evaluation/preflight")
                cmd = complete_pipeline_cmd(dict(qs), preflight=True)
                proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
                try:
                    payload = json.loads(proc.stdout or "{}")
                except Exception:
                    payload = {"ok": False, "missing": ["Could not parse preflight output"], "output": proc.stdout + proc.stderr}
                payload["exit_code"] = proc.returncode
                payload["cmd"] = cmd
                self.send_json(payload, 200 if proc.returncode in {0, 2} else 500)
                return
            if parsed.path == "/api/run/complete-pipeline":
                if qs.get("project_id", [""])[0]:
                    raise ValueError("Uploaded projects must use /api/run/project-evaluation")
                cmd = complete_pipeline_cmd(dict(qs), preflight=False)
                self.send_json(launch_job(cmd))
                return
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
                cmd = [sys.executable, "scripts/ingest_nvidia_rag_documents.py", "--collection", qs.get("collection", ["multimodal_data"])[0], "--path", qs.get("path", ["data/pdfs"])[0], "--limit", limit or "5", "--batch-size", qs.get("batch_size", ["2"])[0]]
                if qs.get("create_collection", ["0"])[0] == "1":
                    cmd.append("--create-collection")
                if qs.get("poll", ["0"])[0] == "1":
                    cmd.append("--poll")
            elif parsed.path == "/api/run/nvidia-benchmark":
                gt = qs.get("groundtruth", [""])[0] or os.getenv("WNS_GROUNDTRUTH_PATH", "")
                if not gt:
                    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
                    gt = str(candidates[-1]) if candidates else "data/groundtruth/groundtruth_500.csv"
                cmd = [sys.executable, "scripts/run_nvidia_rag_benchmark.py", "--groundtruth", gt, "--collection", qs.get("collection", ["multimodal_data"])[0], "--limit", limit or "25", "--top-k", qs.get("top_k", ["10"])[0], "--reranker-top-k", qs.get("reranker_top_k", ["5"])[0]]
                if qs.get("disable_reranker", ["0"])[0] == "1":
                    cmd.append("--disable-reranker")
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5011
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    server = ThreadingHTTPServer((host, port), Handler)
    shown_host = socket.gethostbyname(socket.gethostname()) if host == "0.0.0.0" else host
    print(f"WNS benchmark dashboard: http://{shown_host}:{port} (bind={host})")
    server.serve_forever()


if __name__ == "__main__":
    main()
