#!/usr/bin/env python3
"""Serve the WNS benchmark dashboard using only the Python standard library."""

from __future__ import annotations

import csv
import cgi
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.core.config import generate_matrix, load_benchmark_config
from source.benchmark_pipeline import load_chunks_from_workbook
from scripts.dashboard_source_catalog import (
    DEFAULT_DATASET_ID,
    NONE_GROUNDTRUTH_ID,
    build_source_catalog,
    register_groundtruth_upload,
    resolve_dataset,
    resolve_groundtruth,
)
from source.services.project_documents import (
    ProjectDocumentStorageError,
    ProjectDocumentValidationError,
    extract_project_documents,
    write_project_documents,
)
from source.services.project_run_results import (
    ProjectRunResultService,
    ProjectRunResultsError,
)
from source.services.project_workspace import (
    ProjectWorkspace,
    UnsupportedUploadError,
    UploadError,
    UploadLimits,
    UploadStorageError,
    UploadTooLargeError,
    UploadValidationError,
)
from scripts.dashboard_snapshot_state import publish_document_readiness
from scripts.pipeline_run_contract import parse_query_upload
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


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    if re.fullmatch(r"[0-9]+", raw or "") is None:
        return default
    return int(raw)


def dashboard_upload_limits() -> UploadLimits:
    """Build upload limits from documented nonnegative dashboard settings."""
    max_upload = _nonnegative_env_int("DASHBOARD_MAX_UPLOAD_MB", 100) * 1024 * 1024
    return UploadLimits(
        max_upload_bytes=max_upload,
        max_json_bytes=_nonnegative_env_int("DASHBOARD_MAX_JSON_KB", 1024) * 1024,
        max_zip_entries=_nonnegative_env_int("DASHBOARD_MAX_ZIP_FILES", 500),
        max_zip_entry_bytes=max_upload,
        max_zip_expanded_bytes=_nonnegative_env_int("DASHBOARD_MAX_ZIP_EXPANDED_MB", 500) * 1024 * 1024,
        max_zip_ratio=_nonnegative_env_int("DASHBOARD_MAX_ZIP_RATIO", 100),
    )


def api_error(code: str, message: str, request_id: str) -> dict:
    """Return the one public error envelope used by the upload endpoint."""
    return {"error": {"code": code, "message": message, "request_id": request_id}}


_MAX_CONTENT_LENGTH_DIGITS = 20


class _BoundedRequestReader:
    """Expose at most the declared request bytes without buffering or read-ahead."""

    def __init__(self, source: Any, limit: int):
        self.source = source
        self.remaining = limit
        self.consumed = 0
        self.truncated = False
        self.tail = b""

    def _size(self, requested: int | None) -> int:
        if requested is None or requested < 0:
            return self.remaining
        return min(requested, self.remaining)

    def _record(self, chunk: bytes, allowed: int) -> bytes:
        if not isinstance(chunk, bytes) or len(chunk) > allowed:
            raise OSError("invalid bounded request stream")
        self.remaining -= len(chunk)
        self.consumed += len(chunk)
        self.tail = (self.tail + chunk)[-256:]
        if not chunk and self.remaining:
            self.truncated = True
        return chunk

    def read(self, size: int | None = -1) -> bytes:
        allowed = self._size(size)
        if allowed <= 0:
            return b""
        return self._record(self.source.read(allowed), allowed)

    def readline(self, size: int | None = -1) -> bytes:
        allowed = self._size(size)
        if allowed <= 0:
            return b""
        return self._record(self.source.readline(allowed), allowed)


def _header_values(headers: Any, name: str) -> list[Any]:
    """Return every instance of a header when the headers object supports it."""
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name)
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]
    value = headers.get(name)
    return [] if value is None else [value]


def _parse_content_length(headers: Any) -> int | None:
    values = _header_values(headers, "Content-Length")
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    raw = values[0]
    if len(raw) > _MAX_CONTENT_LENGTH_DIGITS or re.fullmatch(r"[0-9]+", raw) is None:
        return None
    try:
        return int(raw)
    except (OverflowError, ValueError):
        return None


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


def benchmark_row_is_evaluated(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    error_code = str(row.get("error_code") or row.get("error") or "").strip()
    if (
        status in {"failed", "error", "skipped", "cancelled", "interrupted"}
        or error_code
    ):
        return False
    raw_query_count = row.get("evaluated_queries", row.get("query_count"))
    try:
        query_count = float(raw_query_count or 0)
    except (TypeError, ValueError):
        query_count = 0.0
    metric_recorded = any(
        row.get(field) is not None and row.get(field) != ""
        for field in (
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "precision_at_5",
            "ndcg_at_5",
        )
    )
    return query_count > 0 or metric_recorded


def official_evaluated_count(evaluation: dict[str, Any]) -> int:
    official = official_matrix_keys()
    seen: set[tuple[str, str, str, str]] = set()
    sources = (
        ("groundtruth_eval", evaluation.get("summary", [])),
        ("groundtruth_eval_reranked", evaluation.get("reranked", {}).get("summary", [])),
        ("benchmark_reference", evaluation.get("benchmark_reference", {}).get("summary", [])),
    )
    for source, rows in sources:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = normalized_benchmark_row(row, source)
            if not benchmark_row_is_evaluated(normalized):
                continue
            key = (
                normalized["sheet"],
                normalized["embedding"],
                normalized["store"],
                normalized["reranker"],
            )
            if key in official:
                seen.add(key)
    return len(seen)


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
    def value_or_empty(*names: str) -> Any:
        for name in names:
            value = row.get(name)
            if value is not None and value != "":
                return value
        return ""

    latency_ms = row.get("avg_latency_ms") or row.get("avg_query_latency_ms") or row.get("p50_query_latency_ms") or 0
    try:
        latency_s = float(latency_ms) / 1000.0
    except (TypeError, ValueError):
        latency_s = 0.0
    embedding = row.get("embedding") or row.get("embedding_model") or ""
    reranker = canonical_reranker_name(row.get("reranker") or row.get("reranking_model") or "none")
    return {
        "source": source,
        "sheet": row.get("chunker") or row.get("chunking_method") or row.get("sheet") or "",
        "embedding": embedding,
        "store": row.get("vector_store") or row.get("vector_database") or row.get("store") or "",
        "reranker": reranker,
        "status": value_or_empty("status"),
        "error_code": value_or_empty("error_code", "error"),
        "evaluated_queries": value_or_empty("query_count", "evaluated_queries"),
        "recall_at_1": value_or_empty("recall_at_1"),
        "recall_at_3": value_or_empty("recall_at_3"),
        "recall_at_5": value_or_empty("recall_at_5"),
        "recall_at_10": value_or_empty("recall_at_10"),
        "mrr": value_or_empty("mrr"),
        "precision_at_5": value_or_empty("precision_at_5"),
        "ndcg_at_5": value_or_empty("ndcg_at_5", "ndcg_at_10"),
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


def official_matrix_keys() -> set[tuple[str, str, str, str]]:
    cfg = load_benchmark_config(CONFIG_PATH)
    return {
        (row["chunker"], row["embedding"], row["vector_store"], canonical_reranker_name(row["reranker"]))
        for row in generate_matrix(cfg)
    }


def read_benchmark_reference() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    official_keys = official_matrix_keys()
    skipped_non_official = 0
    for source, path in benchmark_reference_sources():
        for row in sort_summary(read_csv(path)):
            normalized = normalized_benchmark_row(row, source)
            key = (normalized["sheet"], normalized["embedding"], normalized["store"], normalized["reranker"])
            if key not in official_keys:
                skipped_non_official += 1
                continue
            if key in seen:
                continue
            if not benchmark_row_is_evaluated(normalized):
                continue
            seen.add(key)
            rows.append(normalized)
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


def validated_positive_depth(name: str, value: str, *, maximum: int = 100) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} depth must be a positive integer") from None
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{name} depth must be between 1 and {maximum}")
    return parsed


def validated_multi(name: str, values: list[str], allowed: list[str] | tuple[str, ...]) -> list[str]:
    permitted = {str(value) for value in allowed}
    selected = [value for value in values if value and value != "all"]
    invalid = [value for value in selected if value not in permitted]
    if invalid:
        raise ValueError(f"Unsupported {name}: {', '.join(invalid)}")
    return selected


LEGACY_ACTION_PATHS = {
    "/api/run/parse-pdfs-mineru",
    "/api/run/ingest-selected",
    "/api/run/retrieval-smoke",
    "/api/run/full-gt-retrieval",
    "/api/run/reranker-smoke",
    "/api/run/evaluate-reranked-groundtruth",
    "/api/run/evaluate-groundtruth",
    "/api/run/evaluate-hallucination",
}
SCORED_LEGACY_ACTION_PATHS = {
    "/api/run/full-gt-retrieval",
    "/api/run/evaluate-reranked-groundtruth",
    "/api/run/evaluate-groundtruth",
    "/api/run/evaluate-hallucination",
}


def validate_legacy_action_request(path: str, qs: dict[str, list[str]]) -> None:
    if path not in LEGACY_ACTION_PATHS:
        return
    dataset_id = qs.get("dataset_id", [DEFAULT_DATASET_ID])[0] or DEFAULT_DATASET_ID
    if dataset_id != DEFAULT_DATASET_ID:
        raise ValueError("Legacy stage actions support only the default dataset; use the complete pipeline for project datasets")
    groundtruth_id = qs.get("groundtruth_id", [NONE_GROUNDTRUTH_ID])[0] or NONE_GROUNDTRUTH_ID
    if path in SCORED_LEGACY_ACTION_PATHS and groundtruth_id == NONE_GROUNDTRUTH_ID:
        raise ValueError("This scored legacy action requires a ground truth source; use evidence retrieval instead")


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


def _typed_queries(qs: dict[str, list[str]]) -> list[str]:
    return [query.strip() for raw in qs.get("query", []) for query in raw.splitlines() if query.strip()]


def _resolved_ready_dataset(root: Path, qs: dict[str, list[str]]):
    source_id = qs.get("dataset_id", [DEFAULT_DATASET_ID])[0] or DEFAULT_DATASET_ID
    dataset = resolve_dataset(root, source_id)
    if not dataset.ready or dataset.workbook_path is None or dataset.document_path is None:
        raise ValueError(f"Dataset source is not ready: {source_id}")
    return dataset


def complete_pipeline_cmd(
    qs: dict[str, list[str]],
    preflight: bool = False,
    *,
    root: Path = ROOT,
) -> list[str]:
    dataset = _resolved_ready_dataset(root, qs)
    groundtruth_id = qs.get("groundtruth_id", [NONE_GROUNDTRUTH_ID])[0] or NONE_GROUNDTRUTH_ID
    cmd = [sys.executable, "scripts/run_complete_pipeline.py"]
    if preflight:
        cmd.append("--preflight-only")
    selected_sheets = qs.get("sheet", [])
    if not selected_sheets and dataset.kind != "default":
        selected_sheets = list(dataset.sheets)
    selected_sheets = validated_multi("sheet", selected_sheets, dataset.sheets)
    options = benchmark_options()
    selected_embeddings = validated_multi("embedding", qs.get("embedding", []), options["embeddings"])
    selected_stores = validated_multi("store", qs.get("store", []), options["vector_stores"])
    rerankers = validated_multi("reranker", qs.get("reranker", []), [*options["rerankers"], "none"])
    add_multi(cmd, "--sheets", selected_sheets)
    add_multi(cmd, "--embeddings", selected_embeddings)
    add_multi(cmd, "--stores", selected_stores)
    if rerankers:
        add_multi(cmd, "--rerankers", rerankers)
    cmd += ["--workbook", str(dataset.workbook_path)]
    if dataset.kind == "default":
        cmd += ["--pdf-audit", str(root / "data" / "pdf_extraction_audit.csv")]
    else:
        cmd.append("--skip-extraction-audit")
    if groundtruth_id == NONE_GROUNDTRUTH_ID:
        queries = _typed_queries(qs)
        if not queries:
            raise ValueError("Evidence-only execution requires at least one typed query")
        cmd.append("--evidence-only")
        for query in queries:
            cmd += ["--query", query]
    else:
        groundtruth = resolve_groundtruth(root, groundtruth_id)
        if groundtruth is None or groundtruth.path is None:
            raise ValueError(f"Ground-truth source is not runnable: {groundtruth_id}")
        cmd += ["--groundtruth", str(groundtruth.path)]
    retrieval_top_k = validated_positive_depth(
        "Retrieval Top K",
        qs.get("retrieval_top_k", qs.get("top_k", ["10"]))[0] or "10",
    )
    reranked_output_k = validated_positive_depth(
        "Reranked Output K",
        qs.get("reranked_output_k", qs.get("reranker_top_k", ["5"]))[0] or "5",
    )
    if reranked_output_k > retrieval_top_k:
        raise ValueError("Reranked Output K cannot exceed Retrieval Top K")
    cmd += ["--top-k", str(retrieval_top_k), "--reranked-output-k", str(reranked_output_k)]
    query_limit = qs.get("query_limit", ["0"])[0] or "0"
    cmd += ["--query-limit", query_limit]
    if qs.get("fresh_run", ["0"])[0] == "1":
        cmd.append("--fresh-run")
    if qs.get("allow_partial_extraction", ["0"])[0] == "1":
        cmd.append("--allow-partial-extraction")
    return cmd


def nvidia_ingest_cmd(qs: dict[str, list[str]], *, root: Path = ROOT) -> list[str]:
    dataset = _resolved_ready_dataset(root, qs)
    cmd = [
        sys.executable,
        "scripts/ingest_nvidia_rag_documents.py",
        "--collection",
        qs.get("collection", ["multimodal_data"])[0],
        "--path",
        str(dataset.document_path),
        "--limit",
        qs.get("limit", ["5"])[0] or "5",
        "--batch-size",
        qs.get("batch_size", ["2"])[0],
    ]
    if qs.get("create_collection", ["0"])[0] == "1":
        cmd.append("--create-collection")
    if qs.get("poll", ["0"])[0] == "1":
        cmd.append("--poll")
    return cmd


def nvidia_benchmark_cmd(qs: dict[str, list[str]], *, root: Path = ROOT) -> list[str]:
    groundtruth_id = qs.get("groundtruth_id", [NONE_GROUNDTRUTH_ID])[0] or NONE_GROUNDTRUTH_ID
    if groundtruth_id == NONE_GROUNDTRUTH_ID:
        raise ValueError("NVIDIA benchmark requires a ground truth source; use smoke for evidence-only queries")
    groundtruth = resolve_groundtruth(root, groundtruth_id)
    if groundtruth is None or groundtruth.path is None:
        raise ValueError(f"Ground-truth source is not runnable: {groundtruth_id}")
    cmd = [
        sys.executable,
        "scripts/run_nvidia_rag_benchmark.py",
        "--groundtruth",
        str(groundtruth.path),
        "--collection",
        qs.get("collection", ["multimodal_data"])[0],
        "--limit",
        qs.get("limit", ["25"])[0] or "25",
        "--top-k",
        qs.get("top_k", ["10"])[0],
        "--reranker-top-k",
        qs.get("reranker_top_k", ["5"])[0],
    ]
    if qs.get("disable_reranker", ["0"])[0] == "1":
        cmd.append("--disable-reranker")
    return cmd


def public_command(cmd: list[str]) -> list[str]:
    private_value_flags = {"--workbook", "--groundtruth", "--pdf-audit", "--benchmark-input", "--path"}
    public: list[str] = []
    redact_next = False
    for value in cmd:
        if redact_next:
            public.append("<server-resolved>")
            redact_next = False
            continue
        public.append(value)
        redact_next = value in private_value_flags
    return public


def launch_job(cmd: list[str]) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    log_path = JOB_DIR / f"{job_id}.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
    log_handle.close()
    job = {"job_id": job_id, "cmd": cmd, "log_path": str(log_path), "process": proc, "started_at": datetime.now().isoformat()}
    JOBS[job_id] = job
    return job_status(job_id)


def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        return {"error": "unknown job_id", "job_id": job_id}
    proc = job["process"]
    exit_code = proc.poll()
    log_path = Path(job["log_path"])
    output = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    return {
        "job_id": job_id,
        "running": exit_code is None,
        "exit_code": exit_code,
        "cmd": public_command(job["cmd"]),
        "started_at": job["started_at"],
        "output": output[-20000:],
        "log_path": str(log_path.relative_to(ROOT)),
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
    return ProjectWorkspace(USER_PROJECTS_DIR).new_project_id(label)


def project_root(project_id: str) -> Path:
    return ProjectWorkspace(USER_PROJECTS_DIR).project_root(project_id)


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


def read_text_upload(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            raise UploadStorageError(UploadStorageError.public_message) from None
        try:
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[page {i}]\n{text}")
        except Exception:
            raise UploadValidationError(UploadValidationError.public_message) from None
        extracted = "\n\n".join(pages)
        if not extracted.strip():
            raise UploadValidationError(UploadValidationError.public_message)
        return extracted
    return ""


def split_project_chunks(text: str, source_name: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"\s+", " ", text or " ").strip()
    if not cleaned:
        return []
    raw_parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9])", text) if p and p.strip()]
    if not raw_parts:
        raw_parts = [cleaned]
    chunks: list[dict[str, Any]] = []
    chunk_id = 1
    for part in raw_parts:
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        # Keep small uploaded demo docs searchable, but avoid pathological giant rows.
        for start in range(0, len(part), 1400):
            paragraph = part[start:start + 1400].strip()
            if not paragraph:
                continue
            page_match = re.search(r"\[page\s+(\d+)\]", paragraph, re.I)
            chunks.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "pdf_name": source_name,
                "paragraph": paragraph,
                "page_number": page_match.group(1) if page_match else "",
                "source_type": "uploaded_document",
                "parser_method": "dashboard_upload_text_index",
            })
            chunk_id += 1
    return chunks


def write_project_workbook(
    project_dir: Path,
    chunks: list[dict[str, Any]],
    display_project_dir: Path | None = None,
) -> str:
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
    display_root = display_project_dir or project_dir
    return display_path(display_root / out.relative_to(project_dir))


def rebuild_project_search_index(
    project_dir: Path,
    display_project_dir: Path | None = None,
) -> dict[str, Any]:
    raw_dir = project_dir / "raw_uploads"
    extracted_text_dir = project_dir / "extracted_text"
    extracted_text_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".csv"}]
    chunks: list[dict[str, Any]] = []
    source_files: list[str] = []
    for path in sorted(candidates):
        text = read_text_upload(path)
        if not text.strip():
            continue
        rel_name = path.name
        source_relative = path.relative_to(project_dir)
        source_files.append(str(source_relative))
        text_relative = path.relative_to(raw_dir).with_suffix(".txt")
        text_path = extracted_text_dir / text_relative
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        for chunk in split_project_chunks(text, rel_name):
            chunk["id"] = len(chunks) + 1
            chunk["chunk_id"] = chunk["id"]
            chunk["project_source_path"] = str(source_relative)
            chunks.append(chunk)
    index_path = project_dir / "search_index.json"
    index = {
        "created_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "source_files": source_files,
        "chunks": chunks,
        "workbook": write_project_workbook(project_dir, chunks, display_project_dir),
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def create_user_project_upload(
    original_name: str,
    content: bytes,
    label: str = "",
    limits: UploadLimits | None = None,
) -> dict[str, Any]:
    if Path(original_name).suffix.lower() not in {".txt", ".pdf", ".zip"}:
        raise UnsupportedUploadError(UnsupportedUploadError.public_message)
    workspace = ProjectWorkspace(USER_PROJECTS_DIR, limits or dashboard_upload_limits())
    finalized_context: dict[str, Any] = {}

    def finalize_upload(staging: Path, final_root: Path, published: dict) -> dict:
        saved_relative = Path(published["saved_path"])
        saved_path = staging / saved_relative
        if staging not in saved_path.resolve().parents:
            raise UploadValidationError(UploadValidationError.public_message)
        final_saved_path = final_root / saved_relative

        # Keep the historical dashboard layout alias while the strict service uses indexes/.
        (staging / "vector_indexes").mkdir()
        extracted_relative = [Path(relative) for relative in published.get("extracted_files", [])]
        for relative in extracted_relative:
            candidate = staging / relative
            if staging not in candidate.resolve().parents:
                raise UploadValidationError(UploadValidationError.public_message)
        extracted = [display_path(final_root / relative) for relative in extracted_relative]

        saved_suffix = saved_path.suffix.lower()
        document_sources: list[Path] = []
        if saved_suffix in {".txt", ".pdf"}:
            document_sources = [saved_path]
        elif saved_suffix == ".zip":
            if not extracted_relative or any(
                relative.suffix.lower() not in {".txt", ".pdf"}
                for relative in extracted_relative
            ):
                raise UploadValidationError(UploadValidationError.public_message)
            document_sources = [staging / relative for relative in extracted_relative]

        if not document_sources:
            raise UploadValidationError(UploadValidationError.public_message)

        documents = []
        source_records: list[dict[str, Any]] = []
        extraction_failures: list[dict[str, str]] = []
        for source_path in sorted(
            document_sources,
            key=lambda path: path.relative_to(staging / "raw_uploads").as_posix(),
        ):
            relative = source_path.relative_to(staging / "raw_uploads")
            if relative.parts and relative.parts[0] == "extracted":
                relative = Path(*relative.parts[1:])
            source_name = relative.as_posix()
            try:
                raw_source = source_path.read_bytes()
            except OSError:
                raise UploadStorageError(UploadStorageError.public_message) from None
            source_record = {
                "source_name": source_name,
                "raw_sha256": hashlib.sha256(raw_source).hexdigest(),
                "raw_size_bytes": len(raw_source),
                "parser_versions": [],
                "page_count": 0,
            }
            try:
                source_documents = extract_project_documents(staging, [source_path])
            except ProjectDocumentStorageError:
                raise UploadStorageError(UploadStorageError.public_message) from None
            except ProjectDocumentValidationError:
                source_records.append({**source_record, "status": "failed"})
                extraction_failures.append(
                    {"source_name": source_name, "code": "extraction_failed"}
                )
                continue
            documents.extend(source_documents)
            source_records.append(
                {
                    **source_record,
                    "status": "complete",
                    "parser_versions": sorted(
                        {
                            str(document.metadata["parser_method"])
                            for document in source_documents
                        }
                    ),
                    "page_count": len(source_documents),
                }
            )

        if extraction_failures:
            corpus_manifest: dict[str, Any] = {
                "ok": False,
                "extraction_status": "failed",
                "document_count": 0,
                "source_count": len(document_sources),
                "page_count": sum(record["page_count"] for record in source_records),
                "corpus_sha256": None,
                "parser_versions": [],
                "sources": source_records,
                "extraction_failures": extraction_failures,
            }
        else:
            corpus_manifest = {
                **write_project_documents(
                    staging / "extracted_text" / "documents.jsonl",
                    documents,
                ),
                "extraction_status": "complete",
                "page_count": len(documents),
                "parser_versions": sorted(
                    {str(document.metadata["parser_method"]) for document in documents}
                ),
                "sources": source_records,
                "extraction_failures": [],
            }

        manifest = {
            **published,
            "saved_path": display_path(final_saved_path),
            "extracted_files": extracted[:200],
            "storage_layout": project_storage_layout(final_root),
            "question_file": "",
            "search_index": "",
            "canonical_corpus": (
                display_path(final_root / "extracted_text" / "documents.jsonl")
                if corpus_manifest["extraction_status"] == "complete"
                else ""
            ),
            **corpus_manifest,
            "chunk_count": 0,
            "workbook": "",
            "manifest_path": display_path(final_root / "manifest.json"),
        }
        finalized_context.update(
            saved_path=final_saved_path,
            root=final_root,
            extracted=extracted,
        )
        return manifest

    published = workspace.create_upload(
        original_name=original_name,
        content=content,
        label=label,
        finalizer=finalize_upload,
    )
    return {
        **published,
        "next_steps": upload_next_steps(
            finalized_context["saved_path"],
            finalized_context["root"],
            finalized_context["extracted"],
        ),
    }



def list_user_projects() -> list[dict[str, Any]]:
    if not USER_PROJECTS_DIR.exists():
        return []
    projects: list[dict[str, Any]] = []
    for manifest_path in sorted(USER_PROJECTS_DIR.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest = {"project_id": manifest_path.parent.name, "error": str(exc)}
        projects.append(manifest)
    return projects


def query_user_project(project_id: str, query: str, top_k: int = 5) -> dict[str, Any]:
    return ProjectWorkspace(USER_PROJECTS_DIR).lexical_preview(project_id, query, top_k)


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
                    configured = (
                        report.get("official_matrix_rows")
                        if isinstance(report, dict)
                        else None
                    )
                    if (
                        isinstance(configured, bool)
                        or not isinstance(configured, int)
                        or configured < 0
                    ):
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
                        limit=_bounded_result_integer(
                            query, "limit", minimum=1, maximum=100
                        ),
                        offset=_bounded_result_integer(
                            query, "offset", minimum=0, maximum=10_000_000
                        ),
                    )
                self.send_json(payload)
            except ProjectRunResultsError as exc:
                self.send_json(
                    {"error": {"code": exc.code, "message": exc.public_message}},
                    exc.status,
                )
            except Exception:
                self.send_json(
                    {
                        "error": {
                            "code": "run_unavailable",
                            "message": "Result service is unavailable",
                        }
                    },
                    500,
                )
            return
        if parsed.path == "/api/run/status":
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
            document_repository = publish_document_readiness(ROOT, read_document_repository())
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
                    "document_repository_latest_attempt": document_repository.get("latest_attempt", {}),
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
        if parsed.path == "/api/upload-dataset":
            request_id = uuid.uuid4().hex
            content_length = _parse_content_length(self.headers)
            if content_length is None:
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed upload request", request_id),
                    400,
                )
                return

            try:
                limits = dashboard_upload_limits()
            except Exception:
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return

            if content_length > limits.max_upload_bytes:
                self.close_connection = True
                self.send_json(
                    api_error(
                        "upload_too_large",
                        "Upload exceeds the configured size limit",
                        request_id,
                    ),
                    413,
                )
                return

            raw_content_type = self.headers.get("Content-Type", "")
            try:
                media_type, content_params = cgi.parse_header(raw_content_type)
            except (TypeError, ValueError):
                media_type = ""
                content_params = {}
            if media_type.lower() != "multipart/form-data":
                self.close_connection = True
                self.send_json(
                    api_error(
                        "unsupported_media_type",
                        "Content-Type must be multipart/form-data",
                        request_id,
                    ),
                    415,
                )
                return

            boundary = content_params.get("boundary")
            if (
                not isinstance(boundary, str)
                or not boundary
                or len(boundary) > 70
                or any(ord(char) < 33 or ord(char) > 126 for char in boundary)
            ):
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed upload request", request_id),
                    400,
                )
                return
            terminal_boundary = b"\r\n--" + boundary.encode("ascii") + b"--"

            bounded_reader = _BoundedRequestReader(self.rfile, content_length)
            try:
                form = cgi.FieldStorage(
                    fp=cast(Any, bounded_reader),
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": raw_content_type,
                        "CONTENT_LENGTH": str(content_length),
                    },
                )
                if bounded_reader.remaining != 0 or not (
                    bounded_reader.tail.endswith(terminal_boundary)
                    or bounded_reader.tail.endswith(terminal_boundary + b"\r\n")
                ):
                    raise ValueError("truncated multipart request")
                file_item = form["file"] if "file" in form else None
                original = getattr(file_item, "filename", "") if file_item is not None else ""
                file_object = getattr(file_item, "file", None) if file_item is not None else None
                if not isinstance(original, str) or not original or file_object is None:
                    raise ValueError("missing upload file")
                content = file_object.read(limits.max_upload_bytes + 1)
                if not isinstance(content, bytes):
                    raise ValueError("upload content must be bytes")
                if len(content) > limits.max_upload_bytes:
                    self.send_json(
                        api_error(
                            "upload_too_large",
                            "Upload exceeds the configured size limit",
                            request_id,
                        ),
                        413,
                    )
                    return
                label_field = form.getfirst("label", Path(original).stem)
                upload_type = str(form.getfirst("upload_type", "dataset"))
                if upload_type not in {"dataset", "groundtruth", "queries"}:
                    raise ValueError("invalid upload type")
            except (AttributeError, EOFError, KeyError, OSError, TypeError, ValueError):
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed upload request", request_id),
                    400,
                )
                return
            except Exception:
                self.close_connection = True
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return

            try:
                if upload_type == "queries":
                    payload = parse_query_upload(original, content)
                elif upload_type == "groundtruth":
                    payload = register_groundtruth_upload(
                        ROOT, original, content, str(label_field)
                    )
                else:
                    payload = create_user_project_upload(
                        original_name=original,
                        content=content,
                        label=str(label_field),
                    )
                    project_id = payload.get("project_id")
                    if project_id:
                        payload["source_id"] = f"project:{project_id}"
            except ValueError:
                self.send_json(
                    api_error("upload_validation_failed", "Upload validation failed", request_id),
                    422,
                )
                return
            except UploadTooLargeError:
                self.send_json(
                    api_error(
                        "upload_too_large",
                        "Upload exceeds the configured size limit",
                        request_id,
                    ),
                    413,
                )
                return
            except UnsupportedUploadError:
                self.send_json(
                    api_error("unsupported_media_type", "Unsupported upload type", request_id),
                    415,
                )
                return
            except UploadValidationError:
                self.send_json(
                    api_error("upload_validation_failed", "Upload validation failed", request_id),
                    422,
                )
                return
            except (UploadStorageError, UploadError):
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return
            except Exception:
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return

            self.send_json(payload)
            return
        if parsed.path == "/api/project-query":
            request_id = uuid.uuid4().hex
            content_length = _parse_content_length(self.headers)
            if content_length is None:
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed project query request", request_id),
                    400,
                )
                return
            media_type = str(self.headers.get("Content-Type", "")).partition(";")[0].strip().lower()
            if media_type != "application/json":
                self.send_json(
                    api_error("unsupported_media_type", "Project query request must be JSON", request_id),
                    415,
                )
                return
            try:
                limits = dashboard_upload_limits()
            except Exception:
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return
            if content_length > limits.max_json_bytes:
                self.close_connection = True
                self.send_json(
                    api_error("request_too_large", "Project query request is too large", request_id),
                    413,
                )
                return

            reader = _BoundedRequestReader(self.rfile, content_length)
            body = reader.read(content_length)
            if reader.remaining != 0:
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed project query request", request_id),
                    400,
                )
                return
            try:
                payload = json.loads(body.decode("utf-8"))
                if (
                    not isinstance(payload, dict)
                    or not set(payload).issubset({"project_id", "query", "top_k"})
                ):
                    raise ValueError("invalid project query request")
                project_id = payload.get("project_id")
                query = payload.get("query")
                top_k = payload.get("top_k", 5)
                if (
                    not isinstance(project_id, str)
                    or not isinstance(query, str)
                    or isinstance(top_k, bool)
                    or not isinstance(top_k, int)
                ):
                    raise ValueError("invalid project query fields")
                result = query_user_project(
                    project_id=project_id,
                    query=query,
                    top_k=top_k,
                )
            except FileNotFoundError:
                self.send_json(
                    api_error("project_not_found", "Project was not found", request_id),
                    404,
                )
                return
            except (UnicodeError, json.JSONDecodeError):
                self.close_connection = True
                self.send_json(
                    api_error("malformed_request", "Malformed project query request", request_id),
                    400,
                )
                return
            except ValueError:
                self.send_json(
                    api_error("invalid_request", "Invalid project query request", request_id),
                    400,
                )
                return
            except UploadValidationError:
                self.send_json(
                    api_error("project_data_invalid", "Project data is invalid", request_id),
                    422,
                )
                return
            except UploadError:
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return
            except Exception:
                self.send_json(api_error("internal_error", "Internal server error", request_id), 500)
                return
            self.send_json(result)
            return
        qs = parse_qs(parsed.query)
        limit = qs.get("limit", ["50"])[0]
        max_runs = qs.get("max_runs", ["0"])[0]
        try:
            validate_legacy_action_request(parsed.path, qs)
            if parsed.path == "/api/run/preflight-complete-pipeline":
                cmd = complete_pipeline_cmd(qs, preflight=True)
                proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
                try:
                    payload = json.loads(proc.stdout or "{}")
                except Exception:
                    payload = {"ok": False, "missing": ["Could not parse preflight output"], "output": proc.stdout + proc.stderr}
                payload["exit_code"] = proc.returncode
                for private_key in ("workbook", "benchmark_input", "pdf_audit", "groundtruth"):
                    payload.pop(private_key, None)
                payload["dataset_id"] = qs.get("dataset_id", [DEFAULT_DATASET_ID])[0]
                payload["groundtruth_id"] = qs.get("groundtruth_id", [NONE_GROUNDTRUTH_ID])[0]
                payload["cmd"] = public_command(cmd)
                self.send_json(payload, 200 if proc.returncode in {0, 2} else 500)
                return
            if parsed.path == "/api/run/complete-pipeline":
                cmd = complete_pipeline_cmd(qs, preflight=False)
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
                cmd = nvidia_ingest_cmd(qs)
            elif parsed.path == "/api/run/nvidia-benchmark":
                cmd = nvidia_benchmark_cmd(qs)
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
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
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
