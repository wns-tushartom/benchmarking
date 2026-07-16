#!/usr/bin/env python3
"""Run the complete live WNS benchmark pipeline for selected or all options.

Stages:
1. prerequisite check
2. optional fresh artifact archive
3. vector DB ingestion
4. ground-truth retrieval
5. no-reranker evaluation
6. selected reranker pass
7. reranked evaluation
8. reranker lift analysis

This script prints progress immediately with flush=True so the dashboard can stream
live output while the run is still executing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.core.config import load_benchmark_config
from scripts.wns_env import load_env_files, service_base_from_endpoint

CONFIG_PATH = ROOT / "configs" / "benchmark.local.json"
WORKBOOK_PATH = ROOT / "data" / "chunking_methods_output_v2.xlsx"
BENCHMARK_INPUT_PATH = ROOT / "data" / "benchmark_input.csv"
PDF_AUDIT_PATH = ROOT / "data" / "pdf_extraction_audit.csv"
GROUNDTRUTH_DIR = ROOT / "data" / "groundtruth"
LIVE_EMBEDDINGS = ["gte_multilingual_base", "jina_v3"]
LIVE_RERANKERS = ["bge-reranker-base", "qwen3_4b_rerank", "Amazon Rerank v1"]
ARCHIVE_DIRS = [
    "data/faiss_indexes",
    "data/retrieval_smoke",
    "data/reranker_smoke",
    "data/evaluation",
    "data/evaluation_reranked",
    "data/reranker_analysis",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_env() -> None:
    load_env_files(ROOT)


def cfg_options() -> dict[str, list[str]]:
    cfg = load_benchmark_config(CONFIG_PATH)
    matrix = cfg.get("matrix", {})
    return {
        "sheets": list(matrix.get("chunkers", [])),
        "embeddings": list(matrix.get("embeddings", [])),
        "stores": list(matrix.get("vector_stores", [])),
        "rerankers": list(matrix.get("rerankers", [])),
    }


def normalize_reranker(value: str) -> str:
    raw = str(value or "").strip()
    normalized = "_".join(part for part in raw.lower().replace(":", " ").replace("-", " ").split() if part)
    if normalized in {"none", "no_reranker", "baseline"}:
        return "none"
    if normalized in {"qwen", "qwen3", "qwen3_4b", "qwen3_4b_rerank", "qwen3_4b_reranker", "qwen3_reranker_4b_seq_cls"}:
        return "qwen3_4b_rerank"
    if normalized in {"amazon", "amazon_rerank", "amazon_rerank_v1", "amazon_rerank_v1_0", "amazon_bedrock", "amazon_bedrock_rerank"} or ("amazon" in normalized and "rerank" in normalized):
        return "Amazon Rerank v1"
    if normalized in {"bge", "bge_reranker", "bge_reranker_base"}:
        return "bge-reranker-base"
    return raw


def choose(values: list[str] | None, allowed: list[str], normalize=None) -> list[str]:
    raw = [normalize(v) if normalize else v for v in (values or []) if v]
    if not raw or "all" in raw:
        return allowed
    return [v for v in raw if v in allowed or v == "none"]


def find_groundtruth(value: str) -> Path | None:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        return path if path.exists() else None
    candidates = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))
    return candidates[-1] if candidates else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extraction_audit_issues(audit_path: Path | None = None) -> list[str]:
    audit_path = audit_path or PDF_AUDIT_PATH
    if not audit_path.exists():
        return [f"{audit_path} missing. Run extraction/chunking before live benchmark ingestion."]
    rows = read_csv_rows(audit_path)
    if not rows:
        return [f"{audit_path} is empty. Extraction audit evidence is required before live benchmark ingestion."]
    bad_statuses = {"failed", "text_only_review", "needs_ocr", "partial_ocr_review"}
    issues: list[str] = []
    for row in rows:
        pdf_name = row.get("pdf_name") or row.get("filename") or row.get("file") or "unknown_pdf"
        status = str(row.get("status", "")).strip()
        parser_method = str(row.get("parser_method") or row.get("parser") or "")
        needs_review = str(row.get("needs_ocr_review", "")).strip().lower() in {"1", "true", "yes"}
        row_count_raw = str(row.get("row_count", row.get("rows", ""))).strip()
        row_count_zero = row_count_raw.isdigit() and int(row_count_raw) == 0
        if status in bad_statuses or parser_method == "PyPDF2_fallback" or needs_review or row_count_zero:
            issues.append(f"{pdf_name}: status={status or 'unknown'} parser={parser_method or 'unknown'} row_count={row_count_raw or 'unknown'}")
    return issues[:50]


def workbook_sheet_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as zf:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            return [sheet.attrib.get("name", "") for sheet in wb.findall("a:sheets/a:sheet", ns)]
    except Exception:
        return []


def normalized_queries(values: list[str] | None) -> list[str]:
    return [query for value in (values or []) if (query := str(value).strip())]


def dataset_sheet_options(workbook_path: Path, configured_sheets: list[str]) -> list[str]:
    if workbook_path.resolve() == WORKBOOK_PATH.resolve():
        return configured_sheets
    return workbook_sheet_names(workbook_path)


def chunking_gap(
    sheets: list[str],
    workbook_path: Path = WORKBOOK_PATH,
    benchmark_input_path: Path = BENCHMARK_INPUT_PATH,
) -> tuple[list[str], list[str], bool]:
    available = workbook_sheet_names(workbook_path)
    missing = [sheet for sheet in sheets if sheet not in available]
    can_rebuild = (
        benchmark_input_path.exists()
        and workbook_path.resolve() == WORKBOOK_PATH.resolve()
        and benchmark_input_path.resolve() == BENCHMARK_INPUT_PATH.resolve()
    )
    return available, missing, can_rebuild


def ensure_chunking_workbook(
    sheets: list[str],
    workbook_path: Path = WORKBOOK_PATH,
    benchmark_input_path: Path = BENCHMARK_INPUT_PATH,
) -> None:
    available, missing, can_rebuild = chunking_gap(sheets, workbook_path, benchmark_input_path)
    if not missing:
        return
    if not can_rebuild:
        raise RuntimeError(
            f"Selected chunker sheet(s) are missing from {workbook_path} and cannot be rebuilt here: "
            + ", ".join(missing)
        )
    log(
        "chunking workbook missing selected sheet(s): "
        + ", ".join(missing)
        + "; rebuilding from data/benchmark_input.csv. Available before rebuild: "
        + ", ".join(available or ["none"])
    )
    run_cmd([sys.executable, "scripts/run_chunking_pipeline.py", "--mode", "chunk-only"], "chunking workbook rebuild")
    available_after, missing_after, _ = chunking_gap(sheets, workbook_path, benchmark_input_path)
    if missing_after:
        raise RuntimeError(
            "Chunking rebuild finished but selected sheet(s) are still missing: "
            + ", ".join(missing_after)
            + ". Available sheets: "
            + ", ".join(available_after or ["none"])
        )


def url_ok(url: str, method: str = "GET") -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except Exception as exc:
        return False, repr(exc)


def pgvector_auth_ok() -> tuple[bool, str]:
    dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        return False, "PGVECTOR_DSN or DATABASE_URL is not set"
    try:
        import psycopg  # type: ignore[import-not-found]
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1").fetchone()
        return True, "SELECT 1 OK"
    except Exception as exc:
        return False, repr(exc)


def faiss_import_ok() -> tuple[bool, str]:
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception as exc:
        return False, "faiss-cpu missing: " + repr(exc)
    return True, f"faiss import OK version={getattr(faiss, '__version__', 'unknown')}"


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    load_env()
    options = cfg_options()
    workbook_path = Path(args.workbook).resolve() if getattr(args, "workbook", "") else WORKBOOK_PATH
    benchmark_input_path = Path(args.benchmark_input).resolve() if getattr(args, "benchmark_input", "") else BENCHMARK_INPUT_PATH
    audit_path = Path(args.pdf_audit).resolve() if getattr(args, "pdf_audit", "") else PDF_AUDIT_PATH
    sheets = choose(args.sheets, dataset_sheet_options(workbook_path, options["sheets"]))
    embeddings = choose(args.embeddings, [e for e in options["embeddings"] if e in LIVE_EMBEDDINGS])
    stores = choose(args.stores, options["stores"])
    evidence_only = bool(getattr(args, "evidence_only", False))
    if evidence_only:
        rerankers: list[str] = []
    elif args.rerankers and "none" in [normalize_reranker(r) for r in args.rerankers]:
        rerankers = []
    else:
        rerankers = choose(args.rerankers, LIVE_RERANKERS, normalize=normalize_reranker)
    gt = None if evidence_only else find_groundtruth(args.groundtruth)
    missing: list[str] = []
    warnings: list[str] = []

    available_sheets, missing_sheets, can_rebuild_chunking = chunking_gap(sheets, workbook_path, benchmark_input_path)
    if not workbook_path.exists() and not can_rebuild_chunking:
        missing.append(f"Missing selected workbook: {workbook_path}")
    elif missing_sheets:
        message = (
            f"Selected chunker sheet(s) are missing from {workbook_path}: "
            + ", ".join(missing_sheets)
            + ". Available sheets: "
            + ", ".join(available_sheets or ["none"])
        )
        if can_rebuild_chunking:
            warnings.append(message + ". Full run will rebuild chunking from the selected benchmark input before ingestion.")
        else:
            missing.append(message + ". Regenerate the selected dataset workbook first.")
    queries = normalized_queries(getattr(args, "query", []))
    if evidence_only and not queries:
        missing.append("Evidence-only mode requires at least one --query value.")
    elif not evidence_only and not gt:
        missing.append("Missing ground-truth CSV/XLSX under data/groundtruth/ or in the selected groundtruth path.")
    audit_issues = [] if getattr(args, "skip_extraction_audit", False) else extraction_audit_issues(audit_path)
    if audit_issues:
        message = "Extraction audit is not final-product clean: " + "; ".join(audit_issues)
        if args.allow_partial_extraction:
            warnings.append(message + ". Proceeding only because --allow-partial-extraction was set.")
        else:
            missing.append(message)
    if not sheets:
        missing.append("No chunker selected.")
    if not embeddings:
        missing.append("No live embedding selected. Supported live endpoints here: gte_multilingual_base, jina_v3.")
    if not stores:
        missing.append("No vector DB selected.")

    unsupported_embeddings = [e for e in (args.embeddings or []) if e not in {"all", *LIVE_EMBEDDINGS}]
    if unsupported_embeddings:
        missing.append("Selected embedding is not supported by the live DB ingestion runner yet: " + ", ".join(unsupported_embeddings))

    unsupported_rerankers = [r for r in [normalize_reranker(v) for v in (args.rerankers or [])] if r not in {"all", "none", *LIVE_RERANKERS}]
    if unsupported_rerankers:
        missing.append("Selected reranker is not supported by the live reranker runner yet: " + ", ".join(unsupported_rerankers))

    embedding_envs = {"gte_multilingual_base": "GTE_EMBEDDING_URL", "jina_v3": "JINA_EMBEDDING_URL"}
    for embedding in embeddings:
        env_name = embedding_envs.get(embedding)
        if env_name and not os.getenv(env_name):
            missing.append(f"{embedding} selected but {env_name} is not set. Run setup_all_on_vm.sh or set it explicitly.")
    reranker_envs = {"bge-reranker-base": "BGE_RERANK_URL", "qwen3_4b_rerank": "QWEN_RERANK_URL"}
    for reranker in rerankers:
        env_name = reranker_envs.get(reranker)
        if env_name and not os.getenv(env_name):
            missing.append(f"{reranker} selected but {env_name} is not set. Run setup_all_on_vm.sh or set it explicitly.")
        if reranker == "Amazon Rerank v1":
            if not (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")):
                missing.append("Amazon Rerank v1 selected but AWS_REGION/AWS_DEFAULT_REGION is not set.")
            try:
                import boto3  # type: ignore[import-not-found]  # noqa: F401
            except Exception as exc:
                missing.append("Amazon Rerank v1 selected but boto3 import failed: " + repr(exc))

    service_checks: list[dict[str, Any]] = []
    model_adapter = service_base_from_endpoint(os.getenv("MODEL_ADAPTER_URL", "http://127.0.0.1:5000"))
    check_targets = [("model_adapter", model_adapter + "/health", True)]
    if "Qdrant" in stores:
        check_targets.append(("qdrant", os.getenv("QDRANT_URL", "http://127.0.0.1:5001").rstrip("/") + "/healthz", True))
    if "Weaviate" in stores:
        check_targets.append(("weaviate", os.getenv("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/") + "/v1/.well-known/ready", True))
    for name, url, required in check_targets:
        ok, note = url_ok(url)
        service_checks.append({"name": name, "url": url, "ok": ok, "note": note})
        if not ok:
            message = f"{name} did not pass health check: {note}"
            if required:
                missing.append(message)
            else:
                warnings.append(message)
    if "PGVector" in stores:
        ok, note = pgvector_auth_ok()
        service_checks.append({"name": "pgvector", "url": "PGVECTOR_DSN/DATABASE_URL", "ok": ok, "note": note})
        if not ok:
            missing.append("PGVector selected but DSN/auth check failed: " + note)
    if "FAISS" in stores:
        ok, note = faiss_import_ok()
        service_checks.append({"name": "faiss", "url": "in-process", "ok": ok, "note": note})
        if not ok:
            missing.append("FAISS selected but faiss-cpu import failed: " + note)

    combo_count = len(sheets) * len(embeddings) * len(stores)
    return {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "service_checks": service_checks,
        "groundtruth": str(gt.relative_to(ROOT)) if gt and gt.is_relative_to(ROOT) else str(gt or ""),
        "mode": "evidence_only" if evidence_only else "evaluated",
        "workbook": str(workbook_path),
        "benchmark_input": str(benchmark_input_path),
        "pdf_audit": "" if getattr(args, "skip_extraction_audit", False) else str(audit_path),
        "queries": queries,
        "sheets": sheets,
        "embeddings": embeddings,
        "stores": stores,
        "rerankers": rerankers,
        "combo_count": combo_count,
        "query_limit": args.query_limit,
        "chunk_limit": 0,
        "top_k": args.top_k,
        "retrieval_top_k": args.top_k,
        "reranked_output_k": args.reranked_output_k,
        "fresh_run": args.fresh_run,
        "allow_partial_extraction": args.allow_partial_extraction,
    }


def run_manifest_payload(run_id: str, info: dict[str, Any], *, status: str) -> dict[str, Any]:
    """Build stable run metadata without browser-controlled chunk limiting."""
    return {
        "run_id": run_id,
        "status": status,
        "mode": info.get("mode", "evaluated"),
        "sheets": list(info.get("sheets") or []),
        "embeddings": list(info.get("embeddings") or []),
        "stores": list(info.get("stores") or []),
        "rerankers": list(info.get("rerankers") or []),
        "combo_count": int(info.get("combo_count") or 0),
        "query_limit": int(info.get("query_limit") or 0),
        "chunk_limit": 0,
        "retrieval_top_k": int(info.get("retrieval_top_k") or info.get("top_k") or 10),
        "reranked_output_k": int(info.get("reranked_output_k") or 5),
    }


def write_run_manifest(root: Path, run_id: str, info: dict[str, Any], *, status: str) -> Path:
    path = root / "data" / "pipeline_runs" / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run_manifest_payload(run_id, info, status=status), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def archive_fresh_dirs(run_id: str) -> None:
    archive_root = ROOT / "data" / "pipeline_archives" / run_id
    moved = 0
    for rel in ARCHIVE_DIRS:
        path = ROOT / rel
        if not path.exists():
            continue
        target = archive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved += 1
        log(f"archived {rel} -> {target.relative_to(ROOT)}")
    if not moved:
        log("no previous fixed-path run artifacts to archive")


def run_cmd(cmd: list[str], stage: str) -> None:
    log(f"START {stage}")
    log("CMD " + " ".join(cmd))
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    code = proc.wait()
    elapsed = time.perf_counter() - started
    if code != 0:
        raise RuntimeError(f"{stage} failed with exit code {code} after {elapsed:.1f}s")
    log(f"DONE {stage} seconds={elapsed:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", nargs="*", default=["all"])
    parser.add_argument("--embeddings", nargs="*", default=["all"])
    parser.add_argument("--stores", nargs="*", default=["all"])
    parser.add_argument("--rerankers", nargs="*", default=["all"])
    parser.add_argument("--groundtruth", default="")
    parser.add_argument("--workbook", default="")
    parser.add_argument("--benchmark-input", default="")
    parser.add_argument("--pdf-audit", default="")
    parser.add_argument("--skip-extraction-audit", action="store_true")
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval candidates per query")
    parser.add_argument("--reranked-output-k", type=int, default=5, help="Final reranked hits retained per query")
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--allow-partial-extraction", action="store_true", help="Dangerous/debug only: allow live pipeline to run with missing/text-only/partial extraction audit rows.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.reranked_output_k < 1:
        parser.error("retrieval and reranked output depths must be positive")
    if args.reranked_output_k > args.top_k:
        parser.error("--reranked-output-k cannot exceed --top-k")

    os.chdir(ROOT)
    info = preflight(args)
    if args.preflight_only:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return 0 if info["ok"] else 2
    if not info["ok"]:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    info["chunk_limit"] = 0
    info["retrieval_top_k"] = args.top_k
    info["reranked_output_k"] = args.reranked_output_k
    write_run_manifest(ROOT, run_id, info, status="running")
    log("COMPLETE PIPELINE START")
    log(json.dumps(info, ensure_ascii=False))
    if info["warnings"]:
        for warning in info["warnings"]:
            log("WARNING " + warning)
    if args.fresh_run and not args.evidence_only:
        archive_fresh_dirs(run_id)

    sheets = info["sheets"]
    embeddings = info["embeddings"]
    stores = info["stores"]
    rerankers = info["rerankers"]
    max_combos = str(info["combo_count"])
    workbook = Path(info.get("workbook") or args.workbook or WORKBOOK_PATH).resolve()
    benchmark_input = Path(info.get("benchmark_input") or args.benchmark_input or BENCHMARK_INPUT_PATH).resolve()
    evidence_only = bool(args.evidence_only or info.get("mode") == "evidence_only")

    ensure_chunking_workbook(sheets, workbook, benchmark_input)

    ingest_cmd = [sys.executable, "scripts/run_long_db_ingestion.py", "--run-id", run_id, "--skip-existing-store-success", "--workbook", str(workbook), "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores]
    run_cmd(ingest_cmd, "ingestion")

    retrieval_base = [sys.executable, "scripts/run_retrieval_smoke_from_vm_dbs.py", "--run-id", run_id]
    if evidence_only:
        queries = normalized_queries(list(info.get("queries") or args.query))
        retrieval_cmd = [*retrieval_base, "--queries", *queries, "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores, "--top-k", str(args.top_k), "--max-combos", max_combos]
        if args.query_limit:
            retrieval_cmd += ["--query-limit", str(args.query_limit)]
        run_cmd(retrieval_cmd, "evidence-only retrieval")
        run_cmd([sys.executable, "scripts/collect_vm_dashboard_artifacts.py"], "dashboard artifact refresh")
        write_run_manifest(ROOT, run_id, info, status="completed")
        log("COMPLETE PIPELINE FINISHED mode=evidence_only metrics=not_applicable")
        return 0

    groundtruth_value = info["groundtruth"]
    gt_path = Path(groundtruth_value)
    gt = str(ROOT / gt_path if not gt_path.is_absolute() else gt_path)
    retrieval_cmd = [*retrieval_base, "--queries-file", gt, "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores, "--top-k", str(args.top_k), "--max-combos", max_combos]
    if args.query_limit:
        retrieval_cmd += ["--query-limit", str(args.query_limit)]
    run_cmd(retrieval_cmd, "ground-truth retrieval")

    run_cmd([sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt, "--smoke-dir", "data/retrieval_smoke", "--out-dir", "data/evaluation"], "no-reranker evaluation")

    if rerankers:
        run_cmd([sys.executable, "scripts/run_reranker_smoke_from_retrieval.py", "--rerankers", *rerankers, "--candidate-k", str(args.top_k), "--top-k", str(args.reranked_output_k), "--limit-artifacts", "0"], "reranker pass")
        run_cmd([sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt, "--smoke-dir", "data/reranker_smoke", "--out-dir", "data/evaluation_reranked"], "reranked evaluation")
        if "qwen3_4b_rerank" in rerankers:
            run_cmd([sys.executable, "scripts/analyze_reranker_lift.py", "--reranker", "qwen3_4b_rerank"], "Qwen vs no-reranker analysis")
    else:
        log("reranker stage skipped because reranker=none")

    run_cmd([sys.executable, "scripts/collect_vm_dashboard_artifacts.py"], "dashboard artifact refresh")
    write_run_manifest(ROOT, run_id, info, status="completed")
    log("COMPLETE PIPELINE FINISHED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("FAILED " + repr(exc))
        raise
