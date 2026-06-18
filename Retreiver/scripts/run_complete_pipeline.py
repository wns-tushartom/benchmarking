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
LIVE_RERANKERS = ["bge-reranker-base", "qwen3_4b_rerank"]
ARCHIVE_DIRS = [
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


def choose(values: list[str] | None, allowed: list[str]) -> list[str]:
    raw = [v for v in (values or []) if v]
    if not raw or "all" in raw:
        return allowed
    return [v for v in raw if v in allowed]


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


def extraction_audit_issues() -> list[str]:
    if not PDF_AUDIT_PATH.exists():
        return ["data/pdf_extraction_audit.csv missing. Run scripts/run_chunking_pipeline.py --mode all with MinerU/layout extraction before live benchmark ingestion."]
    rows = read_csv_rows(PDF_AUDIT_PATH)
    if not rows:
        return ["data/pdf_extraction_audit.csv is empty. Extraction audit evidence is required before live benchmark ingestion."]
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


def chunking_gap(sheets: list[str]) -> tuple[list[str], list[str], bool]:
    available = workbook_sheet_names(WORKBOOK_PATH)
    missing = [sheet for sheet in sheets if sheet not in available]
    can_rebuild = BENCHMARK_INPUT_PATH.exists()
    return available, missing, can_rebuild


def ensure_chunking_workbook(sheets: list[str]) -> None:
    available, missing, can_rebuild = chunking_gap(sheets)
    if not missing:
        return
    if not can_rebuild:
        raise RuntimeError(
            "Selected chunker sheet(s) are missing and data/benchmark_input.csv is unavailable: "
            + ", ".join(missing)
        )
    log(
        "chunking workbook missing selected sheet(s): "
        + ", ".join(missing)
        + "; rebuilding from data/benchmark_input.csv. Available before rebuild: "
        + ", ".join(available or ["none"])
    )
    run_cmd([sys.executable, "scripts/run_chunking_pipeline.py", "--mode", "chunk-only"], "chunking workbook rebuild")
    available_after, missing_after, _ = chunking_gap(sheets)
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


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    load_env()
    options = cfg_options()
    sheets = choose(args.sheets, options["sheets"])
    embeddings = choose(args.embeddings, [e for e in options["embeddings"] if e in LIVE_EMBEDDINGS])
    stores = choose(args.stores, options["stores"])
    if args.rerankers and "none" in args.rerankers:
        rerankers: list[str] = []
    else:
        rerankers = choose(args.rerankers, LIVE_RERANKERS)
    gt = find_groundtruth(args.groundtruth)
    missing: list[str] = []
    warnings: list[str] = []

    available_sheets, missing_sheets, can_rebuild_chunking = chunking_gap(sheets)
    if not WORKBOOK_PATH.exists() and not can_rebuild_chunking:
        missing.append("Missing data/chunking_methods_output_v2.xlsx and data/benchmark_input.csv. Run PDF extraction/chunking first.")
    elif missing_sheets:
        message = (
            "Selected chunker sheet(s) are missing from data/chunking_methods_output_v2.xlsx: "
            + ", ".join(missing_sheets)
            + ". Available sheets: "
            + ", ".join(available_sheets or ["none"])
        )
        if can_rebuild_chunking:
            warnings.append(message + ". Full run will rebuild chunking from data/benchmark_input.csv before ingestion.")
        else:
            missing.append(message + ". Run PDF extraction/chunking first.")
    if not gt:
        missing.append("Missing ground-truth CSV/XLSX under data/groundtruth/ or in the selected groundtruth path.")
    audit_issues = extraction_audit_issues()
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

    unsupported_rerankers = [r for r in (args.rerankers or []) if r not in {"all", "none", *LIVE_RERANKERS}]
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

    combo_count = len(sheets) * len(embeddings) * len(stores)
    return {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "service_checks": service_checks,
        "groundtruth": str(gt.relative_to(ROOT)) if gt and gt.is_relative_to(ROOT) else str(gt or ""),
        "sheets": sheets,
        "embeddings": embeddings,
        "stores": stores,
        "rerankers": rerankers,
        "combo_count": combo_count,
        "query_limit": args.query_limit,
        "chunk_limit": args.chunk_limit,
        "top_k": args.top_k,
        "fresh_run": args.fresh_run,
        "allow_partial_extraction": args.allow_partial_extraction,
    }


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
    parser.add_argument("--chunk-limit", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--allow-partial-extraction", action="store_true", help="Dangerous/debug only: allow live pipeline to run with missing/text-only/partial extraction audit rows.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    info = preflight(args)
    if args.preflight_only:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return 0 if info["ok"] else 2
    if not info["ok"]:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log("COMPLETE PIPELINE START")
    log(json.dumps(info, ensure_ascii=False))
    if info["warnings"]:
        for warning in info["warnings"]:
            log("WARNING " + warning)
    archive_fresh_dirs(run_id)

    sheets = info["sheets"]
    embeddings = info["embeddings"]
    stores = info["stores"]
    rerankers = info["rerankers"]
    gt = str(ROOT / info["groundtruth"] if not Path(info["groundtruth"]).is_absolute() else Path(info["groundtruth"]))
    max_combos = str(info["combo_count"])

    ensure_chunking_workbook(sheets)

    ingest_cmd = [sys.executable, "scripts/run_long_db_ingestion.py", "--run-id", run_id, "--skip-existing-store-success", "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores]
    if args.chunk_limit:
        ingest_cmd += ["--limit", str(args.chunk_limit)]
    run_cmd(ingest_cmd, "ingestion")

    retrieval_cmd = [sys.executable, "scripts/run_retrieval_smoke_from_vm_dbs.py", "--run-id", run_id, "--queries-file", gt, "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores, "--top-k", str(args.top_k), "--max-combos", max_combos]
    if args.query_limit:
        retrieval_cmd += ["--query-limit", str(args.query_limit)]
    run_cmd(retrieval_cmd, "ground-truth retrieval")

    run_cmd([sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt, "--smoke-dir", "data/retrieval_smoke", "--out-dir", "data/evaluation"], "no-reranker evaluation")

    if rerankers:
        run_cmd([sys.executable, "scripts/run_reranker_smoke_from_retrieval.py", "--rerankers", *rerankers, "--top-k", str(args.top_k), "--limit-artifacts", "0"], "reranker pass")
        run_cmd([sys.executable, "scripts/evaluate_retrieval_groundtruth.py", "--groundtruth", gt, "--smoke-dir", "data/reranker_smoke", "--out-dir", "data/evaluation_reranked"], "reranked evaluation")
        if "qwen3_4b_rerank" in rerankers:
            run_cmd([sys.executable, "scripts/analyze_reranker_lift.py", "--reranker", "qwen3_4b_rerank"], "Qwen vs no-reranker analysis")
    else:
        log("reranker stage skipped because reranker=none")

    run_cmd([sys.executable, "scripts/collect_vm_dashboard_artifacts.py"], "dashboard artifact refresh")
    log("COMPLETE PIPELINE FINISHED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("FAILED " + repr(exc))
        raise
