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
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.core.config import load_benchmark_config

CONFIG_PATH = ROOT / "configs" / "benchmark.local.json"
WORKBOOK_PATH = ROOT / "data" / "chunking_methods_output_v2.xlsx"
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
    for name in [".env", ".env.project-smiley-nvidia", ".env.vm.generated"]:
        path = ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


def url_ok(url: str, method: str = "GET") -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 500, f"HTTP {resp.status}"
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

    if not WORKBOOK_PATH.exists():
        missing.append("Missing data/chunking_methods_output_v2.xlsx. Run chunking first.")
    if not gt:
        missing.append("Missing ground-truth CSV/XLSX under data/groundtruth/ or in the selected groundtruth path.")
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

    service_checks: list[dict[str, Any]] = []
    model_adapter = os.getenv("MODEL_ADAPTER_URL", "http://127.0.0.1:5000").rstrip("/")
    if model_adapter.endswith(("/embed/gte", "/embed/jina", "/rerank/bge", "/rerank/qwen")):
        model_adapter = model_adapter.rsplit("/", 2)[0]
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
    if "PGVector" in stores and not (os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")):
        missing.append("PGVector selected but PGVECTOR_DSN or DATABASE_URL is not set.")

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
        log("fresh run requested; no previous run artifacts to archive")


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
    if args.fresh_run:
        archive_fresh_dirs(run_id)

    sheets = info["sheets"]
    embeddings = info["embeddings"]
    stores = info["stores"]
    rerankers = info["rerankers"]
    gt = str(ROOT / info["groundtruth"] if not Path(info["groundtruth"]).is_absolute() else Path(info["groundtruth"]))
    max_combos = str(info["combo_count"])

    ingest_cmd = [sys.executable, "scripts/run_long_db_ingestion.py", "--skip-existing-store-success", "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores]
    if args.chunk_limit:
        ingest_cmd += ["--limit", str(args.chunk_limit)]
    run_cmd(ingest_cmd, "ingestion")

    retrieval_cmd = [sys.executable, "scripts/run_retrieval_smoke_from_vm_dbs.py", "--queries-file", gt, "--sheets", *sheets, "--embeddings", *embeddings, "--stores", *stores, "--top-k", str(args.top_k), "--max-combos", max_combos]
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
