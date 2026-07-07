#!/usr/bin/env python3
"""Collect WNS VM benchmark artifacts for dashboard sync.

Run this on the WNS VM from the Retreiver repo root or from ~/benchmarking.
It packages operational artifacts that the dashboard reads:
- data/modular_runs/*/modular_summary.csv
- data/full_benchmark/benchmark_summary.csv
- data/evaluation*/ ground-truth and reranked metric files
- data/db_ingestion_runs/*/summary.csv
- data/reranker_smoke/*.json
- data/retrieval_smoke/*.json
- data/embedding_cache/*_meta.json
- progress/timing markdown files

Output: ~/wns_dashboard_artifacts_<timestamp>.zip
"""
from __future__ import annotations

import csv
import json
import os
import socket
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "Retreiver", cwd / "benchmarking" / "Retreiver", Path.home() / "benchmarking" / "Retreiver"]
    for c in candidates:
        if (c / "scripts").exists() and (c / "data").exists():
            return c
    raise SystemExit("Could not find Retreiver root. cd into ~/benchmarking/Retreiver or pass artifacts manually.")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_url(name: str, url: str, timeout: float = 4.0) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read(300).decode("utf-8", errors="replace")
        return {"name": name, "url": url, "ok": True, "status": resp.status, "latency_ms": round((time.perf_counter() - start) * 1000, 2), "sample": body[:160]}
    except Exception as exc:
        return {"name": name, "url": url, "ok": False, "error": repr(exc), "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


def model_adapter_health_url() -> str:
    """Return a health URL even if env accidentally points at an endpoint path.

    VM shells often export MODEL_ADAPTER_URL as the service base URL, but some
    runs only have GTE/Jina/reranker endpoint URLs. Normalize known adapter
    paths back to the FastAPI base so the dashboard does not check
    /embed/gte/health and report a misleading 404.
    """
    known_suffixes = ("/health", "/embed/gte", "/embed/jina", "/rerank/bge", "/rerank/qwen")
    candidates = [
        os.getenv("MODEL_ADAPTER_URL", ""),
        os.getenv("GTE_EMBEDDING_URL", ""),
        os.getenv("JINA_EMBEDDING_URL", ""),
        os.getenv("BGE_RERANK_URL", ""),
        os.getenv("QWEN_RERANK_URL", ""),
        "http://127.0.0.1:5000",
    ]
    for raw in candidates:
        url = (raw or "").strip().rstrip("/")
        if not url:
            continue
        for suffix in known_suffixes:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        return url.rstrip("/") + "/health"
    return "http://127.0.0.1:5000/health"


def build_snapshot(root: Path) -> dict[str, Any]:
    ingestion_rows: list[dict[str, str]] = []
    for path in sorted((root / "data" / "db_ingestion_runs").glob("*/summary.csv")):
        run_id = path.parent.name
        for row in read_csv(path):
            row["run_id"] = run_id
            ingestion_rows.append(row)
    ok_rows = [r for r in ingestion_rows if r.get("status") == "ok"]
    combos = {(r.get("sheet"), r.get("embedding"), r.get("store")) for r in ok_rows}
    service_urls = {
        "model_adapter": model_adapter_health_url(),
        "qdrant": os.getenv("QDRANT_URL", "http://127.0.0.1:5001").rstrip("/") + "/healthz",
        "weaviate": os.getenv("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/") + "/v1/.well-known/ready",
    }
    health = [check_url(name, url) for name, url in service_urls.items()]
    return {
        "created_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "root": str(root),
        "ingestion": {
            "row_count": len(ingestion_rows),
            "ok_rows": len(ok_rows),
            "failure_rows": len(ingestion_rows) - len(ok_rows),
            "combo_count": len(combos),
            "chunkers": sorted({r.get("sheet", "") for r in ok_rows if r.get("sheet")}),
            "embeddings": sorted({r.get("embedding", "") for r in ok_rows if r.get("embedding")}),
            "stores": sorted({r.get("store", "") for r in ok_rows if r.get("store")}),
            "latest_run_id": ingestion_rows[-1].get("run_id") if ingestion_rows else None,
        },
        "health": health,
    }


def main() -> int:
    root = find_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = root / "data" / "vm_dashboard_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(build_snapshot(root), indent=2), encoding="utf-8")

    out = Path.home() / f"wns_dashboard_artifacts_{stamp}.zip"
    patterns = [
        "data/modular_runs/**/modular_summary.csv",
        "data/modular_runs/**/analysis.json",
        "data/modular_runs/**/status.json",
        "data/full_benchmark/benchmark_summary.csv",
        "data/full_benchmark/benchmark_report.json",
        "data/evaluation/*.csv",
        "data/evaluation/*.json",
        "data/evaluation_reranked/*.csv",
        "data/evaluation_reranked/*.json",
        "data/reranker_analysis/*.csv",
        "data/reranker_analysis/*.json",
        "data/hallucination/*.csv",
        "data/hallucination/*.json",
        "data/nvidia_rag/*.csv",
        "data/nvidia_rag/*.json",
        "data/db_ingestion_runs/*/summary.csv",
        "data/db_ingestion_runs/*/status.json",
        "data/db_ingestion_runs/*/config.json",
        "data/reranker_smoke/*.json",
        "data/retrieval_smoke/*.json",
        "data/embedding_cache/*_meta.json",
        "data/vm_dashboard_snapshot.json",
        "WNS_VM_PROGRESS_*.md",
        "JINA_TIMINGS_*.md",
        "RERANKER_SMOKE_*.md",
    ]
    added = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for pat in patterns:
            for path in root.glob(pat):
                if path.is_file():
                    zf.write(path, path.relative_to(root))
                    added += 1
    print(f"root={root}")
    print(f"snapshot={snapshot_path}")
    print(f"zip={out}")
    print(f"files_added={added}")
    print("If using Jupyter, download this zip from your home directory and upload/copy it to the dashboard machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
