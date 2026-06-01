#!/usr/bin/env python3
"""Collect WNS VM benchmark artifacts for dashboard sync.

Run this on the WNS VM from the Retreiver repo root or from ~/benchmarking.
It packages operational artifacts that the dashboard reads:
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
        "model_adapter": os.getenv("MODEL_ADAPTER_URL", "http://127.0.0.1:5000").rstrip("/") + "/health",
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
