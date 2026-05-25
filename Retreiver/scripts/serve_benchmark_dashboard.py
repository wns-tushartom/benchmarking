#!/usr/bin/env python3
"""Serve the WNS benchmark dashboard using only the Python standard library."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from benchmarking.core.config import generate_matrix, load_benchmark_config

WEB = ROOT / "web"
FULL_DIR = ROOT / "data" / "full_benchmark"
MODULAR_DIR = ROOT / "data" / "modular_runs" / "latest"
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
        "mode": cfg.get("experiment", {}).get("mode", "local_offline_fallback"),
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


def sort_summary(rows: list[dict]) -> list[dict]:
    def key(r: dict) -> tuple:
        try:
            recall = float(r.get("recall_at_5") or 0)
        except ValueError:
            recall = 0.0
        try:
            latency = float(r.get("avg_query_latency_ms") or r.get("avg_latency_ms") or 999999)
        except ValueError:
            latency = 999999.0
        return (-recall, latency)
    return sorted(rows, key=key)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
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
                "files": [f for f in files if (ROOT / f).exists()],
                "options": benchmark_options(),
            })
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
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
            else:
                self.send_json({"error": "unknown endpoint"}, 404)
                return
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
            self.send_json({"exit_code": proc.returncode, "output": proc.stdout + proc.stderr})
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"WNS benchmark dashboard: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
