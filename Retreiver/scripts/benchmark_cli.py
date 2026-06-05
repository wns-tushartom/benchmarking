#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.core.config import generate_matrix, load_benchmark_config
from benchmarking.core.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="WNS modular benchmark CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("config", nargs="?", default="configs/benchmark.local.json")

    matrix = sub.add_parser("matrix")
    matrix.add_argument("config", nargs="?", default="configs/benchmark.local.json")

    run = sub.add_parser("run")
    run.add_argument("config", nargs="?", default="configs/benchmark.local.json")
    run.add_argument("--output-dir", default="data/modular_runs/latest")
    run.add_argument("--max-runs", type=int, default=0)
    run.add_argument("--limit-queries", type=int, default=0)
    run.add_argument("--chunker", default="all")
    run.add_argument("--embedding", default="all")
    run.add_argument("--vector-store", default="all")
    run.add_argument("--index-type", default="all")
    run.add_argument("--retrieval-method", default="all")
    run.add_argument("--reranker", default="all")

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config

    if args.cmd == "validate":
        cfg = load_benchmark_config(config_path)
        rows = generate_matrix(cfg)
        print(json.dumps({"ok": True, "experiment": cfg["experiment"]["name"], "matrix_count": len(rows)}, indent=2))
    elif args.cmd == "matrix":
        cfg = load_benchmark_config(config_path)
        print(json.dumps(generate_matrix(cfg), indent=2))
    elif args.cmd == "run":
        selections = {
            "chunker": args.chunker,
            "embedding": args.embedding,
            "vector_store": args.vector_store,
            "index_type": args.index_type,
            "retrieval_method": args.retrieval_method,
            "reranker": args.reranker,
        }
        analysis = run_experiment(
            config_path,
            root,
            root / args.output_dir,
            max_runs=args.max_runs,
            limit_queries=args.limit_queries,
            selections=selections,
        )
        print(json.dumps({"ok": True, "best_config": analysis.get("best_config", {}), "report": str(root / args.output_dir / "MODULAR_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
