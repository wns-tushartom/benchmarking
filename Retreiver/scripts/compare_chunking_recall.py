#!/usr/bin/env python3
"""Compare chunking methods by recall with local deterministic retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.benchmark_pipeline import (  # noqa: E402
    CANDIDATE_CHUNKING_METHODS,
    OFFICIAL_CHUNKING_METHODS,
    load_query_cases,
    run_one_config,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-model", default="jina_v3")
    parser.add_argument("--vector-database", default="Qdrant")
    parser.add_argument("--reranking-model", default="bge-reranker-base")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    queries = load_query_cases(root, limit=args.limit)
    methods = OFFICIAL_CHUNKING_METHODS + CANDIDATE_CHUNKING_METHODS

    details = []
    summaries = []
    for idx, method in enumerate(methods, 1):
        print(f"[{idx}/{len(methods)}] {method}")
        config = {
            "benchmark_run_id": f"chunk_{idx:02d}",
            "chunking_method": method,
            "embedding_model": args.embedding_model,
            "vector_database": args.vector_database,
            "index_type": "HNSW",
            "retrieval_method": "Cosine Similarity",
            "reranking_model": args.reranking_model,
        }
        detail, summary = run_one_config(root, config, queries, top_k=args.top_k)
        details.extend(detail)
        summaries.append(summary)

    out_dir = root / "data"
    write_csv(out_dir / "chunking_recall_results.csv", details)
    write_csv(out_dir / "chunking_recall_summary.csv", summaries)

    ranked = sorted(summaries, key=lambda r: (float(r["recall_at_5"]), float(r["recall_at_10"])), reverse=True)
    print("\nChunking recall ranking:")
    for r in ranked:
        print(
            f"{r['chunking_method']}: R@5={r['recall_at_5']} R@10={r['recall_at_10']} "
            f"chunks={r['chunk_count']} avg_ms={r['avg_query_latency_ms']}"
        )

    candidates = [r for r in ranked if r["chunking_method"] in CANDIDATE_CHUNKING_METHODS]
    if candidates:
        print(f"\nCandidate winner: {candidates[0]['chunking_method']}")


if __name__ == "__main__":
    main()
