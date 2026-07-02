#!/usr/bin/env python3
"""Run the WNS full benchmark matrix in local/offline mode.

This script executes every requested combination using deterministic local adapters so the
pipeline, outputs, frontend, and recall math work even before paid APIs or DB containers
are configured.

Official default matrix:
5 chunking × 3 embeddings × 4 vector stores × 3 rerankers = 180 runs.

Use --include-candidates to include semantic_split.
Use --include-milvus to include Milvus as an extra baseline DB.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.benchmark_pipeline import (  # noqa: E402
    CANDIDATE_CHUNKING_METHODS,
    EMBEDDING_MODELS,
    INDEX_TYPE,
    OFFICIAL_CHUNKING_METHODS,
    RERANKING_MODELS,
    RETRIEVAL_METHOD,
    VECTOR_DATABASES,
    DeterministicEmbedder,
    LocalVectorIndex,
    Reranker,
    SearchHit,
    load_chunks_from_workbook,
    load_query_cases,
    make_matrix,
    percentile,
    recall_at_k,
    text_overlap_score,
    write_csv,
    write_json,
)


def avg(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_config(
    config: Dict[str, str],
    query_count: int,
    chunk_count: int,
    embedding_dimension: int,
    embedding_latency_s: float,
    upsert_latency_s: float,
    latencies: List[float],
    rerank_latencies: List[float],
    recalls: Dict[int, List[float]],
) -> Dict[str, Any]:
    return {
        **config,
        "mode": "legacy_local_reference",
        "query_count": query_count,
        "chunk_count": chunk_count,
        "embedding_dimension": embedding_dimension,
        "embedding_latency_s": round(embedding_latency_s, 6),
        "upsert_latency_s": round(upsert_latency_s, 6),
        "avg_query_latency_ms": round(avg(latencies) * 1000, 6),
        "p50_query_latency_ms": round(percentile(latencies, 0.50) * 1000, 6),
        "p95_query_latency_ms": round(percentile(latencies, 0.95) * 1000, 6),
        "p99_query_latency_ms": round(percentile(latencies, 0.99) * 1000, 6),
        "avg_rerank_latency_ms": round(avg(rerank_latencies) * 1000, 6),
        "recall_at_1": round(avg(recalls[1]), 6),
        "recall_at_3": round(avg(recalls[3]), 6),
        "recall_at_5": round(avg(recalls[5]), 6),
        "recall_at_10": round(avg(recalls[10]), 6),
        "errors": "",
        "notes": "Offline comparable benchmark, useful for pipeline QA and relative chunk/rerank signal, not final provider latency.",
    }


def evaluate_hits(
    config: Dict[str, str],
    queries: list,
    base_hits_by_query: List[Tuple[List[SearchHit], float]],
    reranker: Reranker,
    chunk_count: int,
    top_k: int,
    overlap_threshold: float = 0.22,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[float]], List[float], List[float]]:
    detail_rows: List[Dict[str, Any]] = []
    recalls = {1: [], 3: [], 5: [], 10: []}
    latencies: List[float] = []
    rerank_latencies: List[float] = []

    for q, (base_hits, search_latency) in zip(queries, base_hits_by_query):
        hits, rerank_latency = reranker.rerank(q.query, list(base_hits))
        hits = hits[:top_k]
        hit_flags = [text_overlap_score(hit.chunk.paragraph, q.expected_text) >= overlap_threshold for hit in hits]
        for k in recalls:
            recalls[k].append(recall_at_k(hit_flags, k))
        latencies.append(search_latency)
        rerank_latencies.append(rerank_latency)
        detail_rows.append({
            **config,
            "query_id": q.id,
            "query": q.query,
            "chunk_count": chunk_count,
            "top_k": top_k,
            "retrieved_ids": "|".join(str(h.chunk.id) for h in hits),
            "retrieved_scores": "|".join(f"{h.score:.4f}" for h in hits),
            "hit_at_1": int(recall_at_k(hit_flags, 1)),
            "hit_at_3": int(recall_at_k(hit_flags, 3)),
            "hit_at_5": int(recall_at_k(hit_flags, 5)),
            "hit_at_10": int(recall_at_k(hit_flags, 10)),
            "search_latency_ms": round(search_latency * 1000, 4),
            "rerank_latency_ms": round(rerank_latency * 1000, 4),
            "mode": "legacy_local_reference",
            "notes": "Local deterministic embeddings/vector adapters used. Replace with provider adapters for production numbers.",
        })
    return detail_rows, recalls, latencies, rerank_latencies


def run_fast_local_matrix(root: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    queries = load_query_cases(root, limit=args.limit)
    chunking_methods = OFFICIAL_CHUNKING_METHODS + (CANDIDATE_CHUNKING_METHODS if args.include_candidates else [])
    dbs = VECTOR_DATABASES + (["Milvus"] if args.include_milvus else [])
    matrix = make_matrix(include_candidates=args.include_candidates, include_milvus=args.include_milvus)
    if args.max_runs > 0:
        matrix = matrix[: args.max_runs]
        allowed_ids = {m["benchmark_run_id"] for m in matrix}
    else:
        allowed_ids = None

    out_dir = root / args.output_dir
    workbook = root / "data" / "chunking_methods_output_v2.xlsx"
    all_detail: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    started = time.perf_counter()
    run_id = 1

    print(f"Loaded {len(queries)} query cases")
    print(f"Running {len(matrix)} benchmark configurations")

    for chunking in chunking_methods:
        chunks = load_chunks_from_workbook(workbook, chunking)
        for embedding_model in EMBEDDING_MODELS:
            embedder = DeterministicEmbedder(embedding_model)
            embed_start = time.perf_counter()
            chunk_vectors = embedder.embed_many([c.paragraph for c in chunks])
            query_vectors = embedder.embed_many([q.query for q in queries])
            embedding_latency_s = time.perf_counter() - embed_start

            # In local/offline fallback mode the DB labels use the same in-process vector
            # adapter. Search once per chunking+embedding pair, then reuse identical base
            # hits across Qdrant/PGVector/Weaviate/FAISS labels. Provider-mode adapters should
            # benchmark each real DB independently; this optimization is only for local QA.
            shared_index = LocalVectorIndex("local_shared", INDEX_TYPE)
            upsert_latency_s = shared_index.upsert(chunks, chunk_vectors)
            base_hits_by_query = [shared_index.search(qv, top_k=max(args.top_k, 10)) for qv in query_vectors]

            for db in dbs:
                for rerank_model in RERANKING_MODELS:
                    config = {
                        "benchmark_run_id": f"run_{run_id:04d}",
                        "chunking_method": chunking,
                        "embedding_model": embedding_model,
                        "vector_database": db,
                        "index_type": INDEX_TYPE,
                        "retrieval_method": RETRIEVAL_METHOD,
                        "reranking_model": rerank_model,
                    }
                    run_id += 1
                    if allowed_ids is not None and config["benchmark_run_id"] not in allowed_ids:
                        continue

                    label = f"{config['chunking_method']} | {config['embedding_model']} | {config['vector_database']} | {config['reranking_model']}"
                    print(f"[{len(all_summary) + 1}/{len(matrix)}] {label}")
                    detail, recalls, latencies, rerank_latencies = evaluate_hits(
                        config, queries, base_hits_by_query, Reranker(rerank_model), len(chunks), args.top_k
                    )
                    summary = summarize_config(
                        config,
                        len(queries),
                        len(chunks),
                        embedder.dimensions,
                        embedding_latency_s,
                        upsert_latency_s,
                        latencies,
                        rerank_latencies,
                        recalls,
                    )
                    all_detail.extend(detail)
                    all_summary.append(summary)

                    # Status is cheap and useful during long runs. CSVs are written at the end
                    # because rewriting the growing detail file on every configuration is O(n²)
                    # and makes all-query runs unnecessarily slow.
                    write_json(out_dir / "latest_status.json", {
                        "completed_runs": len(all_summary),
                        "total_runs": len(matrix),
                        "query_count": len(queries),
                        "elapsed_s": round(time.perf_counter() - started, 3),
                        "last_config": config,
                    })

    elapsed = time.perf_counter() - started
    return all_detail, all_summary, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Max query cases. 0 means all.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--include-milvus", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Debug cap on matrix runs. 0 means all.")
    parser.add_argument("--output-dir", default="data/full_benchmark")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detail, all_summary, elapsed = run_fast_local_matrix(root, args)
    ranked = sorted(all_summary, key=lambda r: (float(r.get("recall_at_5") or 0), -float(r.get("avg_query_latency_ms") or 0)), reverse=True)
    write_json(out_dir / "benchmark_report.json", {
        "mode": "legacy_local_reference",
        "total_runs": len(all_summary),
        "query_count": int(all_summary[0]["query_count"]) if all_summary else 0,
        "elapsed_s": round(elapsed, 3),
        "best_by_recall_at_5": ranked[:10],
        "warning": "Local deterministic adapters were used. Use provider/native adapters before treating latency as production truth.",
    })
    write_csv(out_dir / "benchmark_summary.csv", all_summary)
    write_csv(out_dir / "benchmark_details.csv", all_detail)
    print(f"Done in {elapsed:.2f}s")
    print(f"Summary: {out_dir / 'benchmark_summary.csv'}")
    print(f"Details: {out_dir / 'benchmark_details.csv'}")
    if ranked:
        print("Best Recall@5:", ranked[0]["benchmark_run_id"], ranked[0]["recall_at_5"])


if __name__ == "__main__":
    main()
