from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from benchmarking.adapters.local import load_query_cases
from benchmarking.core.config import config_hash, dataset_hash, generate_matrix, load_benchmark_config, selected_config, technique
from benchmarking.core.metrics import bootstrap_ci, mean, mrr, ndcg_at_k, precision_at_k, recall_at_k
from benchmarking.core.registry import default_registry
from scripts.wns_env import load_env_files


def load_env_file(root: Path) -> None:
    load_env_files(root)


def run_experiment(
    config_path: Path,
    root: Path,
    output_dir: Path,
    max_runs: int = 0,
    limit_queries: int = 0,
    selections: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    load_env_file(root)
    official_cfg = load_benchmark_config(config_path)
    cfg = official_cfg
    run_selection: Dict[str, str] = {}
    if selections:
        cfg = selected_config(cfg, selections)
        run_selection = {k: v for k, v in selections.items() if v and v != "all"}
        cfg.setdefault("experiment", {})["selection"] = run_selection
    registry = default_registry()
    matrix = generate_matrix(cfg)
    if max_runs:
        matrix = matrix[:max_runs]
    exp = cfg["experiment"]
    queries = load_query_cases(root, exp["dataset"])
    if limit_queries:
        queries = queries[:limit_queries]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        cfg,
        config_path,
        root,
        len(queries),
        len(matrix),
        official_config=official_cfg,
        selection=run_selection,
    )
    manifest.update({
        "status": "running",
        "run_id": output_dir.name,
        "dataset_id": "dataset:wns-default",
        "groundtruth_id": "groundtruth:repository:qa_text_test.csv",
    })
    write_json_atomic(output_dir / "manifest.json", manifest)

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    top_k = int(exp.get("top_k", 10))
    eval_cfg = cfg.get("evaluation", {})

    chunk_cache: Dict[str, Any] = {}
    embedding_cache: Dict[Tuple[str, str], Any] = {}

    for idx, row in enumerate(matrix, 1):
        combo_label = f"[{idx}/{len(matrix)}] {row['chunker']} | {row['embedding']} | {row['vector_store']} | {row['reranker']}"
        print(f"START {combo_label}", flush=True)
        chunker_cfg = technique(cfg, "chunkers", row["chunker"])
        embedding_cfg = technique(cfg, "embeddings", row["embedding"])
        vector_cfg = technique(cfg, "vector_stores", row["vector_store"])
        reranker_cfg = technique(cfg, "rerankers", row["reranker"])

        chunk_key = row["chunker"]
        if chunk_key not in chunk_cache:
            print(f"  chunking start: {row['chunker']}", flush=True)
            chunk_cls = registry.get("chunker", chunker_cfg["adapter"])
            chunk_params = {k: v for k, v in chunker_cfg.items() if k not in {"adapter", "sheet_name"}}
            chunk_cache[chunk_key] = chunk_cls(root=root, sheet_name=chunker_cfg.get("sheet_name", row["chunker"]), **chunk_params).chunk()
        chunks = chunk_cache[chunk_key]
        print(f"  chunks ready: {len(chunks)}", flush=True)

        embed_key = (row["chunker"], row["embedding"])
        if embed_key not in embedding_cache:
            print(f"  embedding start: {row['embedding']} chunks={len(chunks)} queries={len(queries)}", flush=True)
            embed_cls = registry.get("embedding", embedding_cfg["adapter"])
            embedder = embed_cls(model_name=row["embedding"], **embedding_cfg)
            embed_start = time.perf_counter()
            chunk_vectors = embedder.embed_many([c.paragraph for c in chunks])
            query_vectors = embedder.embed_many([q.query for q in queries])
            embedding_cache[embed_key] = (embedder, chunk_vectors, query_vectors, time.perf_counter() - embed_start)
            print(f"  embedding done: {row['embedding']} seconds={embedding_cache[embed_key][3]:.2f}", flush=True)
        else:
            print(f"  embedding cached: {row['embedding']}", flush=True)
        embedder, chunk_vectors, query_vectors, embedding_latency_s = embedding_cache[embed_key]

        print(f"  upsert start: {row['vector_store']} vectors={len(chunk_vectors)}", flush=True)
        vector_cls = registry.get("vector_store", vector_cfg["adapter"])
        store = vector_cls(name=row["vector_store"], **vector_cfg)
        upsert_metrics = store.upsert(chunks, chunk_vectors)
        print(f"  upsert done: {row['vector_store']} seconds={float(upsert_metrics.get('upsert_latency_s', 0)):.2f}", flush=True)
        reranker_cls = registry.get("reranker", reranker_cfg["adapter"])
        reranker = reranker_cls(name=row["reranker"], **reranker_cfg)
        evaluator_cls = registry.get("evaluator", row["evaluator"])
        evaluator = evaluator_cls(overlap_threshold=eval_cfg.get("overlap_threshold", 0.22))

        metric_lists = {"recall_at_1": [], "recall_at_3": [], "recall_at_5": [], "recall_at_10": [], "mrr": [], "ndcg_at_5": [], "ndcg_at_10": [], "precision_at_5": []}
        first_relevant_ranks: List[int] = []
        no_hit_queries = 0
        latencies = []
        category_stats: Dict[str, List[float]] = {}
        examples_missed = []
        examples_hit = []

        print(f"  queries start: {len(queries)} using {row['reranker']}", flush=True)
        for query_idx, (q, qv) in enumerate(zip(queries, query_vectors), 1):
            if query_idx == 1 or query_idx % 5 == 0 or query_idx == len(queries):
                print(f"    query {query_idx}/{len(queries)}: {q.id}", flush=True)
            search_start = time.perf_counter()
            base_hits = store.search(qv, top_k=max(top_k, 20))
            reranked = reranker.rerank(q.query, base_hits, top_k=top_k)
            latency = time.perf_counter() - search_start
            flags = evaluator.flags(q, reranked)
            first_relevant = next((rank for rank, relevant in enumerate(flags, 1) if relevant), None)
            if first_relevant is None:
                no_hit_queries += 1
            else:
                first_relevant_ranks.append(first_relevant)
            values = {
                "recall_at_1": recall_at_k(flags, 1),
                "recall_at_3": recall_at_k(flags, 3),
                "recall_at_5": recall_at_k(flags, 5),
                "recall_at_10": recall_at_k(flags, 10),
                "mrr": mrr(flags),
                "ndcg_at_5": ndcg_at_k(flags, 5),
                "ndcg_at_10": ndcg_at_k(flags, 10),
                "precision_at_5": precision_at_k(flags, 5),
            }
            for k, v in values.items():
                metric_lists[k].append(v)
            category_stats.setdefault(q.category, []).append(values["recall_at_5"])
            latencies.append(latency)
            if values["recall_at_5"] and len(examples_hit) < 5:
                examples_hit.append({"query_id": q.id, "query": q.query, "category": q.category, "top_ids": [h.chunk.id for h in reranked[:5]]})
            if not values["recall_at_5"] and len(examples_missed) < 10:
                examples_missed.append({"query_id": q.id, "query": q.query, "category": q.category, "top_ids": [h.chunk.id for h in reranked[:5]]})
            detail_rows.append({
                **row,
                "query_id": q.id,
                "query": q.query,
                "category": q.category,
                "top_ids": "|".join(str(h.chunk.id) for h in reranked),
                "top_scores": "|".join(f"{h.score:.4f}" for h in reranked),
                **values,
                "latency_ms": round(latency * 1000, 4),
            })

        lo, hi = bootstrap_ci(metric_lists["recall_at_5"], iterations=int(eval_cfg.get("bootstrap_iterations", 200)), seed=int(exp.get("random_seed", 42)))
        summary_rows.append({
            **row,
            "mode": exp.get("mode", "vm_remote_required"),
            "query_count": len(queries),
            "chunk_count": len(chunks),
            "embedding_dimension": getattr(embedder, "dimensions", 0),
            "embedding_latency_s": round(embedding_latency_s, 6),
            "upsert_latency_s": round(float(upsert_metrics.get("upsert_latency_s", 0)), 6),
            "avg_latency_ms": round(mean(latencies) * 1000, 6),
            "avg_latency_seconds": round(mean(latencies), 6),
            "avg_first_relevant_rank": round(mean(first_relevant_ranks), 6) if first_relevant_ranks else 0.0,
            "no_hit_queries": no_hit_queries,
            "status": "completed",
            **{k: round(mean(v), 6) for k, v in metric_lists.items()},
            "recall_at_5_ci_low": round(lo, 6),
            "recall_at_5_ci_high": round(hi, 6),
            "category_recall_at_5": json.dumps({k: round(mean(v), 6) for k, v in sorted(category_stats.items())}, ensure_ascii=False),
            "hit_examples": json.dumps(examples_hit, ensure_ascii=False),
            "miss_examples": json.dumps(examples_missed, ensure_ascii=False),
            "errors": "",
        })
        write_csv(output_dir / "modular_summary.csv", summary_rows)
        write_csv(output_dir / "modular_details.csv", detail_rows)
        (output_dir / "status.json").write_text(json.dumps({"completed_runs": idx, "total_runs": len(matrix), "elapsed_s": round(time.perf_counter() - started, 3)}, indent=2), encoding="utf-8")

    manifest["status"] = "completed"
    analysis = analyze(summary_rows, detail_rows, manifest)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "MODULAR_REPORT.md").write_text(render_markdown_report(analysis), encoding="utf-8")
    try:
        artifact_root = output_dir.resolve().relative_to((root / "data" / "modular_runs").resolve()).as_posix()
    except ValueError:
        artifact_root = output_dir.resolve().as_posix()
    manifest["artifact_root"] = artifact_root
    manifest["artifact_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in (
            "modular_summary.csv",
            "modular_details.csv",
            "analysis.json",
            "MODULAR_REPORT.md",
        )
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return analysis


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_manifest(
    cfg: Dict[str, Any],
    config_path: Path,
    root: Path,
    query_count: int,
    matrix_count: int,
    *,
    official_config: Dict[str, Any] | None = None,
    selection: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    dataset = root / cfg["experiment"]["dataset"]
    corpus = root / cfg["experiment"].get("corpus_workbook", "")
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = "unknown"
    selected_hash = config_hash(cfg)
    contract_hash = config_hash(official_config if official_config is not None else cfg)
    return {
        "experiment": cfg["experiment"],
        "config_path": str(config_path),
        "config_hash": selected_hash,
        "official_matrix_contract_hash": contract_hash,
        "selected_run_config_hash": selected_hash,
        "selection": dict(selection or {}),
        "dataset_hash": dataset_hash([dataset, corpus]),
        "query_count": query_count,
        "matrix_count": matrix_count,
        "git_sha": git_sha,
        "python": platform.python_version(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider_readiness": provider_readiness(),
    }


def provider_readiness() -> Dict[str, bool]:
    keys = [
        "OPENAI_API_KEY",
        "JINA_EMBEDDING_URL",
        "JINA_API_KEY",
        "GTE_EMBEDDING_URL",
        "HF_TOKEN",
        "QDRANT_URL",
        "PGVECTOR_DSN",
        "DATABASE_URL",
        "WEAVIATE_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "QWEN_RERANK_URL",
        "BGE_RERANK_URL",
    ]
    return {k: bool(os.environ.get(k)) for k in keys}


def analyze(summary_rows: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Dict[str, Any]:
    ranked = sorted(summary_rows, key=lambda r: (float(r["recall_at_5"]), float(r["mrr"]), -float(r["avg_latency_ms"])), reverse=True)
    best = ranked[0] if ranked else {}
    runner_up = ranked[1] if len(ranked) > 1 else {}
    return {
        "manifest": manifest,
        "best_config": best,
        "runner_up": runner_up,
        "top10": ranked[:10],
        "why_best": why_best(best, runner_up),
        "pareto": pareto_front(summary_rows),
        "by_chunker": group_summary(summary_rows, "chunker"),
        "by_embedding": group_summary(summary_rows, "embedding"),
        "by_reranker": group_summary(summary_rows, "reranker"),
        "by_index_type": group_summary(summary_rows, "index_type"),
        "by_retrieval_method": group_summary(summary_rows, "retrieval_method"),
        "detail_count": len(detail_rows),
    }


def why_best(best: Dict[str, Any], runner_up: Dict[str, Any]) -> List[str]:
    if not best:
        return []
    reasons = [f"Highest Recall@5: {best.get('recall_at_5')}"]
    if runner_up:
        delta = float(best["recall_at_5"]) - float(runner_up["recall_at_5"])
        reasons.append(f"Recall@5 delta versus runner-up: {delta:.6f}")
        overlap = not (float(best["recall_at_5_ci_low"]) > float(runner_up["recall_at_5_ci_high"]) or float(runner_up["recall_at_5_ci_low"]) > float(best["recall_at_5_ci_high"]))
        reasons.append("Confidence interval status: statistical tie likely" if overlap else "Confidence interval status: clear separation")
    reasons.append(f"Chunk count: {best.get('chunk_count')}, avg latency: {best.get('avg_latency_ms')} ms")
    return reasons


def group_summary(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    out = []
    for name, rs in groups.items():
        out.append({
            "name": name,
            "count": len(rs),
            "recall_at_5": round(mean(float(r["recall_at_5"]) for r in rs), 6),
            "mrr": round(mean(float(r["mrr"]) for r in rs), 6),
            "avg_latency_ms": round(mean(float(r["avg_latency_ms"]) for r in rs), 6),
            "chunk_count": round(mean(float(r["chunk_count"]) for r in rs), 1),
        })
    return sorted(out, key=lambda x: (x["recall_at_5"], x["mrr"], -x["avg_latency_ms"]), reverse=True)


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front = []
    for r in rows:
        r_quality = float(r["recall_at_5"])
        r_latency = float(r["avg_latency_ms"])
        dominated = False
        for o in rows:
            if o is r:
                continue
            if float(o["recall_at_5"]) >= r_quality and float(o["avg_latency_ms"]) <= r_latency and (float(o["recall_at_5"]) > r_quality or float(o["avg_latency_ms"]) < r_latency):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (-float(r["recall_at_5"]), float(r["avg_latency_ms"])))[:25]


def render_markdown_report(analysis: Dict[str, Any]) -> str:
    best = analysis.get("best_config", {})
    lines = ["# Modular WNS Benchmark Report", "", "## Best configuration", ""]
    if best:
        lines += [
            f"- Chunker: `{best['chunker']}`",
            f"- Embedding: `{best['embedding']}`",
            f"- Vector store: `{best['vector_store']}`",
            f"- Index type: `{best.get('index_type', 'HNSW')}`",
            f"- Retrieval method: `{best.get('retrieval_method', best.get('retriever', 'Cosine Similarity'))}`",
            f"- Reranker: `{best['reranker']}`",
            f"- Recall@5: **{best['recall_at_5']}**",
            f"- MRR: **{best['mrr']}**",
            f"- nDCG@10: **{best['ndcg_at_10']}**",
            f"- Avg latency: **{best['avg_latency_ms']} ms**",
            "",
            "## Why it won",
        ]
        lines += [f"- {r}" for r in analysis.get("why_best", [])]
    lines += ["", "## Top 5", ""]
    for i, r in enumerate(analysis.get("top10", [])[:5], 1):
        lines.append(f"{i}. `{r['chunker']}` / `{r['embedding']}` / `{r['vector_store']}` / `{r.get('index_type', 'HNSW')}` / `{r.get('retrieval_method', r.get('retriever', 'Cosine Similarity'))}` / `{r['reranker']}`: R@5={r['recall_at_5']}, MRR={r['mrr']}, latency={r['avg_latency_ms']} ms")
    lines += ["", "## Pareto front", ""]
    for r in analysis.get("pareto", [])[:8]:
        lines.append(f"- `{r['benchmark_run_id']}`: R@5={r['recall_at_5']}, latency={r['avg_latency_ms']} ms, `{r['chunker']}` + `{r['embedding']}`")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
