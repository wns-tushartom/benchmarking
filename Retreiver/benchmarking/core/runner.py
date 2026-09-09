from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from benchmarking.adapters.local import load_query_cases
from benchmarking.core.config import config_hash, dataset_hash, generate_matrix, load_benchmark_config, selected_config, technique
from benchmarking.core.metrics import bootstrap_ci, mean, mrr, ndcg_at_k, precision_at_k, recall_at_k
from benchmarking.core.registry import default_registry
from benchmarking.retrieval import BM25Retriever, DenseCosineRetriever, HybridRRFRetriever
from scripts.wns_env import load_env_files


def load_env_file(root: Path) -> None:
    load_env_files(root)


def validate_output_directory(config: Dict[str, Any], root: Path, output_dir: Path) -> Path:
    """Keep candidate artifacts inside the exact configured candidate namespace."""
    lane = str(config.get("experiment", {}).get("output_lane", "")).strip()
    if not lane:
        return output_dir.resolve()

    lane_path = Path(lane)
    modular_runs_root = root.resolve() / "data" / "modular_runs"
    lane_root = modular_runs_root / lane
    lexical_output = Path(os.path.abspath(output_dir))
    if len(lane_path.parts) != 1 or lane_path.name in {"", ".", ".."}:
        raise ValueError(f"candidate output lane must be one directory under {modular_runs_root}; got {lane!r}")
    if modular_runs_root.is_symlink() or lane_root.is_symlink() or lexical_output.is_symlink():
        raise ValueError("candidate output directory must not contain symlinks")

    if isinstance(config.get("portfolio"), dict):
        try:
            relative = lexical_output.relative_to(lane_root)
        except ValueError as exc:
            raise ValueError(
                f"portfolio output directory must be under {lane_root}; got {lexical_output}"
            ) from exc
        if len(relative.parts) != 2:
            raise ValueError(
                f"portfolio output directory must be portfolio_id/batch_id under {lane_root}; got {lexical_output}"
            )
        portfolio_name, batch_name = relative.parts
        if not re.fullmatch(r"portfolio_[0-9a-f]{64}", portfolio_name) or not re.fullmatch(
            r"batch_[0-9a-f]{64}", batch_name
        ):
            raise ValueError("portfolio output directory has invalid portfolio or batch identity")
        portfolio_root = lane_root / portfolio_name
        if portfolio_root.is_symlink():
            raise ValueError("candidate output directory must not contain symlinks")
    else:
        if lexical_output.parent != lane_root or lexical_output.name in {"", ".", "..", "latest"}:
            raise ValueError(
                f"candidate output directory must be a direct run directory under {lane_root}; got {lexical_output}"
            )

    resolved_lane_root = lane_root.resolve()
    resolved_output = lexical_output.resolve()
    if resolved_lane_root != lane_root or resolved_output != lexical_output:
        raise ValueError("candidate output directory must not contain symlinks")
    return lexical_output


def select_declared_combinations(
    matrix: List[Dict[str, str]],
    combination_ids: Tuple[str, ...] | List[str],
    *,
    maximum: int = 250,
) -> List[Dict[str, str]]:
    """Select an exact declared batch without truncation, guessing, or reordering."""
    requested = list(combination_ids)
    if not requested:
        raise ValueError("declared combination IDs must not be empty")
    if len(requested) > maximum:
        raise ValueError(f"declared batch exceeds maximum of {maximum} combinations")
    if len(requested) != len(set(requested)):
        raise ValueError("declared batch contains duplicate combination IDs")
    by_id = {
        str(row.get("combination_id")): row
        for row in matrix
        if isinstance(row.get("combination_id"), str)
    }
    unknown = [combination_id for combination_id in requested if combination_id not in by_id]
    if unknown:
        raise ValueError(f"declared batch contains unknown combination IDs: {unknown[:3]}")
    return [by_id[combination_id] for combination_id in requested]


def vector_store_parameters(
    vector_cfg: Dict[str, Any], row: Dict[str, str], output_dir: Path
) -> Dict[str, Any]:
    """Build adapter parameters and isolate embedded TurboVec persistence."""
    params = {
        key: value
        for key, value in vector_cfg.items()
        if key not in {"adapter", "index_type"}
    }
    params["index_type"] = row.get(
        "index_type", str(vector_cfg.get("index_type", "HNSW"))
    )
    if vector_cfg.get("adapter") == "turbovec":
        identity = {
            "chunker": row["chunker"],
            "embedding": row["embedding"],
            "vector_store": row["vector_store"],
            "index_type": params["index_type"],
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        params["index_dir"] = os.fspath(output_dir / "vector_indexes" / f"index_{digest}")
    return params


def response_metadata(adapter: Any) -> Dict[str, str]:
    """Return a JSON-safe snapshot of provider metadata exposed by an adapter."""
    raw = getattr(adapter, "last_response_metadata", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)[:512]
        for key, value in raw.items()
        if isinstance(value, (str, int, float)) and str(value)
    }


def response_metadata_history(adapter: Any) -> List[Dict[str, str]]:
    raw = getattr(adapter, "response_metadata_history", None)
    if not isinstance(raw, list):
        metadata = response_metadata(adapter)
        return [metadata] if metadata else []
    history = []
    for item in raw:
        if isinstance(item, dict):
            history.append({
                str(key): str(value)[:512]
                for key, value in item.items()
                if isinstance(value, (str, int, float)) and str(value)
            })
    return history


def response_metadata_since(adapter: Any, start: int) -> List[Dict[str, str]]:
    return response_metadata_history(adapter)[max(0, int(start)):]


def embed_many_for_role(adapter: Any, texts: List[str], role: str) -> List[List[float]]:
    """Use model-defined query/document formatting when an embedding adapter exposes it."""
    role_aware = getattr(adapter, "embed_many_with_role", None)
    if callable(role_aware):
        return cast(List[List[float]], role_aware(texts, role=role))
    return adapter.embed_many(texts)


def source_provenance(root: Path) -> Dict[str, Any]:
    """Describe the exact source state without claiming a dirty tree is a commit."""
    try:
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="surrogateescape").split("\0")
    except Exception:
        return {
            "base_git_commit": "unknown",
            "worktree_dirty": True,
            "working_tree_fingerprint": "unknown",
        }

    digest = hashlib.sha256()
    digest.update(base_commit.encode("utf-8"))
    digest.update(b"\0")
    digest.update(tracked_diff)
    untracked_files = []
    for relative_path in sorted(path for path in untracked if path):
        path = root / relative_path
        if path.is_file():
            file_hash = sha256_file(path)
            untracked_files.append({"path": relative_path, "sha256": file_hash})
            digest.update(relative_path.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(file_hash.encode("ascii"))
    return {
        "base_git_commit": base_commit,
        "worktree_dirty": bool(tracked_diff or untracked_files),
        "working_tree_fingerprint": digest.hexdigest(),
        "untracked_files": untracked_files,
    }


def portfolio_result_row(
    row: Dict[str, Any],
    portfolio_context: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {**row, **(portfolio_context or {})}


def run_experiment(
    config_path: Path,
    root: Path,
    output_dir: Path,
    max_runs: int = 0,
    limit_queries: int = 0,
    selections: Dict[str, str] | None = None,
    combination_ids: Tuple[str, ...] | List[str] | None = None,
    portfolio_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    official_cfg = load_benchmark_config(config_path)
    output_dir = validate_output_directory(official_cfg, root, output_dir)
    is_portfolio = isinstance(official_cfg.get("portfolio"), dict)
    if is_portfolio and combination_ids is None:
        raise ValueError("portfolio runs require one declared portfolio batch")
    if combination_ids is not None and max_runs:
        raise ValueError("max_runs is forbidden for a declared portfolio batch")
    if combination_ids is not None and selections:
        selected_values = [value for value in selections.values() if value and value != "all"]
        if selected_values:
            raise ValueError("matrix selections are forbidden for a declared portfolio batch")
    if combination_ids is not None:
        from benchmarking.core.portfolio import build_portfolio_plan

        plan = build_portfolio_plan(official_cfg)
        declared = tuple(combination_ids)
        matching_batch = next(
            (batch for batch in plan.batches if batch.combination_ids == declared),
            None,
        )
        if matching_batch is None:
            raise ValueError("declared combination IDs do not match one immutable portfolio batch")
        expected_context = {
            "portfolio_id": plan.portfolio_id,
            "portfolio_hash": plan.portfolio_hash,
            "batch_id": matching_batch.batch_id,
            "promotion_status": "not_accepted",
        }
        if portfolio_context != expected_context:
            raise ValueError("portfolio context does not match the immutable portfolio plan")
        expected_output = (
            root.resolve()
            / "data"
            / "modular_runs"
            / str(official_cfg["experiment"]["output_lane"])
            / plan.portfolio_id
            / matching_batch.batch_id
        )
        if output_dir != expected_output:
            raise ValueError("output directory does not match the immutable portfolio batch")
    if portfolio_context is not None:
        parent = output_dir.parent
        if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent:
            raise ValueError("canonical portfolio root must already exist")
        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    load_env_file(root)
    cfg = official_cfg
    run_selection: Dict[str, str] = {}
    if selections:
        cfg = selected_config(cfg, selections)
        run_selection = {k: v for k, v in selections.items() if v and v != "all"}
        cfg.setdefault("experiment", {})["selection"] = run_selection
    registry = default_registry()
    matrix = generate_matrix(cfg)
    if combination_ids is not None:
        matrix = select_declared_combinations(matrix, combination_ids)
    elif max_runs:
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
    if portfolio_context is not None:
        manifest["portfolio"] = dict(portfolio_context)
        manifest["promotion_status"] = "not_accepted"
        write_json_atomic(output_dir / "config_snapshot.json", official_cfg)
        write_json_atomic(
            output_dir / "combination_manifest.json",
            {
                "schema_version": 1,
                **portfolio_context,
                "combination_count": len(matrix),
                "combination_ids": [row["combination_id"] for row in matrix],
                "combinations": matrix,
            },
        )
        write_json_atomic(
            output_dir / "provider_readiness_receipt.json",
            {
                "schema_version": 1,
                **portfolio_context,
                "provider_readiness": manifest["provider_readiness"],
                "state": "checked",
            },
        )
    write_json_atomic(output_dir / "manifest.json", manifest)

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    top_k = int(exp.get("top_k", 10))
    eval_cfg = cfg.get("evaluation", {})

    chunk_cache: Dict[str, Any] = {}
    embedding_cache: Dict[Tuple[str, str], Any] = {}
    vector_store_cache: Dict[Tuple[str, str, str, str], Tuple[Any, Dict[str, Any]]] = {}
    bm25_cache: Dict[str, BM25Retriever] = {}

    for idx, row in enumerate(matrix, 1):
        result_row = portfolio_result_row(row, portfolio_context)
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
            metadata_start = len(response_metadata_history(embedder))
            chunk_vectors = embed_many_for_role(embedder, [c.paragraph for c in chunks], role="document")
            chunk_response_metadata = response_metadata_history(embedder)[metadata_start:]
            query_metadata_start = metadata_start + len(chunk_response_metadata)
            query_vectors = embed_many_for_role(embedder, [q.query for q in queries], role="query")
            query_response_metadata = response_metadata_history(embedder)[query_metadata_start:]
            embedding_cache[embed_key] = (
                embedder,
                chunk_vectors,
                query_vectors,
                time.perf_counter() - embed_start,
                {
                    "chunk_embeddings": chunk_response_metadata,
                    "query_embeddings": query_response_metadata,
                },
            )
            print(f"  embedding done: {row['embedding']} seconds={embedding_cache[embed_key][3]:.2f}", flush=True)
        else:
            print(f"  embedding cached: {row['embedding']}", flush=True)
        embedder, chunk_vectors, query_vectors, embedding_latency_s, embedding_response_metadata = embedding_cache[embed_key]

        store_key = (
            row["chunker"],
            row["embedding"],
            row["vector_store"],
            row["index_type"],
        )
        store_cache_hit = store_key in vector_store_cache
        if not store_cache_hit:
            print(f"  upsert start: {row['vector_store']} vectors={len(chunk_vectors)}", flush=True)
            vector_cls = registry.get("vector_store", vector_cfg["adapter"])
            vector_params = vector_store_parameters(vector_cfg, row, output_dir)
            store = vector_cls(name=row["vector_store"], **vector_params)
            upsert_metrics = store.upsert(chunks, chunk_vectors)
            vector_store_cache[store_key] = (store, upsert_metrics)
            print(f"  upsert done: {row['vector_store']} seconds={float(upsert_metrics.get('upsert_latency_s', 0)):.2f}", flush=True)
        else:
            store, upsert_metrics = vector_store_cache[store_key]
            print(f"  vector index cached: {row['vector_store']}", flush=True)
        dense_retriever = DenseCosineRetriever(store)
        retrieval_method = row.get("retrieval_method", "Cosine Similarity")
        candidate_depth = int(exp.get("candidate_depth", max(top_k, 20)))
        fusion_depth = int(exp.get("fusion_depth", max(top_k, 20)))
        hybrid_retriever: HybridRRFRetriever | None = None
        if retrieval_method == "BM25 + Dense + RRF":
            bm25_retriever = bm25_cache.setdefault(chunk_key, BM25Retriever(chunks))
            hybrid_retriever = HybridRRFRetriever(
                dense=dense_retriever,
                bm25=bm25_retriever,
                rrf_k=int(exp.get("rrf_k", 60)),
            )
        elif retrieval_method not in {"Cosine Similarity", "Dense Cosine"}:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")
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
        reranker_response_metadata = []

        print(f"  queries start: {len(queries)} using {row['reranker']}", flush=True)
        for query_idx, (q, qv) in enumerate(zip(queries, query_vectors), 1):
            if query_idx == 1 or query_idx % 5 == 0 or query_idx == len(queries):
                print(f"    query {query_idx}/{len(queries)}: {q.id}", flush=True)
            search_start = time.perf_counter()
            if hybrid_retriever is not None:
                base_hits = hybrid_retriever.search(
                    q.query,
                    qv,
                    top_k=fusion_depth,
                    candidate_depth=candidate_depth,
                )
            else:
                base_hits = dense_retriever.search(qv, top_k=candidate_depth)[:fusion_depth]
            reranker_metadata_start = len(response_metadata_history(reranker))
            reranked = reranker.rerank(q.query, base_hits, top_k=top_k)
            query_reranker_metadata = response_metadata_since(reranker, reranker_metadata_start)
            if query_reranker_metadata:
                reranker_response_metadata.extend(
                    {"query_id": q.id, **metadata}
                    for metadata in query_reranker_metadata
                )
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
                **result_row,
                "query_id": q.id,
                "query": q.query,
                "category": q.category,
                "top_ids": "|".join(str(h.chunk.id) for h in reranked),
                "top_scores": "|".join(f"{h.score:.4f}" for h in reranked),
                "retrieval_provenance": json.dumps([h.provenance or {} for h in reranked], ensure_ascii=False),
                "provider_response_metadata": json.dumps(
                    {"embedding": embedding_response_metadata, "reranker": query_reranker_metadata},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **values,
                "latency_ms": round(latency * 1000, 4),
            })

        lo, hi = bootstrap_ci(metric_lists["recall_at_5"], iterations=int(eval_cfg.get("bootstrap_iterations", 200)), seed=int(exp.get("random_seed", 42)))
        summary_rows.append({
            **result_row,
            "mode": exp.get("mode", "vm_remote_required"),
            "query_count": len(queries),
            "chunk_count": len(chunks),
            "embedding_dimension": getattr(embedder, "dimensions", 0),
            "embedding_latency_s": round(embedding_latency_s, 6),
            "embedding_response_metadata": json.dumps(embedding_response_metadata, ensure_ascii=False, sort_keys=True),
            "reranker_response_metadata": json.dumps(reranker_response_metadata, ensure_ascii=False, sort_keys=True),
            "upsert_latency_s": round(float(upsert_metrics.get("upsert_latency_s", 0)), 6),
            "vector_store_cache_hit": store_cache_hit,
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
    if portfolio_context is not None:
        for name in (
            "config_snapshot.json",
            "combination_manifest.json",
            "provider_readiness_receipt.json",
        ):
            manifest["artifact_sha256"][name] = sha256_file(output_dir / name)
    write_json_atomic(output_dir / "manifest.json", manifest)
    if portfolio_context is not None:
        write_json_atomic(
            output_dir / "completion_receipt.json",
            {
                "schema_version": 1,
                **portfolio_context,
                "state": "completed",
                "combination_count": len(matrix),
                "manifest_sha256": sha256_file(output_dir / "manifest.json"),
                "artifact_sha256": dict(manifest["artifact_sha256"]),
            },
        )
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
    provenance = source_provenance(root)
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
        "git_sha": provenance["base_git_commit"][:12],
        "source_provenance": provenance,
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
        "NEMOTRON_3_EMBED_1B_BF16_URL",
        "NEMOTRON_3_EMBED_1B_NVFP4_URL",
        "NEMOTRON_3_EMBED_8B_BF16_URL",
        "HF_TOKEN",
        "QDRANT_URL",
        "PGVECTOR_DSN",
        "DATABASE_URL",
        "WEAVIATE_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "QWEN_RERANK_URL",
        "BGE_RERANK_URL",
        "NEMOTRON_RERANK_URL",
        "GTE_MODERNBERT_RERANK_URL",
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
