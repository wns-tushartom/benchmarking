#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_registry(repo: Path) -> None:
    path = repo / "benchmarking/core/registry.py"
    text = path.read_text(encoding="utf-8")
    new_import = (
        "    from benchmarking.adapters.local import LocalChunkWorkbookAdapter, LocalHashEmbeddingAdapter, LocalVectorStoreAdapter, WeightedOverlapReranker, OverlapEvaluator\n"
        "    from benchmarking.adapters.remote_embeddings import RemoteHTTPEmbeddingAdapter\n"
        "    from benchmarking.adapters.vector_qdrant import QdrantVectorStoreAdapter\n"
        "    from benchmarking.adapters.vector_pgvector import PGVectorStoreAdapter\n"
        "    from benchmarking.adapters.vector_weaviate import WeaviateVectorStoreAdapter\n"
    )
    old_import = "    from benchmarking.adapters.local import LocalChunkWorkbookAdapter, LocalHashEmbeddingAdapter, LocalVectorStoreAdapter, WeightedOverlapReranker, OverlapEvaluator\n"
    if "RemoteHTTPEmbeddingAdapter" not in text:
        text = text.replace(old_import, new_import)
    registrations = [
        '    r.register("embedding", "remote_http", RemoteHTTPEmbeddingAdapter)\n',
        '    r.register("vector_store", "qdrant", QdrantVectorStoreAdapter)\n',
        '    r.register("vector_store", "pgvector", PGVectorStoreAdapter)\n',
        '    r.register("vector_store", "weaviate", WeaviateVectorStoreAdapter)\n',
    ]
    marker = '    r.register("evaluator", "overlap_relevance", OverlapEvaluator)\n'
    for reg in registrations:
        if reg.strip() not in text:
            text = text.replace(marker, reg + marker)
    path.write_text(text, encoding="utf-8")


def patch_config(repo: Path) -> None:
    path = repo / "configs/benchmark.local.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    embeddings = cfg["techniques"]["embeddings"]
    embeddings["jina_v3"].update({
        "adapter": "remote_http",
        "endpoint_env": "JINA_EMBEDDING_URL",
        "api_key_env": "JINA_API_KEY",
        "timeout_seconds": 120,
        "batch_size": 32,
        "provider_ready": True,
    })
    embeddings["gte_multilingual_base"].update({
        "adapter": "remote_http",
        "endpoint_env": "GTE_EMBEDDING_URL",
        "api_key_env": "HF_TOKEN",
        "timeout_seconds": 120,
        "batch_size": 32,
        "provider_ready": True,
    })
    vector_stores = cfg["techniques"]["vector_stores"]
    vector_stores["Qdrant"].update({"adapter": "qdrant", "provider_ready_env": "QDRANT_URL"})
    vector_stores["PGVector"].update({"adapter": "pgvector", "provider_ready_env": "PGVECTOR_DSN"})
    vector_stores["Weaviate"].update({"adapter": "weaviate", "provider_ready_env": "WEAVIATE_URL"})
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def patch_requirements(repo: Path) -> None:
    path = repo / "requirements-benchmark.txt"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    needed = [
        "qdrant-client>=1.9",
        "psycopg[binary]>=3.1",
        "requests>=2.31",
    ]
    lines = existing.splitlines()
    normalized = {line.split("==")[0].split(">=")[0].strip().lower() for line in lines if line.strip() and not line.startswith("#")}
    for dep in needed:
        base = dep.split(">=")[0].split("[")[0].lower()
        if base not in normalized:
            lines.append(dep)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_adapters(repo: Path) -> None:
    target = repo / "benchmarking/adapters"
    target.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "adapters").glob("*.py"):
        shutil.copy2(src, target / src.name)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/patch_benchmark_repo.py /path/to/Retreiver", file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).expanduser().resolve()
    required = [repo / "benchmarking/core/registry.py", repo / "configs/benchmark.local.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required benchmark repo files:", *missing, sep="\n- ", file=sys.stderr)
        return 1
    copy_adapters(repo)
    patch_registry(repo)
    patch_config(repo)
    patch_requirements(repo)
    print(f"Installed real VM adapters into {repo}")
    print("Next: python -m pip install -r requirements-benchmark.txt && python scripts/benchmark_cli.py validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
