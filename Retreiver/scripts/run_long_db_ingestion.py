#!/usr/bin/env python3
"""Long-running WNS DB ingestion runner.

Purpose:
- Read chunk sheets from data/chunking_methods_output_v2.xlsx
- Generate/cached embeddings per sheet/model
- Upsert cached vectors into Qdrant, PGVector, and Weaviate
- Run a small search check
- Write resumable progress artifacts under data/db_ingestion_runs/<run_id>/

Example:
  .venv-vm/bin/python scripts/run_long_db_ingestion.py \
    --sheets fixed_tok1200_ov150 Heading_sections_l2 entity_heuristic_w6 \
    --embeddings gte_multilingual_base \
    --stores Qdrant PGVector Weaviate \
    --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.adapters.remote_embeddings import RemoteHTTPEmbeddingAdapter
from benchmarking.adapters.vector_pgvector import PGVectorStoreAdapter
from benchmarking.adapters.vector_qdrant import QdrantVectorStoreAdapter
from benchmarking.adapters.vector_weaviate import WeaviateVectorStoreAdapter
from benchmarking.core.schemas import Chunk

EMBEDDING_CONFIGS: dict[str, dict[str, Any]] = {
    "gte_multilingual_base": {
        "endpoint_env": "GTE_EMBEDDING_URL",
        "dimensions": 768,
        "default_batch_size": 4,
    },
    "jina_v3": {
        "endpoint_env": "JINA_EMBEDDING_URL",
        "dimensions": 1024,
        "default_batch_size": 2,
    },
}

DEFAULT_SHEETS = [
    "fixed_tok1200_ov150",
    "Heading_sections_l2",
    "entity_heuristic_w6",
    "entity_heuristic_w5",
    "entity_heuristic_w4",
    "semantic_split",
]
DEFAULT_STORES = ["Qdrant", "PGVector", "Weaviate"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def load_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_chunks(workbook: Path, sheet: str, limit: int = 0) -> list[Chunk]:
    df = pd.read_excel(workbook, sheet_name=sheet)
    if limit:
        df = df.head(limit)
    chunks: list[Chunk] = []
    for r in df.itertuples(index=False):
        paragraph = str(getattr(r, "paragraph", "") or "").strip()
        if not paragraph:
            continue
        chunks.append(
            Chunk(
                id=int(getattr(r, "id")),
                pdf_name=str(getattr(r, "pdf_name")),
                paragraph=paragraph,
                parent_id=str(getattr(r, "id")),
                metadata={"sheet": sheet},
            )
        )
    return chunks


def get_vectors(
    chunks: list[Chunk],
    embedding_name: str,
    batch_size: int,
    cache_path: Path,
    force: bool = False,
) -> list[list[float]]:
    if cache_path.exists() and not force:
        vectors = np.load(cache_path).tolist()
        log(f"loaded_cached_vectors path={cache_path} count={len(vectors)} dim={len(vectors[0]) if vectors else 0}")
        return vectors

    cfg = EMBEDDING_CONFIGS[embedding_name]
    embedder = RemoteHTTPEmbeddingAdapter(
        model_name=embedding_name,
        endpoint_env=cfg["endpoint_env"],
        dimensions=cfg["dimensions"],
        batch_size=batch_size,
    )
    texts = [c.paragraph for c in chunks]
    start = time.perf_counter()
    vectors = embedder.embed_many(texts)
    elapsed = time.perf_counter() - start
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.asarray(vectors, dtype="float32"))
    log(f"embedded model={embedding_name} count={len(vectors)} dim={len(vectors[0]) if vectors else 0} seconds={elapsed:.2f} cache={cache_path}")
    return vectors


def make_store(store_name: str, sheet: str, embedding: str):
    prefix = f"wns_{safe_name(sheet)[:24]}_{safe_name(embedding)[:16]}"
    if store_name == "Qdrant":
        return QdrantVectorStoreAdapter(name="Qdrant", collection_prefix=prefix)
    if store_name == "PGVector":
        return PGVectorStoreAdapter(name="PGVector", table_prefix=prefix)
    if store_name == "Weaviate":
        return WeaviateVectorStoreAdapter(name="Weaviate", class_prefix="Wns" + safe_name(sheet)[:20] + safe_name(embedding)[:10])
    raise ValueError(f"unknown store: {store_name}")


def upsert_and_check(store_name: str, sheet: str, embedding: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, Any]:
    store = make_store(store_name, sheet, embedding)
    start = time.perf_counter()
    metrics = store.upsert(chunks, vectors)
    hits = store.search(vectors[0], top_k=5) if vectors else []
    elapsed = time.perf_counter() - start
    return {
        "store": store_name,
        "sheet": sheet,
        "embedding": embedding,
        "chunk_count": len(chunks),
        "vector_count": len(vectors),
        "upsert_latency_s": round(float(metrics.get("upsert_latency_s", 0)), 6),
        "total_store_seconds": round(elapsed, 6),
        "search_hits": len(hits),
        "top_hit_pdf": hits[0].chunk.pdf_name if hits else "",
        "top_hit_score": round(float(hits[0].score), 6) if hits else 0,
        "collection_or_table": getattr(store, "collection", getattr(store, "table", getattr(store, "class_name", ""))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", default="data/chunking_methods_output_v2.xlsx")
    parser.add_argument("--sheets", nargs="+", default=DEFAULT_SHEETS)
    parser.add_argument("--embeddings", nargs="+", default=["gte_multilingual_base"])
    parser.add_argument("--stores", nargs="+", default=DEFAULT_STORES)
    parser.add_argument("--batch-size", type=int, default=0, help="Embedding batch size. 0 = model default")
    parser.add_argument("--limit", type=int, default=0, help="Limit chunks per sheet for smoke runs")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--skip-existing-store-success", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    load_env(ROOT)

    workbook = ROOT / args.workbook
    if not workbook.exists():
        raise FileNotFoundError(workbook)

    run_dir = ROOT / "data" / "db_ingestion_runs" / args.run_id
    cache_dir = ROOT / "data" / "embedding_cache"
    summary_csv = run_dir / "summary.csv"
    status_json = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    log(f"run_id={args.run_id} workbook={workbook}")
    write_json(run_dir / "config.json", vars(args))

    completed_keys: set[tuple[str, str, str]] = set()
    if args.skip_existing_store_success and summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        for r in prev.to_dict("records"):
            if str(r.get("status")) == "ok":
                completed_keys.add((str(r.get("sheet")), str(r.get("embedding")), str(r.get("store"))))

    total_tasks = len(args.sheets) * len(args.embeddings) * len(args.stores)
    done_tasks = 0
    failures: list[dict[str, Any]] = []

    for sheet in args.sheets:
        log(f"LOAD sheet={sheet}")
        chunks = load_chunks(workbook, sheet, limit=args.limit)
        log(f"chunks sheet={sheet} count={len(chunks)}")
        for embedding in args.embeddings:
            if embedding not in EMBEDDING_CONFIGS:
                raise ValueError(f"Unknown embedding {embedding}. Known: {sorted(EMBEDDING_CONFIGS)}")
            batch_size = args.batch_size or int(EMBEDDING_CONFIGS[embedding]["default_batch_size"])
            vec_path = cache_dir / f"{safe_name(sheet)}_{safe_name(embedding)}_vectors.npy"
            meta_path = cache_dir / f"{safe_name(sheet)}_{safe_name(embedding)}_meta.json"
            try:
                vectors = get_vectors(chunks, embedding, batch_size, vec_path, force=args.force_embed)
                write_json(meta_path, {"sheet": sheet, "embedding": embedding, "count": len(vectors), "dim": len(vectors[0]) if vectors else 0, "batch_size": batch_size})
            except Exception as exc:
                row = {"status": "embedding_failed", "sheet": sheet, "embedding": embedding, "store": "", "error": repr(exc), "created_at": datetime.now().isoformat()}
                append_csv(summary_csv, row)
                failures.append(row)
                log(f"ERROR embedding sheet={sheet} embedding={embedding}: {exc!r}")
                continue

            for store_name in args.stores:
                key = (sheet, embedding, store_name)
                if key in completed_keys:
                    log(f"SKIP existing success sheet={sheet} embedding={embedding} store={store_name}")
                    continue
                done_tasks += 1
                write_json(status_json, {"current": {"sheet": sheet, "embedding": embedding, "store": store_name}, "done_tasks": done_tasks, "total_tasks": total_tasks, "failures": len(failures), "updated_at": datetime.now().isoformat()})
                try:
                    log(f"UPSERT sheet={sheet} embedding={embedding} store={store_name}")
                    result = upsert_and_check(store_name, sheet, embedding, chunks, vectors)
                    result.update({"status": "ok", "error": "", "created_at": datetime.now().isoformat()})
                    append_csv(summary_csv, result)
                    log(f"OK sheet={sheet} embedding={embedding} store={store_name} hits={result['search_hits']} seconds={result['total_store_seconds']}")
                except Exception as exc:
                    row = {"status": "store_failed", "sheet": sheet, "embedding": embedding, "store": store_name, "chunk_count": len(chunks), "vector_count": len(vectors), "error": repr(exc), "created_at": datetime.now().isoformat()}
                    append_csv(summary_csv, row)
                    failures.append(row)
                    log(f"ERROR store sheet={sheet} embedding={embedding} store={store_name}: {exc!r}")

    write_json(status_json, {"status": "done", "summary_csv": str(summary_csv), "failures": failures, "updated_at": datetime.now().isoformat()})
    log(f"DONE summary={summary_csv} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
