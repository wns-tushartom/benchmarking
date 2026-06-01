#!/usr/bin/env python3
"""Run live retrieval smoke tests against persisted WNS vector DB collections.

Run on the VM from the Retreiver repo root after DB ingestion completed.
Uses data/db_ingestion_runs/*/summary.csv to discover collection/table/class names.
Writes JSON artifacts to data/retrieval_smoke/ for the dashboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.adapters.remote_embeddings import RemoteHTTPEmbeddingAdapter
from benchmarking.adapters.vector_pgvector import _vec

EMBEDDING_CONFIGS: dict[str, dict[str, Any]] = {
    "gte_multilingual_base": {
        "endpoint_env": "GTE_EMBEDDING_URL",
        "dimensions": 768,
        "default_batch_size": 1,
    },
    "jina_v3": {
        "endpoint_env": "JINA_EMBEDDING_URL",
        "dimensions": 1024,
        "default_batch_size": 1,
    },
}


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

DEFAULT_QUERIES = [
    "refund old ticket and issue new ticket",
    "schedule change alternate option",
    "NACO refund process",
    "EMD refund after reissue",
    "Farelogix refund scenario",
]
QUERY_COLS = ["query", "question", "user_query", "prompt"]


def load_queries_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(h or "").strip().lower() for h in rows[0]]
        idx = next((i for i, h in enumerate(headers) if h in QUERY_COLS), 0)
        return [str(r[idx]).strip() for r in rows[1:] if len(r) > idx and r[idx] and str(r[idx]).strip()]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames:
            fields = [x.lower().strip() for x in reader.fieldnames]
            col = next((reader.fieldnames[i] for i, h in enumerate(fields) if h in QUERY_COLS), reader.fieldnames[0])
            return [str(r.get(col, "")).strip() for r in rows if str(r.get(col, "")).strip()]
        f.seek(0)
        return [line.strip() for line in f if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_ok_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((ROOT / "data" / "db_ingestion_runs").glob("*/summary.csv")):
        run_id = path.parent.name
        for row in read_csv(path):
            if row.get("status") == "ok" and row.get("collection_or_table"):
                row["run_id"] = run_id
                rows.append(row)
    # Keep newest row for each sheet, embedding, store.
    picked: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        picked[(row.get("sheet", ""), row.get("embedding", ""), row.get("store", ""))] = row
    return list(picked.values())


def embed_query(query: str, embedding: str) -> list[float]:
    cfg = EMBEDDING_CONFIGS[embedding]
    adapter = RemoteHTTPEmbeddingAdapter(
        model_name=embedding,
        endpoint_env=cfg["endpoint_env"],
        dimensions=cfg["dimensions"],
        batch_size=1,
    )
    return adapter.embed_many([query])[0]


def search_qdrant(collection: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    from qdrant_client import QdrantClient
    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:5001")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=60)
    if hasattr(client, "search"):
        results = client.search(collection_name=collection, query_vector=vector, limit=top_k, with_payload=True)
    else:
        response = client.query_points(collection_name=collection, query=vector, limit=top_k, with_payload=True)
        results = getattr(response, "points", response)
    out = []
    for i, item in enumerate(results, 1):
        p = item.payload or {}
        out.append({"rank": i, "score": float(item.score), "pdf_name": p.get("pdf_name", ""), "chunk_id": p.get("chunk_id", ""), "paragraph": p.get("paragraph", "")})
    return out


def search_pgvector(table: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    import psycopg
    dsn = os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("PGVECTOR_DSN or DATABASE_URL is required")
    q = _vec(vector)
    sql = f'SELECT chunk_id, pdf_name, paragraph, 1 - (embedding <=> %s::vector) AS score FROM "{table}" ORDER BY embedding <=> %s::vector LIMIT %s'
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(sql, (q, q, top_k)).fetchall()
    return [{"rank": i, "score": float(row[3]), "pdf_name": row[1], "chunk_id": row[0], "paragraph": row[2]} for i, row in enumerate(rows, 1)]


def weaviate_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_weaviate(class_name: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    url = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/")
    gql = {"query": "{ Get { %s(nearVector:{vector:%s} limit:%d) { chunk_id pdf_name paragraph _additional { distance certainty } } } }" % (class_name, json.dumps([float(v) for v in vector]), int(top_k))}
    data = weaviate_request(f"{url}/v1/graphql", gql)
    rows = data.get("data", {}).get("Get", {}).get(class_name, [])
    out = []
    for i, row in enumerate(rows, 1):
        add = row.get("_additional", {})
        score = add.get("certainty")
        if score is None and add.get("distance") is not None:
            score = 1.0 - float(add["distance"])
        out.append({"rank": i, "score": float(score or 0), "pdf_name": row.get("pdf_name", ""), "chunk_id": row.get("chunk_id", ""), "paragraph": row.get("paragraph", "")})
    return out


def run_one(row: dict[str, str], query: str, top_k: int) -> dict[str, Any]:
    vector = embed_query(query, row["embedding"])
    start = time.perf_counter()
    store = row["store"]
    target = row["collection_or_table"]
    if store == "Qdrant":
        hits = search_qdrant(target, vector, top_k)
    elif store == "PGVector":
        hits = search_pgvector(target, vector, top_k)
    elif store == "Weaviate":
        hits = search_weaviate(target, vector, top_k)
    else:
        raise RuntimeError(f"Unsupported store: {store}")
    return {
        "created_at": datetime.now().isoformat(),
        "query": query,
        "sheet": row["sheet"],
        "embedding": row["embedding"],
        "store": store,
        "collection_or_table": target,
        "top_k": top_k,
        "retrieval_seconds": round(time.perf_counter() - start, 6),
        "retrieved_count": len(hits),
        "hits": hits,
        "run_id": row.get("run_id", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="*", default=None)
    parser.add_argument("--queries-file", default="", help="CSV/XLSX/TXT query list; uses query/question column when present")
    parser.add_argument("--sheets", nargs="*", default=["fixed_tok1200_ov150", "Heading_sections_l2", "semantic_split"])
    parser.add_argument("--embeddings", nargs="*", default=["gte_multilingual_base", "jina_v3"])
    parser.add_argument("--stores", nargs="*", default=["Qdrant", "PGVector", "Weaviate"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-combos", type=int, default=18)
    args = parser.parse_args()

    os.chdir(ROOT)
    load_env(ROOT)
    out_dir = ROOT / "data" / "retrieval_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in latest_ok_rows() if r.get("sheet") in args.sheets and r.get("embedding") in args.embeddings and r.get("store") in args.stores]
    rows = rows[: args.max_combos]
    queries = load_queries_file(Path(args.queries_file)) if args.queries_file else (args.queries or DEFAULT_QUERIES)
    all_results = []
    errors = []
    for row in rows:
        for query in queries:
            try:
                result = run_one(row, query, args.top_k)
                name = f"{safe_name(row['sheet'])}_{safe_name(row['embedding'])}_{safe_name(row['store'])}_{safe_name(query)[:40]}.json"
                (out_dir / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"OK {row['sheet']} {row['embedding']} {row['store']} query={query!r} hits={result['retrieved_count']} seconds={result['retrieval_seconds']}", flush=True)
                all_results.append(result)
            except Exception as exc:
                err = {"row": row, "query": query, "error": repr(exc)}
                errors.append(err)
                print(f"ERROR {row.get('sheet')} {row.get('embedding')} {row.get('store')} query={query!r}: {exc!r}", flush=True)
    summary = {"created_at": datetime.now().isoformat(), "result_count": len(all_results), "error_count": len(errors), "errors": errors}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
