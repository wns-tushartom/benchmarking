from __future__ import annotations

import os
import time
from typing import Any, List

from benchmarking.core.schemas import Chunk, SearchHit


def _vec(values: List[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"


class PGVectorStoreAdapter:
    def __init__(self, name: str, dsn_env: str = "PGVECTOR_DSN", table_prefix: str = "wns_benchmark", **_: Any):
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError("psycopg[binary] is required for PGVectorStoreAdapter. Install requirements-benchmark.txt.") from exc
        self.psycopg = psycopg
        self.name = name
        self.dsn = os.environ.get(dsn_env) or os.environ.get("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError(f"{dsn_env} or DATABASE_URL is required for PGVector")
        self.table = f"{table_prefix}_{os.getpid()}_{int(time.time())}"
        self.dimensions = 0

    def reset_collection(self, schema: Any = None) -> None:
        with self.psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{self.table}"')

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        start = time.perf_counter()
        self.dimensions = len(vectors[0])
        with self.psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f'DROP TABLE IF EXISTS "{self.table}"')
            conn.execute(f'CREATE TABLE "{self.table}" (id BIGINT PRIMARY KEY, chunk_id TEXT, pdf_name TEXT, paragraph TEXT, page_number TEXT, source_type TEXT, parser_method TEXT, embedding vector({self.dimensions}))')
            rows = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
                metadata = chunk.metadata or {}
                rows.append((i, str(chunk.id), chunk.pdf_name, chunk.paragraph, str(metadata.get("page_number", "")), str(metadata.get("source_type", "")), str(metadata.get("parser_method", "")), _vec(vector)))
            with conn.cursor() as cur:
                cur.executemany(f'INSERT INTO "{self.table}" (id, chunk_id, pdf_name, paragraph, page_number, source_type, parser_method, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)', rows)
            conn.execute(f'CREATE INDEX "{self.table}_hnsw" ON "{self.table}" USING hnsw (embedding vector_cosine_ops)')
        return {"upsert_latency_s": time.perf_counter() - start, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        sql = f'SELECT chunk_id, pdf_name, paragraph, page_number, source_type, parser_method, 1 - (embedding <=> %s::vector) AS score FROM "{self.table}" ORDER BY embedding <=> %s::vector LIMIT %s'
        q = _vec(query_vector)
        with self.psycopg.connect(self.dsn) as conn:
            rows = conn.execute(sql, (q, q, top_k)).fetchall()
        return [SearchHit(Chunk(id=int(row[0]) if str(row[0]).isdigit() else i, pdf_name=row[1], paragraph=row[2], parent_id=str(row[0]), metadata={"store": "PGVector", "page_number": row[3] or "", "source_type": row[4] or "", "parser_method": row[5] or ""}), float(row[6])) for i, row in enumerate(rows, 1)]
