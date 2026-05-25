from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from benchmarking.core.schemas import Chunk, SearchHit


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return cleaned or "wns_benchmark"


def vector_literal(vector: List[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


class PGVectorStoreAdapter:
    def __init__(self, name: str, dsn: str = "", table_prefix: str = "wns_benchmark", **_: Any):
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError("psycopg[binary] is required. Run: python -m pip install 'psycopg[binary]'") from exc
        self.psycopg = psycopg
        self.name = name
        self.dsn = dsn or os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError("Missing PGVECTOR_DSN or DATABASE_URL")
        self.table_name = f"{safe_name(table_prefix)}_{safe_name(name)}_{os.getpid()}"
        self.chunks_by_id: Dict[int, Chunk] = {}

    def _connect(self):
        return self.psycopg.connect(self.dsn)

    def reset_collection(self, schema: Any = None) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
            conn.commit()

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        started = time.perf_counter()
        dim = len(vectors[0])
        self.chunks_by_id = {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
                cur.execute(
                    f'''
                    CREATE TABLE "{self.table_name}" (
                        id integer PRIMARY KEY,
                        chunk_id integer,
                        pdf_name text,
                        paragraph text,
                        parent_id text,
                        metadata jsonb,
                        embedding vector({dim})
                    )
                    '''
                )
                cur.execute(f'CREATE INDEX "{self.table_name}_hnsw_idx" ON "{self.table_name}" USING hnsw (embedding vector_cosine_ops)')
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
                    self.chunks_by_id[idx] = chunk
                    cur.execute(
                        f'INSERT INTO "{self.table_name}" (id, chunk_id, pdf_name, paragraph, parent_id, metadata, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s::vector)',
                        (
                            idx,
                            int(chunk.id),
                            chunk.pdf_name,
                            chunk.paragraph,
                            chunk.parent_id,
                            json.dumps(chunk.metadata or {}),
                            vector_literal(vector),
                        ),
                    )
            conn.commit()
        return {"upsert_latency_s": time.perf_counter() - started, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        qvec = vector_literal(query_vector)
        hits: List[SearchHit] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'''
                    SELECT id, chunk_id, pdf_name, paragraph, parent_id, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM "{self.table_name}"
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    ''',
                    (qvec, qvec, int(top_k)),
                )
                for row in cur.fetchall():
                    rid, chunk_id, pdf_name, paragraph, parent_id, metadata, score = row
                    chunk = self.chunks_by_id.get(int(rid)) or Chunk(
                        id=int(chunk_id),
                        pdf_name=str(pdf_name),
                        paragraph=str(paragraph),
                        parent_id=str(parent_id or ""),
                        metadata=metadata or {},
                    )
                    hits.append(SearchHit(chunk=chunk, score=float(score)))
        return hits
