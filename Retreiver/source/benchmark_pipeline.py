from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

try:
    import openpyxl
except Exception:  # pragma: no cover
    openpyxl = None

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

OFFICIAL_CHUNKING_METHODS = [
    "entity_heuristic_w6",
    "entity_heuristic_w5",
    "entity_heuristic_w4",
    "Heading_sections_l2",
    "fixed_tok1200_ov150",
]
CANDIDATE_CHUNKING_METHODS = ["semantic_split"]
EMBEDDING_MODELS = ["jina_v3", "gte_multilingual_base", "openai_text-embedding-3-large"]
VECTOR_DATABASES = ["Qdrant", "PGVector", "Weaviate"]
RERANKING_MODELS = ["Amazon Rerank v1", "Qwen3:4B Rerank", "bge-reranker-base"]
INDEX_TYPE = "HNSW"
RETRIEVAL_METHOD = "Cosine Similarity"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your",
    "have", "has", "had", "not", "but", "can", "will", "all", "any", "our", "their", "then",
    "than", "into", "out", "when", "where", "what", "which", "who", "how", "why", "step", "page",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or "")) if t.lower() not in STOPWORDS]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class DeterministicEmbedder:
    """Local deterministic embedding fallback.

    This lets the full benchmark matrix run without external API keys. It is not a replacement for
    provider-native embeddings. The model name changes dimensions and hashing salt so each requested
    embedding model produces a distinct, stable vector space for pipeline validation.
    """

    DEFAULT_DIMS = {
        "jina_v3": 384,
        "gte_multilingual_base": 384,
        "openai_text-embedding-3-large": 512,
    }

    def __init__(self, model_name: str, dimensions: int | None = None):
        self.model_name = model_name
        self.dimensions = dimensions or self.DEFAULT_DIMS.get(model_name, 384)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        toks = tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            for feature in (tok, f"{tok[:4]}#prefix", f"{tok[-4:]}#suffix"):
                digest = hashlib.blake2b(f"{self.model_name}:{feature}".encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def text_overlap_score(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def recall_at_k(hits: Sequence[bool], k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


@dataclass
class Chunk:
    id: int
    pdf_name: str
    paragraph: str


@dataclass
class QueryCase:
    id: int
    query: str
    expected_text: str
    pdf_name: str = ""


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class LocalVectorIndex:
    """In-process vector DB adapter used for all DB labels in local/offline mode."""

    def __init__(self, database_name: str, index_type: str = INDEX_TYPE):
        self.database_name = database_name
        self.index_type = index_type
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> float:
        start = time.perf_counter()
        self.chunks = chunks
        self.vectors = vectors
        return time.perf_counter() - start

    def search(self, query_vector: List[float], top_k: int = 10) -> Tuple[List[SearchHit], float]:
        start = time.perf_counter()
        scored = [SearchHit(chunk=c, score=cosine_similarity(query_vector, v)) for c, v in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k], time.perf_counter() - start


class Reranker:
    def __init__(self, name: str):
        self.name = name

    def rerank(self, query: str, hits: List[SearchHit]) -> Tuple[List[SearchHit], float]:
        start = time.perf_counter()
        if self.name == "Amazon Rerank v1":
            weight_vector, weight_overlap = 0.65, 0.35
        elif self.name == "Qwen3:4B Rerank":
            weight_vector, weight_overlap = 0.50, 0.50
        else:  # bge-reranker-base
            weight_vector, weight_overlap = 0.40, 0.60
        rescored = []
        for hit in hits:
            overlap = text_overlap_score(query, hit.chunk.paragraph)
            rescored.append(SearchHit(hit.chunk, weight_vector * hit.score + weight_overlap * overlap))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored, time.perf_counter() - start


def load_chunks_from_workbook(path: Path, sheet_name: str) -> List[Chunk]:
    if openpyxl is not None:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}")
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
    else:
        rows = read_xlsx_sheet_stdlib(path, sheet_name)
    if not rows:
        return []
    headers = [str(h) for h in rows[0]]
    idx = {h: i for i, h in enumerate(headers)}
    required = ["id", "pdf_name", "paragraph"]
    missing = [c for c in required if c not in idx]
    if missing:
        raise ValueError(f"Sheet {sheet_name} missing columns: {missing}")
    chunks: List[Chunk] = []
    for row in rows[1:]:
        paragraph = str(row[idx["paragraph"]] or "").strip()
        if paragraph:
            chunks.append(Chunk(int(row[idx["id"]]), str(row[idx["pdf_name"]] or ""), paragraph))
    return chunks


def _xml_text(node: ET.Element, ns: Dict[str, str]) -> str:
    parts = [t.text or "" for t in node.findall(".//main:t", ns)]
    return "".join(parts)


def _cell_ref_to_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return max(0, col - 1)


def read_xlsx_sheet_stdlib(path: Path, sheet_name: str) -> List[Tuple[Any, ...]]:
    """Read one .xlsx sheet using only Python stdlib.

    This is intentionally small and covers normal worksheet cells, shared strings, inline strings,
    and numeric cells. It exists so benchmark scripts work on clean machines without requiring
    `pip install openpyxl`.
    """
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = [_xml_text(si, ns) for si in root.findall("main:si", ns)]

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_by_id = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", ns)
        }
        target = None
        available = []
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            name = sheet.attrib.get("name", "")
            available.append(name)
            if name == sheet_name:
                rel_id = sheet.attrib.get(f"{{{ns['rel']}}}id")
                target = rel_by_id.get(rel_id or "")
                break
        if target is None:
            raise ValueError(f"Sheet {sheet_name!r} not found in {path}. Available sheets: {available}")
        normalized_target = target.lstrip("/")
        sheet_path = normalized_target if normalized_target.startswith("xl/") else "xl/" + normalized_target
        sheet_xml = ET.fromstring(zf.read(sheet_path))

        rows: List[Tuple[Any, ...]] = []
        for row in sheet_xml.findall("main:sheetData/main:row", ns):
            values: List[Any] = []
            for cell in row.findall("main:c", ns):
                col_idx = _cell_ref_to_col_index(cell.attrib.get("r", ""))
                while len(values) < col_idx:
                    values.append(None)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", ns)
                if cell_type == "s":
                    raw = value_node.text if value_node is not None else ""
                    value = shared_strings[int(raw)] if raw else ""
                elif cell_type == "inlineStr":
                    value = _xml_text(cell, ns)
                else:
                    value = value_node.text if value_node is not None else ""
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                values.append(value)
            rows.append(tuple(values))
        return rows


def sniff_delimiter(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    return "|" if "|" in first_line else ","


def load_query_cases(root: Path, limit: int = 0) -> List[QueryCase]:
    candidates = [root / "data" / "qa_text_test.csv", root / "data" / "query.csv"]
    for path in candidates:
        if not path.exists():
            continue
        delim = sniff_delimiter(path)
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            rows = list(reader)
        if not rows:
            continue
        cols = set(rows[0].keys())
        cases: List[QueryCase] = []
        if {"question", "ground_truth", "context"}.issubset(cols):
            for i, row in enumerate(rows, 1):
                q = row.get("question", "").strip()
                expected = (row.get("context") or row.get("ground_truth") or "").strip()
                if q and expected:
                    cases.append(QueryCase(i, q, expected))
        elif {"query", "answer"}.issubset(cols):
            for i, row in enumerate(rows, 1):
                q = row.get("query", "").strip()
                expected = row.get("answer", "").strip()
                if q and expected:
                    cases.append(QueryCase(i, q, expected))
        if cases:
            return cases[:limit] if limit else cases
    raise FileNotFoundError("No usable query ground truth found in data/qa_text_test.csv or data/query.csv")


def make_matrix(include_candidates: bool = False, include_milvus: bool = False) -> List[Dict[str, str]]:
    chunking = OFFICIAL_CHUNKING_METHODS + (CANDIDATE_CHUNKING_METHODS if include_candidates else [])
    dbs = VECTOR_DATABASES + (["Milvus"] if include_milvus else [])
    matrix = []
    run_id = 1
    for chunk in chunking:
        for emb in EMBEDDING_MODELS:
            for db in dbs:
                for rerank in RERANKING_MODELS:
                    matrix.append({
                        "benchmark_run_id": f"run_{run_id:04d}",
                        "chunking_method": chunk,
                        "embedding_model": emb,
                        "vector_database": db,
                        "index_type": INDEX_TYPE,
                        "retrieval_method": RETRIEVAL_METHOD,
                        "reranking_model": rerank,
                    })
                    run_id += 1
    return matrix


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def run_one_config(root: Path, config: Dict[str, str], queries: List[QueryCase], top_k: int = 10, overlap_threshold: float = 0.22) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    workbook = root / "data" / "chunking_methods_output_v2.xlsx"
    chunks = load_chunks_from_workbook(workbook, config["chunking_method"])
    embedder = DeterministicEmbedder(config["embedding_model"])
    index = LocalVectorIndex(config["vector_database"], config["index_type"])
    reranker = Reranker(config["reranking_model"])

    embed_start = time.perf_counter()
    vectors = embedder.embed_many([c.paragraph for c in chunks])
    embedding_latency_s = time.perf_counter() - embed_start
    upsert_latency_s = index.upsert(chunks, vectors)

    detail_rows: List[Dict[str, Any]] = []
    recalls = {1: [], 3: [], 5: [], 10: []}
    latencies: List[float] = []
    rerank_latencies: List[float] = []

    for q in queries:
        qvec = embedder.embed(q.query)
        hits, search_latency = index.search(qvec, top_k=max(top_k, 10))
        hits, rerank_latency = reranker.rerank(q.query, hits)
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
            "chunk_count": len(chunks),
            "top_k": top_k,
            "retrieved_ids": "|".join(str(h.chunk.id) for h in hits),
            "retrieved_scores": "|".join(f"{h.score:.4f}" for h in hits),
            "hit_at_1": int(recall_at_k(hit_flags, 1)),
            "hit_at_3": int(recall_at_k(hit_flags, 3)),
            "hit_at_5": int(recall_at_k(hit_flags, 5)),
            "hit_at_10": int(recall_at_k(hit_flags, 10)),
            "search_latency_ms": round(search_latency * 1000, 4),
            "rerank_latency_ms": round(rerank_latency * 1000, 4),
            "mode": "local_offline_fallback",
            "notes": "Local deterministic embeddings/vector adapters used. Replace with provider adapters for production numbers.",
        })

    def avg(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        **config,
        "mode": "local_offline_fallback",
        "query_count": len(queries),
        "chunk_count": len(chunks),
        "embedding_dimension": embedder.dimensions,
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
    return detail_rows, summary


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
