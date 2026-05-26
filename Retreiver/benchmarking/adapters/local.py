from __future__ import annotations

import csv
import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmarking.core.schemas import Chunk, QueryCase, SearchHit
from source.benchmark_pipeline import load_chunks_from_workbook, sniff_delimiter

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "you", "your", "have", "has", "had", "not", "but", "can", "will", "all", "any", "our", "their", "then", "than", "into", "out", "when", "where", "what", "which", "who", "how", "why", "step", "page"}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text or "")) if t.lower() not in STOPWORDS]


def text_overlap_score(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class LocalChunkWorkbookAdapter:
    def __init__(self, root: Path, sheet_name: str, workbook: str = "data/chunking_methods_output_v2.xlsx", **_: Any):
        self.root = root
        self.sheet_name = sheet_name
        self.workbook = workbook
        self.name = sheet_name

    def chunk(self) -> List[Chunk]:
        rows = load_chunks_from_workbook(self.root / self.workbook, self.sheet_name)
        return [Chunk(id=r.id, pdf_name=r.pdf_name, paragraph=r.paragraph, parent_id=str(r.id), metadata={"sheet": self.sheet_name}) for r in rows]


class LocalHashEmbeddingAdapter:
    def __init__(self, model_name: str, dimensions: int = 384, **_: Any):
        self.name = model_name
        self.model_name = model_name
        self.dimensions = int(dimensions)
        self.cost_per_1k_tokens = None

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
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class LocalVectorStoreAdapter:
    def __init__(self, name: str, **_: Any):
        self.name = name
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def reset_collection(self, schema: Any = None) -> None:
        self.chunks = []
        self.vectors = []

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        start = time.perf_counter()
        self.chunks = chunks
        self.vectors = vectors
        return {"upsert_latency_s": time.perf_counter() - start, "vector_count": len(vectors)}

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        scored = [SearchHit(chunk=c, score=cosine_similarity(query_vector, v)) for c, v in zip(self.chunks, self.vectors)]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


class PassthroughReranker:
    def __init__(self, name: str = "none", **_: Any):
        self.name = name

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        return hits[:top_k]


class WeightedOverlapReranker:
    def __init__(self, name: str, vector_weight: float = 0.5, overlap_weight: float = 0.5, **_: Any):
        self.name = name
        self.vector_weight = float(vector_weight)
        self.overlap_weight = float(overlap_weight)

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        rescored = []
        for hit in hits:
            overlap = text_overlap_score(query, hit.chunk.paragraph)
            rescored.append(SearchHit(hit.chunk, self.vector_weight * hit.score + self.overlap_weight * overlap))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]


class OverlapEvaluator:
    def __init__(self, overlap_threshold: float = 0.22, **_: Any):
        self.name = "overlap_relevance"
        self.overlap_threshold = float(overlap_threshold)

    def flags(self, query: QueryCase, hits: List[SearchHit]) -> List[bool]:
        return [text_overlap_score(hit.chunk.paragraph, query.expected_text) >= self.overlap_threshold for hit in hits]


def load_query_cases(root: Path, dataset: str) -> List[QueryCase]:
    path = root / dataset
    delim = sniff_delimiter(path)
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    cases: List[QueryCase] = []
    for i, row in enumerate(rows, 1):
        q = (row.get("question") or row.get("query") or "").strip()
        expected = (row.get("context") or row.get("ground_truth") or row.get("answer") or "").strip()
        category = infer_category(q + " " + expected)
        if q and expected:
            cases.append(QueryCase(i, q, expected, category=category, metadata={"source": path.name}))
    return cases


def infer_category(text: str) -> str:
    t = text.lower()
    for category, terms in {
        "refund": ["refund", "chargeback"],
        "voucher": ["voucher", "emd"],
        "rebooking": ["rebook", "booking", "change flight"],
        "pnr": ["pnr"],
        "passport_dob": ["passport", "date of birth", "dob"],
        "ticketing": ["ticket", "open ticket"],
    }.items():
        if any(term in t for term in terms):
            return category
    return "general"
