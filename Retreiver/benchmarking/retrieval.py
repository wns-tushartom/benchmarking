"""First-stage retrieval implementations for the isolated candidate lane.

Scores from BM25 and dense cosine are deliberately never blended directly.
Hybrid retrieval combines rank positions using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Iterable, Protocol, Sequence

from rank_bm25 import BM25Okapi

from benchmarking.core.schemas import Chunk, SearchHit


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class DenseStore(Protocol):
    def search(self, query_vector: Sequence[float], top_k: int) -> list[SearchHit]: ...


def tokenize(text: str) -> list[str]:
    """Use a transparent lexical tokenizer suitable for BM25 candidate retrieval."""
    return _TOKEN_RE.findall(text.lower())


def rrf_score(rank: int, rrf_k: int = 60) -> float:
    if rank < 1:
        raise ValueError("RRF rank must be at least 1")
    if rrf_k < 0:
        raise ValueError("RRF constant must be non-negative")
    return 1.0 / (rrf_k + rank)


class BM25Retriever:
    """Lexical BM25 retrieval over the supplied chunks, with rank provenance."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        self._index = BM25Okapi([tokenize(chunk.paragraph) for chunk in self.chunks])

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k < 1:
            return []
        scores = self._index.get_scores(tokenize(query))
        ordered = sorted(range(len(self.chunks)), key=lambda index: (-float(scores[index]), index))[:top_k]
        return [
            SearchHit(
                chunk=self.chunks[index],
                score=float(scores[index]),
                provenance={"bm25_rank": rank, "bm25_score": float(scores[index])},
            )
            for rank, index in enumerate(ordered, start=1)
        ]


class DenseCosineRetriever:
    """Wrap the existing vector store and attach dense rank/score provenance."""

    def __init__(self, store: DenseStore) -> None:
        self.store = store

    def search(self, query_vector: Sequence[float], top_k: int) -> list[SearchHit]:
        if top_k < 1:
            return []
        return [
            replace(
                hit,
                provenance={**(hit.provenance or {}), "dense_rank": rank, "dense_score": hit.score},
            )
            for rank, hit in enumerate(self.store.search(query_vector, top_k=top_k), start=1)
        ]


class HybridRRFRetriever:
    """Fuse independent BM25 and dense candidate lists with deterministic RRF."""

    def __init__(self, *, dense: DenseCosineRetriever, bm25: BM25Retriever, rrf_k: int = 60) -> None:
        if rrf_k < 0:
            raise ValueError("RRF constant must be non-negative")
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_vector: Sequence[float],
        top_k: int,
        candidate_depth: int | None = None,
    ) -> list[SearchHit]:
        if top_k < 1:
            return []
        branch_depth = candidate_depth or top_k
        dense_hits = self.dense.search(query_vector, top_k=branch_depth)
        bm25_hits = self.bm25.search(query, top_k=branch_depth)
        candidates: dict[int, dict[str, Any]] = {}

        for hit in dense_hits:
            data = candidates.setdefault(hit.chunk.id, {"chunk": hit.chunk, "provenance": {}})
            data["provenance"].update(hit.provenance or {})
        for hit in bm25_hits:
            data = candidates.setdefault(hit.chunk.id, {"chunk": hit.chunk, "provenance": {}})
            data["provenance"].update(hit.provenance or {})

        fused: list[SearchHit] = []
        for data in candidates.values():
            provenance = data["provenance"]
            score = sum(
                rrf_score(rank, self.rrf_k)
                for key, rank in (("dense_rank", provenance.get("dense_rank")), ("bm25_rank", provenance.get("bm25_rank")))
                if rank is not None
            )
            provenance["rrf_score"] = score
            fused.append(SearchHit(chunk=data["chunk"], score=score, provenance=provenance))

        fused.sort(
            key=lambda hit: (
                -hit.score,
                min((hit.provenance or {}).get("dense_rank", 10**9), (hit.provenance or {}).get("bm25_rank", 10**9)),
                hit.chunk.id,
            )
        )
        return fused[:top_k]
