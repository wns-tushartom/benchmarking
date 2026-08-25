from __future__ import annotations

import pytest

from benchmarking.core.schemas import Chunk, SearchHit
from benchmarking.retrieval import BM25Retriever, DenseCosineRetriever, HybridRRFRetriever, rrf_score


CHUNKS = [
    Chunk(id=1, pdf_name="policy.pdf", paragraph="PAX-REF-992 refund policy allows full fare refunds."),
    Chunk(id=2, pdf_name="baggage.pdf", paragraph="Cabin baggage allowance and excess baggage process."),
    Chunk(id=3, pdf_name="policy.pdf", paragraph="Refund requests require original ticket number."),
]


class StaticDenseStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        assert query_vector == [1.0, 0.0]
        return self.hits[:top_k]


def test_bm25_returns_lexical_hit_without_embedding_or_vector_store() -> None:
    retriever = BM25Retriever(CHUNKS)

    hits = retriever.search("PAX-REF-992 refund", top_k=1)

    assert [hit.chunk.id for hit in hits] == [1]
    assert hits[0].provenance["bm25_rank"] == 1
    assert hits[0].provenance["bm25_score"] > 0
    assert "dense_rank" not in hits[0].provenance


def test_dense_cosine_preserves_rank_score_and_marks_provenance() -> None:
    store = StaticDenseStore([
        SearchHit(chunk=CHUNKS[1], score=0.9),
        SearchHit(chunk=CHUNKS[0], score=0.8),
    ])

    hits = DenseCosineRetriever(store).search([1.0, 0.0], top_k=2)

    assert [hit.chunk.id for hit in hits] == [2, 1]
    assert hits[0].provenance == {"dense_rank": 1, "dense_score": 0.9}


def test_hybrid_rrf_fuses_ranks_and_preserves_all_branch_fields() -> None:
    store = StaticDenseStore([
        SearchHit(chunk=CHUNKS[0], score=0.9),
        SearchHit(chunk=CHUNKS[1], score=0.8),
    ])
    retriever = HybridRRFRetriever(
        dense=DenseCosineRetriever(store),
        bm25=BM25Retriever(CHUNKS),
        rrf_k=60,
    )

    hits = retriever.search("PAX-REF-992 baggage", [1.0, 0.0], top_k=3)

    assert [hit.chunk.id for hit in hits] == [1, 2, 3]
    first = hits[0].provenance
    assert first["dense_rank"] == 1
    assert first["bm25_rank"] == 1
    assert first["dense_score"] == pytest.approx(0.9)
    assert first["rrf_score"] == pytest.approx(rrf_score(1, 60) + rrf_score(1, 60))


def test_rrf_score_rejects_invalid_rank_or_constant() -> None:
    with pytest.raises(ValueError):
        rrf_score(0, 60)
    with pytest.raises(ValueError):
        rrf_score(1, -1)
