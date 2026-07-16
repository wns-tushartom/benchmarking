from __future__ import annotations

from typing import Any

import scripts.run_reranker_smoke_from_retrieval as reranker


def test_rerank_one_limits_candidates_and_persists_both_depths(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {
            "model": "test-reranker",
            "results": [
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.8},
            ],
        }

    monkeypatch.setattr(reranker, "post_json", fake_post)
    payload = {
        "query": "refund rule",
        "top_k": 20,
        "hits": [
            {"rank": index + 1, "paragraph": f"candidate {index}"}
            for index in range(20)
        ],
    }

    result = reranker.rerank_one(
        payload,
        "bge-reranker-base",
        output_k=2,
        candidate_k=7,
    )

    assert len(captured["documents"]) == 7
    assert captured["top_k"] == 2
    assert result["retrieval_top_k"] == 20
    assert result["reranked_output_k"] == 2
    assert result["candidate_count"] == 7
    assert result["retrieved_count"] == 2
    assert len(result["hits"]) == 2


def test_rerank_one_candidate_limit_zero_uses_all_retrieved_hits(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(payload)
        return {"results": []}

    monkeypatch.setattr(reranker, "post_json", fake_post)
    payload = {
        "query": "refund rule",
        "top_k": 4,
        "hits": [{"rank": index + 1, "paragraph": str(index)} for index in range(4)],
    }

    result = reranker.rerank_one(
        payload,
        "bge-reranker-base",
        output_k=2,
        candidate_k=0,
    )

    assert len(captured["documents"]) == 4
    assert result["candidate_count"] == 4
