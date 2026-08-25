from __future__ import annotations

import json
from pathlib import Path

from benchmarking.core.config import generate_matrix, load_benchmark_config


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG = REPO_ROOT / "configs" / "benchmark.local.json"
CANDIDATE_CONFIG = REPO_ROOT / "configs" / "benchmark.retrieval-reranker-candidates.json"
OFFICIAL_CATALOG = REPO_ROOT / "configs" / "project_matrix_catalog.json"
CANDIDATE_CATALOG = REPO_ROOT / "configs" / "project_matrix_candidate_catalog.json"


def test_candidate_matrix_is_exactly_fifty_and_official_matrix_stays_180() -> None:
    official = load_benchmark_config(OFFICIAL_CONFIG)
    candidate = load_benchmark_config(CANDIDATE_CONFIG)

    assert len(generate_matrix(official)) == 180
    rows = generate_matrix(candidate)
    assert len(rows) == 50
    assert {row["retrieval_method"] for row in rows} == {
        "GTE Dense Cosine",
        "BM25 + GTE Dense + RRF",
    }
    assert {row["reranker"] for row in rows} == {
        "none",
        "bge-reranker-base",
        "Qwen3:4B Rerank",
        "nemotron_rerank_1b",
        "gte_modernbert_base",
    }


def test_candidate_catalog_is_distinct_and_does_not_reclassify_official_models() -> None:
    official = json.loads(OFFICIAL_CATALOG.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_CATALOG.read_text(encoding="utf-8"))

    assert "nemotron_rerank_1b" not in official["matrix"]["rerankers"]
    assert "gte_modernbert_base" not in official["matrix"]["rerankers"]
    assert candidate["candidate_lane"] == "retrieval_reranker_candidates"
    assert candidate["matrix"]["retrieval_methods"] == [
        "GTE Dense Cosine",
        "BM25 + GTE Dense + RRF",
    ]
