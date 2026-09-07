from __future__ import annotations

from pathlib import Path

from benchmarking.core.config import generate_matrix_catalog, load_benchmark_config
from benchmarking.core.portfolio import build_portfolio_plan


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = ROOT / "configs" / "benchmark.meeting-400-candidates.json"

EXPECTED_MATRIX = {
    "chunkers": [
        "entity_heuristic_w6",
        "entity_heuristic_w5",
        "entity_heuristic_w4",
        "Heading_sections_l2",
        "fixed_tok1200_ov150",
    ],
    "embeddings": [
        "nemotron_3_embed_1b_bf16",
        "nemotron_3_embed_1b_nvfp4",
        "nemotron_3_embed_8b_bf16",
    ],
    "vector_stores": ["Qdrant", "FAISS"],
    "index_types": ["HNSW"],
    "retrieval_methods": ["Dense Cosine", "BM25 + Dense + RRF"],
    "rerankers": [
        "none",
        "bge-reranker-base",
        "Qwen3:4B Rerank",
        "nemotron_rerank_1b",
        "gte_modernbert_base",
    ],
    "evaluators": ["overlap_relevance"],
}
EXPECTED_RERANKER_GROUPS = {
    "A": ["none", "bge-reranker-base", "Qwen3:4B Rerank"],
    "B": ["nemotron_rerank_1b", "gte_modernbert_base"],
}


def _config() -> dict:
    return load_benchmark_config(CANDIDATE_CONFIG)


def test_meeting_candidate_config_has_exact_immutable_contract() -> None:
    config = _config()

    assert config["schema_version"] == 1
    assert config["experiment"] == {
        "name": "wns-meeting-400-candidates",
        "description": (
            "Immutable meeting candidate portfolio; planned only, not accepted official evidence"
        ),
        "mode": "vm_remote_required",
        "output_lane": "meeting-400-candidates",
        "promotion_status": "not_accepted",
        "dataset": "data/groundtruth_500.csv",
        "corpus_workbook": "data/chunking_methods_output_225.xlsx",
        "top_k": 10,
        "random_seed": 42,
        "repetitions": 1,
        "candidate_depth": 50,
        "fusion_depth": 20,
        "rrf_k": 60,
    }
    assert config["matrix"] == EXPECTED_MATRIX
    assert config["portfolio"] == {
        "schema_version": 1,
        "max_combinations_per_batch": 250,
        "reranker_groups": EXPECTED_RERANKER_GROUPS,
    }
    assert config["compatibility"]["vector_store_index_types"] == {
        "Qdrant": ["HNSW"],
        "FAISS": ["HNSW"],
    }


def test_meeting_candidate_config_expands_to_exact_300_and_six_batches() -> None:
    config = _config()
    catalog = generate_matrix_catalog(config)
    first = build_portfolio_plan(config)
    second = build_portfolio_plan(config)

    assert len(catalog["configured"]) == 300
    assert catalog["excluded"] == []
    assert first == second
    assert len(first.combination_ids) == 300
    assert len(set(first.combination_ids)) == 300
    assert [
        (batch.embedding, batch.reranker_group, len(batch.combination_ids))
        for batch in first.batches
    ] == [
        (embedding, group, 60 if group == "A" else 40)
        for embedding in EXPECTED_MATRIX["embeddings"]
        for group in ("A", "B")
    ]
    assert all(
        len(batch.combination_ids) <= first.max_combinations_per_batch <= 250
        for batch in first.batches
    )
    covered = [
        combination_id
        for batch in first.batches
        for combination_id in batch.combination_ids
    ]
    assert len(covered) == len(set(covered)) == 300
    assert set(covered) == set(first.combination_ids)


def test_meeting_candidate_config_declares_only_candidate_methods() -> None:
    config = _config()
    techniques = config["techniques"]

    assert list(techniques["chunkers"]) == EXPECTED_MATRIX["chunkers"]
    assert {
        name: technique.get("workbook")
        for name, technique in techniques["chunkers"].items()
    } == {
        name: config["experiment"]["corpus_workbook"]
        for name in EXPECTED_MATRIX["chunkers"]
    }
    assert list(techniques["embeddings"]) == EXPECTED_MATRIX["embeddings"]
    assert list(techniques["vector_stores"]) == EXPECTED_MATRIX["vector_stores"]
    assert list(techniques["index_types"]) == EXPECTED_MATRIX["index_types"]
    assert list(techniques["retrieval_methods"]) == EXPECTED_MATRIX["retrieval_methods"]
    assert list(techniques["rerankers"]) == EXPECTED_MATRIX["rerankers"]
