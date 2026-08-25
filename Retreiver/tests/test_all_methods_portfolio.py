from __future__ import annotations

import json
from pathlib import Path

from benchmarking.core.config import (
    generate_matrix,
    generate_matrix_catalog,
    load_benchmark_config,
)
from benchmarking.core.portfolio import canonical_combination_id


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_CONFIG = ROOT / "configs" / "benchmark.all-methods-portfolio.json"
OFFICIAL_CONFIG = ROOT / "configs" / "benchmark.local.json"

EXPECTED_CHUNKERS = [
    "entity_heuristic_w6",
    "entity_heuristic_w5",
    "entity_heuristic_w4",
    "Heading_sections_l2",
    "fixed_tok1200_ov150",
    "semantic_split",
]
EXPECTED_EMBEDDINGS = [
    "jina_v3",
    "gte_multilingual_base",
    "openai_text-embedding-3-large",
    "nemotron_3_embed_1b_bf16",
    "nemotron_3_embed_1b_nvfp4",
    "nemotron_3_embed_8b_bf16",
]
EXPECTED_STORES = ["Qdrant", "PGVector", "Weaviate", "FAISS", "TurboVec"]
EXPECTED_INDEXES = ["HNSW", "TurboQuant4bit"]
EXPECTED_RETRIEVALS = ["Dense Cosine", "BM25 + Dense + RRF"]
EXPECTED_RERANKERS = [
    "none",
    "Amazon Rerank v1",
    "bge-reranker-base",
    "Qwen3:4B Rerank",
    "nemotron_rerank_1b",
    "gte_modernbert_base",
]
EXPECTED_PAIRS = {
    ("Qdrant", "HNSW"),
    ("PGVector", "HNSW"),
    ("Weaviate", "HNSW"),
    ("FAISS", "HNSW"),
    ("TurboVec", "TurboQuant4bit"),
}


def test_portfolio_config_has_exact_approved_inventory() -> None:
    config = load_benchmark_config(PORTFOLIO_CONFIG)
    matrix = config["matrix"]

    assert matrix["chunkers"] == EXPECTED_CHUNKERS
    assert matrix["embeddings"] == EXPECTED_EMBEDDINGS
    assert matrix["vector_stores"] == EXPECTED_STORES
    assert matrix["index_types"] == EXPECTED_INDEXES
    assert matrix["retrieval_methods"] == EXPECTED_RETRIEVALS
    assert "Cosine Similarity" not in matrix["retrieval_methods"]
    assert matrix["rerankers"] == EXPECTED_RERANKERS


def test_compatibility_catalog_has_2160_valid_and_explicit_exclusions() -> None:
    config = load_benchmark_config(PORTFOLIO_CONFIG)
    catalog = generate_matrix_catalog(config)

    configured = catalog["configured"]
    excluded = catalog["excluded"]
    assert len(configured) == 2160
    assert len(excluded) == 2160
    assert len(generate_matrix(config)) == 2160
    assert {(row["vector_store"], row["index_type"]) for row in configured} == EXPECTED_PAIRS
    assert all(row["state"] == "configured" for row in configured)
    assert all(row["state"] == "excluded" for row in excluded)
    assert {row["reason_code"] for row in excluded} == {
        "incompatible_vector_store_index"
    }
    assert all(row["combination_id"] for row in configured + excluded)

    invalid_pairs = {(row["vector_store"], row["index_type"]) for row in excluded}
    assert ("TurboVec", "HNSW") in invalid_pairs
    assert {
        (store, "TurboQuant4bit") for store in EXPECTED_STORES if store != "TurboVec"
    }.issubset(invalid_pairs)


def test_combination_id_is_stable_under_key_reordering() -> None:
    row = generate_matrix(load_benchmark_config(PORTFOLIO_CONFIG))[0]
    reordered = dict(reversed(list(row.items())))

    assert canonical_combination_id(row) == canonical_combination_id(reordered)
    assert canonical_combination_id(row) == row["combination_id"]


def test_configs_without_compatibility_keep_cartesian_behavior() -> None:
    config = {
        "matrix": {
            "chunkers": ["chunk"],
            "embeddings": ["embed"],
            "vector_stores": ["A", "B"],
            "index_types": ["X", "Y"],
            "retrieval_methods": ["dense"],
            "rerankers": ["none"],
        }
    }

    rows = generate_matrix(config)
    assert len(rows) == 4
    assert {(row["vector_store"], row["index_type"]) for row in rows} == {
        ("A", "X"),
        ("A", "Y"),
        ("B", "X"),
        ("B", "Y"),
    }
    assert all("combination_id" not in row for row in rows)


def test_official_config_remains_exactly_180() -> None:
    assert len(generate_matrix(load_benchmark_config(OFFICIAL_CONFIG))) == 180


def test_compatibility_mapping_must_cover_declared_stores() -> None:
    config = load_benchmark_config(PORTFOLIO_CONFIG)
    malformed = json.loads(json.dumps(config))
    del malformed["compatibility"]["vector_store_index_types"]["TurboVec"]

    try:
        generate_matrix_catalog(malformed)
    except ValueError as exc:
        assert "TurboVec" in str(exc)
    else:  # pragma: no cover - fail with an actionable message
        raise AssertionError("incomplete compatibility mapping was accepted")
