from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarking.adapters import remote_embeddings, remote_rerankers
from benchmarking.core import runner
from benchmarking.core.schemas import Chunk, SearchHit
from scripts import benchmark_cli


def test_candidate_cli_rejects_official_output_directory(monkeypatch) -> None:
    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(benchmark_cli, "run_experiment", unexpected_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_cli.py",
            "run",
            "configs/benchmark.retrieval-reranker-candidates.json",
            "--output-dir",
            "data/modular_runs/latest",
        ],
    )

    with pytest.raises(ValueError, match="candidate output directory"):
        benchmark_cli.main()

    assert called is False


def test_candidate_runner_rejects_official_output_directory(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/benchmark.retrieval-reranker-candidates.json"

    def unexpected_environment_load(*args, **kwargs):
        pytest.fail("candidate output validation must run before environment or artifact access")

    monkeypatch.setattr(runner, "load_env_file", unexpected_environment_load)

    with pytest.raises(ValueError, match="candidate output directory"):
        runner.run_experiment(config_path, root, root / "data/modular_runs/latest")


def test_remote_reranker_rejects_substituted_model(monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_RERANK_URL", "http://vm/rerank/nemotron")
    monkeypatch.setattr(
        remote_rerankers,
        "_post_json",
        lambda *args, **kwargs: {
            "model": "substituted/not-nemotron",
            "scores": [0.1, 0.9],
        },
    )
    adapter = remote_rerankers.RemoteHTTPRerankerAdapter(
        name="nemotron_rerank_1b",
        endpoint_env="NEMOTRON_RERANK_URL",
        model="nvidia/llama-nemotron-rerank-1b-v2",
        require_response_model=True,
    )
    hits = [
        SearchHit(Chunk(1, "a.pdf", "a", "1"), 0.0),
        SearchHit(Chunk(2, "b.pdf", "b", "2"), 0.0),
    ]

    with pytest.raises(RuntimeError, match="model mismatch"):
        adapter.rerank("q", hits, 2)


def test_remote_embedding_rejects_substituted_model(monkeypatch) -> None:
    monkeypatch.setenv("GTE_EMBEDDING_URL", "http://vm/embed/gte")
    monkeypatch.setattr(
        remote_embeddings,
        "_post_json",
        lambda *args, **kwargs: {
            "model": "substituted/not-gte",
            "embeddings": [[0.1, 0.2]],
        },
    )
    adapter = remote_embeddings.RemoteHTTPEmbeddingAdapter(
        model_name="gte_multilingual_base",
        endpoint_env="GTE_EMBEDDING_URL",
        require_response_model=True,
    )

    with pytest.raises(RuntimeError, match="model mismatch"):
        adapter.embed_many(["policy"])


def test_remote_reranking_preserves_hybrid_provenance(monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_RERANK_URL", "http://vm/rerank/nemotron")
    monkeypatch.setattr(
        remote_rerankers,
        "_post_json",
        lambda *args, **kwargs: {
            "model": "nvidia/llama-nemotron-rerank-1b-v2",
            "request_id": "req-rerank-1",
            "scores": [0.9],
        },
    )
    adapter = remote_rerankers.RemoteHTTPRerankerAdapter(
        name="nemotron_rerank_1b",
        endpoint_env="NEMOTRON_RERANK_URL",
        model="nvidia/llama-nemotron-rerank-1b-v2",
        require_response_model=True,
    )
    provenance = {"dense_rank": 1, "bm25_rank": 2, "rrf_score": 0.03}
    hit = SearchHit(Chunk(1, "a.pdf", "a", "1"), 0.0, provenance)

    reranked = adapter.rerank("q", [hit], 1)

    assert reranked[0].provenance == provenance
    assert adapter.response_metadata_history == [
        {"model": "nvidia/llama-nemotron-rerank-1b-v2", "request_id": "req-rerank-1"}
    ]


def test_manifest_records_dirty_worktree_source_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/benchmark.retrieval-reranker-candidates.json"
    cfg = runner.load_benchmark_config(config_path)

    manifest = runner.build_manifest(cfg, config_path, root, query_count=1, matrix_count=1)

    source = manifest["source_provenance"]
    assert source["base_git_commit"]
    assert isinstance(source["worktree_dirty"], bool)
    assert len(source["working_tree_fingerprint"]) == 64
    assert "NEMOTRON_RERANK_URL" in manifest["provider_readiness"]
    assert "GTE_MODERNBERT_RERANK_URL" in manifest["provider_readiness"]


def test_runner_serializes_remote_response_metadata() -> None:
    class Adapter:
        last_response_metadata = {
            "request_id": "req-verified",
            "model": "nvidia/llama-nemotron-rerank-1b-v2",
        }

    assert runner.response_metadata(Adapter()) == {
        "request_id": "req-verified",
        "model": "nvidia/llama-nemotron-rerank-1b-v2",
    }


def test_runner_serializes_all_remote_response_metadata() -> None:
    class Adapter:
        response_metadata_history = [
            {"request_id": "req-chunks", "model": "Alibaba-NLP/gte-multilingual-base"},
            {"request_id": "req-queries", "model": "Alibaba-NLP/gte-multilingual-base"},
        ]

    assert runner.response_metadata_history(Adapter()) == [
        {"request_id": "req-chunks", "model": "Alibaba-NLP/gte-multilingual-base"},
        {"request_id": "req-queries", "model": "Alibaba-NLP/gte-multilingual-base"},
    ]


def test_runner_keeps_every_reranker_response_for_one_query() -> None:
    class Adapter:
        response_metadata_history = [
            {"request_id": "req-previous-query", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
            {"request_id": "req-first-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
            {"request_id": "req-fallback-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
        ]

    assert runner.response_metadata_since(Adapter(), 1) == [
        {"request_id": "req-first-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
        {"request_id": "req-fallback-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
    ]


def test_runner_persists_every_reranker_fallback_response_in_candidate_artifacts(monkeypatch, tmp_path: Path) -> None:
    query = SimpleNamespace(id="q-1", query="policy", category="coverage")
    chunk = Chunk(1, "policy.pdf", "policy", "1")
    row = {
        "chunker": "chunker",
        "embedding": "embedding",
        "vector_store": "store",
        "reranker": "reranker",
        "evaluator": "evaluator",
        "retrieval_method": "Cosine Similarity",
    }

    class Chunker:
        def __init__(self, **_: object) -> None:
            pass

        def chunk(self):
            return [chunk]

    class Embedder:
        dimensions = 1
        response_metadata_history: list[dict[str, str]] = []

        def __init__(self, **_: object) -> None:
            pass

        def embed_many(self, values):
            return [[1.0] for _ in values]

    class Store:
        def __init__(self, **_: object) -> None:
            pass

        def upsert(self, *_: object):
            return {"upsert_latency_s": 0.0}

    class Reranker:
        def __init__(self, **_: object) -> None:
            self.response_metadata_history: list[dict[str, str]] = []

        def rerank(self, _query: str, hits, top_k: int):
            self.response_metadata_history.extend([
                {"request_id": "req-first-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
                {"request_id": "req-fallback-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
            ])
            return hits

    class Evaluator:
        def __init__(self, **_: object) -> None:
            pass

        def flags(self, _query, _hits):
            return [True]

    class Registry:
        def get(self, group: str, _adapter: str):
            return {"chunker": Chunker, "embedding": Embedder, "vector_store": Store, "reranker": Reranker, "evaluator": Evaluator}[group]

    class DenseRetriever:
        def __init__(self, _store: object) -> None:
            pass

        def search(self, _vector, top_k: int):
            return [SearchHit(chunk, 1.0)][:top_k]

    output_dir = tmp_path / "candidate-run"
    cfg = {"experiment": {"dataset": "ignored.csv", "top_k": 1}, "evaluation": {}}
    monkeypatch.setattr(runner, "load_benchmark_config", lambda _: cfg)
    monkeypatch.setattr(runner, "validate_output_directory", lambda *_: output_dir)
    monkeypatch.setattr(runner, "load_env_file", lambda _: None)
    monkeypatch.setattr(runner, "default_registry", lambda: Registry())
    monkeypatch.setattr(runner, "generate_matrix", lambda _: [row])
    monkeypatch.setattr(runner, "load_query_cases", lambda *_: [query])
    monkeypatch.setattr(runner, "build_manifest", lambda *_, **__: {})
    monkeypatch.setattr(runner, "technique", lambda _cfg, group, _name: {"adapter": group})
    monkeypatch.setattr(runner, "DenseCosineRetriever", DenseRetriever)
    monkeypatch.setattr(runner, "bootstrap_ci", lambda *_args, **_kwargs: (1.0, 1.0))
    monkeypatch.setattr(runner, "analyze", lambda *_: {})
    monkeypatch.setattr(runner, "render_markdown_report", lambda _: "")

    runner.run_experiment(tmp_path / "config.json", tmp_path, output_dir)

    with (output_dir / "modular_details.csv").open(newline="", encoding="utf-8") as handle:
        detail_metadata = json.loads(next(csv.DictReader(handle))["provider_response_metadata"])["reranker"]
    with (output_dir / "modular_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_metadata = json.loads(next(csv.DictReader(handle))["reranker_response_metadata"])

    expected = [
        {"request_id": "req-first-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
        {"request_id": "req-fallback-payload", "model": "nvidia/llama-nemotron-rerank-1b-v2"},
    ]
    assert detail_metadata == expected
    assert summary_metadata == [{"query_id": "q-1", **metadata} for metadata in expected]
