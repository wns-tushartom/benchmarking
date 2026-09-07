from pathlib import Path

import pytest
import scripts.serve_benchmark_dashboard as dashboard


def test_dashboard_exposes_candidate_lane_without_changing_official_options(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(dashboard, "ROOT", root)
    monkeypatch.setattr(dashboard, "CONFIG_PATH", root / "configs" / "benchmark.local.json")
    monkeypatch.setattr(
        dashboard,
        "CANDIDATE_CONFIG_PATH",
        root / "configs" / "benchmark.meeting-400-candidates.json",
    )

    options = dashboard.benchmark_options()

    assert options["matrix_count"] == 180
    assert options["retrieval_methods"] == ["Cosine Similarity"]
    assert options["candidate_lane"]["matrix_count"] == 300
    assert options["candidate_lane"]["retrieval_methods"] == [
        "Dense Cosine",
        "BM25 + Dense + RRF",
    ]
    assert options["candidate_lane"]["embeddings"] == [
        "nemotron_3_embed_1b_bf16",
        "nemotron_3_embed_1b_nvfp4",
        "nemotron_3_embed_8b_bf16",
    ]
    assert options["candidate_lane"]["vector_stores"] == ["Qdrant", "FAISS"]
    assert "nemotron_rerank_1b" in options["candidate_lane"]["rerankers"]
    assert "gte_modernbert_base" in options["candidate_lane"]["rerankers"]


def test_selected_cli_args_for_candidate_request_preserve_retrieval_and_store() -> None:
    args = dashboard.selected_cli_args({
        "chunker": ["entity_heuristic_w6"],
        "embedding": ["nemotron_3_embed_1b_bf16"],
        "store": ["FAISS"],
        "retrieval_method": ["BM25 + Dense + RRF"],
        "reranker": ["nemotron_rerank_1b"],
    })

    assert args == [
        "--chunker", "entity_heuristic_w6",
        "--embedding", "nemotron_3_embed_1b_bf16",
        "--vector-store", "FAISS",
        "--retrieval-method", "BM25 + Dense + RRF",
        "--reranker", "nemotron_rerank_1b",
    ]


def test_candidate_benchmark_command_maps_dashboard_sheet_to_cli_chunker() -> None:
    cmd = dashboard.candidate_benchmark_command(
        {
            "sheet": ["entity_heuristic_w6"],
            "embedding": ["nemotron_3_embed_1b_bf16"],
            "store": ["FAISS"],
            "retrieval_method": ["BM25 + Dense + RRF"],
            "reranker": ["nemotron_rerank_1b"],
        },
        limit_queries="5",
        max_runs="1",
        run_id="20260810-0642",
    )

    assert "--chunker" in cmd
    assert cmd[cmd.index("--chunker") + 1] == "entity_heuristic_w6"


def test_candidate_benchmark_command_has_isolated_receipted_output_dir() -> None:
    cmd = dashboard.candidate_benchmark_command(
        {
            "chunker": ["entity_heuristic_w6"],
            "embedding": ["nemotron_3_embed_1b_bf16"],
            "store": ["FAISS"],
            "retrieval_method": ["BM25 + Dense + RRF"],
            "reranker": ["nemotron_rerank_1b"],
        },
        limit_queries="5",
        max_runs="1",
        run_id="20260810-0642",
    )

    assert cmd == [
        dashboard.sys.executable,
        "scripts/benchmark_cli.py",
        "run",
        "configs/benchmark.meeting-400-candidates.json",
        "--limit-queries", "5",
        "--max-runs", "1",
        "--output-dir", "data/modular_runs/meeting-400-candidates/20260810-0642",
        "--chunker", "entity_heuristic_w6",
        "--embedding", "nemotron_3_embed_1b_bf16",
        "--vector-store", "FAISS",
        "--retrieval-method", "BM25 + Dense + RRF",
        "--reranker", "nemotron_rerank_1b",
    ]


def test_candidate_query_limit_respects_frontend_query_limit_parameter() -> None:
    assert dashboard.candidate_query_limit({"query_limit": ["0"], "limit": ["50"]}) == "0"


def test_candidate_command_rejects_silent_partial_multiselect() -> None:
    with pytest.raises(ValueError, match="one chunker or All chunkers"):
        dashboard.candidate_benchmark_command(
            {
                "chunker": ["entity_heuristic_w6", "entity_heuristic_w5"],
                "embedding": ["nemotron_3_embed_1b_bf16"],
                "store": ["FAISS"],
                "retrieval_method": ["BM25 + Dense + RRF"],
                "reranker": ["all"],
            },
            limit_queries="0",
            max_runs="10",
            run_id="20260810-0642",
        )


def test_candidate_preflight_validates_candidate_config_only() -> None:
    assert dashboard.candidate_preflight_command() == [
        dashboard.sys.executable,
        "scripts/benchmark_cli.py",
        "validate",
        "configs/benchmark.meeting-400-candidates.json",
    ]


def test_candidate_lane_request_rejects_non_candidate_fixed_axes() -> None:
    with pytest.raises(ValueError, match="embedding"):
        dashboard.validate_candidate_request({
            "sheet": ["entity_heuristic_w6"],
            "embedding": ["jina-v3"],
            "store": ["FAISS"],
            "retrieval_method": ["BM25 + Dense + RRF"],
            "reranker": ["none"],
        })


def test_official_route_rejects_a_candidate_retrieval_method() -> None:
    with pytest.raises(ValueError, match="candidate lane"):
        dashboard.validate_modular_lane_request(
            {"retrieval_method": ["BM25 + Dense + RRF"]},
            candidate_lane=False,
        )


def test_dashboard_returns_client_error_for_invalid_candidate_request() -> None:
    source = Path(dashboard.__file__).read_text(encoding="utf-8")
    assert 'except ValueError as exc:\n            self.send_json({"error": str(exc)}, 400)' in source


def test_candidate_artifacts_are_excluded_from_official_metric_sources(tmp_path, monkeypatch) -> None:
    latest = tmp_path / "modular_runs" / "latest"
    candidate = tmp_path / "modular_runs" / "meeting-400-candidates" / "candidate-run"
    latest.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (latest / "modular_summary.csv").write_text("chunker\n", encoding="utf-8")
    candidate_summary = candidate / "modular_summary.csv"
    candidate_summary.write_text("chunker\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "MODULAR_DIR", latest)

    source_paths = {path for _source, path in dashboard.official_metric_sources()}

    assert candidate_summary not in source_paths
