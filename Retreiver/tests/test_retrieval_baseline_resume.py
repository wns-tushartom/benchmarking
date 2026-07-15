import json
from pathlib import Path


def test_retrieval_runner_builds_openai_query_embedding_adapter(monkeypatch) -> None:
    import scripts.run_retrieval_smoke_from_vm_dbs as runner

    monkeypatch.setenv("OPENAI_API_KEY", "unit-secret")
    adapter = runner.make_query_embedding_adapter("openai_text-embedding-3-large")

    assert adapter.__class__.__name__ == "OpenAIEmbeddingAdapter"
    assert adapter.dimensions == 3072


def test_completed_retrieval_artifact_is_skipped_only_for_matching_success(tmp_path: Path) -> None:
    import scripts.run_retrieval_smoke_from_vm_dbs as runner

    row = {"sheet": "Heading_sections_l2", "embedding": "jina_v3", "store": "FAISS"}
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({
        **row,
        "query": "refund ticket",
        "retrieved_count": 5,
        "hits": [{"rank": 1}],
    }), encoding="utf-8")

    assert runner.completed_artifact_matches(path, row, "refund ticket") is True
    assert runner.completed_artifact_matches(path, row, "different query") is False
    path.write_text("{broken", encoding="utf-8")
    assert runner.completed_artifact_matches(path, row, "refund ticket") is False


def test_retrieval_artifact_names_do_not_collide_on_long_query_prefixes(tmp_path: Path) -> None:
    import scripts.run_retrieval_smoke_from_vm_dbs as runner

    row = {"sheet": "Heading_sections_l2", "embedding": "jina_v3", "store": "FAISS"}
    prefix = "same query prefix repeated for forty chars--"
    first = runner.artifact_path(tmp_path, row, prefix + "first")
    second = runner.artifact_path(tmp_path, row, prefix + "second")

    assert first != second
    assert first.suffix == ".json"
    assert runner.legacy_artifact_path(tmp_path, row, prefix + "first") == runner.legacy_artifact_path(tmp_path, row, prefix + "second")


def test_retrieval_runner_cli_supports_skip_existing_success_and_required_coverage() -> None:
    source = Path("scripts/run_retrieval_smoke_from_vm_dbs.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--skip-existing-success", action="store_true")' in source
    assert 'parser.add_argument("--require-combos", type=int, default=0)' in source
    assert "required_combo_count_mismatch" in source
    assert "SKIP existing" in source
