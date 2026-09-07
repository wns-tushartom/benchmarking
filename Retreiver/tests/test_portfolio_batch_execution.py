from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarking.core import runner
from benchmarking.core.config import generate_matrix, load_benchmark_config, technique
from benchmarking.core.portfolio import PortfolioPlan, build_portfolio_plan
from benchmarking.core.registry import default_registry
from scripts import benchmark_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.all-methods-portfolio.json"


def test_portfolio_result_rows_include_immutable_context():
    row = {"combination_id": "combo_123", "status": "completed"}
    context = {
        "portfolio_id": "portfolio_abc",
        "portfolio_hash": "abc",
        "batch_id": "batch_def",
        "promotion_status": "not_accepted",
    }

    assert runner.portfolio_result_row(row, context) == {**row, **context}
    assert runner.portfolio_result_row(row, None) == row


def test_default_registry_exposes_turbovec_without_importing_optional_package():
    cls = default_registry().get("vector_store", "turbovec")
    assert cls.__name__ == "TurboVecVectorStoreAdapter"


def test_portfolio_config_has_executable_contract_for_every_declared_method():
    cfg = load_benchmark_config(CONFIG)
    assert cfg["experiment"]["dataset"] == "data/qa_text_test.csv"
    assert cfg["experiment"]["output_lane"] == "all-methods-portfolio"
    kind_by_axis = {
        "chunkers": "chunkers",
        "embeddings": "embeddings",
        "vector_stores": "vector_stores",
        "rerankers": "rerankers",
    }
    for axis, kind in kind_by_axis.items():
        for name in cfg["matrix"][axis]:
            assert technique(cfg, kind, name)["adapter"]
    assert cfg["techniques"]["vector_stores"]["TurboVec"] == {
        "adapter": "turbovec",
        "execution_location": "in_process",
        "index_type": "TurboQuant4bit",
        "bits": 4,
    }
    assert cfg["techniques"]["embeddings"]["nemotron_3_embed_8b_bf16"]["dimensions"] == 4096
    assert cfg["techniques"]["rerankers"]["gte_modernbert_base"]["model"] == "Alibaba-NLP/gte-reranker-modernbert-base"


def _repo_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    config_path = root / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG.read_bytes())
    return root, config_path


def test_plan_is_written_only_to_immutable_candidate_portfolio_root(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    plan, path = benchmark_cli.create_portfolio_plan(root, config_path)

    assert path == root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id / "portfolio_plan.json"
    assert PortfolioPlan.from_json(path.read_text(encoding="utf-8")) == plan
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (root / "data" / "modular_runs" / "latest").exists()


def test_plan_writer_rejects_existing_different_or_symlinked_plan(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    plan = build_portfolio_plan(load_benchmark_config(config_path))
    portfolio_root = root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id
    portfolio_root.mkdir(parents=True)
    plan_path = portfolio_root / "portfolio_plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        benchmark_cli.create_portfolio_plan(root, config_path)

    plan_path.unlink()
    target = tmp_path / "target.json"
    target.write_text(plan.to_json(), encoding="utf-8")
    plan_path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        benchmark_cli.create_portfolio_plan(root, config_path)


def test_load_plan_recomputes_current_config_and_rejects_tampering(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    plan, path = benchmark_cli.create_portfolio_plan(root, config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["batches"][0]["combination_ids"][0] = "combination_" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="plan"):
        benchmark_cli.load_verified_portfolio_plan(root, config_path, path)


def test_portfolio_output_directory_requires_exact_two_level_identity(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    cfg = load_benchmark_config(config_path)
    plan = build_portfolio_plan(cfg)
    batch = plan.batches[0]
    expected = root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id / batch.batch_id
    assert runner.validate_output_directory(cfg, root, expected) == expected

    invalid = [
        root / "data" / "modular_runs" / "latest",
        root / "data" / "modular_runs" / "all-methods-portfolio" / "latest" / batch.batch_id,
        root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id,
        expected / "nested",
        root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id / "not-a-batch",
    ]
    for path in invalid:
        with pytest.raises(ValueError, match="portfolio output directory"):
            runner.validate_output_directory(cfg, root, path)


def test_portfolio_output_rejects_symlink_escape(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    cfg = load_benchmark_config(config_path)
    plan = build_portfolio_plan(cfg)
    official = root / "data" / "modular_runs" / "latest"
    official.mkdir(parents=True)
    lane = root / "data" / "modular_runs" / "all-methods-portfolio"
    lane.parent.mkdir(parents=True, exist_ok=True)
    lane.symlink_to(official, target_is_directory=True)
    output = lane / plan.portfolio_id / plan.batches[0].batch_id
    with pytest.raises(ValueError, match="symlink"):
        runner.validate_output_directory(cfg, root, output)


def test_exact_batch_selection_rejects_unknown_duplicate_missing_and_over_cap():
    cfg = load_benchmark_config(CONFIG)
    matrix = generate_matrix(cfg)
    plan = build_portfolio_plan(cfg)
    batch = plan.batches[0]
    selected = runner.select_declared_combinations(matrix, batch.combination_ids, maximum=250)
    assert len(selected) == 180
    assert {row["combination_id"] for row in selected} == set(batch.combination_ids)

    with pytest.raises(ValueError, match="duplicate"):
        runner.select_declared_combinations(matrix, [batch.combination_ids[0]] * 2)
    with pytest.raises(ValueError, match="unknown"):
        runner.select_declared_combinations(matrix, ["combination_" + "f" * 64])
    with pytest.raises(ValueError, match="exceeds"):
        runner.select_declared_combinations(matrix, plan.combination_ids[:251], maximum=250)


def test_run_batch_passes_exact_ids_and_never_uses_max_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, config_path = _repo_fixture(tmp_path)
    plan, plan_path = benchmark_cli.create_portfolio_plan(root, config_path)
    batch = plan.batches[3]
    calls = []

    def fake_run(config_path_arg, root_arg, output_dir, **kwargs):
        calls.append((config_path_arg, root_arg, output_dir, kwargs))
        return {"best_config": {}}

    monkeypatch.setattr(benchmark_cli, "run_experiment", fake_run)
    result = benchmark_cli.run_portfolio_batch(
        root,
        config_path,
        plan_path,
        batch.batch_id,
        limit_queries=3,
    )

    assert result["batch_id"] == batch.batch_id
    assert result["combination_count"] == 180
    _, _, output_dir, kwargs = calls[0]
    assert output_dir == root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id / batch.batch_id
    assert kwargs["max_runs"] == 0
    assert kwargs["combination_ids"] == batch.combination_ids
    assert kwargs["limit_queries"] == 3
    assert kwargs["portfolio_context"]["promotion_status"] == "not_accepted"


def test_run_batch_rejects_unknown_batch_before_runner_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, config_path = _repo_fixture(tmp_path)
    _, plan_path = benchmark_cli.create_portfolio_plan(root, config_path)
    monkeypatch.setattr(benchmark_cli, "run_experiment", lambda *args, **kwargs: pytest.fail("runner must not start"))
    with pytest.raises(ValueError, match="unknown batch"):
        benchmark_cli.run_portfolio_batch(root, config_path, plan_path, "batch_" + "0" * 64)


def test_runner_refuses_max_runs_with_declared_combination_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, config_path = _repo_fixture(tmp_path)
    plan = build_portfolio_plan(load_benchmark_config(config_path))
    monkeypatch.setattr(runner, "load_env_file", lambda *_: pytest.fail("must fail before environment access"))
    with pytest.raises(ValueError, match="max_runs"):
        runner.run_experiment(
            config_path,
            root,
            root / "data" / "modular_runs" / "all-methods-portfolio" / plan.portfolio_id / plan.batches[0].batch_id,
            max_runs=1,
            combination_ids=plan.batches[0].combination_ids,
        )


def test_runner_requires_declared_batch_for_every_portfolio_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path = _repo_fixture(tmp_path)
    plan = build_portfolio_plan(load_benchmark_config(config_path))
    monkeypatch.setattr(
        runner, "load_env_file", lambda *_: pytest.fail("must fail before environment access")
    )
    with pytest.raises(ValueError, match="declared portfolio batch"):
        runner.run_experiment(
            config_path,
            root,
            root
            / "data"
            / "modular_runs"
            / "all-methods-portfolio"
            / plan.portfolio_id
            / plan.batches[0].batch_id,
            max_runs=1,
        )


def test_runner_binds_output_path_to_exact_immutable_batch_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path = _repo_fixture(tmp_path)
    plan = build_portfolio_plan(load_benchmark_config(config_path))
    batch = plan.batches[0]
    context = {
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "batch_id": batch.batch_id,
        "promotion_status": "not_accepted",
    }
    monkeypatch.setattr(
        runner, "load_env_file", lambda *_: pytest.fail("must fail before environment access")
    )
    wrong = (
        root
        / "data"
        / "modular_runs"
        / "all-methods-portfolio"
        / ("portfolio_" + "a" * 64)
        / ("batch_" + "b" * 64)
    )
    with pytest.raises(ValueError, match="immutable portfolio batch"):
        runner.run_experiment(
            config_path,
            root,
            wrong,
            combination_ids=batch.combination_ids,
            portfolio_context=context,
        )


def test_plan_create_and_load_reject_symlinked_ancestor(tmp_path: Path):
    root, config_path = _repo_fixture(tmp_path)
    external = tmp_path / "external-data"
    external.mkdir()
    (root / "data").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        benchmark_cli.create_portfolio_plan(root, config_path)

    (root / "data").unlink()
    _, plan_path = benchmark_cli.create_portfolio_plan(root, config_path)
    real_data = tmp_path / "real-data"
    (root / "data").rename(real_data)
    (root / "data").symlink_to(real_data, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        benchmark_cli.load_verified_portfolio_plan(root, config_path, plan_path)
