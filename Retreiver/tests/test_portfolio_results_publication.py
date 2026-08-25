from __future__ import annotations

import csv
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarking.core.portfolio import PORTFOLIO_STATES, PortfolioPlan
from scripts import benchmark_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.all-methods-portfolio.json"
PUBLISHER_PATH = ROOT / "scripts" / "publish_portfolio_results.py"
ARTIFACT_NAMES = (
    "modular_summary.csv",
    "modular_details.csv",
    "analysis.json",
    "MODULAR_REPORT.md",
    "config_snapshot.json",
    "combination_manifest.json",
    "provider_readiness_receipt.json",
)


def _publisher():
    assert PUBLISHER_PATH.is_file(), "portfolio publisher has not been implemented"
    importlib.invalidate_caches()
    return importlib.import_module("scripts.publish_portfolio_results")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, PortfolioPlan, Path]:
    root = tmp_path / "repo"
    config_path = root / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG.read_bytes())
    plan, plan_path = benchmark_cli.create_portfolio_plan(root, config_path)
    return root, config_path, plan, plan_path


def _write_valid_batch(
    portfolio_root: Path,
    plan: PortfolioPlan,
    batch_index: int,
    *,
    manifest_ids: tuple[str, ...] | None = None,
    summary_ids: tuple[str, ...] | None = None,
    promotion_status: str = "not_accepted",
) -> Path:
    batch = plan.batches[batch_index]
    batch_dir = portfolio_root / batch.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = batch.combination_ids
    manifest_ids = expected_ids if manifest_ids is None else manifest_ids
    summary_ids = expected_ids if summary_ids is None else summary_ids
    context = {
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "batch_id": batch.batch_id,
        "promotion_status": promotion_status,
    }

    with (batch_dir / "modular_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("combination_id", "batch_id", "promotion_status", "status", "score"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row_index, combination_id in enumerate(summary_ids):
            writer.writerow({
                "combination_id": combination_id,
                "batch_id": batch.batch_id,
                "promotion_status": promotion_status,
                "status": "completed",
                "score": f"{batch_index}.{row_index:03d}",
            })
    (batch_dir / "modular_details.csv").write_text(
        "combination_id,query_id\n", encoding="utf-8"
    )
    _write_json(batch_dir / "analysis.json", {"candidate": True})
    (batch_dir / "MODULAR_REPORT.md").write_text("# Candidate batch\n", encoding="utf-8")
    _write_json(batch_dir / "config_snapshot.json", {"candidate": True})
    _write_json(
        batch_dir / "combination_manifest.json",
        {
            "schema_version": 1,
            **context,
            "combination_count": len(manifest_ids),
            "combination_ids": list(manifest_ids),
            "combinations": [{"combination_id": value} for value in manifest_ids],
        },
    )
    _write_json(
        batch_dir / "provider_readiness_receipt.json",
        {"schema_version": 1, **context, "provider_readiness": {}, "state": "checked"},
    )
    artifact_sha256 = {name: _sha256(batch_dir / name) for name in ARTIFACT_NAMES}
    _write_json(
        batch_dir / "manifest.json",
        {
            "status": "completed",
            "promotion_status": promotion_status,
            "portfolio": context,
            "artifact_root": (
                f"all-methods-portfolio/{plan.portfolio_id}/{batch.batch_id}"
            ),
            "artifact_sha256": artifact_sha256,
        },
    )
    _write_json(
        batch_dir / "completion_receipt.json",
        {
            "schema_version": 1,
            **context,
            "state": "completed",
            "combination_count": len(expected_ids),
            "manifest_sha256": _sha256(batch_dir / "manifest.json"),
            "artifact_sha256": artifact_sha256,
        },
    )
    return batch_dir


def _write_all_valid(portfolio_root: Path, plan: PortfolioPlan) -> None:
    for batch_index in range(len(plan.batches)):
        _write_valid_batch(portfolio_root, plan, batch_index)


def test_cli_exposes_portfolio_status_and_run_next_commands():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_cli.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "portfolio-status" in result.stdout
    assert "run-next" in result.stdout


def test_status_uses_plan_order_and_never_calls_incomplete_receipt_completed(tmp_path: Path):
    _, _, plan, plan_path = _fixture(tmp_path)
    portfolio_root = plan_path.parent
    _write_valid_batch(portfolio_root, plan, 0)
    failed_dir = portfolio_root / plan.batches[1].batch_id
    failed_dir.mkdir()
    (failed_dir / "completion_receipt.json").write_text("{}", encoding="utf-8")

    status = benchmark_cli.portfolio_status(plan_path)

    assert status["portfolio_id"] == plan.portfolio_id
    assert status["configured_combination_count"] == 2160
    assert status["excluded_combination_count"] == 0
    assert status["batch_count"] == 12
    assert status["state_values"] == sorted(PORTFOLIO_STATES)
    assert [item["state"] for item in status["batches"][:3]] == [
        "completed",
        "failed",
        "not_run",
    ]
    assert status["selected_batch_id"] == plan.batches[1].batch_id
    assert status["batches"][1]["selected"] is True
    assert status["state_counts"] == {"completed": 1, "failed": 1, "not_run": 10}
    assert status["combination_state_counts"] == {
        "configured": 2160,
        "excluded": 0,
        "selected": 180,
        "completed": 180,
        "failed": 180,
        "not_run": 1800,
    }


def test_status_rejects_reordered_manifest_even_when_all_hashes_are_fresh(tmp_path: Path):
    _, _, plan, plan_path = _fixture(tmp_path)
    _write_valid_batch(
        plan_path.parent,
        plan,
        0,
        manifest_ids=tuple(reversed(plan.batches[0].combination_ids)),
    )

    status = benchmark_cli.portfolio_status(plan_path)

    assert status["batches"][0]["state"] == "failed"
    assert "ordered combination IDs" in status["batches"][0]["error"]


def test_run_next_skips_only_verified_completed_batches_and_retries_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, config_path, plan, plan_path = _fixture(tmp_path)
    _write_valid_batch(plan_path.parent, plan, 0)
    _write_valid_batch(
        plan_path.parent,
        plan,
        1,
        manifest_ids=tuple(reversed(plan.batches[1].combination_ids)),
    )
    calls: list[str] = []

    def fake_run(*args, **kwargs):
        calls.append(args[3])
        return {"ok": True, "batch_id": args[3]}

    monkeypatch.setattr(benchmark_cli, "run_portfolio_batch", fake_run)
    result = benchmark_cli.run_next_portfolio_batch(root, config_path, plan_path, limit_queries=2)

    assert calls == [plan.batches[1].batch_id]
    assert result["batch_id"] == plan.batches[1].batch_id
    assert result["resumed_from_state"] == "failed"

    _write_valid_batch(plan_path.parent, plan, 1)
    for batch_index in range(2, len(plan.batches)):
        _write_valid_batch(plan_path.parent, plan, batch_index)
    calls.clear()
    result = benchmark_cli.run_next_portfolio_batch(root, config_path, plan_path)
    assert result == {
        "ok": True,
        "portfolio_id": plan.portfolio_id,
        "state": "completed",
        "message": "all portfolio batches are already completed",
    }
    assert calls == []


def test_publication_requires_every_expected_batch(tmp_path: Path):
    _, _, _, plan_path = _fixture(tmp_path)

    with pytest.raises(ValueError, match="all 12 expected batch directories"):
        _publisher().publish_portfolio_results(plan_path.parent)


def test_publication_writes_candidate_only_exact_coverage_and_is_idempotent(tmp_path: Path):
    root, _, plan, plan_path = _fixture(tmp_path)
    portfolio_root = plan_path.parent
    _write_all_valid(portfolio_root, plan)

    result = _publisher().publish_portfolio_results(portfolio_root)

    receipt_path = portfolio_root / "portfolio_receipt.json"
    aggregate_path = portfolio_root / "aggregate_candidate_summary.csv"
    assert result["portfolio_receipt"] == str(receipt_path)
    assert result["aggregate_candidate_summary"] == str(aggregate_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promotion_status"] == "not_accepted"
    assert receipt["batch_count"] == 12
    assert receipt["combination_count"] == 2160
    assert receipt["aggregate_candidate_summary_sha256"] == _sha256(aggregate_path)
    with aggregate_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_order = [
        combination_id
        for batch in plan.batches
        for combination_id in batch.combination_ids
    ]
    assert [row["combination_id"] for row in rows] == expected_order
    assert len({row["combination_id"] for row in rows}) == 2160
    assert {row["promotion_status"] for row in rows} == {"not_accepted"}
    assert not (root / "data" / "modular_runs" / "latest").exists()

    before = receipt_path.read_bytes(), aggregate_path.read_bytes()
    second = _publisher().publish_portfolio_results(plan_path)
    assert second == result
    assert (receipt_path.read_bytes(), aggregate_path.read_bytes()) == before


def test_publication_fails_closed_on_artifact_hash_or_summary_coverage_tampering(tmp_path: Path):
    _, _, plan, plan_path = _fixture(tmp_path)
    portfolio_root = plan_path.parent
    _write_all_valid(portfolio_root, plan)
    first_dir = portfolio_root / plan.batches[0].batch_id
    (first_dir / "modular_summary.csv").write_text("combination_id\ntampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash"):
        _publisher().publish_portfolio_results(portfolio_root)

    shutil.rmtree(first_dir)
    duplicate_ids = list(plan.batches[0].combination_ids)
    duplicate_ids[1] = duplicate_ids[0]
    _write_valid_batch(
        portfolio_root,
        plan,
        0,
        summary_ids=tuple(duplicate_ids),
    )
    with pytest.raises(ValueError, match="exactly once"):
        _publisher().publish_portfolio_results(portfolio_root)


def test_publication_rejects_wrong_promotion_symlinks_and_noncanonical_lane(tmp_path: Path):
    _, _, plan, plan_path = _fixture(tmp_path)
    portfolio_root = plan_path.parent
    _write_all_valid(portfolio_root, plan)
    wrong_dir = portfolio_root / plan.batches[0].batch_id
    shutil.rmtree(wrong_dir)
    _write_valid_batch(portfolio_root, plan, 0, promotion_status="accepted")
    with pytest.raises(ValueError, match="promotion_status"):
        _publisher().publish_portfolio_results(portfolio_root)

    shutil.rmtree(wrong_dir)
    _write_valid_batch(portfolio_root, plan, 0)
    summary = wrong_dir / "modular_summary.csv"
    target = tmp_path / "aliased-summary.csv"
    target.write_bytes(summary.read_bytes())
    summary.unlink()
    summary.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _publisher().publish_portfolio_results(portfolio_root)

    alias_root = tmp_path / "latest" / plan.portfolio_id
    alias_root.parent.mkdir()
    shutil.copytree(portfolio_root, alias_root, symlinks=True)
    with pytest.raises(ValueError, match="candidate portfolio root"):
        _publisher().publish_portfolio_results(alias_root)


def test_existing_publication_conflict_fails_closed(tmp_path: Path):
    _, _, plan, plan_path = _fixture(tmp_path)
    portfolio_root = plan_path.parent
    _write_all_valid(portfolio_root, plan)
    publisher = _publisher()
    publisher.publish_portfolio_results(portfolio_root)
    (portfolio_root / "aggregate_candidate_summary.csv").write_text(
        "combination_id\nconflict\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="existing publication conflicts"):
        publisher.publish_portfolio_results(portfolio_root)
