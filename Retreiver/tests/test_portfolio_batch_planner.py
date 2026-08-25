from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarking.core.config import load_benchmark_config
from benchmarking.core.portfolio import (
    MAX_BATCH_COMBINATIONS,
    PORTFOLIO_STATES,
    RERANKER_GROUPS,
    PortfolioState,
    build_portfolio_plan,
    reconcile_run_states,
    validate_portfolio_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_CONFIG = ROOT / "configs" / "benchmark.all-methods-portfolio.json"


def _config() -> dict:
    return load_benchmark_config(PORTFOLIO_CONFIG)


def test_planner_builds_deterministic_twelve_by_180_plan() -> None:
    first = build_portfolio_plan(_config())
    second = build_portfolio_plan(_config())

    assert first == second
    assert first.portfolio_hash == second.portfolio_hash
    assert len(first.combination_ids) == 2160
    assert len(first.batches) == 12
    assert len({batch.batch_id for batch in first.batches}) == 12
    assert MAX_BATCH_COMBINATIONS == 250
    assert all(len(batch.combination_ids) == 180 for batch in first.batches)
    assert all(len(batch.combination_ids) <= first.max_combinations_per_batch for batch in first.batches)

    seen = [combination_id for batch in first.batches for combination_id in batch.combination_ids]
    assert len(seen) == len(set(seen)) == 2160
    assert set(seen) == set(first.combination_ids)


def test_each_batch_has_one_embedding_and_one_approved_reranker_group() -> None:
    plan = build_portfolio_plan(_config())

    assert [(batch.embedding, batch.reranker_group) for batch in plan.batches] == [
        (embedding, group)
        for embedding in _config()["matrix"]["embeddings"]
        for group in RERANKER_GROUPS
    ]
    assert all(batch.rerankers == RERANKER_GROUPS[batch.reranker_group] for batch in plan.batches)


def test_reranker_group_dictionary_order_does_not_change_plan() -> None:
    config = _config()
    reordered_groups = dict(reversed(list(config["portfolio"]["reranker_groups"].items())))

    assert build_portfolio_plan(config, reranker_groups=reordered_groups) == build_portfolio_plan(config)


def test_plan_json_is_canonical_and_round_trip_validatable() -> None:
    plan = build_portfolio_plan(_config())
    payload = plan.to_dict()

    assert json.loads(plan.to_json()) == payload
    assert validate_portfolio_plan(plan) is plan
    assert payload["portfolio_hash"] == plan.portfolio_hash
    assert payload["batches"][0]["batch_id"] == plan.batches[0].batch_id


@pytest.mark.parametrize("cap", [0, -1, 251, True, 1.5])
def test_planner_rejects_malformed_or_out_of_policy_caps(cap: object) -> None:
    with pytest.raises(ValueError, match="max.*250|positive integer"):
        build_portfolio_plan(_config(), max_combinations_per_batch=cap)  # type: ignore[arg-type]


def test_planner_rejects_unknown_reranker_group() -> None:
    groups = {**RERANKER_GROUPS, "C": ("none",)}
    with pytest.raises(ValueError, match="reranker group"):
        build_portfolio_plan(_config(), reranker_groups=groups)


def test_planner_rejects_duplicate_group_member() -> None:
    groups = {
        "A": RERANKER_GROUPS["A"],
        "B": (*RERANKER_GROUPS["B"], "none"),
    }
    with pytest.raises(ValueError, match="duplicate reranker"):
        build_portfolio_plan(_config(), reranker_groups=groups)


def test_planner_rejects_missing_reranker_coverage() -> None:
    groups = {
        "A": RERANKER_GROUPS["A"],
        "B": RERANKER_GROUPS["B"][:-1],
    }
    with pytest.raises(ValueError, match="coverage"):
        build_portfolio_plan(_config(), reranker_groups=groups)


def test_plan_validation_rejects_missing_combination_coverage() -> None:
    plan = build_portfolio_plan(_config())
    first_batch = replace(plan.batches[0], combination_ids=plan.batches[0].combination_ids[:-1])
    malformed = replace(plan, batches=(first_batch, *plan.batches[1:]))

    with pytest.raises(ValueError, match="coverage"):
        validate_portfolio_plan(malformed)


def test_planner_rejects_malformed_config() -> None:
    malformed = _config()
    malformed["matrix"].pop("embeddings")

    with pytest.raises(ValueError, match="embeddings"):
        build_portfolio_plan(malformed)


def test_portfolio_and_run_states_are_distinct_and_reconciled_immutably() -> None:
    assert PORTFOLIO_STATES == {
        "configured",
        "excluded",
        "selected",
        "completed",
        "failed",
        "not_run",
    }
    assert len({state.value for state in PortfolioState}) == 6

    plan = build_portfolio_plan(_config())
    batch = plan.batches[0]
    completed_id, failed_id, selected_id = batch.combination_ids[:3]
    before = plan.to_json()
    states = reconcile_run_states(
        plan,
        selected_batch_id=batch.batch_id,
        receipts={completed_id: "completed", failed_id: "failed"},
    )

    assert states[completed_id] is PortfolioState.COMPLETED
    assert states[failed_id] is PortfolioState.FAILED
    assert states[selected_id] is PortfolioState.SELECTED
    assert states[plan.batches[1].combination_ids[0]] is PortfolioState.NOT_RUN
    assert plan.to_json() == before


def test_reconciliation_rejects_unknown_receipts_and_invalid_states() -> None:
    plan = build_portfolio_plan(_config())
    with pytest.raises(ValueError, match="unknown combination"):
        reconcile_run_states(plan, receipts={"not-a-combination": "completed"})
    with pytest.raises(ValueError, match="receipt state"):
        reconcile_run_states(plan, receipts={plan.combination_ids[0]: "selected"})
