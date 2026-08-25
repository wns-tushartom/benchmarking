"""Candidate-only portfolio payloads for the benchmark dashboard."""

from __future__ import annotations

from typing import Any, Mapping

from benchmarking.core.config import generate_matrix_catalog
from benchmarking.core.portfolio import MAX_BATCH_COMBINATIONS, PortfolioPlan
from scripts.dashboard_control_api import required_service_ids_for_batch


_PUBLIC_AXES = (
    "combination_id",
    "chunker",
    "embedding",
    "vector_store",
    "index_type",
    "retrieval_method",
    "reranker",
    "evaluator",
)


def _public_axes(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _PUBLIC_AXES if key in row}


def portfolio_dashboard_payload(
    config: Mapping[str, Any],
    plan: PortfolioPlan,
    status: Mapping[str, Any],
    *,
    include_combinations: bool = False,
) -> dict[str, Any]:
    """Merge immutable planning and receipt state without exposing server paths."""
    catalog = generate_matrix_catalog(dict(config))
    configured = catalog["configured"]
    excluded = catalog["excluded"]
    if len(configured) != len(plan.combination_ids):
        raise ValueError("portfolio configuration no longer matches immutable plan coverage")
    if tuple(row["combination_id"] for row in configured) != plan.combination_ids:
        raise ValueError("portfolio configuration order no longer matches immutable plan")

    observed = {item["batch_id"]: item for item in status.get("batches", [])}
    batch_by_combination: dict[str, Any] = {}
    public_batches: list[dict[str, Any]] = []
    completed = failed = not_run = selected = 0
    for batch in plan.batches:
        if len(batch.combination_ids) > MAX_BATCH_COMBINATIONS:
            raise ValueError("immutable portfolio batch exceeds execution maximum")
        observation = observed.get(batch.batch_id)
        if not isinstance(observation, Mapping):
            raise ValueError("portfolio status omits an immutable batch")
        state = observation.get("state")
        if state not in {"completed", "failed", "not_run"}:
            raise ValueError("portfolio status contains an invalid batch state")
        is_selected = observation.get("selected") is True
        count = len(batch.combination_ids)
        if state == "completed":
            completed += count
        elif state == "failed":
            failed += count
        else:
            not_run += count
        if is_selected:
            selected += count
        for combination_id in batch.combination_ids:
            batch_by_combination[combination_id] = (batch, state, is_selected)
        public_batches.append(
            {
                "batch_id": batch.batch_id,
                "embedding": batch.embedding,
                "reranker_group": batch.reranker_group,
                "rerankers": list(batch.rerankers),
                "combination_count": count,
                "state": state,
                "selected": is_selected,
                "integrity_error": bool(observation.get("error")),
                "required_service_ids": list(required_service_ids_for_batch(batch)),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "promotion_status": "not_accepted",
        "configured_combination_count": len(configured),
        "excluded_combination_count": len(excluded),
        "raw_combination_count": len(configured) + len(excluded),
        "batch_count": len(plan.batches),
        "max_combinations_per_batch": plan.max_combinations_per_batch,
        "selected_batch_id": status.get("selected_batch_id"),
        "combination_state_counts": {
            "configured": len(configured),
            "excluded": len(excluded),
            "selected": selected,
            "completed": completed,
            "failed": failed,
            "not_run": not_run,
        },
        "batches": public_batches,
    }
    if include_combinations:
        configured_rows: list[dict[str, Any]] = []
        for row in configured:
            batch, state, is_selected = batch_by_combination[row["combination_id"]]
            execution_state = "selected" if is_selected and state == "not_run" else state
            configured_rows.append(
                {
                    **_public_axes(row),
                    "state": "configured",
                    "execution_state": execution_state,
                    "batch_id": batch.batch_id,
                    "selected": is_selected,
                }
            )
        payload["configured_combinations"] = configured_rows
        payload["excluded_combinations"] = [
            {
                **_public_axes(row),
                "state": "excluded",
                "reason_code": row.get("reason_code"),
            }
            for row in excluded
        ]
    return payload
