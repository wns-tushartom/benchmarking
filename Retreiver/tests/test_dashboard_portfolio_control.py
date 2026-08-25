from __future__ import annotations

import json
from pathlib import Path

import scripts.serve_benchmark_dashboard as dashboard
from benchmarking.core.config import load_benchmark_config
from benchmarking.core.portfolio import build_portfolio_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.all-methods-portfolio.json"


def not_run_status(plan):
    batches = [
        {
            "batch_id": batch.batch_id,
            "embedding": batch.embedding,
            "reranker_group": batch.reranker_group,
            "combination_count": len(batch.combination_ids),
            "state": "not_run",
            "selected": index == 0,
            "error": None,
        }
        for index, batch in enumerate(plan.batches)
    ]
    return {
        "schema_version": 1,
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "promotion_status": "not_accepted",
        "batch_count": len(plan.batches),
        "max_combinations_per_batch": plan.max_combinations_per_batch,
        "selected_batch_id": plan.batches[0].batch_id,
        "state_counts": {"completed": 0, "failed": 0, "not_run": len(plan.batches)},
        "batches": batches,
    }


def test_portfolio_summary_preserves_configured_excluded_and_batch_states() -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)

    payload = dashboard.portfolio_dashboard_payload(config, plan, not_run_status(plan))

    assert payload["schema_version"] == 1
    assert payload["portfolio_id"] == plan.portfolio_id
    assert payload["promotion_status"] == "not_accepted"
    assert payload["configured_combination_count"] == 2160
    assert payload["excluded_combination_count"] == 2160
    assert payload["raw_combination_count"] == 4320
    assert payload["batch_count"] == 12
    assert len(payload["batches"]) == 12
    assert all(batch["combination_count"] == 180 for batch in payload["batches"])
    assert all(batch["combination_count"] <= 250 for batch in payload["batches"])
    assert payload["combination_state_counts"] == {
        "configured": 2160,
        "excluded": 2160,
        "selected": 180,
        "completed": 0,
        "failed": 0,
        "not_run": 2160,
    }
    assert "/home/" not in json.dumps(payload)


def test_portfolio_details_retains_every_configured_and_excluded_combination() -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)

    payload = dashboard.portfolio_dashboard_payload(
        config,
        plan,
        not_run_status(plan),
        include_combinations=True,
    )

    configured = payload["configured_combinations"]
    excluded = payload["excluded_combinations"]
    assert len(configured) == 2160
    assert len(excluded) == 2160
    assert len({row["combination_id"] for row in configured}) == 2160
    assert all(row["execution_state"] in {"selected", "not_run"} for row in configured)
    assert all(row["state"] == "excluded" for row in excluded)
    assert all(row["reason_code"] == "incompatible_vector_store_index" for row in excluded)
    assert "accepted" not in {row["execution_state"] for row in configured}


class FakeGetRequest:
    def __init__(self, path: str) -> None:
        self.path = path
        self.responses: list[tuple[int, dict]] = []

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.responses.append((status, payload))


def test_get_portfolio_routes_summary_and_detailed_payloads(monkeypatch) -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    status = not_run_status(plan)
    monkeypatch.setattr(dashboard, "load_dashboard_portfolio_plan", lambda: plan)
    monkeypatch.setattr(dashboard, "dashboard_portfolio_status", lambda _plan: (config, status))

    summary_request = FakeGetRequest("/api/portfolio")
    dashboard.Handler.do_GET(summary_request)  # type: ignore[arg-type]
    detailed_request = FakeGetRequest("/api/portfolio/batches")
    dashboard.Handler.do_GET(detailed_request)  # type: ignore[arg-type]

    assert summary_request.responses[0][0] == 200
    assert summary_request.responses[0][1]["configured_combination_count"] == 2160
    assert "configured_combinations" not in summary_request.responses[0][1]
    assert detailed_request.responses[0][0] == 200
    assert len(detailed_request.responses[0][1]["configured_combinations"]) == 2160


def test_launch_portfolio_batch_accepts_only_exact_immutable_batch(monkeypatch) -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    status = not_run_status(plan)
    launched: list[list[str]] = []
    monkeypatch.setattr(dashboard, "load_dashboard_portfolio_plan", lambda: plan)
    monkeypatch.setattr(dashboard, "dashboard_portfolio_status", lambda _plan: (config, status))
    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda command: launched.append(command) or {"job_id": "job-safe", "running": True},
    )

    result = dashboard.launch_portfolio_action(
        "run-batch", {"batch_id": plan.batches[3].batch_id}
    )

    assert result["batch_id"] == plan.batches[3].batch_id
    assert result["promotion_status"] == "not_accepted"
    assert launched and launched[0][1:3] == [
        "scripts/benchmark_cli.py",
        "portfolio-run-batch",
    ]
    assert "--batch-id" in launched[0]
    assert launched[0][launched[0].index("--batch-id") + 1] == plan.batches[3].batch_id
    assert "--plan" in launched[0]
    assert "--limit-queries" not in launched[0]

    try:
        dashboard.launch_portfolio_action(
            "run-batch",
            {"batch_id": plan.batches[0].batch_id, "output_dir": "/tmp/escape"},
        )
    except dashboard.DashboardControlError as exc:
        assert exc.code == "invalid_request"
    else:
        raise AssertionError("browser-supplied output path must be rejected")


def test_launch_run_next_uses_receipt_selected_batch_and_stops_when_complete(monkeypatch) -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    status = not_run_status(plan)
    launched: list[list[str]] = []
    monkeypatch.setattr(dashboard, "load_dashboard_portfolio_plan", lambda: plan)
    monkeypatch.setattr(dashboard, "dashboard_portfolio_status", lambda _plan: (config, status))
    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda command: launched.append(command) or {"job_id": "job-next", "running": True},
    )

    result = dashboard.launch_portfolio_action("run-next", {})
    assert result["batch_id"] == status["selected_batch_id"]
    assert len(launched) == 1

    complete = {**status, "selected_batch_id": None}
    monkeypatch.setattr(dashboard, "dashboard_portfolio_status", lambda _plan: (config, complete))
    result = dashboard.launch_portfolio_action("run-next", {})
    assert result == {
        "ok": True,
        "portfolio_id": plan.portfolio_id,
        "promotion_status": "not_accepted",
        "state": "completed",
        "message": "all portfolio batches are already completed",
    }
    assert len(launched) == 1


def test_post_portfolio_run_requires_control_token(monkeypatch) -> None:
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    monkeypatch.setenv("WNS_ENABLE_ADAPTER_CONTROL", "1")
    monkeypatch.setenv("WNS_ADAPTER_CONTROL_TOKEN", "operator")
    monkeypatch.setattr(
        dashboard,
        "launch_portfolio_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must authenticate first")),
    )
    request = dashboard_control_request = type(
        "Request",
        (),
        {
            "path": "/api/portfolio/run-batch",
            "headers": {"Content-Type": "application/json", "Content-Length": "2"},
            "rfile": __import__("io").BytesIO(b"{}"),
            "responses": [],
            "close_connection": False,
            "send_json": lambda self, payload, status=200: self.responses.append((status, payload)),
        },
    )()

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert dashboard_control_request.responses[0][0] == 401
    assert dashboard_control_request.responses[0][1]["error"]["code"] == "unauthorized"
    assert plan.portfolio_id
