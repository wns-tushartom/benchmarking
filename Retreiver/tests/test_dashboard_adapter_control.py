from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

import scripts.serve_benchmark_dashboard as dashboard


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def statuses(self):
        return [
            {
                "service_id": "jina_embedding",
                "port": 5000,
                "label": "Jina embedding",
                "role": "embedding",
                "status": "offline",
                "ok": False,
                "startable": True,
                "stoppable": True,
                "message": "Not running",
            }
        ]

    def start(self, service_id: str):
        self.calls.append(("start", service_id))
        return {"service_id": service_id, "status": "starting", "ok": True}

    def retry(self, service_id: str):
        self.calls.append(("retry", service_id))
        return {"service_id": service_id, "status": "starting", "ok": True}

    def stop(self, service_id: str):
        self.calls.append(("stop", service_id))
        return {"service_id": service_id, "status": "offline", "ok": True}

    def start_required(self, service_ids):
        ids = tuple(service_ids)
        self.calls.append(("start_required", ids))
        return [{"service_id": service_id, "status": "starting", "ok": True} for service_id in ids]


def enabled_env() -> dict[str, str]:
    return {
        "WNS_ENABLE_ADAPTER_CONTROL": "1",
        "WNS_ADAPTER_CONTROL_TOKEN": "correct horse battery staple",
    }


def bearer(token: str = "correct horse battery staple") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_operator_control_is_disabled_by_default() -> None:
    with pytest.raises(dashboard.DashboardControlError) as exc:
        dashboard.require_operator_control(bearer(), environ={})
    assert exc.value.status == 403
    assert exc.value.code == "control_disabled"


def test_operator_control_requires_exact_constant_time_bearer(monkeypatch) -> None:
    comparisons: list[tuple[str, str]] = []

    def compared(expected: str, supplied: str) -> bool:
        comparisons.append((expected, supplied))
        return expected == supplied

    monkeypatch.setattr(dashboard, "verify_operator_token", compared)
    with pytest.raises(dashboard.DashboardControlError) as exc:
        dashboard.require_operator_control(bearer("wrong"), environ=enabled_env())
    assert exc.value.status == 401
    assert exc.value.code == "unauthorized"
    assert comparisons == [("correct horse battery staple", "wrong")]
    assert dashboard.require_operator_control(bearer(), environ=enabled_env()) is None


@pytest.mark.parametrize("action", ["start", "retry", "stop"])
def test_adapter_mutations_accept_only_an_allowlisted_service_id(action: str) -> None:
    manager = FakeManager()
    result = dashboard.adapter_control_action(
        action,
        {"service_id": "jina_embedding"},
        manager=manager,
    )
    assert result["service_id"] == "jina_embedding"
    assert manager.calls == [(action, "jina_embedding")]

    with pytest.raises(dashboard.DashboardControlError, match="request fields"):
        dashboard.adapter_control_action(
            action,
            {"service_id": "jina_embedding", "port": 5999},
            manager=manager,
        )


def test_adapter_status_payload_contains_only_public_slot_state() -> None:
    payload = dashboard.adapter_status_payload(FakeManager(), control_enabled=False)
    assert payload["schema_version"] == 1
    assert payload["control_enabled"] is False
    assert payload["slot_count"] == 1
    serialized = str(payload).lower()
    for private in ("command", "argv", "environment", "token", "/home/"):
        assert private not in serialized


def test_start_required_derives_services_from_immutable_batch_not_browser_list() -> None:
    manager = FakeManager()
    batch = SimpleNamespace(
        batch_id="batch_001_deadbeefdeadbeef",
        embedding="nemotron_3_embed_1b_bf16",
        rerankers=("none", "bge-reranker-base", "Qwen3:4B Rerank"),
        combination_ids=tuple(f"combination_{index}" for index in range(180)),
    )
    plan = SimpleNamespace(batches=(batch,))

    result = dashboard.adapter_control_action(
        "start-required",
        {"batch_id": batch.batch_id},
        manager=manager,
        plan=plan,
    )
    expected = (
        "nemotron_embed_1b_bf16",
        "pgvector",
        "weaviate_http",
        "qdrant_http",
        "bge_reranker",
        "qwen_reranker",
    )
    assert manager.calls == [("start_required", expected)]
    assert result["batch_id"] == batch.batch_id
    assert tuple(result["required_service_ids"]) == expected

    with pytest.raises(dashboard.DashboardControlError, match="request fields"):
        dashboard.adapter_control_action(
            "start-required",
            {"batch_id": batch.batch_id, "service_ids": ["spare_reserved"]},
            manager=manager,
            plan=plan,
        )


def test_start_required_rejects_unknown_or_oversized_batch() -> None:
    manager = FakeManager()
    batch = SimpleNamespace(
        batch_id="batch_known",
        embedding="jina_v3",
        rerankers=("none",),
        combination_ids=tuple(f"combination_{index}" for index in range(251)),
    )
    plan = SimpleNamespace(batches=(batch,))
    with pytest.raises(dashboard.DashboardControlError, match="maximum"):
        dashboard.adapter_control_action(
            "start-required", {"batch_id": "batch_known"}, manager=manager, plan=plan
        )
    with pytest.raises(dashboard.DashboardControlError, match="unknown batch"):
        dashboard.adapter_control_action(
            "start-required", {"batch_id": "batch_missing"}, manager=manager, plan=plan
        )


class FakeHttpRequest:
    def __init__(
        self,
        path: str,
        payload: object | None = None,
        *,
        authorization: str | None = None,
    ) -> None:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.path = path
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if authorization is not None:
            self.headers["Authorization"] = authorization
        self.rfile = BytesIO(body)
        self.responses: list[tuple[int, dict[str, object]]] = []
        self.close_connection = False

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        self.responses.append((status, payload))


def test_get_adapters_routes_public_status(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "create_adapter_manager", lambda _root: FakeManager())
    monkeypatch.setenv("WNS_ENABLE_ADAPTER_CONTROL", "1")
    request = FakeHttpRequest("/api/adapters")

    dashboard.Handler.do_GET(request)  # type: ignore[arg-type]

    assert request.responses == [
        (
            200,
            {
                "schema_version": 1,
                "control_enabled": True,
                "slot_count": 1,
                "slots": FakeManager().statuses(),
            },
        )
    ]


def test_post_adapter_control_requires_bearer_before_manager_action(monkeypatch) -> None:
    manager = FakeManager()
    monkeypatch.setattr(dashboard, "create_adapter_manager", lambda _root: manager)
    monkeypatch.setenv("WNS_ENABLE_ADAPTER_CONTROL", "1")
    monkeypatch.setenv("WNS_ADAPTER_CONTROL_TOKEN", "correct horse battery staple")
    request = FakeHttpRequest("/api/adapters/start", {"service_id": "jina_embedding"})

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert request.responses[0][0] == 401
    assert request.responses[0][1]["error"]["code"] == "unauthorized"  # type: ignore[index]
    assert manager.calls == []


def test_post_adapter_control_routes_strict_json_to_allowlisted_action(monkeypatch) -> None:
    manager = FakeManager()
    monkeypatch.setattr(dashboard, "create_adapter_manager", lambda _root: manager)
    monkeypatch.setenv("WNS_ENABLE_ADAPTER_CONTROL", "1")
    monkeypatch.setenv("WNS_ADAPTER_CONTROL_TOKEN", "correct horse battery staple")
    request = FakeHttpRequest(
        "/api/adapters/start",
        {"service_id": "jina_embedding"},
        authorization="Bearer correct horse battery staple",
    )

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert request.responses == [
        (200, {"service_id": "jina_embedding", "status": "starting", "ok": True})
    ]
    assert manager.calls == [("start", "jina_embedding")]


def test_post_start_required_uses_server_plan_and_rejects_browser_services(monkeypatch) -> None:
    manager = FakeManager()
    batch = SimpleNamespace(
        batch_id="batch_safe",
        embedding="gte_multilingual_base",
        rerankers=("none",),
        combination_ids=("combination_safe",),
    )
    monkeypatch.setattr(dashboard, "create_adapter_manager", lambda _root: manager)
    monkeypatch.setattr(dashboard, "load_dashboard_portfolio_plan", lambda: SimpleNamespace(batches=(batch,)))
    monkeypatch.setenv("WNS_ENABLE_ADAPTER_CONTROL", "1")
    monkeypatch.setenv("WNS_ADAPTER_CONTROL_TOKEN", "token")
    request = FakeHttpRequest(
        "/api/adapters/start-required",
        {"batch_id": "batch_safe", "service_ids": ["spare_reserved"]},
        authorization="Bearer token",
    )

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert request.responses[0][0] == 400
    assert request.responses[0][1]["error"]["code"] == "invalid_request"  # type: ignore[index]
    assert manager.calls == []
