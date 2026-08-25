from fastapi.testclient import TestClient
import pytest

from scripts import wns_vm_adapter_service as service


def test_jina_profile_advertises_and_serves_only_jina(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "ACTIVE_PROFILE", "jina")
    monkeypatch.setattr(service, "embed", lambda key, req: {"key": key})
    client = TestClient(service.app)

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["profile"] == "jina"
    assert payload["endpoints"] == ["/embed/jina"]
    assert payload["models"] == {"jina": service.JINA_MODEL_NAME}
    assert client.post("/embed/jina", json={"input": ["hello"]}).json() == {"key": "jina"}
    blocked = client.post("/embed/gte", json={"input": ["hello"]})
    assert blocked.status_code == 404
    assert "profile" in blocked.json()["detail"].lower()


def test_reranker_profile_blocks_other_rerankers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "ACTIVE_PROFILE", "nemotron-rerank")
    monkeypatch.setattr(service, "rerank", lambda key, req: {"key": key})
    client = TestClient(service.app)

    assert client.get("/health").json()["endpoints"] == ["/rerank/nemotron"]
    assert client.post("/rerank/nemotron", json={"query": "q", "documents": ["d"]}).status_code == 200
    assert client.post("/rerank/bge", json={"query": "q", "documents": ["d"]}).status_code == 404


def test_legacy_unified_profile_preserves_existing_routes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service, "ACTIVE_PROFILE", "legacy-unified")
    endpoints = service.health()["endpoints"]
    assert set(endpoints) == {
        "/embed/jina",
        "/embed/gte",
        "/rerank/bge",
        "/rerank/qwen",
        "/rerank/nemotron",
        "/rerank/gte-modernbert",
    }


def test_unknown_profile_fails_closed():
    with pytest.raises(RuntimeError, match="WNS_ADAPTER_PROFILE"):
        service.validated_profile("arbitrary-command-profile")
