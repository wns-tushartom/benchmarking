import pytest
from fastapi.testclient import TestClient

from scripts import wns_vm_adapter_service as service


def test_nemotron_formatter_matches_documented_prompt_contract() -> None:
    assert service.format_nemotron_pair("Where is PAX-REF-992?", "Refund policy") == (
        "question:Where is PAX-REF-992? \n \n passage:Refund policy"
    )


def test_candidate_reranker_routes_and_health_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(service, "rerank", lambda key, req: {"key": key, "count": len(req.documents or [])})
    client = TestClient(service.app)

    health = client.get("/health").json()
    assert health["models"]["nemotron_reranker"] == "nvidia/llama-nemotron-rerank-1b-v2"
    assert health["models"]["gte_modernbert_reranker"] == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert "/rerank/nemotron" in health["endpoints"]
    assert "/rerank/gte-modernbert" in health["endpoints"]
    assert client.post("/rerank/nemotron", json={"query": "refund", "documents": ["a", "b"]}).json() == {"key": "nemotron", "count": 2}
    assert client.post("/rerank/gte-modernbert", json={"query": "refund", "documents": ["a"]}).json() == {"key": "gte-modernbert", "count": 1}


def test_reranker_model_ids_are_exact_and_not_embedding_checkpoint() -> None:
    assert service.NEMOTRON_RERANK_MODEL == "nvidia/llama-nemotron-rerank-1b-v2"
    assert service.GTE_MODERNBERT_RERANK_MODEL == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert service.GTE_MODERNBERT_RERANK_MODEL != "Alibaba-NLP/gte-modernbert-base"


def test_candidate_model_environment_mismatch_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_RERANK_MODEL", "substituted/not-nemotron")

    with pytest.raises(RuntimeError, match="NEMOTRON_RERANK_MODEL"):
        service.candidate_model_env("NEMOTRON_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
