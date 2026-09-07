from fastapi.testclient import TestClient
from types import SimpleNamespace

import torch

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


def test_qwen_reranker_uses_single_item_prediction_batches(monkeypatch) -> None:
    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, pairs, **kwargs):
            self.calls.append((pairs, kwargs))
            return [float(index) for index, _ in enumerate(pairs)]

    model = FakeCrossEncoder()
    monkeypatch.setattr(service, "cross_encoder_model", lambda key: model)
    request = service.RerankRequest(query="refund", documents=["one", "two", "three"])

    response = service.rerank("qwen", request)

    assert response["scores"] == [0.0, 1.0, 2.0]
    assert len(model.calls) == 1
    assert model.calls[0][1] == {"batch_size": 1, "show_progress_bar": False}


def test_qwen_model_loads_with_bounded_context_and_checkpoint_dtype(monkeypatch) -> None:
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs) -> None:
            captured.update(model_name=model_name, **kwargs)
            self.tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>", eos_token_id=7)
            self.model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    service.cross_encoder_model.cache_clear()
    try:
        service.cross_encoder_model("qwen")
    finally:
        service.cross_encoder_model.cache_clear()

    assert captured["max_length"] == 4096
    assert captured["automodel_args"] == {"torch_dtype": "auto"}


def test_nemotron_scores_process_documents_in_single_item_batches(monkeypatch) -> None:
    tokenizer_batch_sizes = []

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            tokenizer_batch_sizes.append(len(texts))
            return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

    class FakeModel:
        def __call__(self, **kwargs):
            batch_size = int(kwargs["input_ids"].shape[0])
            return SimpleNamespace(logits=torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1))

    monkeypatch.setattr(service, "nemotron_model", lambda: (FakeTokenizer(), FakeModel()))
    monkeypatch.setattr(service, "DEVICE", "cpu")

    scores = service.nemotron_scores("refund", ["one", "two", "three"])

    assert scores == [0.0, 0.0, 0.0]
    assert tokenizer_batch_sizes == [1, 1, 1]
