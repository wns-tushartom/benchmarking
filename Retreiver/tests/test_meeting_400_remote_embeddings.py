from __future__ import annotations

import pytest

from benchmarking.adapters import remote_embeddings


MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"


def make_adapter(monkeypatch: pytest.MonkeyPatch, **overrides):
    monkeypatch.setenv("TEST_NEMOTRON_URL", "http://127.0.0.1:5010/v1/embeddings")
    options = {
        "model_name": "nemotron_3_embed_1b_bf16",
        "request_model": MODEL,
        "endpoint_env": "TEST_NEMOTRON_URL",
        "dimensions": 3,
        "batch_size": 16,
        "serving_contract": "vllm_openai_compatible",
        "max_input_tokens": 4096,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "require_response_model": True,
        "expected_response_model": MODEL,
    }
    options.update(overrides)
    return remote_embeddings.RemoteHTTPEmbeddingAdapter(**options)


def test_meeting_adapter_uses_exact_model_and_role_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def fake_post(url, payload, headers=None, timeout=120):
        payloads.append(payload)
        inputs = payload["input"]
        return {
            "model": MODEL,
            "data": [{"embedding": [1.0, 2.0, 3.0]} for _ in inputs],
        }

    monkeypatch.setattr(remote_embeddings, "_post_json", fake_post)
    adapter = make_adapter(monkeypatch)

    assert adapter.embed_many_with_role(["refund policy"], "query") == [[1.0, 2.0, 3.0]]
    assert adapter.embed_many_with_role(["refund instructions"], "document") == [[1.0, 2.0, 3.0]]
    assert payloads == [
        {"model": MODEL, "input": ["query: refund policy"], "truncate_prompt_tokens": 4096},
        {"model": MODEL, "input": ["passage: refund instructions"], "truncate_prompt_tokens": 4096},
    ]


def test_meeting_adapter_rejects_wrong_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_embeddings,
        "_post_json",
        lambda *args, **kwargs: {
            "model": "nvidia/Nemotron-3-Embed-1B-NVFP4",
            "data": [{"embedding": [1.0, 2.0, 3.0]}],
        },
    )
    with pytest.raises(RuntimeError, match="model mismatch"):
        make_adapter(monkeypatch).embed_many_with_role(["x"], "query")


def test_meeting_adapter_rejects_wrong_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_embeddings,
        "_post_json",
        lambda *args, **kwargs: {
            "model": MODEL,
            "data": [{"embedding": [1.0, 2.0]}],
        },
    )
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        make_adapter(monkeypatch).embed_many_with_role(["x"], "document")


def test_meeting_adapter_rejects_non_finite_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_embeddings,
        "_post_json",
        lambda *args, **kwargs: {
            "model": MODEL,
            "data": [{"embedding": [1.0, float("nan"), 3.0]}],
        },
    )
    with pytest.raises(RuntimeError, match="finite numeric"):
        make_adapter(monkeypatch).embed_many_with_role(["x"], "query")
