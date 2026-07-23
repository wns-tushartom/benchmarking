import http.client
import urllib.error
from email.message import Message
from io import BytesIO
from unittest.mock import call, patch

import pytest

from benchmarking.adapters import remote_embeddings


def _adapter(*, batch_size: int = 2) -> remote_embeddings.OpenAIEmbeddingAdapter:
    return remote_embeddings.OpenAIEmbeddingAdapter(
        model_name="text-embedding-test",
        dimensions=2,
        batch_size=batch_size,
    )


@pytest.fixture(autouse=True)
def _openai_retry_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("OPENAI_EMBEDDING_RETRY_DELAY_SECONDS", "0.25")


def _embedding_response(*, request_id: str = "request-1", input_tokens: int = 5) -> dict:
    return {
        "request_id": request_id,
        "model": "text-embedding-test",
        "data": [
            {"embedding": [1.0, 2.0]},
            {"embedding": [3.0, 4.0]},
        ],
        "usage": {"input_tokens": input_tokens},
    }


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.openai.test/v1/embeddings",
        status,
        "provider error",
        Message(),
        BytesIO(b'{"error":"try later"}'),
    )


class _TruncatedErrorBody(BytesIO):
    def read(self, _size: int | None = -1) -> bytes:
        raise http.client.IncompleteRead(b'{"error":"truncated', 20)


def _truncated_http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.openai.test/v1/embeddings",
        status,
        "provider error",
        Message(),
        _TruncatedErrorBody(),
    )


class _SuccessfulResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self) -> bytes:
        return (
            b'{"request_id":"request-after-retry","model":"text-embedding-test",'
            b'"data":[{"embedding":[1.0,2.0]},{"embedding":[3.0,4.0]}],'
            b'"usage":{"input_tokens":5}}'
        )


def test_openai_embedding_retries_timeout_with_configured_timeout_and_backoff() -> None:
    response = _embedding_response()
    with (
        patch.object(
            remote_embeddings,
            "_post_json",
            side_effect=[TimeoutError("slow endpoint"), response],
        ) as post,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        vectors = adapter.embed_many(["one", "two"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert post.call_count == 2
    assert [attempt.kwargs["timeout"] for attempt in post.call_args_list] == [17, 17]
    assert sleep.call_args_list == [call(0.25)]
    assert "embedding_input_tokens" not in adapter.last_response_metadata


def test_openai_embedding_retries_incomplete_http_response() -> None:
    response = _embedding_response()
    with (
        patch.object(
            remote_embeddings,
            "_post_json",
            side_effect=[http.client.IncompleteRead(b"partial", 10), response],
        ) as post,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        vectors = adapter.embed_many(["one", "two"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert post.call_count == 2
    assert sleep.call_args_list == [call(0.25)]
    assert "embedding_input_tokens" not in adapter.last_response_metadata


def test_openai_embedding_preserves_metadata_and_exact_cumulative_tokens_without_retry() -> None:
    first_response = {
        "request_id": "request-1",
        "model": "text-embedding-test",
        "data": [{"embedding": [1.0, 2.0]}],
        "usage": {"input_tokens": 5},
    }
    second_response = {
        "request_id": "request-2",
        "model": "text-embedding-test",
        "data": [{"embedding": [3.0, 4.0]}],
        "usage": {"prompt_tokens": 7, "total_tokens": 700},
    }
    with (
        patch.object(
            remote_embeddings,
            "_post_json",
            side_effect=[first_response, second_response],
        ) as post,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter(batch_size=1)
        vectors = adapter.embed_many(["one", "two"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert post.call_count == 2
    assert [attempt.kwargs["timeout"] for attempt in post.call_args_list] == [17, 17]
    sleep.assert_not_called()
    assert adapter.last_response_metadata == {
        "request_id": "request-2",
        "model": "text-embedding-test",
        "embedding_input_tokens": 12,
    }


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_post_json_classifies_transient_http_status_as_retryable(status: int) -> None:
    with patch.object(
        remote_embeddings.urllib.request,
        "urlopen",
        side_effect=_http_error(status),
    ):
        with pytest.raises(
            remote_embeddings.RetryableHTTPError,
            match=rf"HTTP {status} .*try later",
        ):
            remote_embeddings._post_json(
                "https://api.openai.test/v1/embeddings",
                {"input": ["one", "two"]},
            )


def test_openai_embedding_retries_transient_http_and_preserves_response_metadata() -> None:
    with (
        patch.object(
            remote_embeddings.urllib.request,
            "urlopen",
            side_effect=[_http_error(429), _SuccessfulResponse()],
        ) as urlopen,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        vectors = adapter.embed_many(["one", "two"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert urlopen.call_count == 2
    assert sleep.call_args_list == [call(0.25)]
    assert adapter.last_response_metadata == {
        "request_id": "request-after-retry",
        "model": "text-embedding-test",
    }


def test_openai_embedding_fails_nonretryable_http_status_immediately() -> None:
    with (
        patch.object(
            remote_embeddings.urllib.request,
            "urlopen",
            side_effect=_http_error(400),
        ) as urlopen,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        with pytest.raises(RuntimeError, match=r"HTTP 400 .*try later") as raised:
            adapter.embed_many(["one", "two"])

    assert not isinstance(raised.value, remote_embeddings.RetryableHTTPError)
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_openai_embedding_fails_truncated_nonretryable_http_status_immediately() -> None:
    with (
        patch.object(
            remote_embeddings.urllib.request,
            "urlopen",
            side_effect=_truncated_http_error(400),
        ) as urlopen,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        with pytest.raises(RuntimeError, match=r"HTTP 400 .*truncated") as raised:
            adapter.embed_many(["one", "two"])

    assert not isinstance(raised.value, remote_embeddings.RetryableHTTPError)
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_post_json_preserves_retryable_status_when_error_body_is_truncated() -> None:
    with patch.object(
        remote_embeddings.urllib.request,
        "urlopen",
        side_effect=_truncated_http_error(503),
    ):
        with pytest.raises(
            remote_embeddings.RetryableHTTPError,
            match=r"HTTP 503 .*truncated",
        ):
            remote_embeddings._post_json(
                "https://api.openai.test/v1/embeddings",
                {"input": ["one", "two"]},
            )


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("dns unavailable"), OSError("connection reset")],
    ids=["url-error", "os-error"],
)
def test_openai_embedding_retries_direct_network_errors_and_omits_uncertain_tokens(
    failure: OSError,
) -> None:
    response = _embedding_response()
    with (
        patch.object(
            remote_embeddings,
            "_post_json",
            side_effect=[failure, response],
        ) as post,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        vectors = adapter.embed_many(["one", "two"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert post.call_count == 2
    assert sleep.call_args_list == [call(0.25)]
    assert "embedding_input_tokens" not in adapter.last_response_metadata


def test_openai_embedding_exhausts_six_attempts_with_five_backoff_sleeps() -> None:
    with (
        patch.object(
            remote_embeddings,
            "_post_json",
            side_effect=[OSError("connection reset")] * 6,
        ) as post,
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        with pytest.raises(OSError, match="connection reset"):
            adapter.embed_many(["one", "two"])

    assert post.call_count == 6
    assert sleep.call_args_list == [
        call(0.25),
        call(0.5),
        call(1.0),
        call(2.0),
        call(4.0),
    ]
    assert adapter.input_tokens_complete is False


def test_openai_embedding_parses_retry_settings_once_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[int] = []

    def post_then_succeed(*_args: object, **kwargs: object) -> dict:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int)
        timeouts.append(timeout)
        if len(timeouts) == 1:
            monkeypatch.setenv("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "99")
            monkeypatch.setenv("OPENAI_EMBEDDING_RETRY_DELAY_SECONDS", "99")
            raise TimeoutError("slow endpoint")
        return _embedding_response()

    with (
        patch.object(remote_embeddings, "_post_json", side_effect=post_then_succeed),
        patch("time.sleep") as sleep,
    ):
        adapter = _adapter()
        adapter.embed_many(["one", "two"])

    assert timeouts == [17, 17]
    assert sleep.call_args_list == [call(0.25)]


@pytest.mark.parametrize(
    ("name", "value", "requirement"),
    [
        ("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "0", "a positive integer"),
        ("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "-1", "a positive integer"),
        ("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "1.5", "a positive integer"),
        ("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "bad", "a positive integer"),
        (
            "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS",
            "-0.1",
            "a finite, non-negative number",
        ),
        (
            "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS",
            "nan",
            "a finite, non-negative number",
        ),
        (
            "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS",
            "inf",
            "a finite, non-negative number",
        ),
        (
            "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS",
            "bad",
            "a finite, non-negative number",
        ),
    ],
)
def test_openai_embedding_rejects_invalid_retry_settings_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    requirement: str,
) -> None:
    monkeypatch.setenv(name, value)

    with patch.object(remote_embeddings, "_post_json") as post:
        with pytest.raises(
            RuntimeError,
            match=rf"{name} must be {requirement}; got '{value}'",
        ):
            _adapter()

    post.assert_not_called()
