import http.client
import os
import unittest
from unittest.mock import call, patch

from benchmarking.adapters import remote_embeddings


class OpenAIEmbeddingRetryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "17",
                "OPENAI_EMBEDDING_RETRY_DELAY_SECONDS": "0.25",
            },
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def adapter(self):
        return remote_embeddings.OpenAIEmbeddingAdapter(
            model_name="text-embedding-test",
            dimensions=2,
            batch_size=2,
        )

    def assert_retry_then_success(self, first_error):
        response = {
            "data": [
                {"embedding": [1.0, 2.0]},
                {"embedding": [3.0, 4.0]},
            ]
        }
        with (
            patch.object(remote_embeddings, "_post_json", side_effect=[first_error, response]) as post,
            patch.object(remote_embeddings.time, "sleep") as sleep,
        ):
            vectors = self.adapter().embed_many(["one", "two"])

        self.assertEqual(vectors, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["timeout"], 17)
        self.assertEqual(post.call_args_list[1].kwargs["timeout"], 17)
        self.assertEqual(sleep.call_args_list, [call(0.25)])

    def test_retries_timeout_then_succeeds_with_configured_timeout(self):
        self.assert_retry_then_success(TimeoutError("slow endpoint"))

    def test_retries_incomplete_http_response_then_succeeds(self):
        self.assert_retry_then_success(http.client.IncompleteRead(b"partial", 10))


if __name__ == "__main__":
    unittest.main()
