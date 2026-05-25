import unittest
from pathlib import Path
from unittest import mock

import source.benchmark_pipeline as bp
from source.benchmark_pipeline import (
    tokenize,
    cosine_similarity,
    DeterministicEmbedder,
    recall_at_k,
    text_overlap_score,
    load_chunks_from_workbook,
)


class BenchmarkPipelineCoreTests(unittest.TestCase):
    def test_tokenize_normalizes_words(self):
        self.assertEqual(tokenize("PNR refund, Voucher!"), ["pnr", "refund", "voucher"])

    def test_cosine_similarity_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0, places=6)

    def test_deterministic_embedder_is_stable(self):
        emb = DeterministicEmbedder("jina_v3", dimensions=32)
        self.assertEqual(emb.embed("same text"), emb.embed("same text"))
        self.assertEqual(len(emb.embed("same text")), 32)

    def test_text_overlap_score_detects_shared_terms(self):
        score = text_overlap_score("refund voucher pnr", "voucher was sent for one pnr")
        self.assertGreater(score, 0.3)

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k([False, True, False], 1), 0.0)
        self.assertEqual(recall_at_k([False, True, False], 3), 1.0)

    def test_chunk_workbook_reader_works_without_openpyxl(self):
        workbook = Path(__file__).resolve().parents[1] / "data" / "chunking_methods_output_v2.xlsx"
        with mock.patch.object(bp, "openpyxl", None):
            chunks = load_chunks_from_workbook(workbook, "entity_heuristic_w6")
        self.assertGreater(len(chunks), 0)
        self.assertTrue(chunks[0].paragraph)


if __name__ == "__main__":
    unittest.main()
