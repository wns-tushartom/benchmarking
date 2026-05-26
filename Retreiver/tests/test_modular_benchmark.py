import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.core.config import load_benchmark_config, generate_matrix, config_hash, selected_config
from benchmarking.core.metrics import mrr, ndcg_at_k, precision_at_k, bootstrap_ci
from benchmarking.core.registry import Registry
from benchmarking.adapters.local import LocalChunkWorkbookAdapter, LocalHashEmbeddingAdapter, LocalVectorStoreAdapter, WeightedOverlapReranker
from benchmarking.core.runner import run_experiment


def local_smoke_config(root: Path, target: Path) -> Path:
    cfg = load_benchmark_config(root / "configs" / "benchmark.local.json")
    cfg["experiment"]["mode"] = "unit_local_smoke"
    cfg["matrix"] = {
        "chunkers": ["entity_heuristic_w6"],
        "embeddings": ["unit_local_hash"],
        "vector_stores": ["unit_local_vector"],
        "index_types": ["HNSW"],
        "retrieval_methods": ["Cosine Similarity"],
        "rerankers": ["unit_weighted_overlap"],
        "evaluators": ["overlap_relevance"],
    }
    cfg["techniques"]["embeddings"]["unit_local_hash"] = {"adapter": "local_hash", "dimensions": 32}
    cfg["techniques"]["vector_stores"]["unit_local_vector"] = {"adapter": "local_vector"}
    cfg["techniques"]["rerankers"]["unit_weighted_overlap"] = {"adapter": "weighted_overlap"}
    path = target / "unit.local.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class ModularBenchmarkTests(unittest.TestCase):
    def test_config_load_and_matrix_generation(self):
        cfg = load_benchmark_config(Path("configs/benchmark.local.json"))
        matrix = generate_matrix(cfg)
        self.assertGreater(len(matrix), 0)
        self.assertEqual(len(matrix), 135)
        self.assertIn("chunker", matrix[0])
        self.assertIn("index_type", matrix[0])
        self.assertIn("retrieval_method", matrix[0])
        self.assertEqual({row["index_type"] for row in matrix}, {"HNSW"})
        self.assertEqual({row["retrieval_method"] for row in matrix}, {"Cosine Similarity"})
        self.assertEqual(config_hash(cfg), config_hash(cfg))

    def test_selected_config_narrows_matrix(self):
        cfg = load_benchmark_config(Path("configs/benchmark.local.json"))
        narrowed = selected_config(cfg, {
            "chunker": "entity_heuristic_w6",
            "embedding": "jina_v3",
            "vector_store": "Qdrant",
            "index_type": "HNSW",
            "retrieval_method": "Cosine Similarity",
            "reranker": "bge-reranker-base",
        })
        matrix = generate_matrix(narrowed)
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["chunker"], "entity_heuristic_w6")
        self.assertEqual(matrix[0]["reranker"], "bge-reranker-base")

    def test_registry_can_add_new_adapter(self):
        registry = Registry()
        registry.register("embedding", "dummy", object)
        self.assertIs(registry.get("embedding", "dummy"), object)
        self.assertIn("dummy", registry.list("embedding"))

    def test_metrics(self):
        hits = [False, True, False, True]
        self.assertAlmostEqual(mrr(hits), 0.5)
        self.assertAlmostEqual(precision_at_k(hits, 2), 0.5)
        self.assertGreater(ndcg_at_k(hits, 4), 0.0)
        lo, hi = bootstrap_ci([1, 0, 1, 1], iterations=50, seed=1)
        self.assertLessEqual(lo, hi)

    def test_local_adapters_have_standard_contract(self):
        root = Path(__file__).resolve().parents[1]
        chunker = LocalChunkWorkbookAdapter(root=root, sheet_name="semantic_split")
        chunks = chunker.chunk()
        self.assertGreater(len(chunks), 0)
        embedder = LocalHashEmbeddingAdapter(model_name="unit_model", dimensions=32)
        vectors = embedder.embed_many([chunks[0].paragraph])
        store = LocalVectorStoreAdapter(name="unit_store")
        store.upsert([chunks[0]], vectors)
        hits = store.search(embedder.embed("refund voucher"), top_k=1)
        reranked = WeightedOverlapReranker(name="unit_rerank").rerank("refund voucher", hits, top_k=1)
        self.assertEqual(len(reranked), 1)

    def test_modular_experiment_writes_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = local_smoke_config(root, td_path)
            analysis = run_experiment(config_path, root, td_path, max_runs=2, limit_queries=5)
            self.assertIn("best_config", analysis)
            self.assertTrue((Path(td) / "manifest.json").exists())
            self.assertTrue((Path(td) / "modular_summary.csv").exists())
            self.assertTrue((Path(td) / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()
