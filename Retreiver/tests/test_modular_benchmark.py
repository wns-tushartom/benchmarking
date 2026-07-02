import asyncio
import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from benchmarking.core.config import load_benchmark_config, generate_matrix, config_hash, selected_config
from benchmarking.core.metrics import mrr, ndcg_at_k, precision_at_k, bootstrap_ci
from benchmarking.core.registry import Registry
from benchmarking.adapters.local import LocalChunkWorkbookAdapter, LocalHashEmbeddingAdapter, LocalVectorStoreAdapter, WeightedOverlapReranker
from benchmarking.adapters.vector_faiss import FaissVectorStoreAdapter
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
        self.assertEqual(len(matrix), 180)
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
        registry.register("embedding", "unit_test_adapter", object)
        self.assertIs(registry.get("embedding", "unit_test_adapter"), object)
        self.assertIn("unit_test_adapter", registry.list("embedding"))

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

    def test_faiss_adapter_persists_and_searches(self):
        try:
            import faiss  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:
            self.skipTest(f"faiss-cpu not installed: {exc}")
        root = Path(__file__).resolve().parents[1]
        chunks = LocalChunkWorkbookAdapter(root=root, sheet_name="semantic_split").chunk()[:3]
        embedder = LocalHashEmbeddingAdapter(model_name="unit_model", dimensions=32)
        vectors = embedder.embed_many([c.paragraph for c in chunks])
        with tempfile.TemporaryDirectory() as td:
            store = FaissVectorStoreAdapter(name="FAISS", index_dir=td, index_type="HNSW")
            metrics = store.upsert(chunks, vectors)
            self.assertEqual(metrics["vector_count"], len(chunks))
            hits = store.search(vectors[0], top_k=2)
            self.assertGreaterEqual(len(hits), 1)
            loaded = FaissVectorStoreAdapter(name="FAISS", index_dir=td, load_existing=True)
            persisted_hits = loaded.search(vectors[0], top_k=2)
            self.assertGreaterEqual(len(persisted_hits), 1)
            self.assertEqual(persisted_hits[0].chunk.pdf_name, chunks[0].pdf_name)

    def test_workbook_sheet_names_reads_xlsx_without_pandas(self):
        from scripts.run_complete_pipeline import workbook_sheet_names

        workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheets>
    <sheet name="entity_heuristic_w6" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
    <sheet name="fixed_tok1200_ov150" sheetId="2" r:id="rId2" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
  </sheets>
</workbook>'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "book.xlsx"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("xl/workbook.xml", workbook_xml)
            self.assertEqual(workbook_sheet_names(path), ["entity_heuristic_w6", "fixed_tok1200_ov150"])

    def test_vm_generated_env_overrides_stale_env_without_blank_secret_clobber(self):
        from scripts.wns_env import load_env_files

        old_env = {k: os.environ.get(k) for k in ["MODEL_ADAPTER_URL", "OPENAI_API_KEY"]}
        try:
            for k in old_env:
                os.environ.pop(k, None)
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / ".env").write_text("MODEL_ADAPTER_URL=http://old:5000\nOPENAI_API_KEY=real-secret\n", encoding="utf-8")
                (root / ".env.vm.generated").write_text("MODEL_ADAPTER_URL=http://vm:5000\nOPENAI_API_KEY=\n", encoding="utf-8")
                load_env_files(root)
                self.assertEqual(os.environ["MODEL_ADAPTER_URL"], "http://vm:5000")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "real-secret")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_modular_runner_loads_vm_generated_env(self):
        from benchmarking.core.runner import load_env_file

        old_env = {k: os.environ.get(k) for k in ["JINA_EMBEDDING_URL", "OPENAI_API_KEY"]}
        try:
            for k in old_env:
                os.environ.pop(k, None)
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root / ".env").write_text("JINA_EMBEDDING_URL=http://old:5000/embed/jina\nOPENAI_API_KEY=real-secret\n", encoding="utf-8")
                (root / ".env.vm.generated").write_text("JINA_EMBEDDING_URL=http://vm:5000/embed/jina\nOPENAI_API_KEY=\n", encoding="utf-8")
                load_env_file(root)
                self.assertEqual(os.environ["JINA_EMBEDDING_URL"], "http://vm:5000/embed/jina")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "real-secret")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_ingestion_summary_uses_fixed_schema_for_mixed_rows(self):
        from scripts.run_long_db_ingestion import SUMMARY_FIELDS, append_csv

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "summary.csv"
            append_csv(path, {"status": "embedding_failed", "sheet": "s1", "embedding": "gte", "error": "boom"})
            append_csv(path, {"status": "ok", "sheet": "s1", "embedding": "gte", "store": "Qdrant", "collection_or_table": "c1", "search_hits": 3})
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, SUMMARY_FIELDS)
            self.assertEqual(rows[1]["collection_or_table"], "c1")
            self.assertEqual(rows[1]["search_hits"], "3")

    def test_vector_batch_validation_rejects_partial_cached_vectors(self):
        from scripts.run_long_db_ingestion import validate_vectors_for_chunks

        root = Path(__file__).resolve().parents[1]
        chunks = LocalChunkWorkbookAdapter(root=root, sheet_name="semantic_split").chunk()[:2]
        with self.assertRaisesRegex(ValueError, "Vector count mismatch"):
            validate_vectors_for_chunks(chunks, [[0.1, 0.2]], expected_dim=2, embedding="unit")
        with self.assertRaisesRegex(ValueError, "No chunks available"):
            validate_vectors_for_chunks([], [], expected_dim=2, embedding="unit")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_vectors_for_chunks(chunks[:1], [[float("nan"), 0.2]], expected_dim=2, embedding="unit")

    def test_extraction_audit_gate_reports_missing_and_review_rows(self):
        import scripts.run_complete_pipeline as pipeline

        original = pipeline.PDF_AUDIT_PATH
        try:
            with tempfile.TemporaryDirectory() as td:
                audit = Path(td) / "pdf_extraction_audit.csv"
                pipeline.PDF_AUDIT_PATH = audit
                self.assertIn("missing", pipeline.extraction_audit_issues()[0])
                audit.write_text("pdf_name,status,parser_method,row_count,needs_ocr_review\nA.pdf,text_only_review,PyPDF2_fallback,10,true\n", encoding="utf-8")
                issues = pipeline.extraction_audit_issues()
                self.assertEqual(len(issues), 1)
                self.assertIn("A.pdf", issues[0])
                self.assertIn("text_only_review", issues[0])
        finally:
            pipeline.PDF_AUDIT_PATH = original

    def test_complete_pipeline_health_url_rejects_404(self):
        import scripts.run_complete_pipeline as pipeline

        class FakeResponse:
            status = 404
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        original = pipeline.urllib.request.urlopen
        try:
            pipeline.urllib.request.urlopen = lambda *args, **kwargs: FakeResponse()
            ok, note = pipeline.url_ok("http://unit.invalid/health")
            self.assertFalse(ok)
            self.assertEqual(note, "HTTP 404")
        finally:
            pipeline.urllib.request.urlopen = original

    def test_complete_pipeline_archives_fixed_artifact_dirs_by_default(self):
        import scripts.run_complete_pipeline as pipeline

        old_root = pipeline.ROOT
        old_dirs = pipeline.ARCHIVE_DIRS
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                artifact = root / "data" / "retrieval_smoke" / "old.json"
                artifact.parent.mkdir(parents=True)
                artifact.write_text("{}", encoding="utf-8")
                pipeline.ROOT = root
                pipeline.ARCHIVE_DIRS = ["data/retrieval_smoke"]
                pipeline.archive_fresh_dirs("run123")
                self.assertFalse(artifact.exists())
                self.assertTrue((root / "data" / "pipeline_archives" / "run123" / "data" / "retrieval_smoke" / "old.json").exists())
        finally:
            pipeline.ROOT = old_root
            pipeline.ARCHIVE_DIRS = old_dirs

    def test_dashboard_marks_missing_pdf_audit_explicitly(self):
        import scripts.serve_benchmark_dashboard as dashboard

        original = dashboard.PDF_AUDIT_PATH
        try:
            with tempfile.TemporaryDirectory() as td:
                dashboard.PDF_AUDIT_PATH = Path(td) / "missing_audit.csv"
                audit = dashboard.read_pdf_audit()
                self.assertTrue(audit["missing"])
                self.assertEqual(audit["status"], "audit_missing")
                self.assertIn("blocks", audit["message"])
        finally:
            dashboard.PDF_AUDIT_PATH = original

    def test_dashboard_serves_only_named_repository_pdfs(self):
        import scripts.serve_benchmark_dashboard as dashboard

        original = dashboard.PDF_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf_dir = Path(td) / "pdfs"
                pdf_dir.mkdir()
                good = pdf_dir / "A file.pdf"
                good.write_bytes(b"%PDF-1.4\n%%EOF\n")
                dashboard.PDF_DIR = pdf_dir
                self.assertEqual(dashboard.pdf_path_for_name("A file.pdf"), good.resolve())
                self.assertIsNone(dashboard.pdf_path_for_name("../A file.pdf"))
                self.assertIsNone(dashboard.pdf_path_for_name("A file.txt"))
                self.assertIsNone(dashboard.pdf_path_for_name("missing.pdf"))
        finally:
            dashboard.PDF_DIR = original

    def test_nvidia_benchmark_ok_requires_every_query_success(self):
        from scripts.run_nvidia_rag_benchmark import nvidia_success_flags

        self.assertEqual(nvidia_success_flags([]), (False, False, 0))
        self.assertEqual(nvidia_success_flags([{"ok": 1}, {"ok": 1}]), (True, False, 2))
        self.assertEqual(nvidia_success_flags([{"ok": 1}, {"ok": 0}]), (False, True, 1))
        self.assertEqual(nvidia_success_flags([{"ok": 0}]), (False, False, 0))

    def test_dashboard_pdf_links_include_page_fragments_from_hit_metadata(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is required for dashboard JavaScript regression test")
        app_path = Path(__file__).resolve().parents[1] / "web" / "app.js"
        app = app_path.read_text(encoding="utf-8")
        prefix = app.split("function filteredRetrieval", 1)[0]
        probe = prefix + r'''
const rows = buildEvidenceRows([{query:'refund', hits:[{rank:1, pdf_name:'A file.pdf', page_number:'7', paragraph:'Chunk text'}]}], []);
console.log(JSON.stringify({
  directUrl: pdfOpenUrl('A file.pdf', 7),
  directLink: pdfLink('A file.pdf', 'source', 7),
  hitPage: rows[0].evidence_hits[0].page_number,
  topPage: rows[0].top_page_number,
  details: renderEvidenceDetails(rows[0]),
}));
'''
        result = subprocess.run([node, "-e", probe], text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["directUrl"], "/api/pdf?name=A%20file.pdf#page=7")
        self.assertIn("href=\"/api/pdf?name=A%20file.pdf#page=7\"", payload["directLink"])
        self.assertEqual(payload["hitPage"], 7)
        self.assertEqual(payload["topPage"], 7)
        self.assertIn("#page=7", payload["details"])
        self.assertIn("page 7", payload["details"].lower())

    def test_setup_script_generates_usable_pgvector_dsn_not_masked_placeholder(self):
        script = (Path(__file__).resolve().parents[2] / "setup_all_on_vm.sh").read_text(encoding="utf-8")
        self.assertIn("PGVECTOR_DSN=postgresql://wns:${WNS_POSTGRES_PASSWORD}@", script)
        self.assertIn("DATABASE_URL=postgresql://wns:${WNS_POSTGRES_PASSWORD}@", script)
        self.assertNotIn("PGVECTOR_DSN=postgresql://wns:***@", script)
        self.assertNotIn("DATABASE_URL=postgresql://wns:***@", script)

    def test_document_parser_can_disable_pypdf2_after_mineru_failure(self):
        import source.services.document_parser as parser_module
        from source.services.document_parser import DocumentParserService

        service = DocumentParserService(force_backend="mineru")

        async def fail_mineru(*args, **kwargs):
            raise RuntimeError("mineru connection failed")

        async def fallback(*args, **kwargs):
            return "fallback-used"

        service._parse_pdf_with_mineru = fail_mineru
        service._parse_pdf_with_pypdf2 = fallback
        old_pypdf2_available = parser_module.PYPDF2_AVAILABLE
        parser_module.PYPDF2_AVAILABLE = True
        try:
            with tempfile.TemporaryDirectory() as td:
                pdf = Path(td) / "x.pdf"
                pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
                with self.assertRaisesRegex(RuntimeError, "fallback disabled"):
                    asyncio.run(service.parse_pdf(str(pdf), pdf.name, allow_fallback=False))
                self.assertEqual(asyncio.run(service.parse_pdf(str(pdf), pdf.name, allow_fallback=True)), "fallback-used")
        finally:
            parser_module.PYPDF2_AVAILABLE = old_pypdf2_available

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
