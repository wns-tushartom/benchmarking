import asyncio
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

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


def completed_modular_metrics() -> dict:
    return {
        "status": "completed",
        "query_count": "500",
        "recall_at_1": "0.30",
        "recall_at_3": "0.40",
        "recall_at_5": "0.50",
        "recall_at_10": "0.60",
        "mrr": "0.40",
        "precision_at_5": "0.20",
        "ndcg_at_5": "0.45",
        "avg_first_relevant_rank": "2.0",
        "no_hit_queries": "5",
        "avg_latency_ms": "100",
    }


def write_official_manifest(run_dir: Path, dashboard) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()
        for artifact in run_dir.iterdir()
        if artifact.is_file() and artifact.name in {"modular_summary.csv", "modular_details.csv"}
    }
    artifact_root = run_dir.resolve().as_posix()
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed",
        "run_id": run_dir.name,
        "config_hash": config_hash(load_benchmark_config(dashboard.CONFIG_PATH)),
        "dataset_id": dashboard.DEFAULT_DATASET_ID,
        "groundtruth_id": dashboard.OFFICIAL_GROUNDTRUTH_ID,
        "query_count": 500,
        "artifact_root": artifact_root,
        "artifact_sha256": artifacts,
    }), encoding="utf-8")


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

    def test_dashboard_reference_includes_archived_modular_runs_for_faiss(self):
        import scripts.serve_benchmark_dashboard as dashboard

        def write_summary(path: Path, row: dict):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(row))
                writer.writeheader()
                writer.writerow(row)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            latest_dir = root / "data" / "modular_runs" / "latest"
            faiss_dir = root / "data" / "modular_runs" / "faiss_noaws_complete_20260702"
            smoke_dir = root / "data" / "modular_runs" / "selected_smoke"
            full_dir = root / "data" / "full_benchmark"
            write_summary(latest_dir / "modular_summary.csv", {
                "chunker": "entity_heuristic_w6",
                "embedding": "gte_multilingual_base",
                "vector_store": "Qdrant",
                "reranker": "bge-reranker-base",
                **completed_modular_metrics(),
            })
            write_summary(faiss_dir / "modular_summary.csv", {
                "chunker": "entity_heuristic_w6",
                "embedding": "gte_multilingual_base",
                "vector_store": "FAISS",
                "reranker": "bge-reranker-base",
                **completed_modular_metrics(),
                "recall_at_5": "0.55",
                "mrr": "0.45",
                "avg_latency_ms": "80",
            })
            write_summary(smoke_dir / "modular_summary.csv", {
                "chunker": "entity_heuristic_w6",
                "embedding": "gte_multilingual_base",
                "vector_store": "Weaviate",
                "reranker": "bge-reranker-base",
                "query_count": "1",
                "recall_at_5": "1.0",
                "mrr": "1.0",
                "avg_latency_ms": "1",
            })
            full_dir.mkdir(parents=True, exist_ok=True)
            write_official_manifest(latest_dir, dashboard)
            write_official_manifest(faiss_dir, dashboard)

            old_modular_dir = dashboard.MODULAR_DIR
            old_full_dir = dashboard.FULL_DIR
            try:
                dashboard.MODULAR_DIR = latest_dir
                dashboard.FULL_DIR = full_dir
                reference = dashboard.read_benchmark_reference()
            finally:
                dashboard.MODULAR_DIR = old_modular_dir
                dashboard.FULL_DIR = old_full_dir

        stores = {row["store"] for row in reference["summary"]}
        self.assertIn("Qdrant", stores)
        self.assertIn("FAISS", stores)
        self.assertNotIn("Weaviate", stores)
        self.assertEqual(reference["report"]["faiss_rows"], 1)

    def test_frontend_coverage_uses_metric_rows_for_faiss_results(self):
        app_path = Path(__file__).resolve().parents[1] / "web" / "app.js"
        source_state_path = Path(__file__).resolve().parents[1] / "web" / "source-state.js"
        node_script = f'''
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  return {{
    id,
    value: 'all',
    textContent: '',
    innerHTML: '',
    classList: {{toggle(){{}}, add(){{}}, remove(){{}}}},
    addEventListener(){{}},
    querySelectorAll(){{ return []; }},
    setAttribute(){{}},
  }};
}}
const document = {{
  getElementById(id) {{ return elements[id] || (elements[id] = makeElement(id)); }},
  querySelectorAll() {{ return []; }},
  addEventListener() {{}},
  body: {{insertAdjacentHTML(){{}}}},
}};
const context = {{
  console,
  DashboardSourceState: require({str(source_state_path)!r}),
  document,
  window: {{confirm() {{ return true; }}}},
  location: {{hash: '#overview'}},
  history: {{replaceState(){{}}}},
  fetch: async () => ({{ok: true, json: async () => ({{operational: {{}}, files: [], options: {{}}}}), text: async () => ''}}),
  setTimeout(){{}},
}};
vm.createContext(context);
let code = fs.readFileSync({str(app_path)!r}, 'utf8');
code = code.split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
benchmarkOptions = {{
  chunkers: ['entity_heuristic_w6'],
  embeddings: ['gte_multilingual_base'],
  vector_stores: ['Qdrant', 'FAISS'],
  rerankers: ['bge-reranker-base'],
}};
state = {{operational: {{evaluation: {{benchmark_reference: {{summary: [{{
  sheet: 'entity_heuristic_w6',
  embedding: 'gte_multilingual_base',
  store: 'FAISS',
  reranker: 'bge-reranker-base',
  status: 'completed',
  official_provenance: 'trusted',
  evaluated_queries: 500,
  recall_at_1: .3,
  recall_at_3: .4,
  recall_at_5: .5,
  recall_at_10: .6,
  mrr: .4,
  precision_at_5: .2,
  ndcg_at_5: .45,
  avg_first_relevant_rank: 2,
  no_hit_queries: 5,
  avg_latency_seconds: 0.08,
  source: 'modular_run:faiss_noaws_complete_20260702',
}}], report: {{expected_keys:['entity_heuristic_w6|gte_multilingual_base|FAISS|bge-reranker-base']}}}}}}}}}};
renderCoverage([]);
globalThis.__coverageHint = document.getElementById('coverageHint').textContent;
globalThis.__coverageHtml = document.getElementById('coverageTable').innerHTML;
`, context);
if (!context.__coverageHtml.includes('Done · open Metrics')) {{
  console.error(context.__coverageHtml);
  process.exit(1);
}}
if (!context.__coverageHint.includes('1/2')) {{
  console.error(context.__coverageHint);
  process.exit(2);
}}
'''
        proc = subprocess.run(["node", "-e", node_script], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_benchmark_reference_canonicalizes_rerankers_and_dedupes_sources(self):
        import scripts.serve_benchmark_dashboard as dashboard

        def write_summary(path: Path, reranker: str):
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "chunker": "entity_heuristic_w6",
                "embedding": "gte_multilingual_base",
                "vector_store": "Qdrant",
                "reranker": reranker,
                **completed_modular_metrics(),
            }
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(row))
                writer.writeheader()
                writer.writerow(row)

        old_modular_dir = dashboard.MODULAR_DIR
        old_full_dir = dashboard.FULL_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                latest = root / "data" / "modular_runs" / "latest"
                archived = root / "data" / "modular_runs" / "archived_full"
                full_dir = root / "data" / "full_benchmark"
                write_summary(latest / "modular_summary.csv", "qwen3_4b_rerank")
                write_summary(archived / "modular_summary.csv", "Qwen3:4B Rerank")
                write_official_manifest(latest, dashboard)
                write_official_manifest(archived, dashboard)
                full_dir.mkdir(parents=True)
                dashboard.MODULAR_DIR = latest
                dashboard.FULL_DIR = full_dir

                reference = dashboard.read_benchmark_reference()
        finally:
            dashboard.MODULAR_DIR = old_modular_dir
            dashboard.FULL_DIR = old_full_dir

        self.assertEqual(len(reference["summary"]), 1)
        self.assertEqual(reference["summary"][0]["reranker"], "Qwen3:4B Rerank")
        self.assertEqual(reference["report"]["config_rows"], 1)

    def test_frontend_evaluated_rows_merge_reference_canonicalize_and_dedupe(self):
        app_path = Path(__file__).resolve().parents[1] / "web" / "app.js"
        node_script = f'''
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function makeElement(id) {{ return {{id, value:'all', textContent:'', innerHTML:'', classList:{{toggle(){{}}, add(){{}}, remove(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}}; }}
const context = {{console, document: {{addEventListener(){{}}, querySelectorAll(){{return [];}}, getElementById(id){{return elements[id] || (elements[id] = makeElement(id));}}, body: {{insertAdjacentHTML(){{}}}}}}, window: {{}}, location: {{hash: '#overview'}}, history: {{replaceState(){{}}}}, fetch: async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}), setTimeout(){{}}}};
vm.createContext(context);
let code = fs.readFileSync({str(app_path)!r}, 'utf8');
code = code.split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
const rows = evaluatedRows({{
  summary: [{{sheet:'s', embedding:'e', store:'Qdrant', reranker:'qwen3_4b_rerank', recall_at_5:'0.5', mrr:'0.4', ndcg_at_5:'0.3', avg_latency_seconds:'0.2'}}],
  reranked: {{summary: [{{sheet:'s', embedding:'e', store:'Qdrant', reranker:'Qwen3:4B Rerank', recall_at_5:'0.5', mrr:'0.4', ndcg_at_5:'0.3', avg_latency_seconds:'0.2'}}]}},
  benchmark_reference: {{summary: [{{sheet:'s', embedding:'e', store:'FAISS', reranker:'bge-reranker-base', recall_at_5:'0.9', mrr:'0.8', ndcg_at_5:'0.7', avg_latency_seconds:'0.1'}}]}},
}});
globalThis.__rows = rows;
`, context);
const rows = context.__rows;
if (rows.length !== 2) {{ console.error(JSON.stringify(rows)); process.exit(1); }}
if (!rows.some(r => r.store === 'FAISS')) {{ console.error(JSON.stringify(rows)); process.exit(2); }}
if (!rows.some(r => r.reranker === 'Qwen3:4B Rerank')) {{ console.error(JSON.stringify(rows)); process.exit(3); }}
'''
        proc = subprocess.run(["node", "-e", node_script], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_frontend_marks_aws_faiss_missing_lane_blocked(self):
        app_path = Path(__file__).resolve().parents[1] / "web" / "app.js"
        node_script = f'''
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function makeElement(id) {{ return {{id, value:'all', textContent:'', innerHTML:'', classList:{{toggle(){{}}, add(){{}}, remove(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}}; }}
const document = {{getElementById(id){{return elements[id] || (elements[id] = makeElement(id));}}, querySelectorAll(){{return [];}}, addEventListener(){{}}, body:{{insertAdjacentHTML(){{}}}}}};
const context = {{console, document, window:{{confirm(){{return true;}}}}, location:{{hash:'#overview'}}, history:{{replaceState(){{}}}}, fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}), setTimeout(){{}}}};
vm.createContext(context);
let code = fs.readFileSync({str(app_path)!r}, 'utf8');
code = code.split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
benchmarkOptions = {{chunkers:['entity_heuristic_w6'], embeddings:['gte_multilingual_base'], vector_stores:['FAISS'], rerankers:['Amazon Rerank v1']}};
state = {{operational: {{amazon_status: 'Pending AWS/Bedrock credentials', evaluation: {{benchmark_reference: {{summary: []}}}}}}}};
renderCoverage([]);
globalThis.__html = document.getElementById('coverageTable').innerHTML;
`, context);
if (!context.__html.includes('blocked')) {{ console.error(context.__html); process.exit(1); }}
if (context.__html.includes('mini-run-btn')) {{ console.error(context.__html); process.exit(2); }}
'''
        proc = subprocess.run(["node", "-e", node_script], text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_ingestion_supports_openai_embedding_adapter(self):
        import scripts.run_long_db_ingestion as ingestion

        old_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "unit-secret"
            cfg = ingestion.EMBEDDING_CONFIGS["openai_text-embedding-3-large"]
            self.assertEqual(cfg["adapter"], "openai")
            adapter = ingestion.make_embedding_adapter("openai_text-embedding-3-large", batch_size=2)
            self.assertEqual(adapter.__class__.__name__, "OpenAIEmbeddingAdapter")
            self.assertEqual(adapter.dimensions, 3072)
            self.assertEqual(adapter.batch_size, 2)
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key

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

    def test_amazon_bedrock_reranker_uses_text_inline_source_shape(self):
        from benchmarking.adapters.remote_rerankers import AmazonBedrockRerankerAdapter
        from benchmarking.core.schemas import Chunk, SearchHit

        captured: dict = {}

        class FakeBoto3:
            @staticmethod
            def client(service_name: str, region_name: str | None = None):
                self.assertEqual(service_name, "bedrock-agent-runtime")
                self.assertEqual(region_name, "us-west-2")

                class FakeClient:
                    def rerank(self, **kwargs):
                        captured.update(kwargs)
                        return {"results": [{"index": 0, "relevanceScore": 0.7}]}

                return FakeClient()

        old_module = sys.modules.get("boto3")
        old_env = {k: os.environ.get(k) for k in ["AWS_REGION", "AWS_DEFAULT_REGION", "AMAZON_RERANK_MODEL_ID"]}
        sys.modules["boto3"] = FakeBoto3()
        os.environ["AWS_REGION"] = "us-west-2"
        os.environ["AMAZON_RERANK_MODEL_ID"] = "amazon.rerank-v1:0"
        try:
            adapter = AmazonBedrockRerankerAdapter("Amazon Rerank v1")
            hits = [SearchHit(Chunk(id=1, pdf_name="doc.pdf", paragraph="refund text"), 0.1)]
            reranked = adapter.rerank("refund", hits, top_k=1)
        finally:
            if old_module is None:
                sys.modules.pop("boto3", None)
            else:
                sys.modules["boto3"] = old_module
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.assertEqual(len(reranked), 1)
        self.assertEqual(captured["queries"][0]["textQuery"]["text"], "refund")
        source = captured["sources"][0]
        self.assertEqual(source["type"], "INLINE")
        self.assertEqual(source["inlineDocumentSource"]["type"], "TEXT")
        self.assertEqual(source["inlineDocumentSource"]["textDocument"]["text"], "refund text")

    def test_complete_pipeline_normalizes_frontend_reranker_labels(self):
        from scripts.run_complete_pipeline import choose, normalize_reranker, LIVE_RERANKERS

        self.assertEqual(normalize_reranker("Qwen3:4B Rerank"), "qwen3_4b_rerank")
        self.assertEqual(normalize_reranker("Amazon Rerank v1"), "Amazon Rerank v1")
        self.assertIn("Amazon Rerank v1", LIVE_RERANKERS)
        self.assertEqual(
            choose(["Qwen3:4B Rerank", "Amazon Rerank v1", "none"], LIVE_RERANKERS, normalize=normalize_reranker),
            ["qwen3_4b_rerank", "Amazon Rerank v1", "none"],
        )

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
            manifest = json.loads((Path(td) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["run_id"], Path(td).name)
            self.assertEqual(manifest["dataset_id"], "dataset:wns-default")
            self.assertEqual(manifest["groundtruth_id"], "groundtruth:repository:qa_text_test.csv")
            with (Path(td) / "modular_summary.csv").open(encoding="utf-8", newline="") as f:
                self.assertTrue(all(row["status"] == "completed" for row in csv.DictReader(f)))

    def test_modular_experiment_failure_never_publishes_completed_manifest(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            config_path = local_smoke_config(root, td_path)
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg["techniques"]["chunkers"]["entity_heuristic_w6"]["adapter"] = "missing_adapter"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")

            with self.assertRaises(KeyError):
                run_experiment(config_path, root, td_path, max_runs=1, limit_queries=1)

            manifest = json.loads((td_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(manifest["status"], "completed")
            self.assertFalse((td_path / ".manifest.json.tmp").exists())

    def test_selected_official_run_manifest_is_admitted_without_handcrafted_provenance(self):
        import scripts.serve_benchmark_dashboard as dashboard
        from benchmarking.core.registry import Registry
        from benchmarking.core.schemas import Chunk, SearchHit

        class Chunker:
            def __init__(self, **kwargs):
                pass

            def chunk(self):
                return [Chunk(id=1, pdf_name="unit.pdf", paragraph="refund voucher")]

        class Embedder:
            dimensions = 2

            def __init__(self, **kwargs):
                pass

            def embed_many(self, texts):
                return [[1.0, 0.0] for _ in texts]

        class Store:
            def __init__(self, **kwargs):
                self.chunks = []

            def upsert(self, chunks, vectors):
                self.chunks = chunks
                return {"upsert_latency_s": 0.001}

            def search(self, vector, top_k=10):
                return [SearchHit(self.chunks[0], 1.0)]

        class Reranker:
            def __init__(self, **kwargs):
                pass

            def rerank(self, query, hits, top_k=10):
                return hits[:top_k]

        class Evaluator:
            def __init__(self, **kwargs):
                pass

            def flags(self, query, hits):
                return [True for _ in hits]

        registry = Registry()
        cfg = load_benchmark_config(dashboard.CONFIG_PATH)
        selections = {
            "chunker": cfg["matrix"]["chunkers"][0],
            "embedding": cfg["matrix"]["embeddings"][0],
            "vector_store": cfg["matrix"]["vector_stores"][0],
            "index_type": cfg["matrix"]["index_types"][0],
            "retrieval_method": cfg["matrix"]["retrieval_methods"][0],
            "reranker": cfg["matrix"]["rerankers"][0],
            "evaluator": cfg["matrix"]["evaluators"][0],
        }
        registry.register("chunker", cfg["techniques"]["chunkers"][selections["chunker"]]["adapter"], Chunker)
        registry.register("embedding", cfg["techniques"]["embeddings"][selections["embedding"]]["adapter"], Embedder)
        registry.register("vector_store", cfg["techniques"]["vector_stores"][selections["vector_store"]]["adapter"], Store)
        registry.register("reranker", cfg["techniques"]["rerankers"][selections["reranker"]]["adapter"], Reranker)
        registry.register("evaluator", selections["evaluator"], Evaluator)

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "selected-official"
            with patch("benchmarking.core.runner.default_registry", return_value=registry):
                run_experiment(
                    dashboard.CONFIG_PATH,
                    Path(__file__).resolve().parents[1],
                    output,
                    limit_queries=1,
                    selections=selections,
                )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            with patch.object(
                dashboard,
                "benchmark_reference_sources",
                return_value=[("modular_run:selected", output / "modular_summary.csv")],
            ):
                reference = dashboard.read_benchmark_reference()

        self.assertEqual(manifest["official_matrix_contract_hash"], config_hash(cfg))
        self.assertNotEqual(manifest["selected_run_config_hash"], manifest["official_matrix_contract_hash"])
        self.assertEqual(len(reference["summary"]), 1)
        self.assertEqual(reference["summary"][0]["official_provenance"], "trusted")


if __name__ == "__main__":
    unittest.main()
