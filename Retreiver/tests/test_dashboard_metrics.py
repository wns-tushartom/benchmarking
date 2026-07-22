import csv
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_dashboard_metrics_keep_faiss_stage_winner_and_dedupe_rows():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "eval" / "groundtruth_eval_summary.csv", [{"sheet":"Heading_sections_l2", "embedding":"gte_multilingual_base", "store":"Qdrant", "reranker":"qwen3_4b_rerank", "recall_at_5":"0.5", "mrr":"0.4", "ndcg_at_5":"0.3", "avg_latency_seconds":"0.2"}])
        faiss = {"chunker":"Heading_sections_l2", "embedding":"gte_multilingual_base", "vector_store":"FAISS", "reranker":"bge-reranker-base", "query_count":"500", "recall_at_5":"0.9", "mrr":"0.8", "ndcg_at_5":"0.7", "avg_latency_ms":"100"}
        _write_csv(root / "runs" / "latest" / "modular_summary.csv", [faiss])
        _write_csv(root / "runs" / "faiss_noaws_retry" / "modular_summary.csv", [faiss])
        old = dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR
        dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = root / "eval", root / "runs" / "latest", root / "full", root / "gt"
        try:
            evaluation = dashboard.read_evaluation()
        finally:
            dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = old
    assert evaluation["benchmark_reference"]["report"]["config_rows"] == 1
    assert evaluation["benchmark_reference"]["summary"][0]["store"] == "FAISS"

    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}}; function el(id){{return elements[id] ||= {{id,value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
let code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
const evaluation={{summary:[{{sheet:'s',embedding:'e',store:'Qdrant',reranker:'qwen3_4b_rerank',recall_at_5:'0.5',mrr:'0.4',ndcg_at_5:'0.3',avg_latency_seconds:'0.2'}}], benchmark_reference:{{summary:[{{sheet:'s',embedding:'e',store:'Qdrant',reranker:'Qwen3:4B Rerank',recall_at_5:'0.5',mrr:'0.4',ndcg_at_5:'0.3',avg_latency_seconds:'0.2'}},{{sheet:'s',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',recall_at_5:'0.9',mrr:'0.8',ndcg_at_5:'0.7',avg_latency_seconds:'0.1'}}]}}}};
globalThis.__rows=evaluatedRows(evaluation); renderPipelineComparison(evaluation); globalThis.__stage=document.getElementById('stageComparisonChart').innerHTML;
`, context);
if (context.__rows.length !== 2 || !context.__stage.includes('Vector DB component: FAISS')) {{ console.error(JSON.stringify(context.__rows), context.__stage); process.exit(1); }}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_dashboard_broad_artifact_lanes_cover_official_180_without_double_count():
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import generate_matrix, load_benchmark_config

    rows = generate_matrix(load_benchmark_config(Path("configs/benchmark.local.json")))
    def metric(row: dict, modular: bool = False) -> dict:
        base = {"evaluated_queries": "500", "recall_at_5": "0.5", "mrr": "0.4", "ndcg_at_5": "0.3", "avg_latency_seconds": "0.2"}
        if modular:
            return {"chunker": row["chunker"], "embedding": row["embedding"], "vector_store": row["vector_store"], "reranker": row["reranker"], "query_count": "500", "recall_at_5": "0.5", "mrr": "0.4", "ndcg_at_5": "0.3", "avg_latency_ms": "200"}
        return {"sheet": row["chunker"], "embedding": row["embedding"], "store": row["vector_store"], "reranker": row["reranker"], **base}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "evaluation" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[:90]])
        _write_csv(root / "evaluation_reranked" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[90:120]])
        recovery_rows = [metric(r, modular=True) for r in rows[120:]]
        _write_csv(root / "runs" / "official_recovery_missing60_20260720_074219" / "modular_summary.csv", recovery_rows + recovery_rows[:1])
        extra_rows = []
        for embedding in ["jina_v3", "gte_multilingual_base", "openai_text-embedding-3-large"]:
            for store in ["Qdrant", "PGVector", "Weaviate"]:
                for reranker in ["Amazon Rerank v1", "Qwen3:4B Rerank", "bge-reranker-base"]:
                    extra_rows.append({"chunker": "semantic_split", "embedding": embedding, "vector_store": store, "reranker": reranker, "query_count": "500", "recall_at_5": "0.99", "mrr": "0.99", "ndcg_at_5": "0.99", "avg_latency_ms": "200"})
        _write_csv(root / "runs" / "archive" / "non_official_old_experiment" / "modular_summary.csv", extra_rows)
        old = dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR
        dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = root / "evaluation", root / "runs" / "latest", root / "full", root / "gt"
        try:
            evaluation = dashboard.read_evaluation()
            evaluation["reranked"] = dashboard.read_evaluation_dir(root / "evaluation_reranked")
            pipeline_state = dashboard.pipeline_state_rows()
        finally:
            dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = old

    assert len(pipeline_state) == 180
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    payload = json.dumps(evaluation)
    state_payload = json.dumps(pipeline_state)
    js = f"""
const fs=require('fs'), vm=require('vm');
const context={{console,document:{{getElementById(){{return {{value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `globalThis.__rows=evaluatedRows({payload}, {state_payload});`, context);
const keys=new Set(context.__rows.map(r=>`${{r.sheet}}|${{r.embedding}}|${{r.store}}|${{r.reranker||'none'}}`));
const byStore=context.__rows.reduce((a,r)=>(a[r.store]=(a[r.store]||0)+1,a),{{}});
const openai=context.__rows.filter(r=>String(r.embedding).includes('openai')).length;
const amazon=context.__rows.filter(r=>String(r.reranker).includes('Amazon')).length;
if (keys.size !== 180 || context.__rows.length !== 180 || byStore.FAISS !== 45 || openai !== 60 || amazon !== 60) {{ console.error(JSON.stringify({{rows:context.__rows.length, keys:keys.size, byStore, openai, amazon}})); process.exit(1); }}
"""
    with tempfile.TemporaryDirectory() as js_td:
        script_path = Path(js_td) / "verify_metrics.js"
        script_path.write_text(js, encoding="utf-8")
        proc = subprocess.run(["node", str(script_path)], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_metrics_table_does_not_silently_cap_complete_configs_at_60():
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    assert "filteredRows.slice(0, 60)" not in app.read_text(encoding="utf-8")


def test_benchmark_details_generate_aws_faiss_top5_evidence_rows():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        row = {
            "chunker": "Heading_sections_l2",
            "embedding": "gte_multilingual_base",
            "vector_store": "FAISS",
            "reranker": "Amazon Rerank v1",
            "query_id": "1",
            "query": "how to rebook",
            "category": "rebooking",
            "top_ids": "14|20565|26365|26028|25992",
            "top_scores": "0.9|0.8|0.7|0.6|0.5",
        }
        _write_csv(root / "runs" / "aws_combined" / "modular_details.csv", [row])
        old_modular, old_lookup = dashboard.MODULAR_DIR, dashboard.chunk_lookup_for_sheet
        dashboard.MODULAR_DIR = root / "runs" / "latest"
        dashboard.chunk_lookup_for_sheet = lambda sheet: {
            i: SimpleNamespace(id=i, pdf_name=f"doc{i}.pdf", paragraph=f"paragraph {i}")
            for i in [14, 20565, 26365, 26028, 25992]
        }
        try:
            evidence = dashboard.benchmark_detail_evidence(limit_per_combo=3)
        finally:
            dashboard.MODULAR_DIR = old_modular
            dashboard.chunk_lookup_for_sheet = old_lookup

    assert len(evidence) == 1
    assert evidence[0]["store"] == "FAISS"
    assert evidence[0]["reranker"] == "Amazon Rerank v1"
    assert len(evidence[0]["hits"]) == 5
    assert evidence[0]["hits"][0]["pdf_name"] == "doc14.pdf"


def test_frontend_initial_load_uses_lazy_evidence_limits():
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    text = app.read_text(encoding="utf-8")
    assert "retrieval_limit=0" in text
    assert "reranker_limit=0" in text
    assert "detail_evidence_limit=360" in text
    assert "retrieval_limit=60000" not in text
    assert "reranker_limit=60000" not in text


def test_smoke_counts_use_summary_without_loading_json_rows():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        retrieval = root / "retrieval_smoke"
        reranker = root / "reranker_smoke"
        retrieval.mkdir()
        reranker.mkdir()
        (retrieval / "summary.json").write_text(json.dumps({"result_count": 13050}), encoding="utf-8")
        (reranker / "summary.json").write_text(json.dumps({"result_count": 26251}), encoding="utf-8")
        old = dashboard.RETRIEVAL_DIR, dashboard.RERANKER_DIR
        dashboard.RETRIEVAL_DIR, dashboard.RERANKER_DIR = retrieval, reranker
        try:
            assert dashboard.read_retrieval_smokes(limit=0) == []
            assert dashboard.read_reranker_smokes(limit=0) == []
            assert dashboard.retrieval_smoke_count() == 13050
            assert dashboard.reranker_smoke_count() == 26251
        finally:
            dashboard.RETRIEVAL_DIR, dashboard.RERANKER_DIR = old


def test_static_assets_are_cache_busted_and_render_errors_visible():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    server = (root / "scripts" / "serve_benchmark_dashboard.py").read_text(encoding="utf-8")
    assert 'src="/app.js?v=' in index
    assert "Cache-Control" in server and "no-store" in server
    assert "Dashboard render failed" in app


def test_retrieval_filter_rerender_keeps_benchmark_detail_evidence():
    app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")
    assert "function operationalRerankRows()" in app
    assert "benchmark_detail_evidence" in app
    assert "renderRetrieval(state.operational?.retrieval_smokes || [], operationalRerankRows())" in app


def test_evidence_browser_wording_is_clear_about_display_rows_vs_raw_artifacts():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert "Evidence browser" in index
    assert "Retrieval checks" not in index
    assert "evidence display rows loaded" in app
    assert "Raw artifacts on disk" in app
    assert "Not total document chunks" in app
    assert "Retrieved evidence excerpts" in app
    assert "Benchmark final top 5 evidence" in app


def test_document_repository_page_does_not_show_summary_cards():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert "documentRepositoryCards" not in index
    assert "Repository PDFs" not in app
    assert "Chunked documents" not in app
    assert "Review queue" not in app


def test_user_project_upload_is_isolated_and_creates_manifest_and_search_index():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "user_projects"
        try:
            project = dashboard.create_user_project_upload(
                original_name="policy.txt",
                content=b"Refund policy: passengers can rebook cancelled flights or request refund.",
                label="Karthik Demo Data",
            )
        finally:
            dashboard.USER_PROJECTS_DIR = old

        project_root = Path(td) / "user_projects" / project["project_id"]
        assert project["ok"] is True
        assert project["project_id"].startswith("karthik-demo-data_")
        assert (project_root / "raw_uploads" / "policy.txt").exists()
        assert (project_root / "manifest.json").exists()
        assert (project_root / "search_index.json").exists()
        manifest = json.loads((project_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["storage_layout"]["runs"].endswith("/runs")
        assert "data/uploads" not in json.dumps(manifest)


def test_project_query_uses_uploaded_text_and_expands_selected_matrix():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "user_projects"
        try:
            project = dashboard.create_user_project_upload(
                original_name="refund.txt",
                content=b"Refund desk note. Cancelled flight customers may choose a full refund or rebooking.",
                label="Refund Demo",
            )
            result = dashboard.query_user_project(
                project["project_id"],
                "cancelled flight refund",
                {
                    "chunkers": ["fixed_tok1200_ov150", "Heading_sections_l2"],
                    "embeddings": ["gte_multilingual_base"],
                    "vector_stores": ["FAISS", "Qdrant"],
                    "rerankers": ["bge-reranker-base", "none"],
                },
                top_k=3,
            )
        finally:
            dashboard.USER_PROJECTS_DIR = old

        assert result["ok"] is True
        assert result["mode"] == "evidence_only"
        assert result["combo_count"] == 8
        assert result["rows"][0]["query"] == "cancelled flight refund"
        assert result["rows"][0]["hits"][0]["pdf_name"] == "refund.txt"
        assert "Cancelled flight customers" in result["rows"][0]["hits"][0]["paragraph"]


def test_frontend_has_multi_select_controls_project_query_and_matrix_count():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app_path = root / "web" / "app.js"
    app = app_path.read_text(encoding="utf-8")
    assert 'id="runSheet" multiple' in index
    assert 'id="runEmbedding" multiple' in index
    assert 'id="runStore" multiple' in index
    assert 'id="runRerankerMain" multiple' in index
    assert "selectedValues(" in app
    assert "updateSelectedMatrixCount" in app
    assert "/api/project-query" in app
    assert "projectQueryTable" in index
    assert "Evidence-only mode" in index
    assert "Type a specific test query" in index
    assert "id=\"runQueries\"" in index
    assert "Qwen vs no-reranker diagnosis" not in index
    assert "qwenAnalysis" not in index
    assert "renderQwenAnalysis" not in app

    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function select(id, selected) {{
  elements[id] = {{id, selectedOptions: selected.map(value => ({{value}})), textContent:'', innerHTML:'', classList:{{toggle(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}};
}}
['runSheet','runEmbedding','runStore','runRerankerMain'].forEach(id => select(id, ['all']));
const context={{console,document:{{getElementById(id){{return elements[id] || (elements[id]={{id,value:'',selectedOptions:[],textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}});}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app_path)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
benchmarkOptions={{
  chunkers:['c1','c2','c3','c4','c5'],
  embeddings:['e1','e2','e3'],
  vector_stores:['Qdrant','PGVector','Weaviate','FAISS'],
  rerankers:['Amazon Rerank v1','Qwen3:4B Rerank','bge-reranker-base'],
}};
globalThis.__defaultCount=selectedMatrixComboCount();
document.getElementById('runRerankerMain').selectedOptions=[{{value:'none'}}];
globalThis.__noneCount=selectedMatrixComboCount();
`, context);
if (context.__defaultCount !== 180 || context.__noneCount !== 60) {{
  console.error(JSON.stringify({{defaultCount: context.__defaultCount, noneCount: context.__noneCount}}));
  process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
