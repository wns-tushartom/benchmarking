import csv
import hashlib
import json
import re
import subprocess
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest


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
        faiss = {"chunker":"Heading_sections_l2", "embedding":"gte_multilingual_base", "vector_store":"FAISS", "reranker":"bge-reranker-base", "status":"completed", **_complete_official_metrics()}
        _write_csv(root / "runs" / "latest" / "modular_summary.csv", [faiss])
        _write_csv(root / "runs" / "faiss_noaws_retry" / "modular_summary.csv", [faiss])
        old = dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR
        dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = root / "eval", root / "runs" / "latest", root / "full", root / "gt"
        _write_official_manifest(root / "runs" / "latest" / "manifest.json", dashboard)
        _write_official_manifest(root / "runs" / "faiss_noaws_retry" / "manifest.json", dashboard)
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
const metric={{status:'completed',official_provenance:'trusted',evaluated_queries:5,recall_at_1:.5,recall_at_3:.6,recall_at_5:.7,recall_at_10:.8,mrr:.65,precision_at_5:.4,ndcg_at_5:.68,avg_first_relevant_rank:1.5,no_hit_queries:1,avg_latency_seconds:.03}};
const evaluation={{summary:[{{...metric,sheet:'s',embedding:'e',store:'Qdrant',reranker:'qwen3_4b_rerank'}}], benchmark_reference:{{summary:[{{...metric,sheet:'s',embedding:'e',store:'Qdrant',reranker:'Qwen3:4B Rerank'}},{{...metric,sheet:'s',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',recall_at_5:.9,mrr:.8,ndcg_at_5:.7,avg_latency_seconds:.1}}]}}}};
globalThis.__rows=evaluatedRows(evaluation); renderPipelineComparison(evaluation); globalThis.__stage=document.getElementById('stageComparisonChart').innerHTML;
`, context);
if (context.__rows.length !== 2 || !context.__stage.includes('Vector DB component: FAISS')) {{ console.error(JSON.stringify(context.__rows), context.__stage); process.exit(1); }}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_benchmark_reference_prefers_complete_row_over_newer_partial_duplicate() -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    complete = {
        "chunker": "Heading_sections_l2",
        "embedding": "gte_multilingual_base",
        "vector_store": "FAISS",
        "reranker": "bge-reranker-base",
        "query_count": "500",
        "recall_at_1": "0.7",
        "recall_at_3": "0.8",
        "recall_at_5": "0.9",
        "recall_at_10": "0.95",
        "mrr": "0.82",
        "precision_at_5": "0.4",
        "ndcg_at_5": "0.86",
        "avg_first_relevant_rank": "1.7",
        "no_hit_queries": "10",
        "avg_latency_ms": "30",
        "status": "completed",
    }
    partial = {**complete, "mrr": "", "ndcg_at_5": "", "avg_latency_ms": ""}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "runs" / "latest" / "modular_summary.csv", [partial])
        _write_csv(root / "runs" / "archive" / "complete" / "modular_summary.csv", [complete])
        old = dashboard.MODULAR_DIR, dashboard.FULL_DIR
        dashboard.MODULAR_DIR, dashboard.FULL_DIR = root / "runs" / "latest", root / "full"
        _write_official_manifest(root / "runs" / "latest" / "manifest.json", dashboard)
        _write_official_manifest(root / "runs" / "archive" / "complete" / "manifest.json", dashboard)
        try:
            reference = dashboard.read_benchmark_reference()
        finally:
            dashboard.MODULAR_DIR, dashboard.FULL_DIR = old

    assert len(reference["summary"]) == 1
    row = reference["summary"][0]
    assert row["mrr"] == "0.82"
    assert row["ndcg_at_5"] == "0.86"
    assert row["avg_latency_seconds"] == pytest.approx(0.03)
    assert reference["report"]["complete_metric_rows"] == 1
    assert reference["report"]["incomplete_metric_rows"] == 0


def test_dashboard_broad_artifact_lanes_cover_official_180_without_double_count():
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import generate_matrix, load_benchmark_config

    rows = generate_matrix(load_benchmark_config(Path("configs/benchmark.local.json")))
    def metric(row: dict, modular: bool = False) -> dict:
        base = {"status": "completed", "official_provenance": "trusted", **_complete_official_metrics()}
        if modular:
            return {"chunker": row["chunker"], "embedding": row["embedding"], "vector_store": row["vector_store"], "reranker": row["reranker"], **base}
        return {"sheet": row["chunker"], "embedding": row["embedding"], "store": row["vector_store"], "reranker": row["reranker"], **base}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "eval" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[:135]])
        _write_csv(root / "reranked" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[135:165]])
        _write_csv(root / "runs" / "archive" / "faiss_noaws_complete" / "modular_summary.csv", [metric(r, modular=True) for r in rows[165:]])
        extra_rows = []
        for embedding in ["jina_v3", "gte_multilingual_base", "openai_text-embedding-3-large"]:
            for store in ["Qdrant", "PGVector", "Weaviate"]:
                for reranker in ["Amazon Rerank v1", "Qwen3:4B Rerank", "bge-reranker-base"]:
                    extra_rows.append({"chunker": "semantic_split", "embedding": embedding, "vector_store": store, "reranker": reranker, "query_count": "500", "recall_at_5": "0.99", "mrr": "0.99", "avg_latency_ms": "200"})
        _write_csv(root / "runs" / "archive" / "non_official_old_experiment" / "modular_summary.csv", extra_rows)
        old = dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR
        dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = root / "eval", root / "runs" / "latest", root / "full", root / "gt"
        _write_official_manifest(root / "runs" / "archive" / "faiss_noaws_complete" / "manifest.json", dashboard)
        try:
            evaluation = dashboard.read_evaluation()
            evaluation["reranked"] = dashboard.read_evaluation_dir(root / "reranked")
        finally:
            dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = old

    assert len(evaluation["benchmark_reference"]["summary"]) == 15
    assert dashboard.official_evaluated_count(evaluation) == 180
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    payload = json.dumps(evaluation)
    js = f"""
const fs=require('fs'), vm=require('vm');
const context={{console,document:{{getElementById(){{return {{value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}},querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `globalThis.__rows=evaluatedRows({payload});`, context);
const keys=new Set(context.__rows.map(r=>`${{r.sheet}}|${{r.embedding}}|${{r.store}}|${{r.reranker||'none'}}`));
const byStore=context.__rows.reduce((a,r)=>(a[r.store]=(a[r.store]||0)+1,a),{{}});
if (keys.size !== 180 || context.__rows.length !== 180 || byStore.FAISS !== 45) {{ console.error(JSON.stringify({{rows:context.__rows.length, keys:keys.size, byStore}})); process.exit(1); }}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


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
        detail_path = root / "runs" / "aws_combined" / "modular_details.csv"
        _write_csv(detail_path, [row])
        _write_csv(
            root / "runs" / "aws_combined" / "modular_summary.csv",
            [{**row, "status": "completed", **_complete_official_metrics()}],
        )
        old_modular, old_lookup = dashboard.MODULAR_DIR, dashboard.chunk_lookup_for_sheet
        dashboard.MODULAR_DIR = root / "runs" / "latest"
        _write_official_manifest(root / "runs" / "aws_combined" / "manifest.json", dashboard)
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


def test_dataset_status_renderer_defines_evaluation_before_reading_groundtruth_status() -> None:
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    text = app.read_text(encoding="utf-8")
    status_renderer = text.split("function renderDatasetStatusSummary()", 1)[1].split(
        "function renderOperational", 1
    )[0]

    declaration = "const evaluation = op.evaluation || {};"
    status_read = "evaluation?.report?.groundtruth_rows"
    assert declaration in status_renderer
    assert status_renderer.index(declaration) < status_renderer.index(status_read)


def test_completed_coverage_cells_open_the_exact_metrics_combination() -> None:
    app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")

    assert 'coverage-metric-btn' in app
    assert "function showMetricsCombination(" in app
    assert "showPage('metrics')" in app
    assert "qualityComboFilter" in app
    assert "data-combo=" in app
    assert "const label = v.status === 'metrics' ? 'metrics' : 'ok';" not in app


def test_dataset_chunk_labels_are_per_strategy_not_a_fake_global_total() -> None:
    app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")

    assert "chunk_counts_by_strategy" in app
    assert "chunks/strategy" in app
    assert "documents · ${fmtInt(row.chunk_count || 0)} chunks" not in app


def test_metrics_exposes_shared_dataset_groundtruth_and_result_set_selectors() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="metricsDataset"' in index
    assert 'id="metricsGroundtruth"' in index
    assert 'id="metricsResultSet"' in index
    assert 'id="metricsSourceContext"' in index
    assert "function syncSourceSelectorControls()" in app
    assert "target.replaceChildren" in app
    assert "target.innerHTML = source.innerHTML" not in app
    assert "function renderMetricsSource(payload)" in app
    assert "$('metricsDataset')?.addEventListener('input'" in app
    assert "$('metricsGroundtruth')?.addEventListener('input'" in app
    assert "$('metricsResultSet')?.addEventListener('input'" in app


def test_run_pipeline_can_explicitly_allow_audited_text_only_documents() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="runAllowPartialExtraction"' in index
    assert "Allow audited text-only documents" in index
    assert "if ($('runAllowPartialExtraction')?.checked) p.set('allow_partial_extraction', '1');" in app


def test_advanced_pipeline_controls_use_the_six_approved_stage_names_in_order() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")

    expected = [
        "Extraction",
        "Embedding",
        "Vector-store ingestion",
        "Retrieval",
        "Reranking",
        "Evaluation",
    ]
    labels = re.findall(r'data-pipeline-stage="[^"]+"[^>]*>\s*<h3>([^<]+)</h3>', index)
    assert labels == expected
    assert 'id="runParseMineruBtn"' in index
    assert 'id="runIngestBtn"' in index
    assert 'id="runFullGtBtn"' in index
    assert 'id="runRerankerBtn"' in index
    assert 'id="runRerankedEvalBtn"' in index
    assert 'id="runEvalBtn"' in index


def test_grounding_audit_is_visibly_disabled_until_the_feature_is_active() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")

    assert re.search(
        r'<button[^>]+data-page="hallucination"[^>]+disabled[^>]+aria-disabled="true"',
        index,
    )
    assert re.search(r'id="runHallucinationBtn"[^>]+disabled', index)
    assert re.search(r'id="runHallucinationLlmBtn"[^>]+disabled', index)
    assert 'data-feature-state="disabled"' in index
    assert "Not active in this build" in index


def test_preflight_keeps_operator_summary_primary_and_diagnostics_collapsed() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="runReadinessSummary"' in index
    assert re.search(r'<details[^>]+id="runTechnicalDiagnostics"(?![^>]+open)', index)
    assert "Technical readiness details" in index
    assert "Ready to run the selected pipeline." in app
    assert "Resolve before running:" in app
    assert "groundtruth_id" in app
    assert "payload.groundtruth ||" not in app


def test_nvidia_baseline_and_reranked_results_are_independently_visible():
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app_text = app.read_text(encoding="utf-8")
    assert 'id="nvidiaBenchmarkMode"' in index
    assert '<option value="baseline">Baseline</option>' in index
    assert '<option value="reranked">Reranked</option>' in index
    assert "$('nvidiaBenchmarkMode')?.addEventListener('input'" in app_text
    payload = {
        "health": {"ok": True},
        "benchmark": {
            "report": {"ok": False, "created_at": "stale", "error": "stale failed latest report"},
            "summary": [],
            "baseline": {
                "report": {
                    "ok": True,
                    "groundtruth_rows": 500,
                    "evaluated_rows": 500,
                    "successful_queries": 500,
                },
                "summary": [{
                    "pipeline": "NVIDIA RAG Blueprint baseline",
                    "evaluated_queries": 500,
                    "successful_queries": 500,
                    "recall_at_5": 0.81,
                    "mrr": 0.72,
                    "ndcg_at_5": 0.76,
                    "avg_latency_seconds": 0.41,
                }],
            },
            "reranked": {
                "report": {
                    "ok": True,
                    "groundtruth_rows": 500,
                    "evaluated_rows": 498,
                    "successful_queries": 497,
                },
                "summary": [{
                    "pipeline": "NVIDIA RAG Blueprint reranked",
                    "evaluated_queries": 498,
                    "successful_queries": 497,
                    "recall_at_5": 0.91,
                    "mrr": 0.84,
                    "ndcg_at_5": 0.88,
                }],
            },
        },
    }
    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function el(id) {{
  return elements[id] ||= {{id,value:id === 'nvidiaBenchmarkMode' ? 'baseline' : 'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};
}}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
const payload={json.dumps(payload)};
renderNvidiaRag(payload);
globalThis.__baseline=[document.getElementById('nvidiaBenchmarkHint').textContent, document.getElementById('nvidiaBenchmarkCards').innerHTML, document.getElementById('nvidiaBenchmarkTable').innerHTML].join(' ');
document.getElementById('nvidiaBenchmarkMode').value='reranked';
renderNvidiaRag(payload);
globalThis.__reranked=[document.getElementById('nvidiaBenchmarkHint').textContent, document.getElementById('nvidiaBenchmarkCards').innerHTML, document.getElementById('nvidiaBenchmarkTable').innerHTML].join(' ');
`, context);
const baseline=context.__baseline;
const reranked=context.__reranked;
if (!baseline.includes('Baseline') || !baseline.includes('500 evaluated') || !baseline.includes('500 successful') || !baseline.includes('81.0%') || !baseline.includes('0.720') || baseline.includes('91.0%')) {{
  console.error('baseline', baseline); process.exit(1);
}}
if (!reranked.includes('Reranked') || !reranked.includes('498 evaluated') || !reranked.includes('497 successful') || !reranked.includes('91.0%') || !reranked.includes('0.840') || !reranked.includes('—s') || reranked.includes('0.000s') || reranked.includes('81.0%')) {{
  console.error('reranked', reranked); process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_nvidia_legacy_reranked_artifact_is_not_mislabeled_as_baseline():
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    payload = {
        "health": {"ok": True},
        "benchmark": {
            "report": {"ok": True, "reranker_enabled": True, "evaluated_rows": 500, "successful_queries": 500},
            "summary": [{"evaluated_queries": 500, "successful_queries": 500, "recall_at_5": 0.91, "mrr": 0.84}],
        },
    }
    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function el(id) {{ return elements[id] ||= {{id,value:id === 'nvidiaBenchmarkMode' ? 'baseline' : 'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}}; }}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
const payload={json.dumps(payload)};
renderNvidiaRag(payload);
globalThis.__baseline=document.getElementById('nvidiaBenchmarkCards').innerHTML;
document.getElementById('nvidiaBenchmarkMode').value='reranked';
renderNvidiaRag(payload);
globalThis.__reranked=document.getElementById('nvidiaBenchmarkCards').innerHTML;
`, context);
if (context.__baseline.includes('91.0%') || !context.__reranked.includes('91.0%')) process.exit(1);
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_source_mode_disables_incompatible_legacy_actions():
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function el(id) {{ return elements[id] ||= {{id,value:'',textContent:'',innerHTML:'',disabled:false,checked:false,hidden:false,classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}}; }}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
state.sourceCatalog={{datasets:[{{id:'dataset:wns-default',kind:'default'}},{{id:'project:demo',kind:'project'}}]}};
document.getElementById('runDataset').value='project:demo';
document.getElementById('runGroundtruth').value='groundtruth:none';
syncRunMode();
globalThis.__disabled=['runParseMineruBtn','runIngestBtn','runSmokeBtn','runFullGtBtn','runRerankerBtn','runRerankedEvalBtn','runEvalBtn','runHallucinationBtn','runHallucinationLlmBtn'].every(id=>document.getElementById(id).disabled);
globalThis.__primary=document.getElementById('runCompletePipelineBtn').disabled;
`, context);
if (!context.__disabled || context.__primary) process.exit(1);
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_frontend_assets_use_current_cache_key():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    assert "/recommendations.js?v=20260716-extraction-override" in index
    assert "/app.js?v=20260720-provisional-recommendations" in index
    assert "/styles.css?v=20260716-meeting-hardening" in index
    assert "20260709-query-ui" not in index


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


def test_global_dataset_and_groundtruth_context_owns_page_mirrors() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="globalDataset"' in index
    assert 'id="globalGroundtruth"' in index
    assert 'id="globalSourceContext"' in index
    for mirror_id in (
        "runDataset",
        "runGroundtruth",
        "nvidiaDataset",
        "nvidiaGroundtruth",
        "metricsDataset",
        "metricsGroundtruth",
        "recommendationDataset",
        "recommendationGroundtruth",
    ):
        assert f'data-source-mirror><select id="{mirror_id}"' in index
    assert "function applyGlobalSourceContext" in app
    assert "fillSourceSelect('globalDataset'" in app
    assert "fillSourceSelect('globalGroundtruth'" in app
    assert "$('globalDataset')?.addEventListener('change'" in app
    assert "$('globalGroundtruth')?.addEventListener('change'" in app


def test_global_project_selection_updates_document_readiness_kpi() -> None:
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    js = f"""
const fs=require('fs'),vm=require('vm');
const elements={{}};
function el(id) {{
  if (!elements[id]) elements[id]={{
    value:id==='globalDataset'?'project:demo':(id==='globalGroundtruth'?'groundtruth:none':''), textContent:'', innerHTML:'',
    options:[], selectedOptions:[], classList:{{toggle(){{}}}},
    addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}},
  }};
  return elements[id];
}}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
state={{
  operational:{{document_repository:{{ready_count:0,total:6}},known_pdf_count:6}},
  sourceCatalog:{{datasets:[{{id:'project:demo',label:'Demo corpus',kind:'uploaded_project',document_count:5,ready:true,validation:'ready_for_project_matrix'}}]}},
  files:[], options:{{}},
}};
renderDatasetStatusSummary();
globalThis.__result={{status:document.getElementById('pdfStatus').textContent,note:document.getElementById('pdfNote').textContent,groundtruth:document.getElementById('groundTruthStatus').textContent}};
`, context);
if (context.__result.status !== '5/5' || !context.__result.note.includes('Demo corpus') || !context.__result.note.includes('project matrix') || context.__result.groundtruth !== 'Evidence-only') {{
  console.error(JSON.stringify(context.__result)); process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_run_pipeline_depth_controls_are_explicit_and_chunk_limit_is_fixed() -> None:
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    server = (root / "scripts" / "serve_benchmark_dashboard.py").read_text(encoding="utf-8")
    pipeline = (root / "scripts" / "run_complete_pipeline.py").read_text(encoding="utf-8")

    assert 'id="runRetrievalTopK"' in index
    assert 'id="runRerankedOutputK"' in index
    assert 'id="runLimit"' not in index
    assert 'id="rerankerLimit"' not in index
    assert "Chunk limit" not in index
    assert "Reranker artifact limit" not in index
    assert "p.set('retrieval_top_k'" in app
    assert "p.set('reranked_output_k'" in app
    assert "p.set('chunk_limit'" not in app
    assert "$('runTopK')" not in app
    assert "$('rerankerLimit')" not in app
    assert "p.set('limit', '0')" in app
    assert '"--chunk-limit"' not in server
    assert 'parser.add_argument("--chunk-limit"' not in pipeline


def test_dashboard_publishes_document_readiness_without_losing_last_success():
    root = Path(__file__).resolve().parents[1]
    server = (root / "scripts" / "serve_benchmark_dashboard.py").read_text(encoding="utf-8")
    assert "from scripts.dashboard_snapshot_state import publish_document_readiness" in server
    assert "document_repository = publish_document_readiness(ROOT, read_document_repository())" in server
    assert '"document_repository_latest_attempt": document_repository.get("latest_attempt", {})' in server


def test_dashboard_query_upload_reuses_hardened_parser_contract():
    root = Path(__file__).resolve().parents[1]
    server = (root / "scripts" / "serve_benchmark_dashboard.py").read_text(encoding="utf-8")
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert 'upload_type not in {"dataset", "groundtruth", "queries"}' in server
    assert "payload = parse_query_upload(original, content)" in server
    assert 'id="runQueryFile"' in index
    assert 'accept=".txt,.csv,.xlsx"' in index
    assert 'id="runQueryUploadBtn"' in index
    assert "async function uploadRunQueries()" in app
    assert "form.append('upload_type', 'queries')" in app
    assert "$('runQueries').value = payload.queries.join('\\n')" in app


def test_document_repository_reports_audit_reason_and_parser_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    pdf_dir = tmp_path / "data" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "one.pdf").write_bytes(b"%PDF-1.4\n")
    audit_path = tmp_path / "data" / "pdf_extraction_audit.csv"
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(dashboard, "PDF_AUDIT_PATH", audit_path)
    monkeypatch.setattr(dashboard, "read_pdf_chunk_counts", lambda: {"one.pdf": 3})

    missing = dashboard.read_document_repository()
    assert missing["audit_status"] == "missing"
    assert missing["review_reason_counts"] == {"audit_missing": 1}
    assert missing["parser_counts"] == {}

    _write_csv(audit_path, [{
        "pdf_name": "one.pdf",
        "status": "text_only_review",
        "parser_method": "PyPDF2_fallback",
        "needs_ocr_review": "true",
    }])
    fallback = dashboard.read_document_repository()
    assert fallback["audit_status"] == "text_only_fallback"
    assert fallback["parser_counts"] == {"PyPDF2_fallback": 1}
    assert fallback["review_reason_counts"] == {"text_only_fallback": 1}


def test_document_readiness_summary_explains_live_and_preserved_states() -> None:
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    script = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function el(id){{return elements[id] ||= {{id,value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}},add(){{}},remove(){{}}}},addEventListener(){{}},querySelectorAll(){{return[];}},setAttribute(){{}}}};}}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return[];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>{{}},text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
globalThis.__missing=documentReadinessSummary({{total:225,ready_count:0,review_count:225,audit_status:'missing',display_source:'latest_attempt'}});
globalThis.__fallback=documentReadinessSummary({{total:225,ready_count:0,review_count:225,audit_status:'text_only_fallback',parser_counts:{{PyPDF2_fallback:225}},display_source:'latest_attempt'}});
globalThis.__preserved=documentReadinessSummary({{total:225,ready_count:225,review_count:0,audit_status:'clean',display_source:'last_successful',latest_attempt:{{total:225,ready_count:0,review_count:225,audit_status:'missing'}}}});
`, context);
console.log(JSON.stringify({{missing:context.__missing,fallback:context.__fallback,preserved:context.__preserved}}));
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summaries = json.loads(proc.stdout)
    assert "extraction audit missing" in summaries["missing"]
    assert "text-only fallback" in summaries["fallback"]
    assert "preserved clean snapshot" in summaries["preserved"]
    assert "latest attempt: extraction audit missing" in summaries["preserved"]


def test_document_repository_page_does_not_show_summary_cards():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert "documentRepositoryCards" not in index
    assert "Repository PDFs" not in app
    assert "Chunked documents" not in app
    assert "Review queue" not in app


def test_user_project_upload_is_isolated_and_creates_manifest_and_canonical_corpus():
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
        assert (project_root / "extracted_text" / "documents.jsonl").exists()
        assert not (project_root / "search_index.json").exists()
        manifest = json.loads((project_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["storage_layout"]["runs"].endswith("/runs")
        assert "data/uploads" not in json.dumps(manifest)


def test_public_project_upload_payload_excludes_internal_paths_and_layout() -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    internal = {
        "ok": True,
        "schema_version": 1,
        "project_id": "demo_1234",
        "label": "Demo",
        "created_at": "2026-07-17T00:00:00+00:00",
        "bytes": 42,
        "corpus_sha256": "a" * 64,
        "document_count": 1,
        "source_count": 1,
        "page_count": 1,
        "extracted_count": 1,
        "extraction_status": "complete",
        "extraction_failures": ["/private/repo/bad.pdf: provider detail"],
        "chunk_count": 0,
        "sources": [{
            "source_name": "policy.pdf",
            "raw_sha256": "b" * 64,
            "raw_size_bytes": 42,
            "parser_versions": ["pypdf_v1"],
            "page_count": 1,
            "status": "complete",
            "saved_path": "/private/repo/policy.pdf",
        }],
        "manifest_path": "/private/repo/manifest.json",
        "saved_path": "/private/repo/upload.zip",
        "canonical_corpus": "/private/repo/documents.jsonl",
        "storage_layout": {"root": "/private/repo"},
        "next_steps": ["Review /private/repo"],
        "workbook": "/private/repo/corpus.xlsx",
        "search_index": "/private/repo/index.json",
        "question_file": "/private/repo/questions.csv",
    }

    public = dashboard.public_project_upload_payload(internal)

    assert public["source_id"] == "project:demo_1234"
    assert public["extraction_failure_count"] == 1
    assert set(public) == {
        "ok", "schema_version", "project_id", "source_id", "label", "created_at",
        "bytes", "corpus_sha256", "document_count", "source_count", "page_count",
        "extracted_count", "extraction_status", "extraction_failure_count",
        "chunk_count", "sources",
    }
    assert set(public["sources"][0]) == {
        "source_name", "raw_sha256", "raw_size_bytes", "parser_versions",
        "page_count", "status",
    }
    assert "/private/" not in json.dumps(public)


def test_project_query_uses_uploaded_text_without_fake_matrix_rows():
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
                top_k=3,
            )
        finally:
            dashboard.USER_PROJECTS_DIR = old

        assert result["ok"] is True
        assert result["mode"] == "lexical_preview"
        assert "combo_count" not in result
        assert "rows" not in result
        assert "selections" not in result
        assert result["query"] == "cancelled flight refund"
        assert result["hits"][0]["pdf_name"] == "refund.txt"
        assert "Cancelled flight customers" in result["hits"][0]["paragraph"]
        run_root = Path(td) / "user_projects" / project["project_id"] / "runs" / result["run_id"]
        manifest = json.loads((run_root / "manifest.json").read_text())
        assert manifest["mode"] == "lexical_preview"
        assert "selections" not in manifest


def test_frontend_has_multi_select_controls_without_legacy_lexical_preview():
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
    assert "/api/project-query" not in app
    assert "projectQueryTable" not in index
    assert "Lexical preview" not in index
    assert "Search this project" not in index
    assert "function runProjectQuery()" not in app
    assert "function renderProjectQuery(" not in app
    assert "projectSelections()" not in app
    assert "lexical_preview" not in app
    assert "selections: projectSelections()" not in app
    assert "Evidence-only queries" in index
    assert "id=\"runQueries\"" in index
    assert 'id="runDataset"' in index
    assert 'id="nvidiaDataset"' in index
    assert 'id="uploadType"' in index
    assert 'id="nvidiaDataPath"' not in index
    assert "p.set('dataset_id'" in app
    assert "p.set('groundtruth_id'" in app
    assert "form.append('upload_type'" in app
    assert "p.set('path'" not in app
    assert "p.set('groundtruth'" not in app
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
state.sourceCatalog={{datasets:[{{id:'project:demo',kind:'uploaded_project',sheets:['uploaded_chunks']}}],groundtruth:[]}};
document.getElementById('runDataset').value='project:demo';
document.getElementById('runRerankerMain').selectedOptions=[{{value:'all'}}];
globalThis.__projectCount=selectedMatrixComboCount();
`, context);
if (context.__defaultCount !== 180 || context.__noneCount !== 60 || context.__projectCount !== 180) {{
  console.error(JSON.stringify({{defaultCount: context.__defaultCount, noneCount: context.__noneCount, projectCount: context.__projectCount}}));
  process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_project_result_http_endpoints_are_exact_no_store_and_safely_mapped(monkeypatch: pytest.MonkeyPatch):
    import scripts.serve_benchmark_dashboard as dashboard
    from source.services.project_run_results import ProjectRunResultsError

    project_id = "alpha_0123456789abcdef0123456789abcdef"
    run_id = "run_0123456789abcdef0123456789abcdef"

    class FakeResults:
        def result_sources(self, *, official_configured: int, official_evaluated: int):
            return {"official": {"source_type": "official", "configured": official_configured, "evaluated": official_evaluated}, "projects": [{"project_id": project_id, "label": "Alpha"}]}

        def project_runs(self, supplied_project_id: str):
            if supplied_project_id != project_id:
                raise ProjectRunResultsError("not_found", "Project result was not found", 404)
            return [{"project_id": project_id, "run_id": run_id}]

        def project_run_results(self, supplied_project_id: str, supplied_run_id: str):
            if (supplied_project_id, supplied_run_id) != (project_id, run_id):
                raise ProjectRunResultsError("not_found", "Run result was not found", 404)
            return {"source_type": "uploaded_project", "project_id": project_id, "run_id": run_id, "rows": []}

        def project_run_evidence(self, supplied_project_id: str, supplied_run_id: str, combo_id: str, *, limit: int, offset: int):
            if (supplied_project_id, supplied_run_id, combo_id) != (project_id, run_id, "combo"):
                raise ProjectRunResultsError("not_found", "Combination evidence was not found", 404)
            return {"project_id": project_id, "run_id": run_id, "combo_id": combo_id, "rows": [], "next_offset": None, "limit": limit, "offset": offset}

    monkeypatch.setattr(dashboard, "project_result_service", lambda: FakeResults())
    monkeypatch.setattr(
        dashboard,
        "read_evaluation",
        lambda: {
            "summary": [],
            "reranked": {"summary": []},
            "benchmark_reference": {
                "summary": [{}, {}],
                "report": {"config_rows": 0, "official_matrix_rows": 180},
            },
        },
    )
    monkeypatch.setattr(dashboard, "official_evaluated_count", lambda _evaluation: 2)
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        paths = (
            "/api/result-sources",
            f"/api/project-runs?project_id={quote(project_id)}",
            f"/api/project-run-results?project_id={quote(project_id)}&run_id={quote(run_id)}",
            f"/api/project-run-evidence?project_id={quote(project_id)}&run_id={quote(run_id)}&combo_id=combo&limit=2&offset=0",
        )
        payloads = []
        for path in paths:
            with urlopen(base + path, timeout=5) as response:
                assert response.headers["Cache-Control"] == "no-store, max-age=0"
                payloads.append(json.loads(response.read()))
        assert payloads[0]["official"] == {"source_type": "official", "configured": 180, "evaluated": 2}
        assert payloads[1][0]["run_id"] == run_id
        assert payloads[2]["source_type"] == "uploaded_project"
        assert payloads[3]["limit"] == 2

        for bad_path in (
            "/api/project-runs",
            f"/api/project-runs?project_id={quote(project_id)}&extra=/srv/private",
            f"/api/project-run-evidence?project_id={quote(project_id)}&run_id={quote(run_id)}&combo_id=combo&limit=999&offset=0",
            "/api/project-runs?project_id=missing_0123456789abcdef0123456789abcdef",
        ):
            with pytest.raises(HTTPError) as caught:
                urlopen(base + bad_path, timeout=5)
            body = json.loads(caught.value.read())
            assert set(body) == {"error"}
            assert set(body["error"]) == {"code", "message"}
            assert "/srv" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_official_evaluated_count_excludes_failed_or_unmeasured_rows():
    import scripts.serve_benchmark_dashboard as dashboard

    official_key = next(iter(dashboard.official_matrix_keys()))
    chunker, embedding, store, reranker = official_key
    base = {
        "sheet": chunker,
        "embedding": embedding,
        "store": store,
        "reranker": reranker,
    }
    failed = {**base, "status": "failed"}
    failed_with_stale_metrics = {
        **base,
        "status": "completed",
        "query_count": "1",
        "error_code": "reranker_failed",
    }
    unmeasured = {**base, "status": "completed"}
    measured = {
        **base,
        "status": "completed",
        "official_provenance": "trusted",
        **_complete_official_metrics(),
    }

    assert dashboard.official_evaluated_count({
        "summary": [failed],
        "reranked": {"summary": []},
        "benchmark_reference": {"summary": []},
    }) == 0
    assert dashboard.official_evaluated_count({
        "summary": [failed_with_stale_metrics],
        "reranked": {"summary": []},
        "benchmark_reference": {"summary": []},
    }) == 0
    assert dashboard.official_evaluated_count({
        "summary": [unmeasured],
        "reranked": {"summary": []},
        "benchmark_reference": {"summary": []},
    }) == 0
    assert dashboard.official_evaluated_count({
        "summary": [measured],
        "reranked": {"summary": []},
        "benchmark_reference": {"summary": []},
    }) == 1


def test_benchmark_reference_skips_failed_rows_and_uses_measured_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    official_key = next(iter(dashboard.official_matrix_keys()))
    chunker, embedding, store, reranker = official_key
    base = {
        "chunker": chunker,
        "embedding": embedding,
        "vector_store": store,
        "reranker": reranker,
    }
    latest = tmp_path / "latest.csv"
    archive = tmp_path / "archive.csv"
    failed = tmp_path / "failed.csv"
    _write_csv(latest, [{**base, "status": "completed", **_complete_official_metrics(), "query_count": ""}])
    _write_csv(archive, [{**base, "status": "completed", **_complete_official_metrics()}])
    _write_csv(failed, [{**base, "status": "failed", **_complete_official_metrics()}])
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {official_key})
    _write_official_manifest(tmp_path / "manifest.json", dashboard)
    monkeypatch.setattr(
        dashboard,
        "benchmark_reference_sources",
        lambda: [
            ("modular_matrix", latest),
            ("modular_run:archive", archive),
            ("modular_run:failed", failed),
        ],
    )

    reference = dashboard.read_benchmark_reference()

    assert len(reference["summary"]) == 1
    assert reference["summary"][0]["source"] == "modular_run:archive"
    assert reference["summary"][0]["status"] == "completed"
    assert dashboard.official_evaluated_count({
        "summary": [],
        "reranked": {"summary": []},
        "benchmark_reference": reference,
    }) == 1


def _complete_official_metrics() -> dict:
    return {
        "query_count": "5", "recall_at_1": "0.5", "recall_at_3": "0.6",
        "recall_at_5": "0.7", "recall_at_10": "0.8", "mrr": "0.65",
        "precision_at_5": "0.4", "ndcg_at_5": "0.68",
        "avg_first_relevant_rank": "1.5", "no_hit_queries": "1",
        "avg_latency_seconds": "0.03",
    }


def _write_official_manifest(path: Path, dashboard, *, status: str = "completed") -> None:
    from benchmarking.core.config import config_hash, load_benchmark_config

    artifacts = {
        artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()
        for artifact in path.parent.iterdir()
        if artifact.is_file() and artifact.suffix == ".csv"
    }
    artifact_root = path.parent.resolve().as_posix()
    path.write_text(json.dumps({
        "status": status,
        "run_id": path.parent.name,
        "config_hash": config_hash(load_benchmark_config(dashboard.CONFIG_PATH)),
        "dataset_id": dashboard.DEFAULT_DATASET_ID,
        "groundtruth_id": "groundtruth:repository:qa_text_test.csv",
        "query_count": 5,
        "artifact_root": artifact_root,
        "artifact_sha256": artifacts,
    }), encoding="utf-8")


def test_official_reference_keeps_untrusted_rows_diagnostic_and_accepts_only_manifested_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    official_key = next(iter(dashboard.official_matrix_keys()))
    chunker, embedding, store, reranker = official_key
    base = {
        "chunker": chunker, "embedding": embedding, "vector_store": store,
        "reranker": reranker, "status": "completed", **_complete_official_metrics(),
    }
    copied = tmp_path / "runs" / "copied" / "modular_summary.csv"
    trusted = tmp_path / "runs" / "trusted" / "modular_summary.csv"
    _write_csv(copied, [{**base, "recall_at_5": "0.99"}])
    _write_csv(trusted, [{**base, "recall_at_5": "0.70"}])
    _write_official_manifest(trusted.parent / "manifest.json", dashboard)
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {official_key})
    monkeypatch.setattr(dashboard, "benchmark_reference_sources", lambda: [
        ("modular_run:copied", copied), ("modular_run:trusted", trusted),
    ])

    reference = dashboard.read_benchmark_reference()

    assert [row["source"] for row in reference["summary"]] == ["modular_run:trusted"]
    assert reference["summary"][0]["official_provenance"] == "trusted"
    assert reference["report"]["diagnostic_rows"] == 1
    assert reference["diagnostics"][0]["admission_reason"] == "manifest_missing"


def test_selected_manifest_cannot_admit_rows_outside_its_concrete_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import config_hash, generate_matrix, load_benchmark_config, selected_config

    cfg = load_benchmark_config(dashboard.CONFIG_PATH)
    matrix = generate_matrix(cfg)
    selected_row = matrix[0]
    other_row = next(row for row in matrix if row["reranker"] != selected_row["reranker"])
    selected_key = (
        selected_row["chunker"], selected_row["embedding"], selected_row["vector_store"],
        dashboard.canonical_reranker_name(selected_row["reranker"]),
    )
    other_key = (
        other_row["chunker"], other_row["embedding"], other_row["vector_store"],
        dashboard.canonical_reranker_name(other_row["reranker"]),
    )
    selection = {
        "chunker": selected_row["chunker"], "embedding": selected_row["embedding"],
        "vector_store": selected_row["vector_store"], "index_type": selected_row["index_type"],
        "retrieval_method": selected_row["retrieval_method"], "reranker": selected_row["reranker"],
        "evaluator": selected_row["evaluator"],
    }
    concrete = selected_config(cfg, selection)
    concrete.setdefault("experiment", {})["selection"] = selection
    run_dir = tmp_path / "selected"
    summary = run_dir / "modular_summary.csv"
    _write_csv(summary, [{
        "chunker": other_key[0], "embedding": other_key[1], "vector_store": other_key[2],
        "reranker": other_key[3], "status": "completed", **_complete_official_metrics(),
    }])
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed", "run_id": run_dir.name,
        "config_hash": config_hash(concrete),
        "official_matrix_contract_hash": config_hash(cfg),
        "selected_run_config_hash": config_hash(concrete),
        "selection": selection,
        "dataset_id": dashboard.DEFAULT_DATASET_ID,
        "groundtruth_id": dashboard.OFFICIAL_GROUNDTRUTH_ID,
        "query_count": 5,
        "artifact_root": run_dir.resolve().as_posix(),
        "artifact_sha256": {"modular_summary.csv": hashlib.sha256(summary.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {selected_key, other_key})
    monkeypatch.setattr(dashboard, "benchmark_reference_sources", lambda: [("modular_run:selected", summary)])

    reference = dashboard.read_benchmark_reference()

    assert reference["summary"] == []
    assert reference["diagnostics"][0]["admission_reason"] == "not_in_manifest_selection"


def test_official_evaluated_count_requires_explicit_completed_full_metrics_and_trusted_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    official_key = next(iter(dashboard.official_matrix_keys()))
    chunker, embedding, store, reranker = official_key
    base = {
        "sheet": chunker, "embedding": embedding, "store": store,
        "reranker": reranker, **_complete_official_metrics(),
    }
    rows = [
        {**base, "status": "", "official_provenance": "trusted"},
        {**base, "status": "unknown", "official_provenance": "trusted"},
        {**base, "status": "completed", "official_provenance": "untrusted"},
        {**base, "status": "completed", "official_provenance": "trusted", "mrr": ""},
    ]
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {official_key})
    evaluation = {"summary": rows, "reranked": {"summary": []}, "benchmark_reference": {"summary": []}}
    assert dashboard.official_evaluated_count(evaluation) == 0

    rows.append({**base, "status": "completed", "official_provenance": "trusted"})
    assert dashboard.official_evaluated_count(evaluation) == 1


def test_frontend_coverage_does_not_promote_failed_rows_and_prefers_trusted_duplicate() -> None:
    root = Path(__file__).resolve().parents[1]
    app = root / "web" / "app.js"
    source_state = root / "web" / "source-state.js"
    js = f"""
const fs=require('fs'),vm=require('vm');
const elements={{}};
function el(id){{return elements[id] ||= {{id,value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}},add(){{}},remove(){{}}}},addEventListener(){{}},querySelectorAll(){{return[];}},setAttribute(){{}}}};}}
const context={{console,DashboardSourceState:require({str(source_state)!r}),document:{{getElementById:el,querySelectorAll(){{return[];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#overview'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code+`
const metric={{evaluated_queries:500,recall_at_1:.5,recall_at_3:.6,recall_at_5:.7,recall_at_10:.8,mrr:.65,precision_at_5:.4,ndcg_at_5:.68,avg_first_relevant_rank:1.5,no_hit_queries:1,avg_latency_seconds:.03,sheet:'c',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',official_provenance:'trusted'}};
benchmarkOptions={{chunkers:['c'],embeddings:['e'],vector_stores:['FAISS'],rerankers:['bge-reranker-base']}};
const expected='c|e|FAISS|bge-reranker-base';
state={{operational:{{evaluation:{{summary:[
  {{...metric,status:'failed'}},
  {{...metric,status:'completed',source_type:'nvidia',recall_at_5:.99}},
  {{...metric,status:'completed',source_type:'uploaded_project',recall_at_5:.98}},
  {{...metric,status:'completed',sheet:'wrong',recall_at_5:.97}},
  {{...metric,status:'completed'}},
],benchmark_reference:{{report:{{expected_keys:[expected]}}}}}}}}}};
renderCoverage([]); globalThis.__valid=document.getElementById('coverageTable').innerHTML;
state.operational.evaluation.summary=[{{...metric,status:'failed'}},{{...metric,status:'completed',source_type:'nvidia'}}];
renderCoverage([]); globalThis.__failed=document.getElementById('coverageTable').innerHTML;
`,context);
if(!context.__valid.includes('Done · open Metrics') || context.__failed.includes('Done · open Metrics')){{console.error(context.__valid,context.__failed);process.exit(1);}}
"""
    proc = subprocess.run(["node", "-e", js], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
