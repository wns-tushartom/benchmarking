import csv
import json
import subprocess
import tempfile
from pathlib import Path


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
        _write_csv(root / "eval" / "groundtruth_eval_summary.csv", [{"sheet":"s", "embedding":"e", "store":"Qdrant", "reranker":"qwen3_4b_rerank", "recall_at_5":"0.5", "mrr":"0.4", "ndcg_at_5":"0.3", "avg_latency_seconds":"0.2"}])
        faiss = {"chunker":"s", "embedding":"e", "vector_store":"FAISS", "reranker":"bge-reranker-base", "query_count":"500", "recall_at_5":"0.9", "mrr":"0.8", "ndcg_at_10":"0.7", "avg_latency_ms":"100"}
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
        base = {"recall_at_5": "0.5", "mrr": "0.4", "ndcg_at_5": "0.3", "avg_latency_seconds": "0.2"}
        if modular:
            return {"chunker": row["chunker"], "embedding": row["embedding"], "vector_store": row["vector_store"], "reranker": row["reranker"], "query_count": "500", "recall_at_5": "0.5", "mrr": "0.4", "avg_latency_ms": "200"}
        return {"sheet": row["chunker"], "embedding": row["embedding"], "store": row["vector_store"], "reranker": row["reranker"], **base}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "eval" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[:135]])
        _write_csv(root / "reranked" / "groundtruth_eval_summary.csv", [metric(r) for r in rows[135:165]])
        _write_csv(root / "runs" / "archive" / "faiss_noaws_complete" / "modular_summary.csv", [metric(r, modular=True) for r in rows[165:]])
        old = dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR
        dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = root / "eval", root / "runs" / "latest", root / "full", root / "gt"
        try:
            evaluation = dashboard.read_evaluation()
            evaluation["reranked"] = dashboard.read_evaluation_dir(root / "reranked")
        finally:
            dashboard.EVAL_DIR, dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.GROUNDTRUTH_DIR = old

    assert len(evaluation["benchmark_reference"]["summary"]) == 15
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
