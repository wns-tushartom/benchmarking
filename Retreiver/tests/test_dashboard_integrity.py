import csv
import os
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


def _complete_row(**overrides: str) -> dict:
    row = {
        "chunker": "chunker-a",
        "embedding": "embedding-a",
        "vector_store": "FAISS",
        "reranker": "bge-reranker-base",
        "query_count": "12",
        "recall_at_5": "0.75",
        "mrr": "0.70",
        "ndcg_at_5": "0.65",
        "avg_latency_ms": "120",
        "created_at": "2026-07-21T10:00:00",
    }
    row.update(overrides)
    return row


def _with_artifact_sources(dashboard, root: Path, official_keys: set[tuple[str, str, str, str]]):
    old = dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.official_matrix_keys
    dashboard.MODULAR_DIR = root / "runs" / "latest"
    dashboard.FULL_DIR = root / "full"
    dashboard.official_matrix_keys = lambda: official_keys
    return old


def _restore_artifact_sources(dashboard, old) -> None:
    dashboard.MODULAR_DIR, dashboard.FULL_DIR, dashboard.official_matrix_keys = old


def test_prior_complete_result_survives_newer_incomplete_retry():
    import scripts.serve_benchmark_dashboard as dashboard

    key = ("chunker-a", "embedding-a", "FAISS", "bge-reranker-base")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        complete = root / "runs" / "complete_20260721" / "modular_summary.csv"
        retry = root / "runs" / "retry_20260722" / "modular_summary.csv"
        _write_csv(complete, [_complete_row(created_at="2026-07-21T10:00:00")])
        _write_csv(retry, [_complete_row(created_at="2026-07-22T10:00:00", ndcg_at_5="")])
        old = _with_artifact_sources(dashboard, root, {key})
        try:
            rows = dashboard.pipeline_state_rows()
            complete_rows = dashboard.complete_pipeline_metric_rows(rows)
        finally:
            _restore_artifact_sources(dashboard, old)

    assert rows == [
        {
            "key": "chunker-a|embedding-a|FAISS|bge-reranker-base",
            "sheet": "chunker-a",
            "embedding": "embedding-a",
            "store": "FAISS",
            "reranker": "bge-reranker-base",
            "state": "complete",
            "artifact": str(complete),
            "source": "modular_run:complete_20260721",
            "run_at": "2026-07-21T10:00:00",
            "evaluated_queries": 12,
            "candidate_count": 2,
            "recall_at_5": 0.75,
            "mrr": 0.7,
            "ndcg_at_5": 0.65,
            "avg_latency_seconds": 0.12,
        }
    ]
    assert complete_rows == rows


def test_missing_ndcg_is_incomplete_and_not_a_metric_row():
    import scripts.serve_benchmark_dashboard as dashboard

    key = ("chunker-a", "embedding-a", "FAISS", "bge-reranker-base")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_csv(root / "runs" / "latest" / "modular_summary.csv", [_complete_row(ndcg_at_5="")])
        old = _with_artifact_sources(dashboard, root, {key})
        try:
            rows = dashboard.pipeline_state_rows()
            complete_rows = dashboard.complete_pipeline_metric_rows(rows)
        finally:
            _restore_artifact_sources(dashboard, old)

    assert rows[0]["state"] == "incomplete"
    assert rows[0]["candidate_count"] == 1
    assert "ndcg_at_5" not in rows[0]
    assert complete_rows == []


def test_latest_timestamp_less_complete_artifact_uses_mtime_ns():
    import scripts.serve_benchmark_dashboard as dashboard

    key = ("chunker-a", "embedding-a", "FAISS", "bge-reranker-base")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        earlier = root / "runs" / "z_earlier" / "modular_summary.csv"
        newer = root / "runs" / "a_newer" / "modular_summary.csv"
        _write_csv(earlier, [_complete_row(created_at="")])
        _write_csv(newer, [_complete_row(created_at="")])
        second_ns = 1_700_000_000_000_000_000
        os.utime(earlier, ns=(second_ns + 100, second_ns + 100))
        os.utime(newer, ns=(second_ns + 900, second_ns + 900))
        old = _with_artifact_sources(dashboard, root, {key})
        try:
            rows = dashboard.pipeline_state_rows()
        finally:
            _restore_artifact_sources(dashboard, old)

    assert rows[0]["state"] == "complete"
    assert rows[0]["artifact"] == str(newer)
    assert rows[0]["source"] == "modular_run:a_newer"


def test_frontend_pipeline_state_excludes_raw_incomplete_summary_from_rankings():
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    node_script = f"""
const fs = require('fs'), vm = require('vm');
const elements = {{}};
function el(id) {{ return elements[id] ||= {{id, value:'all', textContent:'', innerHTML:'', classList:{{toggle(){{}}, add(){{}}, remove(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}}; }}
const context = {{
  console,
  document: {{getElementById: el, querySelectorAll() {{ return []; }}, addEventListener() {{}}, body: {{insertAdjacentHTML() {{}}}}}},
  window: {{}}, location: {{hash: '#overview'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok: true, json: async () => ({{}}), text: async () => ''}}), setTimeout() {{}}
}};
vm.createContext(context);
const code = fs.readFileSync({str(app)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
  const data = {{
    operational: {{
      pipeline_state: [{{state:'complete', sheet:'complete', embedding:'complete', store:'FAISS', reranker:'none', recall_at_5:'0.7', mrr:'0.6', ndcg_at_5:'0.5', avg_latency_seconds:'0.2', evaluated_queries:'10'}}],
      evaluation: {{
        summary: [{{sheet:'raw', embedding:'raw', store:'Qdrant', reranker:'none', recall_at_5:'0.99', mrr:'0.99', avg_latency_seconds:'0.01'}}]
      }}
    }}
  }};
  const operational = data.operational;
  const evaluation = operational.evaluation;
  globalThis.__rows = evaluatedRows(evaluation, operational.pipeline_state);
  renderEvaluation(evaluation, operational.pipeline_state);
  renderPipelineComparison(evaluation, operational.pipeline_state);
  globalThis.__quality = document.getElementById('qualityTable').innerHTML;
  globalThis.__comparison = document.getElementById('comparisonGrid').innerHTML;
`, context);
if (context.__rows.length !== 1 || context.__rows[0].sheet !== 'complete' || context.__quality.includes('raw') || context.__comparison.includes('raw')) {{
  console.error(JSON.stringify({{rows:context.__rows, quality:context.__quality, comparison:context.__comparison}})); process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", node_script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_frontend_defaults_to_official_wns_and_scopes_project_params_explicitly():
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    node_script = f"""
const fs = require('fs'), vm = require('vm');
const elements = {{}};
function el(id) {{ return elements[id] ||= {{id, value:'', checked:false, textContent:'', innerHTML:'', options:[], selectedOptions:[{{value:'all'}}], classList:{{toggle(){{}}, add(){{}}, remove(){{}}}}, addEventListener(){{}}, querySelectorAll(){{return [];}}, setAttribute(){{}}}}; }}
const context = {{
  console,
  URLSearchParams,
  document: {{getElementById: el, querySelectorAll() {{ return []; }}, addEventListener() {{}}, body: {{insertAdjacentHTML() {{}}}}}},
  window: {{}}, location: {{hash: '#overview'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok: true, json: async () => ({{}}), text: async () => ''}}), setTimeout() {{}}
}};
vm.createContext(context);
const code = fs.readFileSync({str(app)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
  state = {{operational: {{user_projects: [{{project_id:'uploaded-1', label:'Uploaded project', chunk_count:4}}]}}, files:[], options:{{}}}};
  document.getElementById('projectSelect').value = '';
  renderUserProjects(state.operational.user_projects);
  const defaultSelectValue = document.getElementById('projectSelect').value;
  const defaultParams = selectedParams().toString();
  document.getElementById('projectSelect').value = 'uploaded-1';
  const genericParams = selectedParams().toString();
  const projectParams = selectedParams({{projectScoped:true}}).toString();
  globalThis.__result = {{
    defaultSelectValue,
    options: document.getElementById('projectSelect').innerHTML,
    defaultParams, genericParams, projectParams,
  }};
`, context);
const result = context.__result;
if (result.defaultSelectValue !== '' || !result.options.includes('value=""') || !result.options.includes('Official WNS dataset') || result.defaultParams.includes('project_id=') || result.genericParams.includes('project_id=') || !result.projectParams.includes('project_id=uploaded-1')) {{
  console.error(JSON.stringify(result)); process.exit(1);
}}
"""
    proc = subprocess.run(["node", "-e", node_script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_configured_combination_without_source_is_not_run():
    import scripts.serve_benchmark_dashboard as dashboard

    key = ("chunker-z", "embedding-z", "Qdrant", "none")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old = _with_artifact_sources(dashboard, root, {key})
        try:
            rows = dashboard.pipeline_state_rows()
        finally:
            _restore_artifact_sources(dashboard, old)

    assert rows == [
        {
            "key": "chunker-z|embedding-z|Qdrant|none",
            "sheet": "chunker-z",
            "embedding": "embedding-z",
            "store": "Qdrant",
            "reranker": "none",
            "state": "not_run",
            "artifact": "",
            "source": "",
            "run_at": "",
            "evaluated_queries": "",
            "candidate_count": 0,
        }
    ]


if __name__ == "__main__":
    test_prior_complete_result_survives_newer_incomplete_retry()
    test_missing_ndcg_is_incomplete_and_not_a_metric_row()
    test_latest_timestamp_less_complete_artifact_uses_mtime_ns()
    test_frontend_pipeline_state_excludes_raw_incomplete_summary_from_rankings()
    test_configured_combination_without_source_is_not_run()
    print("dashboard integrity tests passed")
