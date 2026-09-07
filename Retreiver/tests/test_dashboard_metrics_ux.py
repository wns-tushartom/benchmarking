import subprocess
from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "web" / "app.js"


def run_dashboard_probe(assertions: str) -> None:
    js = f"""
const fs = require('fs'), vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  return {{
    id, value: 'all', textContent: '', innerHTML: '', selectedOptions: [], listeners: {{}},
    classList: {{ toggle() {{}}, contains() {{ return false; }} }},
    addEventListener(name, handler) {{ this.listeners[name] = handler; }},
    setAttribute() {{}}, closest() {{ return null; }},
  }};
}}
function el(id) {{ return elements[id] ||= makeElement(id); }}
const context = {{
  console,
  document: {{
    getElementById: el,
    querySelectorAll() {{ return []; }},
    addEventListener() {{}},
    body: {{ insertAdjacentHTML() {{}} }},
  }},
  window: {{}}, location: {{hash:'#quality'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok:true, json:async()=>({{}}), text:async()=>''}}), setTimeout() {{}},
}};
vm.createContext(context);
const code = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code, context);
{assertions}
"""
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_metrics_visuals_use_complete_rows_and_render_not_run_cells():
    run_dashboard_probe(
        """
const stateRows = [
  {key:'c1|e1|Qdrant|none',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:'10',recall_at_5:'0.8',mrr:'0.7',ndcg_at_5:'0.75',avg_latency_seconds:'0.2'},
  {key:'c1|e1|FAISS|none',sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',state:'not_run',artifact:'',source:'',run_at:'',evaluated_queries:''},
];
context.renderMetricExplorer({pipeline_state:stateRows}, {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant','FAISS'],rerankers:['none']});
context.renderMetricHeatmap({pipeline_state:stateRows}, {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant','FAISS'],rerankers:['none']});
if (!el('tradeoffChart').innerHTML.includes('<circle') || !el('metricHeatmap').innerHTML.includes('not run') || !el('metricExplorerHint').textContent.includes('1 complete')) process.exit(1);
"""
    )


def test_metrics_empty_state_has_no_plots_or_winner_text():
    run_dashboard_probe(
        """
const stateRows = [
  {sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'not_run',evaluated_queries:''},
  {sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',state:'incomplete',evaluated_queries:'0'},
];
context.renderMetricExplorer({pipeline_state:stateRows}, {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant','FAISS'],rerankers:['none']});
context.renderMetricHeatmap({pipeline_state:stateRows}, {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant','FAISS'],rerankers:['none']});
const output = el('tradeoffChart').innerHTML + el('metricExplorerKpis').innerHTML + el('metricInterpretation').textContent;
if (!output.includes('Metrics unavailable until ground truth evaluation completes') || output.includes('<circle') || /winner/i.test(output)) process.exit(1);
"""
    )


def test_zero_complete_metric_is_displayed_not_relabelled_not_run():
    run_dashboard_probe(
        """
const stateRows = [
  {sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:'10',recall_at_5:'0',mrr:'0',ndcg_at_5:'0',avg_latency_seconds:'0'},
];
el('heatmapMetric').value = 'recall_at_5';
context.renderMetricHeatmap({pipeline_state:stateRows}, {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant'],rerankers:['none']});
const heatmap = el('metricHeatmap').innerHTML;
if (!heatmap.includes('0.000') || heatmap.includes('not run')) process.exit(1);
"""
    )


def test_metrics_controls_preserve_valid_selection_and_pipeline_state_is_authoritative():
    run_dashboard_probe(
        """
const operational = {
  pipeline_state:[{sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:'10',recall_at_5:'0.2',mrr:'0.1',ndcg_at_5:'0.1',avg_latency_seconds:'0.2'}],
  evaluation:{summary:[{sheet:'raw',embedding:'raw',store:'FAISS',reranker:'none',recall_at_5:'0.99',mrr:'0.99',ndcg_at_5:'0.99',avg_latency_seconds:'0.01'}]},
};
const options = {chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant'],rerankers:['none']};
state = {operational, files:[], options};
benchmarkOptions = options;
context.renderMetricExplorer(operational, options);
el('metricXAxis').value = 'mrr';
el('heatmapMetric').value = 'mrr';
if (typeof el('metricXAxis').listeners.input !== 'function' || typeof el('heatmapMetric').listeners.input !== 'function') process.exit(1);
el('metricXAxis').listeners.input();
el('heatmapMetric').listeners.input();
const output = el('tradeoffChart').innerHTML + el('metricHeatmap').innerHTML + el('metricExplorerKpis').innerHTML;
if (el('metricXAxis').value !== 'mrr' || el('heatmapMetric').value !== 'mrr' || output.includes('raw') || output.includes('0.990')) process.exit(1);
"""
    )


if __name__ == "__main__":
    test_metrics_visuals_use_complete_rows_and_render_not_run_cells()
    test_metrics_empty_state_has_no_plots_or_winner_text()
    test_zero_complete_metric_is_displayed_not_relabelled_not_run()
    test_metrics_controls_preserve_valid_selection_and_pipeline_state_is_authoritative()
