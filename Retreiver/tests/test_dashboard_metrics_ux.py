from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.js"
INDEX = ROOT / "web" / "index.html"


ANALYTICS_IDS = {
    "metricExplorerTitle",
    "metricExplorerHint",
    "metricXAxis",
    "metricYAxis",
    "metricColorBy",
    "metricBubbleBy",
    "metricExplorerKpis",
    "tradeoffChart",
    "metricInterpretation",
    "heatmapTitle",
    "metricHeatmapLegend",
    "heatmapMetric",
    "heatmapEmbedding",
    "heatmapReranker",
    "metricHeatmap",
}


def run_dashboard_probe(assertions: str) -> None:
    js = f"""
const fs = require('fs'), vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  return {{
    id, value: 'all', textContent: '', innerHTML: '', selectedOptions: [], options: [], listeners: {{}},
    hidden: false, disabled: false,
    classList: {{ toggle() {{}}, contains() {{ return false; }}, add() {{}}, remove() {{}} }},
    addEventListener(name, handler) {{ this.listeners[name] = handler; }},
    setAttribute() {{}}, closest() {{ return null; }}, replaceChildren() {{}}, scrollIntoView() {{}},
  }};
}}
function el(id) {{ return elements[id] ||= makeElement(id); }}
const context = {{
  console,
  document: {{
    getElementById: el,
    querySelectorAll() {{ return []; }},
    addEventListener() {{}},
    createElement(tag) {{ return makeElement(tag); }},
    body: {{ insertAdjacentHTML() {{}} }},
  }},
  window: {{}}, location: {{hash:'#compare'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok:true, json:async()=>({{}}), text:async()=>''}}),
  setTimeout() {{}}, clearTimeout() {{}}, AbortController,
}};
vm.createContext(context);
const code = fs.readFileSync({str(APP)!r}, 'utf8').split("loadOptions().then(refresh)")[0];
vm.runInContext(code, context);
{assertions}
"""
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def labelled_payload(rows: str, *, source_type: str = "official") -> str:
    return f"""{{
  source_type:{source_type!r}, scoring_mode:'retrieval_labels', mode:'labelled',
  rows:{rows}, recommendations:{{roles:{{quality:{{combo_id:'quality'}},speed:{{combo_id:'speed'}},value:{{combo_id:'value'}}}}}}
}}"""


def test_recommendation_analytics_markup_is_below_top_table_and_not_collapsed() -> None:
    html = INDEX.read_text(encoding="utf-8")
    for element_id in ANALYTICS_IDS:
        assert f'id="{element_id}"' in html
    assert html.index('id="recommendationTable"') < html.index('id="tradeoffChart"')
    assert html.index('id="tradeoffChart"') < html.index('id="pricingLedger"')
    assert 'id="tradeoffDetails"' not in html
    assert "Only complete evaluations" in html
    assert "Missing never means zero" in html
    assert '/styles.css?v=20260723-canonical-dashboard-v4' in html
    assert '/app.js?v=20260723-canonical-dashboard-v4' in html


def test_official_analytics_plot_complete_rows_and_show_all_heatmap_states() -> None:
    payload = labelled_payload("""[
  {combo_id:'quality',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.9,mrr:.8,ndcg_at_5:.85,avg_latency_seconds:.3},
  {combo_id:'speed',sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.7,mrr:.6,ndcg_at_5:.65,avg_latency_seconds:.1},
  {combo_id:'value',sheet:'c2',embedding:'e1',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.8,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.2}
]""")
    run_dashboard_probe(
        f"""
state = {{operational:{{pipeline_state:[
  {{key:'c1|e1|Qdrant|none',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:.9,mrr:.8,ndcg_at_5:.85,avg_latency_seconds:.3}},
  {{key:'c1|e1|FAISS|none',sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:.7,mrr:.6,ndcg_at_5:.65,avg_latency_seconds:.1}},
  {{key:'c2|e1|Qdrant|none',sheet:'c2',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:.8,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.2}},
  {{key:'c2|e1|FAISS|none',sheet:'c2',embedding:'e1',store:'FAISS',reranker:'none',state:'not_run',evaluated_queries:''}}
]}}, options:{{chunkers:['c1','c2'],embeddings:['e1'],vector_stores:['Qdrant','FAISS'],rerankers:['none']}}}};
const payload = {payload};
context.renderRecommendationAnalytics(payload);
const chart = el('tradeoffChart').innerHTML;
const heatmap = el('metricHeatmap').innerHTML;
if ((chart.match(/<circle/g)||[]).length !== 3) throw new Error(chart);
if ((chart.match(/<title>/g)||[]).length !== 3) throw new Error('missing accessible titles');
if (!chart.includes('Faster ←') || !chart.includes('Better ↑')) throw new Error(chart);
if (!chart.includes('Quality') || !chart.includes('Speed') || !chart.includes('Value')) throw new Error(chart);
if (!heatmap.includes('not run') || !heatmap.includes('0.000') && !heatmap.includes('0.900')) throw new Error(heatmap);
if (!el('metricExplorerHint').textContent.includes('3 complete')) throw new Error(el('metricExplorerHint').textContent);
"""
    )


def test_project_analytics_never_read_unrelated_official_pipeline_state() -> None:
    payload = labelled_payload("""[
  {combo_id:'quality',sheet:'project_chunker',embedding:'project_embedding',store:'ProjectDB',reranker:'none',status:'completed',state:'complete',evaluated_queries:12,recall_at_5:.42,mrr:.31,ndcg_at_5:.37,avg_latency_seconds:.22}
]""", source_type="uploaded_project")
    run_dashboard_probe(
        f"""
state = {{operational:{{pipeline_state:[{{combo_id:'official-leak',sheet:'official',embedding:'official',store:'OfficialDB',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:.99,mrr:.99,ndcg_at_5:.99,avg_latency_seconds:.01}}]}},options:{{}}}};
context.renderRecommendationAnalytics({payload});
const output = el('tradeoffChart').innerHTML + el('metricHeatmap').innerHTML + el('metricExplorerKpis').innerHTML;
if (!output.includes('project_chunker') || output.includes('official-leak') || output.includes('OfficialDB') || output.includes('0.990')) throw new Error(output);
if (!el('metricExplorerHint').textContent.includes('1 complete')) throw new Error(el('metricExplorerHint').textContent);
"""
    )


def test_evidence_only_has_no_quality_plot_or_winner_claim() -> None:
    run_dashboard_probe(
        """
const payload = {source_type:'uploaded_project',scoring_mode:'evidence_only',mode:'evidence_only',rows:[
  {combo_id:'evidence',sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',status:'completed',state:'complete',query_count:3,recall_at_5:null,avg_latency_seconds:.1}
]};
context.renderRecommendationAnalytics(payload);
const output = el('tradeoffChart').innerHTML + el('metricExplorerKpis').innerHTML + el('metricInterpretation').textContent;
if (!output.includes('Metrics unavailable until ground truth evaluation completes') || output.includes('<circle') || /winner/i.test(output)) throw new Error(output);
"""
    )


def test_zero_complete_metric_is_measured_not_relabelled_not_run() -> None:
    payload = labelled_payload("""[
  {combo_id:'quality',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:0,mrr:0,ndcg_at_5:0,avg_latency_seconds:0}
]""")
    run_dashboard_probe(
        f"""
state = {{operational:{{pipeline_state:[{{key:'c1|e1|Qdrant|none',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:0,mrr:0,ndcg_at_5:0,avg_latency_seconds:0}}]}},options:{{chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant'],rerankers:['none']}}}};
el('heatmapMetric').value = 'recall_at_5';
context.renderRecommendationAnalytics({payload});
const heatmap = el('metricHeatmap').innerHTML;
if (!heatmap.includes('0.000') || heatmap.includes('not run')) throw new Error(heatmap);
"""
    )


def test_clear_recommendation_view_removes_stale_analytics() -> None:
    payload = labelled_payload("""[
  {combo_id:'quality',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.8,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.2}
]""")
    run_dashboard_probe(
        f"""
state = {{operational:{{pipeline_state:[{{sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:500,recall_at_5:.8,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.2}}]}},options:{{chunkers:['c1'],embeddings:['e1'],vector_stores:['Qdrant'],rerankers:['none']}}}};
context.renderRecommendationAnalytics({payload});
if (!el('tradeoffChart').innerHTML.includes('<circle')) throw new Error('setup did not render');
context.clearRecommendationView('Loading selected result source…');
const output = el('tradeoffChart').innerHTML + el('metricHeatmap').innerHTML;
if (output.includes('<circle') || output.includes('metric-heatmap-grid') || !output.includes('No trade-off data for the active source')) throw new Error(output);
"""
    )


def test_baseline_heatmap_uses_baseline_dimensions_not_official_rerankers() -> None:
    payload = labelled_payload("""[
  {combo_id:'quality',sheet:'baseline_chunker',embedding:'baseline_embedding',store:'FAISS',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.5,mrr:.4,ndcg_at_5:.45,avg_latency_seconds:.1}
]""")[:-1] + ", result_set:'baseline'}"
    run_dashboard_probe(
        f"""
state = {{operational:{{pipeline_state:[{{sheet:'official',embedding:'official',store:'OfficialDB',reranker:'Qwen3:4B Rerank',state:'complete',evaluated_queries:500,recall_at_5:.99}}]}},options:{{chunkers:['official'],embeddings:['official'],vector_stores:['OfficialDB'],rerankers:['Qwen3:4B Rerank']}}}};
context.renderRecommendationAnalytics({payload});
const heatmap = el('metricHeatmap').innerHTML;
if (!heatmap.includes('baseline_chunker') || !heatmap.includes('FAISS') || !heatmap.includes('0.500') || heatmap.includes('OfficialDB')) throw new Error(heatmap);
"""
    )
