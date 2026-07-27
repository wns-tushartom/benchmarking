from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.js"
INDEX = ROOT / "web" / "index.html"
VISUALS = ROOT / "web" / "recommendation-visuals.js"
RECOMMENDATIONS = ROOT / "web" / "recommendations.js"


ANALYTICS_IDS = {
    "metricExplorerTitle",
    "metricExplorerHint",
    "metricXAxis",
    "metricYAxis",
    "metricColorBy",
    "metricBubbleBy",
    "metricQuickView",
    "metricExplorerKpis",
    "tradeoffChart",
    "metricChartTooltip",
    "metricPointDetails",
    "metricInterpretation",
    "heatmapTitle",
    "metricHeatmapLegend",
    "heatmapMetric",
    "heatmapEmbedding",
    "heatmapReranker",
    "metricHeatmap",
}


def run_dashboard_probe(assertions: str) -> None:
    source_state = ROOT / "web" / "source-state.js"
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
  window: {{}},
  RecommendationVisuals: require({str(VISUALS)!r}),
  PipelineRecommendations: require({str(RECOMMENDATIONS)!r}),
  DashboardSourceState: require({str(source_state)!r}),
  location: {{hash:'#compare'}},
  history: {{replaceState() {{}}}},
  fetch: async () => ({{ok:true, json:async()=>({{}}), text:async()=>''}}),
  setTimeout() {{}}, clearTimeout() {{}}, AbortController, URLSearchParams,
}};
vm.createContext(context);
const code = fs.readFileSync({str(APP)!r}, 'utf8').split("loadOptions().then(refresh)")[0];
vm.runInContext(code + `\n  globalThis.__recommendationState = recommendationState;\n  globalThis.__setDashboardState = (value) => {{ state = value; }};\n  globalThis.__setBenchmarkOptions = (value) => {{ benchmarkOptions = value; }};\n  globalThis.DashboardSourceState = DashboardSourceState;\n`, context);
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
    assert '/recommendation-visuals.js?v=20260724-canonical-dashboard-v5.3' in html
    assert html.index('/recommendation-visuals.js?v=20260724-canonical-dashboard-v5.3') < html.index('/app.js?v=20260724-canonical-dashboard-v5.3')
    assert '/styles.css?v=20260724-canonical-dashboard-v5.3' in html
    assert '/app.js?v=20260724-canonical-dashboard-v5.3' in html


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
if ((chart.match(/class="metric-point/g)||[]).length !== 3) throw new Error(chart);
if ((chart.match(/<title>/g)||[]).length !== 3) throw new Error('missing accessible titles');
if (!chart.includes('Faster ←') || !chart.includes('Better ↑')) throw new Error(chart);
if (!chart.includes('Rank 1') || !chart.includes('metric-rank-top')) throw new Error(chart);
if (chart.includes('svg-winner-label')) throw new Error('persistent winner label obscures nearby points');
if (!heatmap.includes('not run') || !heatmap.includes('0.000') && !heatmap.includes('0.900')) throw new Error(heatmap);
if (!heatmap.includes('is-best') || !heatmap.includes('is-worst')) throw new Error(heatmap);
if (!heatmap.includes('data-heatmap-row="c1"') || !heatmap.includes('data-heatmap-column="Qdrant"')) throw new Error(heatmap);
if (!el('metricHeatmapLegend').innerHTML.includes('metric-state-swatch')) throw new Error(el('metricHeatmapLegend').innerHTML);
if (!el('metricExplorerHint').textContent.includes('3 complete')) throw new Error(el('metricExplorerHint').textContent);
"""
    )


def test_heatmap_extrema_are_calculated_within_visible_filter_slice() -> None:
    payload = labelled_payload("""[
  {combo_id:'visible-low',sheet:'c1',embedding:'e-a',store:'Qdrant',reranker:'Amazon Rerank v1',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.70,mrr:.6,ndcg_at_5:.65,avg_latency_seconds:.3},
  {combo_id:'visible-high',sheet:'c2',embedding:'e-a',store:'FAISS',reranker:'Amazon Rerank v1',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.80,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.2},
  {combo_id:'hidden-global-high',sheet:'c1',embedding:'e-b',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.99,mrr:.9,ndcg_at_5:.95,avg_latency_seconds:.1},
  {combo_id:'hidden-global-low',sheet:'c2',embedding:'e-b',store:'FAISS',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.10,mrr:.1,ndcg_at_5:.10,avg_latency_seconds:.4}
]""", source_type="uploaded_project")
    run_dashboard_probe(
        f"""
context.renderRecommendationAnalytics({payload});
const heatmap = el('metricHeatmap').innerHTML;
if (!heatmap.includes('0.700') || !heatmap.includes('0.800')) throw new Error(heatmap);
if (!heatmap.includes('is-best') || !heatmap.includes('is-worst')) throw new Error(heatmap);
"""
    )


def test_heatmap_extrema_and_legend_are_scoped_to_visible_slice() -> None:
    payload = labelled_payload("""[
  {combo_id:'bge-low',sheet:'c1',embedding:'BGE',store:'FAISS',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.7,mrr:.6,ndcg_at_5:.65,avg_latency_seconds:.2},
  {combo_id:'bge-high',sheet:'c1',embedding:'BGE',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.8,mrr:.7,ndcg_at_5:.75,avg_latency_seconds:.3},
  {combo_id:'openai-global-low',sheet:'c1',embedding:'OpenAI',store:'FAISS',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.1,mrr:.1,ndcg_at_5:.1,avg_latency_seconds:.1},
  {combo_id:'openai-global-high',sheet:'c1',embedding:'OpenAI',store:'Qdrant',reranker:'none',status:'completed',state:'complete',evaluated_queries:500,recall_at_5:.9,mrr:.9,ndcg_at_5:.9,avg_latency_seconds:.4}
]""")
    run_dashboard_probe(
        f"""
state = {{options:{{chunkers:['c1'],embeddings:['BGE','OpenAI'],vector_stores:['FAISS','Qdrant'],rerankers:['none']}}}};
context.renderRecommendationSource({payload});
const heatmap = el('metricHeatmap').innerHTML;
const legend = el('metricHeatmapLegend').innerHTML;
if ((heatmap.match(/is-best/g)||[]).length !== 1) throw new Error(heatmap);
if ((heatmap.match(/is-worst/g)||[]).length !== 1) throw new Error(heatmap);
if (!legend.includes('Low 0.700') || !legend.includes('High 0.800')) throw new Error(legend);
"""
    )


def test_relationship_scatter_exposes_rank_shapes_family_and_quick_views() -> None:
    run_dashboard_probe(
        """
const rows = Array.from({length:12}, (_, index) => ({
  combo_id:`combo-${index}`,
  sheet:'chunk-a',
  embedding:'embed-a',
  store:['FAISS','PGVector','Qdrant','Weaviate'][index % 4],
  reranker:['none','bge-reranker-base','Amazon Rerank v1'][Math.floor(index / 4)],
  status:'completed', state:'complete', evaluated_queries:500,
  recall_at_5:.95 - index * .01, mrr:.9 - index * .01, ndcg_at_5:.92 - index * .01,
  avg_latency_seconds:.1 + index * .02,
}));
const payload = {source_type:'uploaded_project',scoring_mode:'retrieval_labels',mode:'labelled',rows};
context.__recommendationState.analyticsResult = {mode:'labelled',completed:rows,rows,roles:{}};
context.renderRecommendationAnalytics(payload);
let chart = el('tradeoffChart').innerHTML;
if ((chart.match(/class="metric-point/g)||[]).length !== 12) throw new Error(chart);
if (!chart.includes('metric-rank-top') || !chart.includes('metric-rank-bottom')) throw new Error(chart);
if (!chart.includes('metric-shape-circle') || !chart.includes('metric-shape-diamond') || !chart.includes('metric-shape-triangle')) throw new Error(chart);
if (!chart.includes('data-family-key=') || !chart.includes('data-rank="1"') || !chart.includes('data-rank="12"')) throw new Error(chart);
if (!el('metricPointDetails').textContent.includes('Select a point')) throw new Error(el('metricPointDetails').textContent);
el('metricQuickView').value = 'top10';
context.renderRecommendationAnalytics(payload);
chart = el('tradeoffChart').innerHTML;
if ((chart.match(/class="metric-point/g)||[]).length !== 10) throw new Error(chart);
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


def test_semantic_job_status_prefers_run_payload_over_exit_code() -> None:
    run_dashboard_probe(
        """
const failed = context.semanticRunStatus({exit_code:0, output:'{"state":"failed","succeeded":0,"failed":1,"run_id":"run_x"}'});
if (failed.label !== 'Failed' || failed.ok !== false) throw new Error(JSON.stringify(failed));
const passed = context.semanticRunStatus({exit_code:0, output:'{"state":"completed","succeeded":1,"failed":0}'});
if (passed.label !== 'Passed' || passed.ok !== true) throw new Error(JSON.stringify(passed));
const partial = context.semanticRunStatus({exit_code:0, output:'{"state":"partial","succeeded":1,"failed":1}'});
if (partial.label !== 'Partial' || partial.ok !== false) throw new Error(JSON.stringify(partial));
const running = context.semanticRunStatus({exit_code:null, running:true, output:''});
if (running.label !== 'Running') throw new Error(JSON.stringify(running));
"""
    )


def test_selected_params_always_submit_query_limit_zero_without_query_limit_control() -> None:
    html = INDEX.read_text(encoding='utf-8')
    assert 'id="runQueryLimit"' not in html
    assert 'id="runQueryField"' in html
    assert 'id="runSmokeBtn"' in html
    run_dashboard_probe(
        """
el('runDataset').value = 'dataset:wns-default';
el('runGroundtruth').value = 'groundtruth:official';
el('runRetrievalTopK').value = '10';
el('runRerankedOutputK').value = '5';
el('runQueries').value = 'alpha\\nbeta';
el('runSheet').selectedOptions = [{value:'fixed_tok1200_ov150'}];
el('runEmbedding').selectedOptions = [{value:'gte_multilingual_base'}];
el('runStore').selectedOptions = [{value:'FAISS'}];
el('runRerankerMain').selectedOptions = [{value:'bge-reranker-base'}];
context.__setBenchmarkOptions({chunkers:['fixed_tok1200_ov150'],embeddings:['gte_multilingual_base'],vector_stores:['FAISS'],rerankers:['bge-reranker-base'],matrix_count:1});
const params = context.selectedParams();
if (params.get('query_limit') !== '0') throw new Error(String(params));
if (el('runQueryField').hidden) throw new Error('typed query field must stay visible');
"""
    )


def test_project_overview_counts_use_active_project_payload_not_official_matrix() -> None:
    run_dashboard_probe(
        """
context.__setDashboardState({operational:{known_matrix_count:180,retrieval_smokes:[{},{}],reranker_smokes:[],benchmark_detail_evidence:[],options_formula:'official'},files:[],options:{},sourceCatalog:{datasets:[{id:'project:test1_abc',label:'test1',kind:'uploaded_project',document_count:5,ready:true,validation:'ready'}],groundtruth:[{id:'groundtruth:project:test1_abc',label:'test1',valid:true,row_count:11}]}});
el('globalDataset').value = 'project:test1_abc';
el('globalGroundtruth').value = 'groundtruth:project:test1_abc';
context.__recommendationState.sourceType = 'uploaded_project';
context.__recommendationState.active = {
  source_type:'uploaded_project', dataset_id:'project:test1_abc', project_id:'test1_abc', run_id:'run_1', combination_count:1,
  rows:[{status:'failed', evidence_count:0, combo_id:'c0'},{status:'completed', evidence_count:11, combo_id:'c1', chunker_id:'entity_heuristic_w6', embedding_id:'gte_multilingual_base', vector_store_id:'FAISS', reranker_id:'bge-reranker-base'}]
};
context.renderOperational();
if (el('comboStatus').textContent !== '2') throw new Error('combo ' + el('comboStatus').textContent);
if (!String(el('comboNote').textContent).toLowerCase().includes('project')) throw new Error(el('comboNote').textContent);
if (el('retrievalStatus').textContent !== '11') throw new Error('evidence ' + el('retrievalStatus').textContent);
if (!String(el('retrievalTable').innerHTML || '').includes('View 11')) throw new Error('project evidence browser missing open button: ' + el('retrievalTable').innerHTML);
"""
    )


def test_project_result_normalization_aliases_matrix_fields_for_metrics() -> None:
    run_dashboard_probe(
        """
const payload = context.DashboardSourceState.normalizeResultPayload({
  source_type:'uploaded_project',
  scoring_mode:'retrieval_labels',
  metric_k:10,
  rows:[{
    status:'completed',
    combo_id:'combo_1',
    chunker_id:'entity_heuristic_w6',
    embedding_id:'gte_multilingual_base',
    vector_store_id:'FAISS',
    reranker_id:'bge-reranker-base',
    recall_at_k:1,
    mrr_at_k:0.9,
    ndcg_at_k:0.8,
    avg_query_latency_s:0.12,
    labelled_queries:11,
    evidence_count:110
  }]
});
const row = payload.rows[0];
if (row.sheet !== 'entity_heuristic_w6') throw new Error('sheet ' + row.sheet);
if (row.embedding !== 'gte_multilingual_base') throw new Error('embedding');
if (row.store !== 'FAISS') throw new Error('store');
if (row.reranker !== 'bge-reranker-base') throw new Error('reranker');
if (row.recall_at_5 !== 1) throw new Error('recall alias');
if (row.mrr !== 0.9) throw new Error('mrr alias');
if (row.ndcg_at_5 !== 0.8) throw new Error('ndcg alias');
if (row.avg_latency_seconds !== 0.12) throw new Error('latency alias');
if (row.evaluated_queries !== 11) throw new Error('evaluated queries');
"""
    )
