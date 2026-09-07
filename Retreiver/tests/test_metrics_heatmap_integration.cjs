const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const index = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const appPath = path.join(root, 'web/app.js');
const app = fs.readFileSync(appPath, 'utf8');
const sourceState = fs.readFileSync(path.join(root, 'web/source-state.js'), 'utf8');
const recommendations = fs.readFileSync(path.join(root, 'web/recommendations.js'), 'utf8');
const visuals = fs.readFileSync(path.join(root, 'web/recommendation-visuals.js'), 'utf8');

for (const id of [
  'metricXAxis', 'metricYAxis', 'metricColorBy', 'metricBubbleBy', 'metricQuickView',
  'metricExplorerKpis', 'tradeoffChart', 'metricPointDetails', 'metricHeatmap',
  'heatmapMetric', 'heatmapEmbedding', 'heatmapReranker', 'heatmapRetrievalMethod',
  'heatmapIndexType',
]) {
  assert.match(index, new RegExp(`id=["']${id}["']`), `missing updated Metrics control #${id}`);
}
assert.ok(index.indexOf('/recommendation-visuals.js') < index.indexOf('/app.js'), 'visual helper must load before app.js');
assert.match(index, /20260907-handoff-v1/, 'updated frontend assets must share a new cache key');
assert.match(index, /id="candidateOperations"/, 'candidate operations markup must be preserved');
assert.match(app, /async function loadCandidateOperations/, 'candidate operations logic must be preserved');
assert.match(app, /function renderRecommendationExplorer/);
assert.match(app, /function renderRecommendationHeatmap/);
assert.match(app, /function recommendationHeatmapIdentity/);

function element(id) {
  return elements[id] ||= {
    id,
    value: '',
    textContent: '',
    innerHTML: '',
    hidden: false,
    options: [],
    selectedOptions: [],
    dataset: {},
    style: {},
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener() {},
    querySelectorAll() { return []; },
    querySelector() { return null; },
    setAttribute() {},
    getBoundingClientRect() { return {left: 0, top: 0}; },
  };
}
const elements = {};
const context = {
  console,
  AbortController,
  URLSearchParams,
  setTimeout,
  clearTimeout,
  document: {
    getElementById: element,
    querySelectorAll() { return []; },
    addEventListener() {},
    body: {insertAdjacentHTML() {}},
  },
  window: {},
  location: {hash: '#compare'},
  history: {replaceState() {}},
  fetch: async () => ({ok: true, json: async () => ({}), text: async () => '{}'}),
};
vm.createContext(context);
vm.runInContext(sourceState, context);
vm.runInContext(recommendations, context);
vm.runInContext(visuals, context);
const appBeforeStartup = app.split('loadOptions().then(refresh)')[0];
vm.runInContext(`${appBeforeStartup}\n` +
  `globalThis.__renderHeatmap = renderRecommendationHeatmap;\n` +
  `globalThis.__completeRows = completeRecommendationRows;\n` +
  `globalThis.__heatmapIdentity = recommendationHeatmapIdentity;`, context);

const base = {
  sheet: 'Heading', embedding: 'GTE', store: 'FAISS', reranker: 'none',
  status: 'completed', evaluated_queries: 500, recall_at_5: 0.8, mrr: 0.7,
  ndcg_at_5: 0.75, avg_latency_seconds: 0.2,
};
const denseHnsw = {...base, retrieval_method: 'Dense Cosine', index_type: 'HNSW'};
const hybridHnsw = {...base, retrieval_method: 'Hybrid RRF', index_type: 'HNSW', recall_at_5: 0.95};
const denseIvf = {...base, retrieval_method: 'Dense Cosine', index_type: 'IVF', recall_at_5: 0.1};
const validZero = {...base, store: 'Qdrant', retrieval_method: 'Dense Cosine', index_type: 'HNSW', recall_at_5: 0};
const historical100 = {...base, sheet: 'Fixed', retrieval_method: 'Dense Cosine', index_type: 'HNSW', evaluated_queries: 100, recall_at_5: 0.99};
const missingMetric = {...base, sheet: 'Fixed', store: 'Qdrant', retrieval_method: 'Dense Cosine', index_type: 'HNSW'};
delete missingMetric.recall_at_5;
const notRun = {
  sheet: 'Semantic', embedding: 'GTE', store: 'FAISS', reranker: 'none',
  retrieval_method: 'Dense Cosine', index_type: 'HNSW', status: 'not_run',
};
const rows = [denseHnsw, hybridHnsw, denseIvf, validZero, historical100, missingMetric, notRun];

assert.notEqual(context.__heatmapIdentity(denseHnsw), context.__heatmapIdentity(hybridHnsw), 'retrieval method must be part of heatmap identity');
assert.notEqual(context.__heatmapIdentity(denseHnsw), context.__heatmapIdentity(denseIvf), 'index type must be part of heatmap identity');
const admitted = context.__completeRows({source_type: 'combined', scoring_mode: 'retrieval_labels', rows});
assert.equal(admitted.length, 4, 'only complete 500-query rows with all required visual metrics are admitted');
assert.ok(!admitted.some(row => row.evaluated_queries === 100), 'historical 100-query rows must not be measured/ranked');

for (const [id, value] of [
  ['heatmapMetric', 'recall_at_5'],
  ['heatmapEmbedding', 'GTE'],
  ['heatmapReranker', 'none'],
  ['heatmapRetrievalMethod', 'dense_cosine'],
  ['heatmapIndexType', 'HNSW'],
]) element(id).value = value;
context.__renderHeatmap({source_type: 'combined', scoring_mode: 'retrieval_labels', rows}, rows);
const html = element('metricHeatmap').innerHTML;
assert.match(html, />0\.800</, 'selected dense/HNSW cell must retain its own value');
assert.doesNotMatch(html, />0\.950</, 'hybrid value must be excluded by the method slice');
assert.doesNotMatch(html, />0\.100</, 'IVF value must be excluded by the index slice');
assert.match(html, />0\.000</, 'a genuinely measured zero must remain visible');
assert.doesNotMatch(html, />0\.990</, '100-query historical result must not be colored as measured');
assert.match(html, />incomplete</, 'complete-labelled rows with ineligible/missing metrics must render incomplete, not zero');
assert.match(html, />not run</, 'configured unrun cells must preserve their lifecycle state');

const recommendationRules = require(path.join(root, 'web/recommendations.js'));
const project500 = {
  combo_id: 'project-500', status: 'completed', evaluated_queries: 500, labelled_queries: 500,
  ndcg_at_k: 0.7, mrr_at_k: 0.7, recall_at_k: 0.8, avg_query_latency_s: 0.2,
};
const project100 = {...project500, combo_id: 'project-100', evaluated_queries: 100, labelled_queries: 100, ndcg_at_k: 0.99};
const ranked = recommendationRules.recommendationsForSource({
  source_type: 'uploaded_project', scoring_mode: 'retrieval_labels', rows: [project100, project500],
});
assert.deepEqual(ranked.completed.map(row => row.combo_id), ['project-500'], '100-query project runs must not enter recommendation ranking');

console.log('PASS: Metrics explorer/heatmap wiring, 500-query admission, missing-state truth, and full configuration identity');
