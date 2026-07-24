# Recommendations Chart Visual Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense four-colour Recommendations dot cloud with a relationship-first scatter plot and ordered heatmap that expose component families, strongest rows, weakest rows, and exact metrics.

**Architecture:** Add one dependency-free UMD helper module for deterministic visual encodings, family keys, rank subsets, collision offsets, and Pareto selection. Keep source ownership and canonical recommendation ordering in the existing modules. `app.js` will render and manage interaction state; CSS will provide accessible colour, shape, rank, tooltip, and heatmap treatments.

**Tech Stack:** Vanilla JavaScript, inline SVG, CSS, Python pytest wrappers around Node/VM probes, Playwright browser QA.

---

## File map

- Create `web/recommendation-visuals.js`: pure reusable visualization rules with CommonJS and browser exports.
- Create `tests/test_recommendation_visuals.py`: unit tests for deterministic colours, reranker shapes, family grouping, ranking subsets, collision spread, and Pareto orientation.
- Modify `web/index.html`: load the helper, add the quick-view control and accessible chart detail region, and bump cache keys.
- Modify `web/app.js`: render ranked/filtered shaped points, family overlays, tooltips/details, legend filtering, and heatmap extrema/state metadata.
- Modify `web/styles.css`: six-colour palette, point/rank/family states, chart tooltip/detail styles, heatmap scale, responsive rules, and focus states.
- Modify `tests/test_dashboard_metrics_ux.py`: guard markup, visual encodings, family interaction output, quick views, and heatmap semantics.
- Modify `tests/test_recommendations_ui.py`: guard helper loading order, cache contract, and responsive selectors.
- Modify `tests/browser_dashboard_qa.spec.js`: capture and assert Recommendations interactions on desktop/mobile.
- Modify `configs/dashboard_release_contract.json` only if its current asset allowlist requires the new helper/cache key.

### Task 1: Pure visualization contract

**Files:**
- Create: `web/recommendation-visuals.js`
- Create: `tests/test_recommendation_visuals.py`

- [ ] **Step 1: Write failing Node-backed unit tests**

Create tests that require the helper and assert the exact public API:

```python
VISUALS = ROOT / "web" / "recommendation-visuals.js"


def run_visual_probe(assertions: str) -> None:
    script = f"""
const V = require({str(VISUALS)!r});
{assertions}
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stable_colours_and_shapes() -> None:
    run_visual_probe("""
const values = ['FAISS','PGVector','Qdrant','Weaviate','OpenSearch','CustomDB'];
const first = V.categoryColorMap(values);
const second = V.categoryColorMap([...values].reverse());
if (new Set(Object.values(first)).size < 6) throw new Error(JSON.stringify(first));
if (JSON.stringify(first) !== JSON.stringify(second)) throw new Error('order-dependent colours');
if (V.rerankerShape('none') !== 'circle') throw new Error('none');
if (V.rerankerShape('bge-reranker-base') !== 'diamond') throw new Error('bge');
if (V.rerankerShape('Amazon Rerank v1') !== 'triangle') throw new Error('amazon');
""")


def test_rank_views_family_keys_and_pareto() -> None:
    run_visual_probe("""
const rows = Array.from({length:12}, (_,i)=>({combo_id:String(i),sheet:'c'+i,embedding:'e',store:'s',reranker:'none',x:i,y:12-i}));
if (V.familyKey(rows[0]) !== 'c0|e|s') throw new Error(V.familyKey(rows[0]));
if (V.applyQuickView(rows,'top10').length !== 10) throw new Error('top10');
if (V.applyQuickView(rows,'bottom10')[0].combo_id !== '2') throw new Error('bottom rank order');
const frontier = V.paretoRows([{id:'a',x:1,y:3},{id:'b',x:2,y:2},{id:'c',x:3,y:1},{id:'d',x:3,y:0}], r=>r.x, r=>r.y, 'lower', 'higher');
if (frontier.some(r=>r.id==='d')) throw new Error(JSON.stringify(frontier));
""")
```

- [ ] **Step 2: Run the unit tests and confirm RED**

Run:

```bash
python3 -m pytest -q tests/test_recommendation_visuals.py
```

Expected: failure because `web/recommendation-visuals.js` does not exist.

- [ ] **Step 3: Implement the pure helper module**

Expose this stable interface:

```javascript
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.RecommendationVisuals = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const palette = ['#63d7ff', '#69f0c0', '#b695ff', '#ffbf69', '#ff7f9f', '#7aa8ff'];
  function categoryColorMap(values) {
    const categories = [...new Set((values || []).map(value => String(value || 'none')))]
      .sort((a, b) => a.localeCompare(b));
    return Object.fromEntries(categories.map((value, index) => [value, palette[index % palette.length]]));
  }
  function rerankerShape(value) {
    const key = String(value || 'none').toLowerCase();
    if (!key || key === 'none' || key.includes('no rerank')) return 'circle';
    if (key.includes('bge')) return 'diamond';
    if (key.includes('amazon') || key.includes('aws')) return 'triangle';
    return 'square';
  }
  function familyKey(row = {}) {
    return [row.chunker_id || row.sheet || row.chunker, row.embedding_id || row.embedding, row.vector_store_id || row.store || row.vector_store].map(v => String(v || 'none')).join('|');
  }
  function applyQuickView(rows, view) {
    if (view === 'top10') return rows.slice(0, 10);
    if (view === 'bottom10') return rows.slice(Math.max(0, rows.length - 10));
    return rows.slice();
  }
  function paretoRows(rows, xValue, yValue, xDirection = 'lower', yDirection = 'higher') {
    const better = (a, b, direction) => direction === 'lower' ? a <= b : a >= b;
    const strict = (a, b, direction) => direction === 'lower' ? a < b : a > b;
    return rows.filter(candidate => !rows.some(other => other !== candidate
      && better(xValue(other), xValue(candidate), xDirection)
      && better(yValue(other), yValue(candidate), yDirection)
      && (strict(xValue(other), xValue(candidate), xDirection) || strict(yValue(other), yValue(candidate), yDirection))));
  }
  function collisionOffsets(rows, xValue, yValue) {
    const groups = new Map();
    rows.forEach(row => {
      const key = `${Number(xValue(row)).toPrecision(8)}|${Number(yValue(row)).toPrecision(8)}`;
      groups.set(key, [...(groups.get(key) || []), row]);
    });
    const offsets = new Map();
    groups.forEach(group => group.forEach((row, index) => {
      const angle = group.length === 1 ? 0 : (Math.PI * 2 * index) / group.length;
      const radius = group.length === 1 ? 0 : Math.min(4, 1.5 + group.length * 0.35);
      offsets.set(row, {dx: Math.cos(angle) * radius, dy: Math.sin(angle) * radius});
    }));
    return offsets;
  }
  return Object.freeze({palette, categoryColorMap, rerankerShape, familyKey, applyQuickView, paretoRows, collisionOffsets});
});
```

- [ ] **Step 4: Run tests and confirm GREEN**

Run:

```bash
python3 -m pytest -q tests/test_recommendation_visuals.py
node --check web/recommendation-visuals.js
```

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add web/recommendation-visuals.js tests/test_recommendation_visuals.py
git commit -m "feat: add recommendation visualization rules"
```

### Task 2: Relationship-first scatter and exact-data interaction

**Files:**
- Modify: `web/index.html:171-184,460-463`
- Modify: `web/app.js:1240-1413`
- Modify: `tests/test_dashboard_metrics_ux.py`
- Modify: `tests/test_recommendations_ui.py`

- [ ] **Step 1: Extend failing markup and rendering tests**

Add `metricQuickView`, `metricPointDetails`, and `metricChartTooltip` to `ANALYTICS_IDS`. Assert helper load order before `app.js`, six stable colour values, circle/diamond/triangle marker output, top/bottom classes, family data attributes, role labels, and exact metric detail text.

```python
assert 'id="metricQuickView"' in html
assert 'id="metricPointDetails"' in html
assert index.index('/recommendation-visuals.js?v=20260724-option1') < index.index('/app.js?v=20260724-canonical-dashboard-v5')
```

In the VM probe, load the helper before `app.js`:

```javascript
context.RecommendationVisuals = require('/absolute/path/to/web/recommendation-visuals.js');
```

Use twelve complete rows so the probe can assert:

```javascript
if (!chart.includes('metric-rank-top') || !chart.includes('metric-rank-bottom')) throw new Error(chart);
if (!chart.includes('data-family-key=')) throw new Error('family keys absent');
if (!chart.includes('metric-shape-diamond') || !chart.includes('metric-shape-triangle')) throw new Error(chart);
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
python3 -m pytest -q tests/test_recommendation_visuals.py tests/test_dashboard_metrics_ux.py tests/test_recommendations_ui.py
```

Expected: failures for missing helper wiring, controls, shapes, rank classes, and detail region.

- [ ] **Step 3: Wire markup and visualization state**

Load the helper before `app.js`, add the quick-view control, and add accessible live regions:

```html
<label>Show<select id="metricQuickView">
  <option value="all">All combinations</option>
  <option value="top10">Top 10</option>
  <option value="bottom10">Bottom 10</option>
  <option value="pareto">Pareto frontier</option>
</select></label>
<div id="tradeoffChart" class="svg-chart metric-tradeoff-chart"></div>
<div id="metricChartTooltip" class="metric-chart-tooltip" hidden></div>
<div id="metricPointDetails" class="metric-point-details" aria-live="polite">Select a point to inspect exact metrics.</div>
```

Add chart state without changing recommendation source state:

```javascript
const recommendationVisualState = {
  quickView: 'all',
  selectedFamily: '',
  selectedCombo: '',
  hiddenCategories: new Set(),
};
```

- [ ] **Step 4: Replace modulo-four scatter rendering**

Derive canonical ranking from `result.completed`, map the complete canonical rows by combo/key, decorate ranks, apply the selected quick view, compute collision offsets, and emit shape-specific SVG. Keep unmodified exact values in titles/details.

```javascript
const rankedRows = (result?.completed || []).map(canonicalMetricRow);
const rankByKey = new Map(rankedRows.map((row, index) => [metricRowKey(row), index + 1]));
let plotted = complete.map(row => ({...row, recommendation_rank: rankByKey.get(metricRowKey(row)) || null}));
if (quickView === 'pareto') plotted = V.paretoRows(plotted, row => recommendationMetricValue(row, xKey), row => recommendationMetricValue(row, yKey), metricDirection(xKey), metricDirection(yKey));
else plotted = V.applyQuickView(plotted.sort((a,b) => a.recommendation_rank - b.recommendation_rank), quickView);
```

Render marker geometry with `<circle>`, rotated `<rect>`, and `<path>` while preserving one `.metric-point` focus target per row. Emit `data-combo-id`, `data-family-key`, `data-category`, `data-rank`, and exact title text.

- [ ] **Step 5: Add family/point interaction handlers**

Use delegated pointer, focus, click, Escape, and empty-chart click handlers. On family activation, add one SVG overlay polyline connecting the matching variants and toggle `is-related` / `is-muted` classes. Point selection fills the detail region with rank, pipeline, active metrics, Recall@5, MRR, nDCG@5, latency, evaluated queries, and winner badges.

- [ ] **Step 6: Run focused tests and confirm GREEN**

```bash
python3 -m pytest -q tests/test_recommendation_visuals.py tests/test_dashboard_metrics_ux.py tests/test_recommendations_ui.py
node --check web/recommendation-visuals.js
node --check web/app.js
```

Expected: all pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add web/index.html web/app.js tests/test_dashboard_metrics_ux.py tests/test_recommendations_ui.py
git commit -m "feat: redesign recommendation tradeoff chart"
```

### Task 3: Colour, heatmap, responsive, and accessibility polish

**Files:**
- Modify: `web/styles.css:58-60`
- Modify: `web/app.js:1415-1455`
- Modify: `tests/test_dashboard_metrics_ux.py`
- Modify: `tests/test_recommendations_ui.py`

- [ ] **Step 1: Add failing style and heatmap assertions**

Assert six explicit category variables, focus-visible treatment, selected/muted/family classes, top/bottom rings, tooltip/detail layout, ordered heatmap buckets, extrema classes, lifecycle state selectors, and mobile rules.

```python
for token in ("--chart-cyan", "--chart-mint", "--chart-violet", "--chart-amber", "--chart-rose", "--chart-blue"):
    assert token in styles
for selector in (".metric-rank-top", ".metric-rank-bottom", ".metric-point.is-muted", ".metric-family-link", ".metric-chart-tooltip", ".metric-heatmap-cell.is-best", ".metric-heatmap-cell.is-worst"):
    assert selector in styles
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
python3 -m pytest -q tests/test_dashboard_metrics_ux.py tests/test_recommendations_ui.py
```

Expected: style and heatmap semantic assertions fail.

- [ ] **Step 3: Implement the full-palette scatter CSS**

Add tinted dark neutrals and chart role variables using the approved six hues. Style shape-independent fills through CSS custom properties, mint numbered halos, rose dashed bottom rings, selected family connectors, legend filter buttons, tooltips, fixed/locked detail state, focus-visible outlines, and short opacity/transform transitions only.

- [ ] **Step 4: Implement ordered heatmap semantics**

Keep exact metric values and add deterministic bucket/extrema/state metadata:

```javascript
const measuredCells = [];
// record each measured cell's raw value before rendering
const bestValue = measuredValues.length ? Math.max(...measuredValues) : null;
const worstValue = measuredValues.length ? Math.min(...measuredValues) : null;
const extrema = measured && metricValue === bestValue ? ' is-best'
  : measured && metricValue === worstValue ? ' is-worst' : '';
```

Render a named scale legend with low/high values and swatches for not run, running, incomplete, and failed. Use rose → amber → cyan → mint for measured buckets. Do not map absent values to bucket zero.

- [ ] **Step 5: Add responsive behavior**

At 760 px, stack five controls, make the detail panel full width below the chart, keep the SVG minimum width inside its own overflow container when necessary, wrap legend filters, and ensure touch targets are at least 40 px high.

- [ ] **Step 6: Run focused and regression tests**

```bash
python3 -m pytest -q \
  tests/test_recommendation_visuals.py \
  tests/test_dashboard_metrics_ux.py \
  tests/test_recommendations_ui.py \
  tests/test_dashboard_metrics.py \
  tests/test_dashboard_v3_contract.py
node --check web/app.js
node --check web/recommendation-visuals.js
git diff --check
```

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add web/app.js web/styles.css tests/test_dashboard_metrics_ux.py tests/test_recommendations_ui.py
git commit -m "style: polish recommendation analytics"
```

### Task 4: Browser QA, release contract, and verified package state

**Files:**
- Modify: `tests/browser_dashboard_qa.spec.js`
- Modify: `configs/dashboard_release_contract.json` if required
- Modify: `web/index.html` for the final cache key only

- [ ] **Step 1: Extend browser QA assertions**

On the Recommendations page, assert:

```javascript
await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(180);
await expect(page.locator('#tradeoffChart .metric-rank-top')).toHaveCount(5);
await expect(page.locator('#tradeoffChart .metric-rank-bottom')).toHaveCount(5);
await page.locator('#metricQuickView').selectOption('top10');
await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(10);
await page.locator('#metricQuickView').selectOption('bottom10');
await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(10);
await page.locator('#metricQuickView').selectOption('all');
await page.locator('#tradeoffChart .metric-point').first().focus();
await expect(page.locator('#metricPointDetails')).not.toContainText('Select a point');
```

Capture dedicated `recommendations-option1.png`, `recommendations-family-selected.png`, and mobile screenshots.

- [ ] **Step 2: Run browser QA against deterministic 180-row fixtures**

Start the canonical dashboard on an unused local port using the existing test data or browser fixture path, then run:

```bash
DASHBOARD_QA_URL=http://127.0.0.1:5098 \
DASHBOARD_QA_ARTIFACTS=/tmp/canonical-dashboard-option1-qa \
npx playwright test tests/browser_dashboard_qa.spec.js
```

Expected: desktop and mobile tests pass with no console errors, page errors, failed requests, or global overflow.

- [ ] **Step 3: Inspect screenshots and patch at least one visual pass if below 8.5/10**

Check hierarchy, point overlap, category distinction, top/bottom visibility, tooltip legibility, heatmap exact values, mobile controls, and clipping. Make only targeted CSS/rendering corrections and rerun focused tests plus screenshots.

- [ ] **Step 4: Bump and enforce the final cache contract**

Use `20260724-canonical-dashboard-v5` for `styles.css`, `recommendation-visuals.js`, and `app.js`. Update the release contract allowlist/assertions if it pins asset strings.

- [ ] **Step 5: Run the final release gate**

```bash
python3 -m pytest -q \
  tests/test_dashboard_v3_contract.py \
  tests/test_dashboard_release_guard.py \
  tests/test_dashboard_metrics.py \
  tests/test_dashboard_metrics_ux.py \
  tests/test_recommendations_ui.py \
  tests/test_dashboard_dataset_lifecycle.py \
  tests/test_dashboard_source_catalog.py \
  tests/test_official_artifact_integrity.py \
  tests/test_modular_benchmark.py \
  tests/test_document_parser_mineru_cli.py \
  tests/test_document_parser_modern_mineru_cli.py \
  tests/test_project_run_results.py \
  tests/test_project_documents.py \
  tests/test_openai_embedding_retry.py \
  tests/test_recommendation_visuals.py
python3 scripts/verify_dashboard_release.py
node --check web/app.js
node --check web/recommendation-visuals.js
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/dashboard_source_catalog.py
git diff --check
```

Expected: all tests pass and verifier prints `DASHBOARD_RELEASE_CONTRACT_OK`.

- [ ] **Step 6: Commit Task 4**

```bash
git add web/index.html tests/browser_dashboard_qa.spec.js configs/dashboard_release_contract.json
git commit -m "test: verify recommendation chart visual upgrade"
```

- [ ] **Step 7: Produce the handoff artifact**

Create a tracked-source ZIP from canonical HEAD excluding `data/`, verify archive integrity, record SHA-256, and provide narrow Windows overlay/commit/push plus VM pull/restart/live-API verification commands. Never stage VM-local `data/benchmark_input.csv`, adapter customizations, logs, indexes, or benchmark runs.
