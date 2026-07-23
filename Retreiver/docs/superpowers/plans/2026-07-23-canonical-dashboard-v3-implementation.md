# Canonical WNS Benchmark Dashboard V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one regression-safe WNS dashboard that combines the newest Recommendations experience with linked ground truth, complete-run integrity, the practical trade-off/heatmap UX, MinerU project evaluation, and truthful 180-combination state.

**Architecture:** Keep the hardened `55ea159` tree as the canonical base. Port behaviors from the divergent integrity lifecycle by adding failing tests and making surgical edits to the canonical files; never replace whole server or frontend files. The static frontend consumes additive source/result contracts from the standard-library backend, and a release verifier blocks packaging when any required page, selector, route, visual, or parity rule disappears.

**Tech Stack:** Python 3 standard library, existing project-isolation services, static HTML/CSS/JavaScript, Node VM tests, pytest/direct Python tests, Playwright Chromium, Git archive/ZIP.

---

## File map

| File | Responsibility |
|---|---|
| `scripts/serve_benchmark_dashboard.py` | Canonical source/result APIs, complete-run state, upload/setup/evaluation routes, selected source context. |
| `scripts/dashboard_source_catalog.py` | Allowlisted datasets, linked ground truths, exact path resolution, official/project source metadata. |
| `source/services/document_parser.py` | MinerU CLI discovery and fail-visible layout-aware PDF parsing. |
| `scripts/run_project_evaluation.py` | Project-scoped retrieval/evaluation runner when the canonical tree does not already provide equivalent matrix execution. |
| `web/index.html` | Stable page inventory, Recommendations source strip/cards/table/analytics targets, linked-GT status targets. |
| `web/app.js` | Source binding, recommendation rendering, complete-run analytics, heatmap, run gating, cache-safe refresh. |
| `web/styles.css` | Existing dark WNS visual language plus analytical layouts and mobile-safe heatmap behavior. |
| `tests/test_dashboard_v3_contract.py` | Cross-lineage release behavior: required pages, routes, linked-GT, 180 states, no fake quality. |
| `tests/test_recommendations_ui.py` | Recommendation cards, top-10 table, source safety, analytical explorer integration. |
| `tests/test_dashboard_metrics.py` | Existing exact metrics/source/result API regression suite; extend, never weaken. |
| `tests/test_dashboard_integrity.py` | Complete-run state selection and latest-artifact integrity. |
| `tests/test_dashboard_dataset_lifecycle.py` | Linked-GT, evidence-only, setup, MinerU, and project-evaluation lifecycle. |
| `tests/test_dashboard_metrics_ux.py` | Browser-independent trade-off and heatmap behavior. |
| `configs/dashboard_release_contract.json` | Machine-readable pages, IDs, routes, matrix formula, and required test commands. |
| `scripts/verify_dashboard_release.py` | Static/runtime release-contract verifier and clean-tree guard. |
| `scripts/inventory_vm_source_drift.py` | Read-only VM source inventory used before pull/package operations. |

Do not delete or replace existing hardened modules such as `source/services/project_*`, `web/recommendations.js`, `web/source-state.js`, official-artifact protection, or project matrix tests.

---

### Task 1: Establish the canonical baseline and complete-run contract

**Files:**
- Create: `tests/test_dashboard_v3_contract.py`
- Create only if missing after the baseline run: `tests/test_dashboard_integrity.py`
- Modify only when a failing assertion proves it necessary: `scripts/serve_benchmark_dashboard.py`
- Modify only when a failing assertion proves it necessary: `web/app.js`

- [ ] **Step 1: Record the clean baseline and run existing source/result tests**

Run:

```bash
git status --short
git rev-parse HEAD
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_metrics.py \
  tests/test_recommendations_ui.py \
  tests/test_project_run_results.py \
  tests/test_dashboard_source_catalog.py
node --check web/app.js
git diff --check
```

Expected: HEAD includes design commit `294a95a`; `git status --short` is empty before new tests. Record every failing pre-existing test without changing assertions to match broken behavior.

- [ ] **Step 2: Write the V3 backend contract test**

Create `tests/test_dashboard_v3_contract.py` with the following initial executable assertions:

```python
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "serve_benchmark_dashboard.py"
INDEX = ROOT / "web" / "index.html"
APP = ROOT / "web" / "app.js"


def test_canonical_backend_exposes_source_and_result_contracts():
    source = SERVER.read_text(encoding="utf-8")
    for route in (
        '"/api/results"',
        '"/api/result-sources"',
        '"/api/projects/"',
    ):
        assert route in source
    assert "source_catalog" in source
    assert "pipeline_state" in source


def test_official_matrix_formula_remains_exactly_180():
    import scripts.serve_benchmark_dashboard as dashboard

    options = dashboard.options_payload()
    chunkers = options.get("chunkers", [])
    embeddings = options.get("embeddings", [])
    stores = options.get("vector_stores", options.get("stores", []))
    rerankers = options.get("rerankers", [])
    assert (len(chunkers), len(embeddings), len(stores), len(rerankers)) == (5, 3, 4, 3)
    assert len(chunkers) * len(embeddings) * len(stores) * len(rerankers) == 180


def test_frontend_never_uses_incomplete_rows_as_quality_winners():
    app = APP.read_text(encoding="utf-8")
    assert "pipeline_state" in app
    assert "state === 'complete'" in app or 'state === "complete"' in app
    assert "not_run" in app
    assert "incomplete" in app
```

If the actual helper is not named `options_payload`, call the existing helper that returns `/api/results.options`; do not create a duplicate options authority merely to satisfy the test.

- [ ] **Step 3: Run the new tests and confirm the exact RED state**

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_dashboard_v3_contract.py
```

Expected: route/source checks may already pass; the 180/options or complete-state assertion may fail. Passing assertions prove preserved canonical behavior and require no production change.

- [ ] **Step 4: Port only missing complete-run helpers**

When the tests prove them absent, port the accepted behavior from commits `13bf6ea`, `8eb387c`, and `5049f26` into the canonical server/app without replacing files. Preserve these signatures and rules:

```python
def pipeline_key(row: dict) -> str: ...
def complete_metric_row(row: dict) -> bool: ...
def pipeline_state_rows(sources=None) -> list[dict]: ...
def complete_pipeline_metric_rows(states: list[dict]) -> list[dict]: ...
```

`pipeline_state_rows` must start from all 180 configured keys, select the latest complete artifact by nanosecond-safe ordering, retain incomplete/failed/not-run candidates, and never allow a newer incomplete retry to replace an older complete result.

In `web/app.js`, quality winners and exact metrics rankings must use:

```javascript
function completeStateRows(rows = []) {
  return rows.filter(row => row.state === 'complete' && Number(row.evaluated_queries) > 0);
}
```

Use `state.operational.pipeline_state` as the authority when present. Legacy summary fallback remains only for old fixture payloads.

- [ ] **Step 5: Verify Task 1 GREEN**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_v3_contract.py \
  tests/test_dashboard_metrics.py \
  tests/test_recommendations_ui.py \
  tests/test_project_run_results.py \
  tests/test_dashboard_source_catalog.py
node --check web/app.js
git diff --check
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1 narrowly**

```bash
git add tests/test_dashboard_v3_contract.py tests/test_dashboard_integrity.py scripts/serve_benchmark_dashboard.py web/app.js
git diff --cached --check
git commit -m "fix: unify dashboard source and complete-run contracts"
```

Stage only paths that actually changed.

---

### Task 2: Lock dataset-to-ground-truth binding and preserve MinerU project evaluation

**Files:**
- Modify: `tests/test_dashboard_v3_contract.py`
- Create or merge: `tests/test_dashboard_dataset_lifecycle.py`
- Create if absent: `tests/test_dashboard_upload_selection.py`
- Create if absent: `tests/test_document_parser_mineru_cli.py`
- Create if absent: `tests/test_document_parser_modern_mineru_cli.py`
- Modify: `scripts/dashboard_source_catalog.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `source/services/document_parser.py`
- Modify: `web/app.js`
- Modify: `web/index.html`

- [ ] **Step 1: Add the linked-GT and evidence-only assertions**

Append to `tests/test_dashboard_v3_contract.py`:

```python
def test_global_dataset_selection_owns_one_linked_groundtruth():
    app = APP.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    assert 'id="globalDataset"' in index
    assert 'id="globalGroundtruth"' in index
    assert "linked_groundtruth" in app or "groundtruth_ids" in app
    assert "Evidence-only" in app or "evidence-only" in app


def test_mineru_project_contract_is_visible_and_fail_closed():
    parser = (ROOT / "source" / "services" / "document_parser.py").read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")
    assert "find_mineru_cli" in parser
    assert "DocumentParserService" in server
    assert "mineru" in server.lower()
```

Add lifecycle behavior tests that create two temporary projects, link a valid GT to only one, and assert the other project cannot select or resolve that GT. Add a PDF upload test that monkeypatches `DocumentParserService.parse_document` and verifies page/table provenance and `parser_method == "MinerU"` are retained.

- [ ] **Step 2: Confirm RED without editing production code**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_v3_contract.py \
  tests/test_dashboard_dataset_lifecycle.py \
  tests/test_dashboard_upload_selection.py \
  tests/test_document_parser_mineru_cli.py \
  tests/test_document_parser_modern_mineru_cli.py
```

Expected: only genuinely missing lifecycle/MinerU contracts fail. If a candidate test assumes the divergent simplified project model and conflicts with hardened project isolation, adapt the fixture to canonical project IDs and services; never weaken isolation.

- [ ] **Step 3: Implement linked-ground-truth metadata at the catalog boundary**

Every dataset catalog row must expose one of:

```python
{
    "id": "dataset:...",
    "label": "...",
    "ready": True,
    "linked_groundtruth_id": "groundtruth:...",
}
```

or:

```python
{
    "id": "dataset:...",
    "label": "...",
    "ready": True,
    "linked_groundtruth_id": "groundtruth:none",
    "mode": "evidence_only",
}
```

The server resolves the linked ID through `dashboard_source_catalog.py`; the browser never submits an arbitrary path. A mismatched dataset/GT pair returns a structured 400 response and never launches evaluation.

- [ ] **Step 4: Make the global selector deterministic and locked**

In `web/app.js`, add or update these pure helpers:

```javascript
function selectedDatasetRecord() {
  const id = $('globalDataset')?.value || 'dataset:wns-default';
  return (state.sourceCatalog?.datasets || []).find(row => row.id === id) || null;
}

function linkedGroundtruthId(dataset = selectedDatasetRecord()) {
  return dataset?.linked_groundtruth_id || 'groundtruth:none';
}

function syncLinkedGroundtruth() {
  const select = $('globalGroundtruth');
  if (!select) return 'groundtruth:none';
  const linkedId = linkedGroundtruthId();
  Array.from(select.options || []).forEach(option => {
    option.disabled = option.value !== linkedId;
  });
  select.value = linkedId;
  select.disabled = true;
  return linkedId;
}
```

Call `syncLinkedGroundtruth()` after dataset options are filled and before dependent source/result requests. The visible label remains readable even though the control is locked. Metrics, Recommendations, and Run Pipeline mirrors must show the same dataset and GT labels.

- [ ] **Step 5: Port the accepted MinerU discovery/evaluation behavior surgically**

Port the accepted `find_mineru_cli()` and modern MinerU CLI fallback behavior from `0365448` into `source/services/document_parser.py`. For PDFs uploaded through Documents:

```python
parsed = asyncio.run(
    DocumentParserService(input_dir=pdf.parent, output_dir=output_dir).parse_document(
        pdf,
        force_backend="mineru",
    )
)
```

Preserve parsed page numbers, tables, image/layout text, and parser method in canonical project documents/chunks. If MinerU is unavailable or parsing fails, return a visible project error; do not call a text-only PDF helper silently.

Prefer the hardened project matrix/evaluation services already present in the canonical tree. Add `scripts/run_project_evaluation.py` only when a failing canonical endpoint test proves no equivalent runner exists.

- [ ] **Step 6: Verify Task 2 GREEN**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_v3_contract.py \
  tests/test_dashboard_dataset_lifecycle.py \
  tests/test_dashboard_upload_selection.py \
  tests/test_document_parser_mineru_cli.py \
  tests/test_document_parser_modern_mineru_cli.py \
  tests/test_project_isolation.py \
  tests/test_project_matrix_server_bridge.py
node --check web/app.js
python -m py_compile scripts/serve_benchmark_dashboard.py source/services/document_parser.py
git diff --check
```

Expected: all selected tests pass; no hardened isolation test is removed or skipped.

- [ ] **Step 7: Commit Task 2 narrowly**

```bash
git add scripts/dashboard_source_catalog.py scripts/serve_benchmark_dashboard.py source/services/document_parser.py web/index.html web/app.js tests/test_dashboard_v3_contract.py tests/test_dashboard_dataset_lifecycle.py tests/test_dashboard_upload_selection.py tests/test_document_parser_mineru_cli.py tests/test_document_parser_modern_mineru_cli.py
git diff --cached --check
git commit -m "feat: bind datasets to ground truth and preserve MinerU evaluation"
```

---

### Task 3: Combine Recommendations with the complete-run trade-off explorer and heatmap

**Files:**
- Modify: `tests/test_recommendations_ui.py`
- Create: `tests/test_dashboard_metrics_ux.py`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`

- [ ] **Step 1: Write the failing combined-page test**

Add assertions to `tests/test_recommendations_ui.py`:

```python
def test_recommendations_page_contains_ranked_decisions_and_analytics():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    app = (root / "web" / "app.js").read_text(encoding="utf-8")
    required_ids = (
        "recommendationStrip", "recommendationTable", "recommendationSourceContext",
        "metricXAxis", "metricYAxis", "metricColorBy", "metricBubbleBy",
        "tradeoffChart", "metricInterpretation", "heatmapMetric",
        "heatmapEmbedding", "heatmapReranker", "metricHeatmap",
    )
    for item in required_ids:
        assert f'id="{item}"' in index
    assert "renderRecommendationAnalytics" in app
    assert "completeRecommendationRows" in app
```

Create `tests/test_dashboard_metrics_ux.py` using the existing Node VM test pattern. Supply one complete row, one incomplete row, and one not-run row. Assert:

- only the complete row creates a scatter point;
- the incomplete and not-run combinations remain visible as state-labelled heatmap cells;
- zero is formatted as `0.000`, not interpreted as missing;
- an evidence-only payload renders no quality winner, scatter point, or quality-colored heatmap cell;
- selected metric/embedding/reranker controls persist during a client-only rerender.

- [ ] **Step 2: Run the new tests and confirm RED**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_recommendations_ui.py \
  tests/test_dashboard_metrics_ux.py
```

Expected: failure because the canonical Recommendations page has the cards/table but not the V3 analytical target IDs/functions.

- [ ] **Step 3: Add the analytical markup below the top-10 table**

In `web/index.html`, keep the recommendation source panel, recommendation strip, top combinations table, and pricing ledger. Replace the old collapsed chart body with stable V3 panels directly below the top-10 table:

```html
<section class="panel metric-explorer-panel" aria-labelledby="metricExplorerTitle">
  <div class="section-head">
    <div><h2 id="metricExplorerTitle">Accuracy, cost, and latency trade-offs</h2><p>Only complete evaluations from the selected dataset and linked ground truth are plotted. Left is faster; higher is better for quality.</p></div>
    <span id="metricExplorerHint" class="mini-stat">Waiting for evaluation</span>
  </div>
  <div class="metric-control-strip">
    <label>X axis<select id="metricXAxis"></select></label>
    <label>Y axis<select id="metricYAxis"></select></label>
    <label>Color by<select id="metricColorBy"></select></label>
    <label>Bubble size<select id="metricBubbleBy"></select></label>
  </div>
  <div id="metricExplorerKpis" class="metric-explorer-kpis"></div>
  <div id="tradeoffChart" class="svg-chart metric-tradeoff-chart"></div>
  <p id="metricInterpretation" class="metric-interpretation"></p>
</section>
<section class="panel metric-heatmap-panel" aria-labelledby="heatmapTitle">
  <div class="section-head"><div><h2 id="heatmapTitle">Configuration heatmap</h2><p>Chunker × vector database for the selected metric, embedding, and reranker. Missing never means zero.</p></div><div id="metricHeatmapLegend" class="metric-heatmap-legend"></div></div>
  <div class="metric-control-strip heatmap-controls">
    <label>Metric<select id="heatmapMetric"></select></label>
    <label>Embedding<select id="heatmapEmbedding"></select></label>
    <label>Reranker<select id="heatmapReranker"></select></label>
  </div>
  <div id="metricHeatmap" class="metric-heatmap" role="region" aria-label="Configuration metric heatmap"></div>
</section>
```

Add `id="recommendationSourceContext"` to the existing source-context paragraph or rename references consistently.

- [ ] **Step 4: Implement source-specific complete rows**

Add these helpers before the new renderers:

```javascript
function completeRecommendationRows(payload = {}) {
  if (payload.scoring_mode !== 'retrieval_labels' && payload.mode !== 'labelled') return [];
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  return rows.filter(row => {
    const status = String(row.status || row.state || '').toLowerCase();
    return ['complete', 'completed'].includes(status)
      && Number(row.evaluated_queries ?? row.query_count ?? 0) > 0;
  });
}

function recommendationStateRows(payload = {}) {
  if (payload.source_type === 'official' && Array.isArray(state.operational?.pipeline_state)) {
    return state.operational.pipeline_state;
  }
  return Array.isArray(payload.rows) ? payload.rows : [];
}
```

Implement `renderRecommendationAnalytics(payload)` by adapting the accepted `renderMetricExplorer` and `renderMetricHeatmap` algorithms from `cdb28ef`. It must use `completeRecommendationRows(payload)` for scatter/KPIs and `recommendationStateRows(payload)` for all configured heatmap states. It must not read unrelated global rows after an uploaded project is selected.

Call it at the end of `renderRecommendationSource(payload)` after the cards, table, and pricing ledger:

```javascript
renderRecommendationAnalytics(payload);
```

All seven control listeners call `renderRecommendationAnalytics(recommendationState.active)` without refetching data.

- [ ] **Step 5: Improve the graph for practical decision reading**

The SVG renderer must include:

- X and Y axis labels and min/max values;
- guidance text `Faster ←` for latency and `Better ↑` for quality;
- one accessible `<title>` per point containing pipeline, X, Y, source, and completion state;
- persistent labels only for Quality, Speed, and Value winners; other points use compact IDs and hover/focus details;
- color legend derived from actual selected grouping values;
- bubble size fixed unless `evaluated_queries` is explicitly selected and present.

Do not add synthetic cost points when cost is unavailable.

- [ ] **Step 6: Add practical heatmap color and mobile behavior**

Use existing design tokens. Add styles equivalent to:

```css
.metric-control-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric-explorer-kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric-heatmap { overflow:auto; scrollbar-gutter:stable; }
.metric-heatmap-grid { min-width:720px; display:grid; }
.metric-heatmap-cell[data-state="not_run"] { opacity:.62; }
.metric-heatmap-cell[data-state="incomplete"],
.metric-heatmap-cell[data-state="running"] { border-color:#fbbf24; }
.metric-heatmap-cell[data-state="failed"] { border-color:#fda4af; }
@media (max-width: 767px) {
  .metric-control-strip, .metric-explorer-kpis { grid-template-columns:1fr; }
  .metric-heatmap-scroll-note { display:block; }
}
```

Measured buckets use the existing cyan-to-mint palette and an explicit low/high legend. Exact values remain inside cells.

- [ ] **Step 7: Verify Task 3 GREEN**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_recommendations_ui.py \
  tests/test_dashboard_metrics_ux.py \
  tests/test_dashboard_metrics.py \
  tests/test_dashboard_v3_contract.py
node --check web/app.js
git diff --check
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 3 narrowly**

```bash
git add web/index.html web/app.js web/styles.css tests/test_recommendations_ui.py tests/test_dashboard_metrics_ux.py tests/test_dashboard_v3_contract.py
git diff --cached --check
git commit -m "feat: add recommendation tradeoffs and configuration heatmap"
```

---

### Task 4: Integrate source refresh, cache safety, and honest VM adapter states

**Files:**
- Modify: `tests/test_dashboard_metrics.py`
- Modify: `tests/test_recommendations_ui.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `web/app.js`

- [ ] **Step 1: Add failing end-to-end contract tests**

Extend existing tests to assert:

1. `/api/results` includes `source_catalog` even when no local evaluation artifacts exist.
2. `/api/result-sources` returns exact official/project sources and `Cache-Control: no-store`.
3. Selecting a dataset aborts stale source/result requests, locks its linked GT, updates Metrics/Recommendations/Run mirrors, then refreshes only the selected result source.
4. A late project result response cannot overwrite a newer official selection.
5. Amazon credential status can block Amazon Rerank but never labels FAISS itself unavailable.

Use the existing fake handler/Node VM patterns in `tests/test_dashboard_metrics.py` and `tests/test_recommendations_ui.py`; do not replace them with text-only assertions when behavior can be executed.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_metrics.py \
  tests/test_recommendations_ui.py
```

- [ ] **Step 3: Make source application one atomic client operation**

`applyGlobalSourceContext()` must:

```javascript
async function applyGlobalSourceContext({syncResults = true} = {}) {
  const generation = ++sourceGeneration;
  const dataset = selectedDatasetRecord();
  const groundtruthId = syncLinkedGroundtruth();
  renderSourceMirrors(dataset, groundtruthId);
  if (!syncResults) return;
  await loadResultSources({generation, datasetId: dataset?.id, groundtruthId});
  if (generation !== sourceGeneration) return;
  await refreshSelectedResult({generation});
}
```

Reuse existing request tokens/AbortControllers and established functions where available. Do not introduce a second state store.

- [ ] **Step 4: Keep the server additive and no-store**

The canonical server must retain all existing hardened routes. Add missing result-source/project routes by calling `ProjectRunResultService`; do not parse project evidence in the HTTP handler. Every result endpoint returns `Cache-Control: no-store` and exact 400/404 mappings without filesystem paths.

In `/api/results`, `source_catalog` is present independent of runtime result completeness. `operational.pipeline_state` always contains the configured official keys, so missing VM artifacts render `not_run` rather than making selectors vanish.

- [ ] **Step 5: Correct the FAISS/Amazon label**

Display adapter state at the component level:

```javascript
const amazonBlocked = reranker === 'amazon_rerank_v1' && !amazonConfigured;
```

The blocked reason must read `Amazon Rerank unavailable: AWS/Bedrock credentials or permissions missing`. FAISS stays runnable for BGE, Qwen, and no-reranker lanes when its own health is available.

- [ ] **Step 6: Verify Task 4 GREEN**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_metrics.py \
  tests/test_recommendations_ui.py \
  tests/test_project_run_results.py \
  tests/test_dashboard_v3_contract.py
node --check web/app.js
python -m py_compile scripts/serve_benchmark_dashboard.py
git diff --check
```

- [ ] **Step 7: Commit Task 4 narrowly**

```bash
git add scripts/serve_benchmark_dashboard.py web/app.js tests/test_dashboard_metrics.py tests/test_recommendations_ui.py tests/test_dashboard_v3_contract.py
git diff --cached --check
git commit -m "fix: keep dashboard sources synchronized and cache safe"
```

---

### Task 5: Add release and VM-to-local parity guardrails

**Files:**
- Create: `configs/dashboard_release_contract.json`
- Create: `scripts/verify_dashboard_release.py`
- Create: `scripts/inventory_vm_source_drift.py`
- Create: `tests/test_dashboard_release_guard.py`
- Modify: `.gitignore` only if generated reports need exclusion

- [ ] **Step 1: Write the release-guard tests**

Create `tests/test_dashboard_release_guard.py`:

```python
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_release_contract_declares_every_required_surface():
    contract = json.loads((ROOT / "configs" / "dashboard_release_contract.json").read_text(encoding="utf-8"))
    assert contract["matrix"] == {"chunkers": 5, "embeddings": 3, "stores": 4, "rerankers": 3, "total": 180}
    assert set(contract["pages"]) == {"overview", "quality", "compare", "run", "evidence", "hallucination", "nvidia", "repository", "upload", "ops"}
    assert {"globalDataset", "globalGroundtruth", "recommendationStrip", "recommendationTable", "tradeoffChart", "metricHeatmap"}.issubset(contract["dom_ids"])


def test_release_verifier_passes_current_tracked_tree():
    result = subprocess.run(
        [sys.executable, "scripts/verify_dashboard_release.py", "--allow-dirty-tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DASHBOARD_RELEASE_CONTRACT_OK" in result.stdout
```

Add unit tests for VM inventory classification using a temporary `git status --porcelain=v1` fixture. Source paths under `scripts/`, `web/`, `source/`, `benchmarking/`, `configs/`, and `tests/` must classify as `merge_back_required`; `data/` and runtime logs classify as `runtime_artifact`.

- [ ] **Step 2: Confirm RED**

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_dashboard_release_guard.py
```

Expected: files do not exist.

- [ ] **Step 3: Create the machine-readable contract**

`configs/dashboard_release_contract.json` must include:

```json
{
  "pages": ["overview", "quality", "compare", "run", "evidence", "hallucination", "nvidia", "repository", "upload", "ops"],
  "dom_ids": ["globalDataset", "globalGroundtruth", "metricsSourceContext", "recommendationSourceContext", "recommendationStrip", "recommendationTable", "metricXAxis", "metricYAxis", "metricColorBy", "metricBubbleBy", "tradeoffChart", "heatmapMetric", "heatmapEmbedding", "heatmapReranker", "metricHeatmap", "runSourceContext", "runCompletePipelineBtn"],
  "api_routes": ["/api/results", "/api/result-sources", "/api/upload-dataset", "/api/run/status", "/api/run/preflight-complete-pipeline", "/api/run/complete-pipeline"],
  "matrix": {"chunkers": 5, "embeddings": 3, "stores": 4, "rerankers": 3, "total": 180},
  "required_files": ["web/recommendations.js", "web/source-state.js", "scripts/dashboard_source_catalog.py", "source/services/project_run_results.py", "source/services/document_parser.py"]
}
```

- [ ] **Step 4: Implement the release verifier**

`scripts/verify_dashboard_release.py` must:

- load the JSON contract;
- check every page panel and DOM ID in `web/index.html`;
- check every route in `scripts/serve_benchmark_dashboard.py`;
- check required files exist;
- import the server/options helper and verify the 5×3×4×3 formula;
- run `node --check web/app.js`;
- reject duplicate HTML IDs;
- reject a dirty tracked source tree unless `--allow-dirty-tests` is set and every dirty path is under `tests/`, `docs/`, or `artifacts/`;
- print `DASHBOARD_RELEASE_CONTRACT_OK` only after all checks pass.

It must never mutate the repository.

- [ ] **Step 5: Implement the read-only VM source inventory**

`scripts/inventory_vm_source_drift.py` reads `git status --porcelain=v1 -z` through `subprocess.run`, classifies changed paths, prints JSON, and exits `2` when any `merge_back_required` path exists. It does not stash, restore, reset, clean, or delete anything.

Its output must include:

```json
{
  "merge_back_required": [],
  "runtime_artifacts": [],
  "unclassified": [],
  "safe_to_pull_source": true
}
```

The VM deployment procedure stops when `safe_to_pull_source` is false and returns the exact source paths that must be merged locally first.

- [ ] **Step 6: Verify Task 5 GREEN**

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_dashboard_release_guard.py
python scripts/verify_dashboard_release.py --allow-dirty-tests
node --check web/app.js
git diff --check
```

- [ ] **Step 7: Commit Task 5 narrowly**

```bash
git add configs/dashboard_release_contract.json scripts/verify_dashboard_release.py scripts/inventory_vm_source_drift.py tests/test_dashboard_release_guard.py .gitignore
git diff --cached --check
git commit -m "test: guard dashboard releases and VM source parity"
```

Stage `.gitignore` only if changed.

---

### Task 6: Full deterministic verification and browser QA

**Files:**
- Modify production files only when a new failing regression test proves a defect
- Create QA artifacts under: `artifacts/dashboard-v3/`

- [ ] **Step 1: Run the complete relevant deterministic suite**

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_dashboard_v3_contract.py \
  tests/test_dashboard_release_guard.py \
  tests/test_dashboard_metrics.py \
  tests/test_dashboard_integrity.py \
  tests/test_dashboard_dataset_lifecycle.py \
  tests/test_dashboard_metrics_ux.py \
  tests/test_recommendations_ui.py \
  tests/test_dashboard_source_catalog.py \
  tests/test_project_run_results.py \
  tests/test_project_isolation.py \
  tests/test_project_matrix_frontend.py \
  tests/test_project_matrix_server_bridge.py \
  tests/test_document_parser_mineru_cli.py \
  tests/test_document_parser_modern_mineru_cli.py \
  tests/test_openai_embedding_retry.py \
  tests/test_modular_benchmark.py
node --check web/app.js
python -m py_compile scripts/serve_benchmark_dashboard.py scripts/inventory_vm_source_drift.py scripts/verify_dashboard_release.py source/services/document_parser.py
git diff --check
```

Expected: all selected tests pass. Do not label unrelated environment-dependent suites green if they were not run.

- [ ] **Step 2: Start the real canonical dashboard on an isolated QA port**

```bash
python scripts/serve_benchmark_dashboard.py --host 127.0.0.1 --port 5098
```

Run as a tracked background process. Verify readiness with:

```bash
curl -fsS http://127.0.0.1:5098/api/results >/dev/null
curl -fsS http://127.0.0.1:5098/api/result-sources >/dev/null
```

- [ ] **Step 3: Probe API truth contracts**

Capture JSON proving:

- `source_catalog.datasets` is non-empty;
- the official dataset has a linked GT;
- options multiply to 180;
- pipeline state has 180 official keys even when complete count is zero locally;
- result sources return HTTP 200 and no-store headers;
- no local fixture claims VM metrics.

- [ ] **Step 4: Run desktop Playwright QA on every page**

Use Chromium with `--no-sandbox`, viewport `1440×1100`. For every fixed page:

- click the real navigation button;
- assert its panel is visible;
- capture `artifacts/dashboard-v3/desktop-<page>.png`;
- record console errors and failed requests;
- assert document width does not exceed viewport except inside marked heatmap/table scrollers.

On Recommendations assert cards, table, trade-off chart/no-data state, heatmap, controls, linked dataset, and linked GT are visible. On Run assert the selected source/GT restatement and exact gating reason are visible.

- [ ] **Step 5: Run mobile Playwright QA on every page**

Repeat at `390×844`, capture `mobile-<page>.png`, and verify:

- no page-level horizontal overflow;
- controls are not clipped;
- heatmap has an explicit scroll note and scrollable inner container;
- recommendation table remains usable;
- dialogs fit and close;
- disabled Grounding Audit is explicit rather than blank.

- [ ] **Step 6: Fix only verified defects with RED tests**

For every browser/API defect, add or extend the narrowest deterministic test, prove it fails, implement the surgical fix, rerun focused tests, then repeat the browser step. Do not make untested visual cleanup changes during this loop.

- [ ] **Step 7: Run final release verifier and commit QA fixes**

```bash
python scripts/verify_dashboard_release.py
git status --short
git diff --check
```

Expected: verifier passes and tracked source tree is clean. Commit any tested QA fixes with a narrow message before proceeding.

---

### Task 7: Build the verified package and prepare safe VM commands

**Files:**
- Create artifact outside the repo: `/home/fate/wns-deliverables/Retreiver-dashboard-v3-<commit>.zip`
- Create artifact outside the repo: `/home/fate/wns-deliverables/Retreiver-dashboard-v3-<commit>.sha256`

- [ ] **Step 1: Verify the canonical commit and clean source tree**

```bash
python scripts/verify_dashboard_release.py
git status --short
git rev-parse --short HEAD
```

Expected: clean tree and release verifier success.

- [ ] **Step 2: Build from Git, not from a mutable folder**

Use `git archive` so only committed canonical files enter the ZIP:

```bash
git archive --format=zip --prefix=Retreiver/ --output=/home/fate/wns-deliverables/Retreiver-dashboard-v3-$(git rev-parse --short HEAD).zip HEAD
sha256sum /home/fate/wns-deliverables/Retreiver-dashboard-v3-$(git rev-parse --short HEAD).zip
```

Save the real hash beside the ZIP. Do not include `.git`, runtime datasets, virtualenvs, caches, or VM result directories outside tracked release files.

- [ ] **Step 3: Run VM source inventory before giving a pull command**

The first VM command is read-only:

```bash
cd ~/benchmarking/Retreiver && .venv-vm/bin/python scripts/inventory_vm_source_drift.py
```

If `safe_to_pull_source` is false, stop and merge those VM source changes back into this canonical branch before any pull.

- [ ] **Step 4: Prepare the safe pull/deploy command only after inventory passes**

The final VM command must preserve runtime data and must not use `git reset`, `git clean`, or broad restore. It may stash only the exact runtime files reported by inventory, pull `feature/global-source-selectors`, restore only those runtime files, run the release verifier, restart with `scripts/start_vm_stack.sh`, and probe both source/result endpoints.

- [ ] **Step 5: Diagnose real 180-combination metrics only after frontend verification**

Use read-only API/artifact probes on the VM to report:

- configured count,
- complete count,
- incomplete/failed/not-run count,
- selected dataset ID,
- linked GT ID and validity,
- result source IDs,
- missing metric fields by combination.

Do not rerun combinations until the probe distinguishes missing artifacts from incomplete evaluation fields.

---

## Final acceptance checklist

- [ ] Canonical base keeps the hardened project/module tree.
- [ ] Dataset automatically locks to its linked ground truth.
- [ ] Evidence-only mode blocks quality claims.
- [ ] Recommendations contains cards, top-10 table, pricing, trade-off graph, and heatmap.
- [ ] Metrics remains the exact table/diagnostics source.
- [ ] Run Pipeline restates dataset/GT and keeps official/project actions separate.
- [ ] Documents uses MinerU for uploaded PDFs without silent fallback.
- [ ] `/api/results` and `/api/result-sources` remain available with no-store behavior.
- [ ] Official configuration count remains 180 independent of local artifacts.
- [ ] Desktop/mobile screenshots exist for every page with no console/request failures.
- [ ] Release verifier passes on a clean commit.
- [ ] VM source inventory runs before pull.
- [ ] ZIP is created from the verified Git commit and has a recorded SHA-256.
