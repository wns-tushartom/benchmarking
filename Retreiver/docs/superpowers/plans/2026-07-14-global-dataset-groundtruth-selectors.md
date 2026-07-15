# Global Dataset and Ground-Truth Selectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add global independently selectable datasets and ground-truth files to standard and NVIDIA pipeline controls, preserve evidence-only metric integrity, and render the completed NVIDIA baseline results.

**Architecture:** Introduce a focused server-side source-catalog module that discovers allowlisted data and ground-truth assets and resolves opaque IDs to private paths. Extend the dashboard API and run endpoints to accept those IDs. Keep the static frontend build-free by rendering selectors and run modes in the existing HTML/JavaScript application.

**Tech Stack:** Python 3.12, stdlib HTTP server, CSV/XLSX metadata inspection, vanilla JavaScript, HTML/CSS, pytest, Node syntax/DOM harness tests.

---

## File map

- Create `scripts/dashboard_source_catalog.py`: source discovery, validation, counts, opaque-ID resolution, upload metadata.
- Modify `scripts/serve_benchmark_dashboard.py`: catalog API, selected-source command dispatch, ground-truth upload route, NVIDIA baseline/reranked artifact payloads.
- Modify `scripts/run_complete_pipeline.py`: accept a trusted resolved workbook/audit and an explicit evidence-only query mode instead of assuming WNS paths.
- Modify `web/index.html`: upload type, standard selectors, NVIDIA selectors, baseline/reranked result selector.
- Modify `web/app.js`: catalog rendering, mode integrity, payload construction, separate NVIDIA result rendering.
- Modify `web/style.css`: selector metadata/status styling using the current dashboard system.
- Create `tests/test_dashboard_source_catalog.py`: backend catalog/resolver/upload/evidence-only tests.
- Modify `tests/test_dashboard_metrics.py`: frontend DOM behavior and NVIDIA 500/500 baseline rendering tests.
- Modify `README.md`: operator-facing selector and evidence-only behavior.

### Task 1: Build the allowlisted source catalog

**Files:**
- Create: `scripts/dashboard_source_catalog.py`
- Create: `tests/test_dashboard_source_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

Create fixtures for a default PDF directory, canonical workbook metadata, one uploaded project manifest/search index, repository ground truth, and uploaded ground truth. Assert:

```python
catalog = build_source_catalog(root)
assert catalog["datasets"][0]["id"] == "dataset:wns-default"
assert catalog["datasets"][0]["document_count"] == 2
assert catalog["datasets"][0]["chunk_count"] == 9
assert catalog["datasets"][1]["id"] == "project:claims_20260714"
assert catalog["groundtruth"][0]["id"] == "groundtruth:none"
assert {g["row_count"] for g in catalog["groundtruth"][1:]} == {2, 3}
assert all("path" not in item for group in catalog.values() for item in group)
```

Add resolver assertions:

```python
resolved = resolve_dataset(root, "project:claims_20260714")
assert resolved.document_count == 1
assert resolved.document_path.is_relative_to(root)
with pytest.raises(ValueError):
    resolve_groundtruth(root, "groundtruth:repository:../../etc/passwd")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py -q`
Expected: FAIL because `scripts.dashboard_source_catalog` does not exist.

- [ ] **Step 3: Implement minimal catalog and resolvers**

Implement immutable records and public serializers:

```python
@dataclass(frozen=True)
class DatasetSource:
    id: str
    label: str
    kind: str
    document_count: int
    chunk_count: int
    document_path: Path
    workbook_path: Path | None = None

@dataclass(frozen=True)
class GroundtruthSource:
    id: str
    label: str
    kind: str
    row_count: int
    path: Path | None
    valid: bool = True
    error: str = ""
```

Expose `build_source_catalog(root)`, `resolve_dataset(root, source_id)`, and `resolve_groundtruth(root, source_id)`. Generate IDs from known roots only; never derive a path directly from an untrusted suffix. Public dictionaries omit all path fields.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_source_catalog.py tests/test_dashboard_source_catalog.py
git commit -m "feat: add dashboard source catalog"
```

### Task 2: Add ground-truth upload registration

**Files:**
- Modify: `scripts/dashboard_source_catalog.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `tests/test_dashboard_source_catalog.py`

- [ ] **Step 1: Write failing upload tests**

Test CSV registration and rejection:

```python
entry = register_groundtruth_upload(root, "claims.csv", b"query,ground_truth\nq1,a1\nq2,a2\n", "Claims GT")
assert entry["kind"] == "uploaded"
assert entry["row_count"] == 2
assert resolve_groundtruth(root, entry["id"]).path.exists()
with pytest.raises(ValueError, match="query column"):
    register_groundtruth_upload(root, "bad.csv", b"answer\na1\n", "Bad")
```

Assert the upload does not create a document project or search index.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py -q`
Expected: FAIL because registration is missing.

- [ ] **Step 3: Implement registration and route**

Store uploads under `data/user_groundtruth/<generated-id>/`, write a manifest with original filename, label, row count, detected columns, and relative stored path. Extend `/api/upload-dataset` to inspect multipart `upload_type`:

```python
if upload_type == "groundtruth":
    payload = register_groundtruth_upload(ROOT, filename, content, label)
else:
    payload = create_user_project_upload(filename, content, label)
```

Return `source_id` in both cases.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/dashboard_source_catalog.py scripts/serve_benchmark_dashboard.py tests/test_dashboard_source_catalog.py
git commit -m "feat: register reusable groundtruth uploads"
```

### Task 3: Wire standard and NVIDIA run dispatch to source IDs

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `scripts/run_complete_pipeline.py`
- Modify: `tests/test_dashboard_source_catalog.py`
- Create: `tests/test_complete_pipeline_sources.py`

- [ ] **Step 1: Write failing command-dispatch tests**

Assert the default source and uploaded project resolve into commands without browser paths:

```python
cmd = complete_pipeline_cmd({
    "dataset_id": ["dataset:wns-default"],
    "groundtruth_id": ["groundtruth:repository:groundtruth_500.csv"],
})
assert "--groundtruth" in cmd
assert "data/groundtruth/groundtruth_500.csv" in cmd
assert "--workbook" in cmd
```

Add runner tests with `subprocess` mocked. Assert an uploaded dataset passes its resolved `chunks/chunking_methods_output_v2.xlsx` to `run_long_db_ingestion.py`. Assert `groundtruth:none` plus typed queries invokes ingestion and retrieval only, and never evaluation, reranker, or lift analysis. Assert NVIDIA ingestion receives the selected dataset directory and NVIDIA benchmark rejects `groundtruth:none`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py tests/test_complete_pipeline_sources.py -q`
Expected: FAIL because command builders and runner source arguments are missing.

- [ ] **Step 3: Implement source-aware dispatch**

Add `source_catalog` to `/api/results`. Resolve IDs before launching jobs. Extend `run_complete_pipeline.py` with `--workbook`, `--benchmark-input`, `--pdf-audit`, `--evidence-only`, and repeatable `--query` arguments. Default these paths to existing WNS artifacts for backward compatibility. Pass the selected workbook to `run_long_db_ingestion.py`. In evidence-only mode run ingestion and `run_retrieval_smoke_from_vm_dbs.py --queries ...`, then stop before evaluation/reranker stages.

Use explicit execution modes:

```python
mode = "evaluated" if gt.path else "evidence_only"
if mode == "evidence_only":
    return launch_complete_pipeline(dataset, None, typed_queries, selections)
return launch_complete_pipeline(dataset, gt, [], selections)
```

Only allow evidence-only execution when at least one typed query is present. For NVIDIA, resolve dataset directory for ingestion and ground-truth path for benchmarking. Reject missing/unknown IDs with HTTP 400 and a concrete message.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_dashboard_source_catalog.py tests/test_complete_pipeline_sources.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/serve_benchmark_dashboard.py scripts/run_complete_pipeline.py tests/test_dashboard_source_catalog.py tests/test_complete_pipeline_sources.py
git commit -m "feat: dispatch pipeline runs from selected sources"
```

### Task 4: Render selectors and enforce evidence-only UI semantics

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/style.css`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing frontend harness test**

Use the existing Node VM harness with a catalog fixture. Assert:

```javascript
renderSourceSelectors(catalog);
if (!document.getElementById('runDataset').innerHTML.includes('2 documents · 9 chunks')) process.exit(1);
document.getElementById('runGroundtruth').value = 'groundtruth:none';
syncRunMode();
if (!document.getElementById('runModeHint').textContent.includes('Evidence-only')) process.exit(1);
if (!document.getElementById('runCompletePipelineBtn').textContent.includes('Run evidence retrieval')) process.exit(1);
if (document.getElementById('qualitySection').dataset.runMetricsVisible === 'true') process.exit(1);
```

Also assert upload form sends `upload_type` and both standard/NVIDIA requests send IDs, not path input values.

- [ ] **Step 2: Run test and verify RED**

Run: `python3 -m pytest tests/test_dashboard_metrics.py -q`
Expected: FAIL because selectors/functions do not exist.

- [ ] **Step 3: Implement HTML and JavaScript**

Add these controls:

```html
<label>Documents / dataset<select id="runDataset"></select><small id="runDatasetMeta"></small></label>
<label>Ground truth<select id="runGroundtruth"></select><small id="runModeHint"></small></label>
```

Mirror them as `nvidiaDataset` and `nvidiaGroundtruth`. Add upload type selection. Populate options from `payload.source_catalog`. Keep document/chunk/row counts in visible labels. Update button copy and disable only scored actions when `groundtruth:none` is active. Do not hide historical evaluated results globally; suppress quality metrics only in the current run-result panel.

- [ ] **Step 4: Style within the existing design system**

Use current panel, label, select, `mini-stat`, muted text, border, and accent tokens. Add no new dependencies, gradients, or decorative cards.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
node --check web/app.js
python3 -m pytest tests/test_dashboard_metrics.py -q
```

Expected: both commands pass.

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/app.js web/style.css tests/test_dashboard_metrics.py
git commit -m "feat: add dataset and groundtruth selectors"
```

### Task 5: Render NVIDIA baseline and reranked evidence independently

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing 500/500 baseline test**

Create temporary `benchmark_baseline_report.json`, summary, and details without reranked artifacts. Assert the API reader returns:

```python
assert payload["baseline"]["report"]["evaluated_rows"] == 500
assert payload["baseline"]["report"]["successful_queries"] == 500
assert payload["reranked"]["report"] == {}
```

In the Node harness, call `renderNvidia(...)` and assert `Baseline / no reranker`, `500/500`, and baseline metric rows appear.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_dashboard_metrics.py -q`
Expected: FAIL because current rendering collapses result modes.

- [ ] **Step 3: Implement mode-separated payload and rendering**

Return `nvidia_rag.benchmark.baseline`, `nvidia_rag.benchmark.reranked`, and compatibility `latest`. Add a result-mode selector when both modes exist; default to reranked when valid, otherwise baseline. Render cards/table from the selected mode and label partial runs with successful/failed counts.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_dashboard_metrics.py -q`
Expected: PASS, including the 500/500 baseline-only fixture.

- [ ] **Step 5: Commit**

```bash
git add scripts/serve_benchmark_dashboard.py web/index.html web/app.js tests/test_dashboard_metrics.py
git commit -m "feat: show NVIDIA benchmark modes separately"
```

### Task 6: Document and verify the complete feature

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document operator behavior**

Describe global independent selectors, upload types, evaluated vs evidence-only behavior, NVIDIA source selection, and separate result modes. State that arbitrary data/ground-truth pairings are allowed but compatibility is not inferred.

- [ ] **Step 2: Run complete automated verification**

Run:

```bash
python3 -m py_compile scripts/dashboard_source_catalog.py scripts/serve_benchmark_dashboard.py
node --check web/app.js
python3 -m pytest tests/test_dashboard_source_catalog.py tests/test_dashboard_metrics.py tests/test_modular_benchmark.py -q
python3 scripts/benchmark_cli.py validate configs/benchmark.local.json
```

Expected: zero failures and official matrix count 180.

- [ ] **Step 3: Run live dashboard verification**

Start the dashboard on a free local port, request `/api/results`, and assert:

```python
assert payload["source_catalog"]["datasets"]
assert payload["source_catalog"]["groundtruth"][0]["id"] == "groundtruth:none"
assert "baseline" in payload["nvidia_rag"]["benchmark"]
```

Capture Run Pipeline and NVIDIA desktop screenshots, inspect them, and check the browser console for errors.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: explain selectable pipeline sources"
```

- [ ] **Step 5: Prepare VM handoff**

Create a narrow patch or archive containing committed source/test/docs changes only. Verify archive listing/integrity and provide VM commands to apply, restart port 5011, request `/api/results`, and inspect both pages.
