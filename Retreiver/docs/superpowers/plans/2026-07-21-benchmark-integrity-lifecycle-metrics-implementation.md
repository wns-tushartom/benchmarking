# Benchmark Integrity, Lifecycle, and Metrics UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the WNS dashboard rank only complete evaluation runs, bind each uploaded dataset to its own validated ground truth, expose a durable setup lifecycle, and render an honest metrics trade-off/heat-map experience.

**Architecture:** Keep `scripts/serve_benchmark_dashboard.py` as the single backend source for cataloguing artifacts, dataset manifests, persisted jobs, and API responses. The static `web/` client renders the returned state without inventing scores, using inline SVG and DOM tables only. `tests/` exercises the Python contracts and runs browser-independent Node VM assertions against the actual client code.

**Tech Stack:** Python 3 standard library, existing `pypdf`/`pandas` optional upload helpers, static HTML/CSS/JavaScript, Node for frontend assertions, Chromium/Playwright for final visual QA.

---

## File map

| File | Responsibility |
|---|---|
| `scripts/serve_benchmark_dashboard.py` | Canonical pipeline state, artifact selection, dataset/ground-truth validation, persisted setup jobs, API routes and run gating. |
| `scripts/run_long_db_ingestion.py` | Existing real embedding and vector-index runner, extended with project-isolated artifact/cache/store destinations. |
| `web/index.html` | Dataset selector and immutable ground-truth status, setup-status surface, Metrics controls, heat-map/scatter targets. |
| `web/app.js` | State-aware rendering, fixed dataset binding, run gating, SVG scatter, heat-map and no-data states. |
| `web/styles.css` | Compact dark dashboard layouts, state badges, metric control strip, responsive plot/table behavior. |
| `tests/test_dashboard_integrity.py` | Pure backend state selection and complete-row gating tests. |
| `tests/test_dashboard_dataset_lifecycle.py` | Dataset-ground-truth and persisted setup job tests. |
| `tests/test_dashboard_metrics_ux.py` | Node VM assertions for metric controls, evaluation-only charts, and not-run heat-map cells. |

The existing `tests/test_dashboard_metrics.py` remains the regression suite for legacy artifact/evidence behavior. Do not rewrite it merely to make the new suite pass.

## Shared test command

The repository does not require pytest for these test files. Run the direct assertion suites from the repo root:

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics_ux.py
```

Expected after all tasks: each command exits `0` with no output.

---

### Task 1: Canonical pipeline state and complete-run gating

**Files:**
- Create: `tests/test_dashboard_integrity.py`
- Modify: `scripts/serve_benchmark_dashboard.py:335-422, 751-770, 1141-1260`

- [ ] **Step 1: Write the failing backend tests**

Create `tests/test_dashboard_integrity.py` with these executable behavior tests. They use real dashboard helpers and a temporary artifact tree, not mock implementations.

```python
import csv
import tempfile
from pathlib import Path


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "chunker": "fixed_tok1200_ov150",
        "embedding": "gte_multilingual_base",
        "vector_store": "Qdrant",
        "reranker": "none",
        "query_count": "500",
        "recall_at_5": "0.80",
        "mrr": "0.70",
        "ndcg_at_5": "0.75",
        "avg_latency_ms": "120",
    }
    row.update(overrides)
    return row


def test_pipeline_state_keeps_latest_complete_when_newer_retry_is_incomplete():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old = dashboard.MODULAR_DIR
        dashboard.MODULAR_DIR = root / "runs" / "latest"
        try:
            _csv(root / "runs" / "20260720_complete" / "modular_summary.csv", [_row(created_at="2026-07-20T10:00:00")])
            _csv(root / "runs" / "20260721_retry" / "modular_summary.csv", [_row(created_at="2026-07-21T10:00:00", ndcg_at_5="")])
            states = dashboard.pipeline_state_rows()
        finally:
            dashboard.MODULAR_DIR = old

    state = next(row for row in states if row["key"].endswith("|none"))
    assert state["state"] == "complete"
    assert state["artifact"].endswith("20260720_complete/modular_summary.csv")
    assert state["candidate_count"] == 2


def test_pipeline_state_marks_missing_metric_row_incomplete_and_excludes_it_from_ranked_rows():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        old = dashboard.MODULAR_DIR
        dashboard.MODULAR_DIR = root / "runs" / "latest"
        try:
            _csv(root / "runs" / "latest" / "modular_summary.csv", [_row(ndcg_at_5="")])
            states = dashboard.pipeline_state_rows()
        finally:
            dashboard.MODULAR_DIR = old

    assert next(row for row in states if row["key"].endswith("|none"))["state"] == "incomplete"
    assert dashboard.complete_pipeline_metric_rows(states) == []


def test_pipeline_state_lists_configured_combination_with_no_artifact_as_not_run():
    import scripts.serve_benchmark_dashboard as dashboard

    old = dashboard.official_matrix_keys
    dashboard.official_matrix_keys = lambda: {("c", "e", "Qdrant", "none")}
    try:
        states = dashboard.pipeline_state_rows(sources=[])
    finally:
        dashboard.official_matrix_keys = old

    assert states == [{
        "key": "c|e|Qdrant|none",
        "sheet": "c", "embedding": "e", "store": "Qdrant", "reranker": "none",
        "state": "not_run", "artifact": "", "source": "", "run_at": "",
        "evaluated_queries": "", "candidate_count": 0,
    }]
```

- [ ] **Step 2: Run the new test file and confirm RED**

Run:

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
```

Expected: failure because `pipeline_state_rows` and `complete_pipeline_metric_rows` do not exist yet.

- [ ] **Step 3: Add the minimal canonical state helpers**

In `scripts/serve_benchmark_dashboard.py`, directly after `canonical_reranker_name`, add constants and helpers with these signatures:

```python
REQUIRED_EVALUATION_FIELDS = (
    "recall_at_5", "mrr", "ndcg_at_5", "avg_latency_seconds", "evaluated_queries",
)


def pipeline_key(row: dict[str, Any]) -> str:
    return "|".join([
        str(row.get("sheet") or ""), str(row.get("embedding") or ""),
        str(row.get("store") or ""), canonical_reranker_name(row.get("reranker") or "none"),
    ])


def numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def complete_metric_row(row: dict[str, Any]) -> bool:
    return all(numeric_value(row.get(field)) is not None for field in REQUIRED_EVALUATION_FIELDS)


def complete_pipeline_metric_rows(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sort_summary([row for row in states if row.get("state") == "complete"])
```

Add `benchmark_state_sources()` beside `benchmark_reference_sources()`. It returns the same summary paths but preserves a source artifact path and uses `path.stat().st_mtime_ns` as the fallback sort value when a row has no ISO timestamp.

Add `pipeline_state_rows(sources: list[tuple[str, Path]] | None = None)`. It must:

1. Start with every `official_matrix_keys()` tuple as a `not_run` state row.
2. Normalize rows from each artifact with `normalized_benchmark_row`.
3. Mark candidate rows `complete` only when `complete_metric_row(normalized)` is true, otherwise `incomplete`.
4. Retain `artifact`, `source`, `run_at`, and `candidate_count`.
5. For a duplicate key, choose the latest *complete* candidate. If there is no complete candidate, choose the latest candidate and retain `incomplete`.
6. Return rows sorted by `key`, so API and tests are deterministic.

Update `read_benchmark_reference()` to source its `summary` from `complete_pipeline_metric_rows(pipeline_state_rows())`. Keep its existing report keys, and add `complete_rows`, `incomplete_rows`, and `not_run_rows` counts.

- [ ] **Step 4: Run RED-to-GREEN verification**

Run:

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
```

Expected: both exit `0`. If the legacy test needs a query count to remain a complete row, update only its fixture helper to include `query_count: "500"`, not production completeness rules.

- [ ] **Step 5: Publish state in the real response without breaking existing keys**

In `/api/results`, calculate once:

```python
pipeline_state = pipeline_state_rows()
```

Add this additive field under `operational`:

```python
"pipeline_state": pipeline_state,
```

Keep `operational.evaluation` and every existing response key intact for compatibility.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/serve_benchmark_dashboard.py tests/test_dashboard_integrity.py tests/test_dashboard_metrics.py
git commit -m "feat: gate benchmark rankings on complete pipeline runs"
```

---

### Task 2: Dataset-ground-truth binding and evidence-only mode

**Files:**
- Create: `tests/test_dashboard_dataset_lifecycle.py`
- Modify: `scripts/serve_benchmark_dashboard.py:800-1096, 1264-1295`
- Modify: `web/index.html:154-198, 310-344`
- Modify: `web/app.js:823-905, 1064-1079`

- [ ] **Step 1: Write failing manifest and query-mode tests**

Create `tests/test_dashboard_dataset_lifecycle.py` with the following tests:

```python
import json
import tempfile
from pathlib import Path


def test_groundtruth_validation_requires_question_and_reference_column():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        valid = Path(td) / "valid.csv"
        valid.write_text("question,ground truth,context\nHow do I rebook?,Rebook from Manage My Booking,policy text\n", encoding="utf-8")
        invalid = Path(td) / "invalid.csv"
        invalid.write_text("question,category\nHow do I rebook?,travel\n", encoding="utf-8")
        assert dashboard.validate_groundtruth_file(valid)["valid"] is True
        result = dashboard.validate_groundtruth_file(invalid)

    assert result["valid"] is False
    assert "reference" in result["reason"].lower()


def test_project_manifest_binds_its_uploaded_groundtruth_and_never_accepts_other_project_path():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            bound = dashboard.attach_project_groundtruth(project["project_id"], "gt.csv", b"question,ground truth,context\nrefund?,refund,policy\n")
            manifest = json.loads((dashboard.project_root(project["project_id"]) / "manifest.json").read_text(encoding="utf-8"))
        finally:
            dashboard.USER_PROJECTS_DIR = old

    assert bound["mode"] == "evaluated"
    assert manifest["groundtruth"]["valid"] is True
    assert manifest["groundtruth"]["path"].endswith("/questions/gt.csv")


def test_project_query_stays_evidence_only_without_valid_bound_groundtruth():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Customers can request a refund.", "Refund")
            payload = dashboard.query_user_project(project["project_id"], "refund")
        finally:
            dashboard.USER_PROJECTS_DIR = old

    assert payload["mode"] == "evidence_only"
    assert payload["recommendation_available"] is False
```

- [ ] **Step 2: Run the new test file and confirm RED**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
```

Expected: failure because validation and attach helpers do not exist.

- [ ] **Step 3: Implement safe, immutable dataset linkage**

Add near `write_project_manifest`:

```python
GROUNDTRUTH_QUERY_COLUMNS = {"question", "query"}
GROUNDTRUTH_REFERENCE_COLUMNS = {
    "ground truth", "ground_truth", "context", "paragraph", "expected_text",
    "answer", "relevant_text",
}


def validate_groundtruth_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        columns = {str(name or "").strip().lower() for name in (csv.DictReader(f).fieldnames or [])}
    if not columns & GROUNDTRUTH_QUERY_COLUMNS:
        return {"valid": False, "reason": "Ground-truth file needs a question or query column.", "columns": sorted(columns)}
    if not columns & GROUNDTRUTH_REFERENCE_COLUMNS:
        return {"valid": False, "reason": "Ground-truth file needs a reference text column such as ground truth or context.", "columns": sorted(columns)}
    return {"valid": True, "reason": "", "columns": sorted(columns)}
```

Implement `attach_project_groundtruth(project_id, original_name, content)`. It must allow `.csv` only in this first slice, write the file under the already selected project’s `questions/` directory, validate it, update only that project’s `manifest.json`, and return `{project_id, groundtruth, mode}`. Never accept a free-form project path or global ground-truth path from the browser.

Make new project manifests include:

```python
"groundtruth": {"path": "", "valid": False, "reason": "No ground truth attached"},
"mode": "evidence_only",
"setup": {"state": "not_started", "job_id": "", "stages": []},
```

In `query_user_project`, load the project manifest and derive `mode` from `manifest["groundtruth"]["valid"]`. It may still return retrieval evidence in both modes, but it must always set `recommendation_available` to `False` until an actual evaluated result artifact exists. Do not claim scored mode merely because a CSV was attached.

- [ ] **Step 4: Add the optional ground-truth upload field and API route**

In `web/index.html`, place an optional file input beside the dataset input:

```html
<label>Ground-truth file, optional
  <input id="datasetGroundtruthFile" name="groundtruth_file" type="file" accept=".csv" />
  <small>Must contain question/query plus ground-truth/context/reference text.</small>
</label>
```

In `uploadDataset`, append it only when selected:

```javascript
const groundtruthFile = $('datasetGroundtruthFile')?.files?.[0];
if (groundtruthFile) form.append('groundtruth_file', groundtruthFile);
```

In `Handler.do_POST` for `/api/upload-dataset`, read `groundtruth_file` when present, call `attach_project_groundtruth` after creating the project, and return the refreshed manifest. Add a narrow `POST /api/project-groundtruth` route for attaching a ground-truth file to an existing `project_id`; it must accept multipart data only and call the same helper.

In `renderUserProjects`, render each option with its real mode, for example `Refund · 12 chunks · evidence only` or `Refund · 12 chunks · ground truth linked`. Add a `change` listener that calls `syncDatasetBinding()`.

`syncDatasetBinding()` must read the selected project from `state.operational.user_projects`, populate a non-editable `#runGroundtruthStatus` value for the run page, disable `#runCompletePipelineBtn` when mode is evidence-only, and show the exact reason. Remove the editable `#runGroundtruth` input from the primary run form. Advanced scripts may retain their own explicit file input outside this product flow.

- [ ] **Step 5: Verify GREEN and regression suite**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
```

Expected: all exit `0`.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/serve_benchmark_dashboard.py web/index.html web/app.js tests/test_dashboard_dataset_lifecycle.py
git commit -m "feat: bind datasets to validated ground truth"
```

---

### Task 3: Persisted background setup lifecycle and execution gating

**Files:**
- Modify: `tests/test_dashboard_dataset_lifecycle.py`
- Modify: `scripts/serve_benchmark_dashboard.py:47-48, 697-748, 946-995, 1141-1260, 1296-1430`
- Modify: `scripts/run_long_db_ingestion.py:274-397`
- Modify: `web/index.html:154-198, 310-344`
- Modify: `web/app.js:908-1037, 1138-1187`

- [ ] **Step 1: Write the failing setup-job test**

Append this test to `tests/test_dashboard_dataset_lifecycle.py`:

```python
def test_setup_job_is_persisted_with_project_matrix_and_stage_provenance():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old_projects, old_jobs = dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR
        dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = Path(td) / "projects", Path(td) / "jobs"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            job = dashboard.create_project_setup_job(
                project["project_id"],
                {"chunkers": ["fixed_tok1200_ov150"], "embeddings": ["gte_multilingual_base"], "vector_stores": ["FAISS"]},
            )
            stored = json.loads((dashboard.JOB_DIR / f"{job['job_id']}.json").read_text(encoding="utf-8"))
            manifest = json.loads((dashboard.project_root(project["project_id"]) / "manifest.json").read_text(encoding="utf-8"))
        finally:
            dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = old_projects, old_jobs

    assert stored["kind"] == "dataset_setup"
    assert stored["project_id"] == project["project_id"]
    assert stored["matrix"]["vector_stores"] == ["FAISS"]
    assert [stage["name"] for stage in stored["stages"]] == ["extract", "chunk", "embed", "index"]
    assert manifest["setup"]["job_id"] == job["job_id"]
```

- [ ] **Step 2: Run it and confirm RED**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
```

Expected: failure because `create_project_setup_job` does not exist.

- [ ] **Step 3: Implement the persisted job record before launching work**

Add these helpers near the existing `launch_job` functions:

```python
SETUP_STAGE_NAMES = ("extract", "chunk", "embed", "index")


def dashboard_job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def write_dashboard_job(record: dict[str, Any]) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_job_path(record["job_id"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
```

Implement `create_project_setup_job(project_id, matrix)`. It creates a unique job id, validates the project, expands only supplied matrix values, writes a record with `kind: "dataset_setup"`, `state: "queued"`, `created_at`, `project_id`, `matrix`, `artifact_root`, and four ordered stage records with `{name, state: "queued", started_at: "", finished_at: "", output: "", error: ""}`. It updates the same project’s manifest `setup` object and writes it immediately.

Refactor `launch_job` and `job_status` so their in-memory object is supplemented by the persisted JSON record. Existing complete pipeline jobs continue to work, but all new job responses include `kind`, `state`, `project_id`, `matrix`, and `stages` where those values exist. On process completion, update `state` to `complete` for exit code 0 or `failed` otherwise, set `finished_at`, and write the record.

Extend `scripts/run_long_db_ingestion.py` before calling it. Add these arguments:

```python
parser.add_argument("--output-dir", default="")
parser.add_argument("--cache-dir", default="")
parser.add_argument("--collection-prefix", default="wns")
parser.add_argument("--faiss-index-root", default="")
```

Resolve non-empty values relative to `ROOT`; otherwise retain existing `data/db_ingestion_runs`, `data/embedding_cache`, and `data/faiss_indexes` defaults. Change `make_store` and `upsert_and_check` to accept `collection_prefix` and `faiss_index_root`, then construct names as follows:

```python
prefix = f"{safe_name(collection_prefix)}_{safe_name(sheet)[:24]}_{safe_name(embedding)[:16]}"
index_dir = faiss_index_root / f"{safe_name(sheet)}_{safe_name(embedding)}"
```

Pass the selected cache/output/prefix/root values through the main ingestion loop. This prevents a user project from reusing the official WNS embedding cache, FAISS index, Qdrant collection, PGVector table, or Weaviate class.

Add a `project_setup_cmd(record)` helper in the dashboard with this exact argument shape:

```python
project_id = record["project_id"]
root = project_root(project_id)
matrix = record["matrix"]
return [
    sys.executable, "scripts/run_long_db_ingestion.py",
    "--workbook", str((root / "chunks" / "chunking_methods_output_v2.xlsx").relative_to(ROOT)),
    "--output-dir", str((root / "runs" / record["job_id"] / "ingestion").relative_to(ROOT)),
    "--cache-dir", str((root / "vector_indexes" / "embedding_cache").relative_to(ROOT)),
    "--collection-prefix", f"project_{safe_label(project_id)}",
    "--faiss-index-root", str((root / "vector_indexes" / "faiss").relative_to(ROOT)),
    "--run-id", record["job_id"], "--skip-existing-store-success",
    "--sheets", *matrix["chunkers"],
    "--embeddings", *matrix["embeddings"],
    "--stores", *matrix["vector_stores"],
]
```

`create_project_setup_job` must set `extract` and `chunk` to `complete` only when the project already has `search_index.json` and the generated workbook. It starts `embed` and `index` as `queued`. `job_status` parses the real runner output: `embedded model=` completes `embed`; `OK sheet=` advances `index`; process exit `0` completes `index` and the job; any `embedding_failed`, `store_failed`, or non-zero exit marks the currently active stage and job as `failed`, retaining the runner error/output. No stage may be marked complete before its artifact or real runner line exists.

- [ ] **Step 4: Add setup routes and gate the run endpoint**

Add these endpoints:

```text
POST /api/project-setup?project_id=<id>&chunker=<...>&embedding=<...>&store=<...>
GET  /api/project-setup/status?job_id=<id>
```

`/api/project-setup` calls `create_project_setup_job`, then launches `project_setup_cmd` with the saved job metadata. It must return the persisted record plus live status.

In `/api/run/preflight-complete-pipeline` and `/api/run/complete-pipeline`, accept `project_id` only as an identifier, resolve its bound ground truth internally, and return HTTP 400 with a structured `{error, mode, reason}` when its manifest has no valid ground truth or setup has not completed. Never trust a client-provided ground-truth path for an uploaded project.

For the official WNS dataset, preserve existing CLI behavior. The gating applies to selected user projects only.

- [ ] **Step 5: Render the lifecycle in the Documents and Run pages**

Add these stable targets to `web/index.html`:

```html
<div id="datasetSetupStatus" class="dataset-setup-status"></div>
<button id="projectSetupBtn" type="button">Prepare selected dataset</button>
<div id="runGroundtruthStatus" class="bound-groundtruth-status"></div>
```

In `web/app.js`, add `selectedProject()`, `syncDatasetBinding()`, `renderDatasetSetup(project)`, `startProjectSetup()`, and `pollProjectSetup(jobId)`. These functions must:

- show extract, chunk, embed, and index stage state from persisted data;
- poll only while `running` is true;
- disable evaluation action until the selected project has valid ground truth and `setup.state === "complete"`;
- describe the project as evidence-only instead of showing disabled metrics without explanation;
- include `project_id` in `selectedParams()` only for a selected user project.

- [ ] **Step 6: Run GREEN verification**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
```

Expected: all exit `0`.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/serve_benchmark_dashboard.py scripts/run_long_db_ingestion.py web/index.html web/app.js tests/test_dashboard_dataset_lifecycle.py
git commit -m "feat: persist dataset setup lifecycle and run gating"
```

---

### Task 4: Evaluation-only trade-off scatter and configuration heat-map

**Files:**
- Create: `tests/test_dashboard_metrics_ux.py`
- Modify: `web/index.html:106-152`
- Modify: `web/app.js:603-815, 944-1002, 1236-1264`
- Modify: `web/styles.css`

- [ ] **Step 1: Write a failing frontend behavior test**

Create `tests/test_dashboard_metrics_ux.py`. It must load the real `web/app.js` into the same minimal Node VM pattern used in `tests/test_dashboard_metrics.py`, then assert these outcomes:

```python
import subprocess
from pathlib import Path


def test_metrics_visuals_use_complete_rows_and_render_not_run_cells():
    app = Path(__file__).resolve().parents[1] / "web" / "app.js"
    js = f"""
const fs=require('fs'), vm=require('vm');
const elements={{}};
function el(id){{return elements[id] ||= {{id,value:'all',textContent:'',innerHTML:'',classList:{{toggle(){{}}}},addEventListener(){{}},querySelectorAll(){{return [];}},setAttribute(){{}}}};}}
const context={{console,document:{{getElementById:el,querySelectorAll(){{return [];}},addEventListener(){{}},body:{{insertAdjacentHTML(){{}}}}}},window:{{}},location:{{hash:'#quality'}},history:{{replaceState(){{}}}},fetch:async()=>({{ok:true,json:async()=>({{}})}}),setTimeout(){{}}}};
vm.createContext(context);
const code=fs.readFileSync({str(app)!r},'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(code, context);
const stateRows=[
  {{key:'c1|e1|Qdrant|none',sheet:'c1',embedding:'e1',store:'Qdrant',reranker:'none',state:'complete',evaluated_queries:'10',recall_at_5:'0.8',mrr:'0.7',ndcg_at_5:'0.75',avg_latency_seconds:'0.2'}},
  {{key:'c1|e1|FAISS|none',sheet:'c1',embedding:'e1',store:'FAISS',reranker:'none',state:'not_run',artifact:'',source:'',run_at:'',evaluated_queries:''}},
];
context.renderMetricExplorer({{pipeline_state:stateRows}}, {{chunkers:['c1'],vector_stores:['Qdrant','FAISS']}});
if (!el('tradeoffChart').innerHTML.includes('circle') || !el('metricHeatmap').innerHTML.includes('not run') || el('metricExplorerHint').textContent.includes('2 complete')) process.exit(1);
"""
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
```

Add a second Node assertion in the same test that passes only `not_run` and `incomplete` rows and expects visible text `Metrics unavailable until ground truth evaluation completes`, no `<circle`, and no winning-method text.

- [ ] **Step 2: Run it and confirm RED**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics_ux.py
```

Expected: failure because `renderMetricExplorer` and its DOM targets do not exist.

- [ ] **Step 3: Add semantic markup and fixed controls**

Replace the current top-10 chart section in `web/index.html` with an additional panel that has these IDs and labels:

```html
<section class="panel metric-explorer-panel">
  <div class="section-head">
    <div><h2>Metric trade-off</h2><p>Compare only complete ground-truth evaluations. Left is faster for latency; higher is better for quality metrics.</p></div>
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
<section class="panel metric-heatmap-panel">
  <div class="section-head"><div><h2>Configuration heat-map</h2><p>Each cell is one configured chunker × vector DB combination for the selected embedding and reranker. Empty means not run, never zero quality.</p></div></div>
  <div class="metric-control-strip">
    <label>Metric<select id="heatmapMetric"></select></label>
    <label>Embedding<select id="heatmapEmbedding"></select></label>
    <label>Reranker<select id="heatmapReranker"></select></label>
  </div>
  <div id="metricHeatmap" class="metric-heatmap" role="region" aria-label="Configuration metric heat-map"></div>
</section>
```

The options are fixed strings: `score`, `recall_at_5`, `mrr`, `ndcg_at_5`, `avg_latency_seconds` for metrics; component choices `sheet`, `embedding`, `store`, `reranker` for color; `none` and `evaluated_queries` for bubble size.

- [ ] **Step 4: Implement pure state filtering and SVG rendering**

In `web/app.js`, add these pure helpers before rendering functions:

```javascript
function completeStateRows(pipelineState = []) {
  return pipelineState.filter(row => row.state === 'complete' && Number(row.evaluated_queries) > 0);
}

function metricValue(row, key) {
  return key === 'score' ? metricScore(row) : num(row[key]);
}

function metricLabel(key) {
  return {{score:'Score', recall_at_5:'Recall@5', mrr:'MRR', ndcg_at_5:'nDCG@5', avg_latency_seconds:'Average latency/query'}}[key] || key;
}
```

Implement `renderMetricExplorer(operational, options)`. It reads `operational.pipeline_state`, fills selectors without overwriting current valid user choices, and uses only `completeStateRows` for KPI values and SVG dots. The scatter must:

- draw a labelled horizontal and vertical axis;
- map metrics to a fixed chart box, with safe `max - min || 1` divisions;
- use color class/data attribute determined by the selected component;
- use `evaluated_queries` for bubble area only when chosen;
- include a text legend/table label so it remains meaningful without hover;
- render the explicit no-data state from Step 1 when no complete rows exist.

Implement `renderMetricHeatmap(operational, options)`. It renders all configured `options.chunkers × options.vector_stores` cells for the selected embedding and reranker. Map matching complete rows to a numeric value and class bucket. Render state labels `not run`, `incomplete`, `running`, or `failed` when no complete value exists. A zero metric remains `0.000`, never `not run`.

Call both functions from `renderOperational()` after `renderEvaluation` and `renderPipelineComparison`. Add `input` listeners for all seven controls to re-render only the metric explorer, not refetch data.

Update `renderEvaluation`, `renderBestMethods`, and `renderPipelineComparison` to source complete rows from `state.operational.pipeline_state` when present. Retain `evaluatedRows()` only as the compatibility fallback for legacy API payload tests.

- [ ] **Step 5: Add responsive styles without changing the established visual language**

In `web/styles.css`, add styles scoped to:

```css
.metric-control-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric-explorer-kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric-heatmap { overflow:auto; }
.metric-heatmap-grid { min-width:680px; display:grid; }
.metric-heatmap-cell[data-state="not_run"] { opacity:.65; }
.metric-heatmap-cell[data-state="incomplete"], .metric-heatmap-cell[data-state="failed"] { border-color:var(--risk-rose, #fda4af); }
```

Use the colors already declared by `DESIGN.md`: cyan for selected/quality, mint for complete, amber for incomplete, rose for failure. Add a mobile media rule below 768px that changes the control and KPI grids to one column and keeps the heat-map in a clearly labelled horizontal scroll container. Do not add external fonts, images, cards full of fake numbers, or new animation dependencies.

- [ ] **Step 6: Verify GREEN and existing regression behavior**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics_ux.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
```

Expected: all exit `0`.

- [ ] **Step 7: Commit Task 4**

```bash
git add web/index.html web/app.js web/styles.css tests/test_dashboard_metrics_ux.py
git commit -m "feat: add complete-run metric tradeoff and heatmap"
```

---

### Task 5: Real API and browser verification

**Files:**
- Modify only if a verified defect is found: `scripts/serve_benchmark_dashboard.py`, `web/index.html`, `web/app.js`, `web/styles.css`, or the test that exposes it.

- [ ] **Step 1: Run all deterministic suites**

```bash
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_integrity.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_dataset_lifecycle.py
PYTHONPATH="$PWD" python3 tests/test_dashboard_metrics_ux.py
```

Expected: four zero exit codes.

- [ ] **Step 2: Start the intended dashboard process and probe the API**

```bash
python3 scripts/serve_benchmark_dashboard.py 5011 127.0.0.1
```

Run it as a tracked long-lived background process. In a separate shell, prove the correct service and new response contract:

```bash
curl -sS http://127.0.0.1:5011/api/results | python3 -c 'import json,sys; p=json.load(sys.stdin); op=p["operational"]; print(bool(op["pipeline_state"]), len(op["user_projects"]), p["options"]["matrix_count"])'
```

Expected shape: `True <non-negative project count> 180`.

- [ ] **Step 3: Capture desktop and mobile browser evidence**

Use Chromium/Playwright against `http://127.0.0.1:5011/`. Before screenshots, assert the document title is `Project Smiley Integrated RAG Pipeline`.

Capture:

```text
artifacts/dashboard-desktop.png   viewport 1440 × 1100, Metrics page
artifacts/dashboard-mobile.png    viewport 390 × 844, Metrics page
```

Verify visually:

- evidence-only projects show no winner, no fake scatter points, and an explanation;
- complete rows are the only plotted/ranked rows;
- heat-map has labelled not-run cells rather than zero metrics;
- no horizontal clipping except the explicit heat-map scroll container on mobile;
- Run Pipeline shows the dataset-bound ground-truth status and setup stage state;
- console has no JavaScript errors or failed API requests.

- [ ] **Step 4: Fix only observed defects and rerun the specific regression test first**

For every browser/API defect, write or extend one focused failing test, demonstrate failure, patch minimal source, rerun the focused test, then rerun all four suites and take replacement screenshots.

- [ ] **Step 5: Commit final verified work**

```bash
git add scripts/serve_benchmark_dashboard.py web/index.html web/app.js web/styles.css tests
git commit -m "test: verify benchmark integrity dashboard flow"
git status --short
```

Expected: clean working tree.

## Plan self-review

- **Spec coverage:** Task 1 implements canonical keys, complete/incomplete/not-run state, provenance, and safe winner selection. Task 2 binds data to validated ground truth and preserves evidence-only operation. Task 3 persists setup provenance and prevents user-project evaluation before setup/ground truth. Task 4 implements the approved scatter/heat-map UX against complete rows only. Task 5 verifies live behavior and responsive UI.
- **Completeness scan:** checked for deferred-work markers and vague implementation instructions. The setup command is intentionally limited to the existing safe preparation path and is explicitly not a full-benchmark substitute.
- **Type consistency:** backend state uses `sheet`, `embedding`, `store`, `reranker`, `state`, `evaluated_queries`, `artifact`, and `run_at` throughout. Frontend reads those same keys and treats `pipeline_state` as authoritative with a legacy fallback only for old API payload tests.
