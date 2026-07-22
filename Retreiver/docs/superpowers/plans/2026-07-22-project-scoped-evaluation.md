# Project-Scoped Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a valid uploaded dataset run a selected local FAISS retrieval evaluation against its bound ground truth through the dashboard, with all inputs and outputs isolated under that project.

**Architecture:** Keep `run_complete_pipeline.py` official-only. Add a project evaluation runner which reads one validated project setup receipt, loads its persisted FAISS index, writes retrieval/evaluation artifacts only under `data/user_projects/<project>/runs/<evaluation-id>/`, and invokes the existing evaluator. The dashboard resolves all filesystem paths from the trusted `project_id`; the browser only sends the selected component tuple.

**Tech Stack:** Python stdlib, existing `FaissVectorStoreAdapter`, existing remote embedding adapter, FAISS, static `web/` dashboard, unittest, Playwright.

---

## File map

- Modify `scripts/run_retrieval_smoke_from_vm_dbs.py`: accept a trusted ingestion receipt and caller-owned output directory rather than only global discovery/output.
- Create `scripts/run_project_evaluation.py`: validate one project-local FAISS lane, retrieve every bound ground-truth query, score with the existing evaluator, persist an immutable project run manifest.
- Modify `scripts/evaluate_retrieval_groundtruth.py`: recognize the corpus `source_document` column as an expected document alias.
- Modify `scripts/serve_benchmark_dashboard.py`: resolve project-scoped commands and route project preflight/run requests without falling into global lanes.
- Modify `web/app.js`: route Run Pipeline preflight/run to project endpoints when a project is selected; retain official endpoint behavior for WNS.
- Modify `tests/test_dashboard_dataset_lifecycle.py`: API command-isolation and preflight tests.
- Create `tests/test_project_evaluation.py`: runner/retrieval/evaluator unit and isolation tests.

### Task 1: Make retrieval smoke project-addressable

**Files:**
- Modify: `scripts/run_retrieval_smoke_from_vm_dbs.py`
- Test: `tests/test_project_evaluation.py`

- [ ] **Step 1: Write failing tests**

```python
def test_retrieval_uses_explicit_project_receipt_and_output_dir(tmp_path):
    receipt = tmp_path / "project" / "runs" / "setup" / "ingestion" / "summary.csv"
    out_dir = tmp_path / "project" / "runs" / "evaluation" / "retrieval"
    # one ok FAISS row with a collection_or_table under tmp_path/project
    result = retrieval.rows_from_receipt(receipt, ["Heading_sections_l2"], ["gte_multilingual_base"], ["FAISS"])
    assert result[0]["collection_or_table"].startswith(str(tmp_path / "project"))
    assert retrieval.output_dir_for(out_dir) == out_dir
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `/tmp/wns-mineru-v3/bin/python -m unittest tests.test_project_evaluation.ProjectRetrievalTests.test_retrieval_uses_explicit_project_receipt_and_output_dir -v`

Expected: FAIL because `rows_from_receipt` and `output_dir_for` do not exist.

- [ ] **Step 3: Implement minimal retrieval inputs**

```python
parser.add_argument("--ingestion-summary", default="")
parser.add_argument("--out-dir", default="data/retrieval_smoke")
# If --ingestion-summary is set, read only that CSV. Otherwise retain latest_ok_rows().
out_dir = ROOT / args.out_dir
```

Use only receipt rows with `status == "ok"`, matching selected tuple. Do not broaden discovery to global summaries when a receipt is supplied.

- [ ] **Step 4: Run focused and existing retrieval tests**

Run: `/tmp/wns-mineru-v3/bin/python -m unittest tests.test_project_evaluation tests.test_modular_benchmark -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_retrieval_smoke_from_vm_dbs.py tests/test_project_evaluation.py
git commit -m "feat: support isolated retrieval receipts"
```

### Task 2: Add isolated project evaluator and document-aware ground truth

**Files:**
- Create: `scripts/run_project_evaluation.py`
- Modify: `scripts/evaluate_retrieval_groundtruth.py`
- Test: `tests/test_project_evaluation.py`

- [ ] **Step 1: Write failing tests**

```python
def test_source_document_is_used_as_expected_pdf():
    row = {"question": "Which paper?", "source_document": "paper.pdf"}
    assert evaluator.load_groundtruth_rows([row])[0]["expected_pdf"] == "paper.pdf"

def test_project_evaluation_command_never_uses_official_lanes(tmp_path):
    cmd = project_runner.build_commands(
        project_root=tmp_path / "project",
        run_id="eval-1",
        groundtruth=tmp_path / "project" / "questions" / "groundtruth.csv",
        ingestion_summary=tmp_path / "project" / "runs" / "setup" / "ingestion" / "summary.csv",
        sheet="Heading_sections_l2",
        embedding="gte_multilingual_base",
        store="FAISS",
        reranker="none",
        top_k=10,
    )
    rendered = " ".join(" ".join(part) for part in cmd)
    assert "data/db_ingestion_runs" not in rendered
    assert "data/retrieval_smoke" not in rendered
    assert "data/evaluation " not in rendered
    assert str(tmp_path / "project" / "runs" / "eval-1") in rendered
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `/tmp/wns-mineru-v3/bin/python -m unittest tests.test_project_evaluation.ProjectEvaluationTests -v`

Expected: FAIL because the project runner and `source_document` normalization do not exist.

- [ ] **Step 3: Implement the minimal runner**

`run_project_evaluation.py` must:

```python
# inputs: --project-root, --groundtruth, --ingestion-summary,
# --sheet, --embedding, --store, --reranker none, --top-k, --run-id, --preflight-only
# validate root-contained receipt/index/chunks and all selected fields
# outputs: <project>/runs/<run-id>/retrieval and <project>/runs/<run-id>/evaluation
# execute retrieval then evaluate_retrieval_groundtruth.py
# write <project>/runs/<run-id>/evaluation_manifest.json with state,
# component tuple, GT SHA-256, receipt, artifact dirs, expected/evaluated/missing query counts.
# state=complete iff evaluator has one summary row, evaluated_queries == GT rows,
# and missing_query_count == 0; otherwise incomplete/failed.
```

Add `source_document` to `PDF_COLS`, so citation-linked document relevance is used before text-overlap fallback.

- [ ] **Step 4: Verify runner with a synthetic local FAISS fixture and then its real project preflight**

Run:

```bash
/tmp/wns-mineru-v3/bin/python -m unittest tests.test_project_evaluation.ProjectEvaluationTests -v
/tmp/wns-mineru-v3/bin/python scripts/run_project_evaluation.py --preflight-only --project-root data/user_projects/public-five-paper-layout-e2e_20260722_022545_83de4f --groundtruth data/user_projects/public-five-paper-layout-e2e_20260722_022545_83de4f/questions/public_five_paper_groundtruth.csv --ingestion-summary data/user_projects/public-five-paper-layout-e2e_20260722_022545_83de4f/runs/20260722_030850_2c774ce8/ingestion/summary.csv --sheet Heading_sections_l2 --embedding gte_multilingual_base --store FAISS --reranker none --top-k 10
```

Expected: unit tests pass; real preflight reports only project-relevant readiness and does not reference global lanes.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_project_evaluation.py scripts/evaluate_retrieval_groundtruth.py tests/test_project_evaluation.py
git commit -m "feat: add isolated project ground-truth evaluation"
```

### Task 3: Route the dashboard safely

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `web/app.js`
- Test: `tests/test_dashboard_dataset_lifecycle.py`

- [ ] **Step 1: Write failing endpoint/command tests**

```python
def test_project_evaluation_command_is_project_local(tmp_path):
    cmd = dashboard.project_evaluation_cmd(project_id, selected)
    assert "scripts/run_project_evaluation.py" in cmd
    assert all("data/user_projects/" in value or value.startswith("scripts/") or value.startswith("--") or value in selected_values for value in cmd)
    assert "run_complete_pipeline.py" not in cmd

def test_official_complete_pipeline_command_is_unchanged():
    assert "scripts/run_complete_pipeline.py" in dashboard.complete_pipeline_cmd({"sheet": ["Heading_sections_l2"]})
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `/tmp/wns-mineru-v3/bin/python -m unittest tests.test_dashboard_dataset_lifecycle -v`

Expected: FAIL because project evaluation command routing does not exist.

- [ ] **Step 3: Implement server/UI routing**

```python
# server: `project_evaluation_cmd(project_id, qs, preflight=False)` resolves manifest,
# bound GT, latest successful matching setup receipt, and project root before spawning.
# GET /api/run/project-evaluation/preflight and /api/run/project-evaluation use it.
# UI: if selectedProject().project_id, choose those URLs; otherwise keep existing
# /api/run/preflight-complete-pipeline and /api/run/complete-pipeline URLs.
```

Never accept path arguments from the browser. Reject evidence-only, setup-incomplete, and tuple-mismatched projects before spawning.

- [ ] **Step 4: Run backend/frontend tests and syntax checks**

Run:

```bash
/tmp/wns-mineru-v3/bin/python -m unittest tests.test_dashboard_dataset_lifecycle tests.test_project_evaluation -v
node --check web/app.js
/tmp/wns-mineru-v3/bin/python -m py_compile scripts/serve_benchmark_dashboard.py scripts/run_project_evaluation.py scripts/run_retrieval_smoke_from_vm_dbs.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/serve_benchmark_dashboard.py web/app.js tests/test_dashboard_dataset_lifecycle.py tests/test_project_evaluation.py
git commit -m "feat: route project evaluations through dashboard"
```

### Task 4: Browser E2E and non-contamination proof

**Files:**
- Create: `/tmp/wns-dashboard-qa/qa-project-evaluation.mjs` (temporary, not committed)
- Test: `tests/test_project_evaluation.py`

- [ ] **Step 1: Add a failing browser assertion to the temporary QA script**

```javascript
await page.getByRole('button', { name: 'Run pipeline', exact: true }).click();
await page.locator('#runPreflightBtn').click();
await page.waitForFunction(() => document.querySelector('#runStatus')?.textContent === 'Ready');
```

Expected: currently fails because the project is routed through the global preflight.

- [ ] **Step 2: Snapshot official lane hashes before live run**

```bash
sha256sum data/db_ingestion_runs/*/summary.csv data/retrieval_smoke/summary.json data/evaluation/groundtruth_eval_report.json 2>/dev/null > /tmp/official-lanes-before.sha256 || true
```

- [ ] **Step 3: Start dashboard with only the real local GTE endpoint configured**

```bash
GTE_EMBEDDING_URL=http://127.0.0.1:5014/embed/gte /tmp/wns-mineru-v3/bin/python scripts/serve_benchmark_dashboard.py 5013 127.0.0.1
```

- [ ] **Step 4: Execute one browser-selected baseline**

Select project `public-five-paper-layout-e2e_20260722_022545_83de4f`, `Heading_sections_l2`, `gte_multilingual_base`, `FAISS`, `none`, then use the dashboard check/run controls. Assert job success, 20 project retrieval artifacts, evaluation report `missing_query_count == 0`, and a summary with `evaluated_queries == 20`.

- [ ] **Step 5: Confirm no official artifact changed, browser clean, and package only after all checks**

```bash
sha256sum data/db_ingestion_runs/*/summary.csv data/retrieval_smoke/summary.json data/evaluation/groundtruth_eval_report.json 2>/dev/null > /tmp/official-lanes-after.sha256 || true
diff -u /tmp/official-lanes-before.sha256 /tmp/official-lanes-after.sha256
```

Expected: no diff. Capture desktop and mobile screenshots, check console/network errors, and retain the project-local result manifest as evidence.
