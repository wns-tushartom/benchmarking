# Official Meeting Product Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a meeting-ready WNS dashboard whose first render is evidence-correct, whose source selection is global, and whose modular pipeline can safely reuse uploaded indexes while presenting an understandable retrieval → reranking → answer/evaluation flow.

**Architecture:** Introduce one canonical frontend source-selection contract and one backend run contract. Normalize official/baseline rows before any stakeholder rendering; represent pipeline stages and reusable project artifacts with immutable fingerprints; split latest attempt from last successful corpus state. Keep compatibility endpoints and artifacts but remove misleading controls from the stakeholder UI.

**Tech Stack:** Static HTML/CSS/JavaScript, Python 3 dashboard server and pipeline scripts, pytest, Node syntax/contract tests, Playwright/browser QA where available.

---

## File structure

**Create**

- `web/source-state.js` — canonical active-source state and official/baseline payload normalization usable from Node tests and the browser.
- `scripts/pipeline_run_contract.py` — depth validation, stage fingerprints, reusable-stage planning, and a small adapter over the existing hardened `source/services/project_questions.py` query parser.
- `scripts/dashboard_snapshot_state.py` — atomic last-success/latest-attempt publication and resolution.
- `tests/test_pipeline_run_contract.py` — backend contract tests.
- `tests/test_dashboard_snapshot_state.py` — readiness publication tests.
- `tests/test_source_state_ui.py` — static and Node-backed canonical source-state tests.

**Modify**

- `web/index.html` — global source bar, modular Run Pipeline labels/controls, inactive Grounding tab, Documents cleanup, cache key.
- `web/app.js` — consume canonical active source, remove page-local source divergence, query upload, stage readiness, concise preflight.
- `web/styles.css` — global source bar, stage flow, disabled tab, responsive QA.
- `web/recommendations.js` — consume canonical rows only; no independent initial payload assumptions.
- `scripts/serve_benchmark_dashboard.py` — source-state API fields, fixed `chunk_limit=0`, reranked depth, query upload endpoint/parse path, snapshot and project-stage metadata.
- `scripts/run_complete_pipeline.py` — explicit reranked output depth, run manifest, reuse plan, latest-attempt/last-success publication.
- `scripts/run_reranker_smoke_from_retrieval.py` — independently retain reranked output K.
- `scripts/dashboard_source_catalog.py` — expose truthful reusable stage/index readiness without paths.
- `source/services/project_workspace.py` — persist project stage manifests and validated index fingerprints.
- `tests/test_dashboard_metrics.py` — UI/server contract regressions.
- `tests/test_recommendations_ui.py` — canonical initial-render and inactive UI contracts.
- `tests/test_pipeline_recommendations.py` — official/baseline row parity.
- `tests/test_complete_pipeline_sources.py` — command, depth, reuse, and manifest behavior.
- `tests/test_dashboard_source_catalog.py` — reusable index readiness.

## Task 1: Canonical source and result payload

- [ ] Add failing Node/static tests proving that initial official render, baseline → official toggle, and Metrics filter reset all resolve the same canonical official row IDs and winner.
- [ ] Run `python -m pytest -q tests/test_source_state_ui.py tests/test_pipeline_recommendations.py tests/test_dashboard_metrics.py -k 'source or initial or recommendation or winner'` and confirm the new regression fails.
- [ ] Implement `web/source-state.js` with normalized `activeSource`, canonical official/baseline payload builders, selection generation, and identity matching.
- [ ] Load `source-state.js` before `recommendations.js` and `app.js`; replace mixed `renderEvaluation(..., true)` initialization with one canonical payload render transaction.
- [ ] Ensure Overview KPIs, Metrics, Recommendations, and comparison charts consume the same payload and reranker canonicalization.
- [ ] Run focused tests and `node --check web/source-state.js && node --check web/app.js`.
- [ ] Commit: `fix: make official result rendering canonical`.

## Task 2: Global Dataset, Ground truth, and Result set bar

- [ ] Add failing tests requiring global `activeDataset`, `activeGroundtruth`, and `activeResultSet` controls beside `refreshBtn`, with no independent Metrics/Recommendations source state.
- [ ] Implement the global source bar and populate it from `/api/results` plus `/api/result-sources`.
- [ ] Preserve page-local component filters, but convert duplicated source selectors to context labels or remove them.
- [ ] Synchronize Run Pipeline and NVIDIA dataset/ground-truth values from global state without merging NVIDIA result rows into official results.
- [ ] Persist selection in session storage only after validating IDs against the fresh server catalog; default to official WNS + preferred validated ground truth + official reranked matrix.
- [ ] Reject stale project-run responses through source-generation and identity checks.
- [ ] Run frontend/source tests and commit: `feat: share one dashboard source context`.

## Task 3: Pipeline depth and chunk-limit contract

- [ ] Add failing tests asserting no `runLimit`/Chunk limit UI exists, `chunk_limit` is always `0`, and Retrieval Top K plus Reranked Output K are independently validated and passed.
- [ ] Implement `PipelineDepth` validation in `scripts/pipeline_run_contract.py`: integers with `retrieval_top_k >= reranked_output_k >= 1`, safe caps, and `chunk_limit=0` compatibility.
- [ ] Replace `runTopK` label with **Retrieval Top K** and add `runRerankedTopK` labeled **Reranked Output K**.
- [ ] Remove `runLimit` and set `chunk_limit=0` server-side regardless of browser input.
- [ ] Add `--reranked-output-k` to `run_complete_pipeline.py` and pass it to `run_reranker_smoke_from_retrieval.py` as the retained result count, not artifact limit.
- [ ] Persist both depths in preflight and run manifests/evidence.
- [ ] Run `tests/test_pipeline_run_contract.py`, `tests/test_complete_pipeline_sources.py`, and dashboard tests; commit: `feat: separate retrieval and reranking depth`.

## Task 4: Query-file input

- [ ] Add failing tests for TXT one-query-per-line, CSV/XLSX query aliases, empty/oversized/invalid files, and filename/path non-disclosure.
- [ ] Reuse `source/services/project_questions.py::parse_question_bytes` through a small `scripts/pipeline_run_contract.py` adapter; do not implement a second parser or weaken its signature, size, row, and duplicate checks.
- [ ] Add a query-file input beside typed evidence-only queries; accept `.txt,.csv,.xlsx` and show accepted count/source label.
- [ ] Add a multipart upload/parse API that returns normalized queries only and stores no arbitrary browser path.
- [ ] Merge parsed and typed queries deterministically with duplicate removal; persist query source/count in the run manifest.
- [ ] Run parser/server/UI tests; commit: `feat: support bulk pipeline query files`.

## Task 5: Last-success versus latest-attempt readiness

- [ ] Add failing tests showing a successful snapshot followed by a zero/failed attempt still resolves the successful document counts while exposing the failure separately.
- [ ] Implement atomic `publish_last_success`, `publish_latest_attempt`, and `read_dashboard_snapshot_state` in `scripts/dashboard_snapshot_state.py` with schema validation and replace-on-success semantics.
- [ ] Integrate publication into complete-pipeline/extraction completion and failure paths.
- [ ] Update `/api/results` to expose both records; make Overview/Documents KPI counts read last-success first and latest-attempt separately.
- [ ] Run snapshot/server/dashboard tests; commit: `fix: preserve last successful corpus readiness`.

## Task 6: Uploaded project reusable-stage manifests

- [ ] Add failing tests for stable corpus fingerprints, embedding/store fingerprints, reachable matching index reuse, config invalidation, and project/path isolation.
- [ ] Implement stage fingerprint helpers and a `ProjectStageManifest` contract in `scripts/pipeline_run_contract.py` / `source/services/project_workspace.py`, then integrate them with `source/services/project_matrix_runner.py`.
- [ ] Store reusable project indexes under project-owned fingerprinted namespaces derived from corpus hash + chunker + embedding + vector-store configuration. Do not reuse the generic official runner's sheet-name cache because uploaded projects commonly share sheet names and could collide.
- [ ] Record extraction, chunking, embedding, and vector-store ingestion outputs after successful uploaded-project matrix runs.
- [ ] Extend catalog entries with public `reusable_stages`, `indexed_store_count`, and truthful readiness—never paths or raw collection credentials.
- [ ] Resolve collections/tables only from the selected project manifest and reject unknown or mismatched IDs.
- [ ] Run project/catalog/security tests; commit: `feat: track reusable uploaded pipeline stages`.

## Task 7: Reuse-aware run planner

- [ ] Add failing tests showing a matching uploaded index skips extraction/chunking/embedding/ingestion and starts at retrieval, while stale or unreachable indexes rebuild the required stages.
- [ ] Implement a stage planner returning `ready_reusable`, `required`, `stale`, or `blocked` for each modular stage.
- [ ] Integrate the plan into the isolated `ProjectMatrixRunner` for uploaded datasets and into complete-pipeline preflight selection. Preserve the official default-corpus runner unchanged; never route uploaded work through its shared sheet-name cache.
- [ ] Add an explicit **Rebuild selected stages** advanced action; reuse remains default.
- [ ] Ensure evidence-only mode may reuse indexes but still suppresses scored evaluation.
- [ ] Run source/run integration tests; commit: `feat: reuse uploaded indexes across runs`.

## Task 8: Modular Run Pipeline and concise readiness UX

- [ ] Add failing UI tests for the ordered stage labels: Extraction, Chunking, Embedding, Vector-store ingestion, Retrieval, Reranking, Answer generation, Evaluation.
- [ ] Render a compact stage-flow summary with reuse/required/blocked status from preflight.
- [ ] Rename **Check what is needed** to **Check readiness**.
- [ ] Render concise action cards in the main panel; move service checks, libraries, ports, commands, and raw extraction-audit text into closed **Technical details** and Operations.
- [ ] Preserve the explicit audited text-only override and concise document-level warning.
- [ ] Group advanced actions under the correct stage headings and remove ambiguous artifact-limit labels.
- [ ] Run UI/server tests; commit: `feat: simplify modular pipeline controls`.

## Task 9: Grounding state, answer-generation contract, and Lexical Preview removal

- [ ] Add failing tests requiring Grounding Audit to be disabled/grey unless real answer-generation plus grounding artifacts exist, and requiring Lexical Preview to be absent from stakeholder HTML.
- [ ] Add provider/runtime capability metadata for Gemini, OpenAI, Anthropic, and open-source answer adapters without exposing secrets.
- [ ] Add the answer-generation manifest contract and unavailable/ready stage display; do not issue paid provider calls unless configured.
- [ ] Disable the Grounding tab and public grounding buttons when the required real artifacts are absent.
- [ ] Remove Lexical Preview panel/event wiring from Documents; retain internal backend compatibility temporarily without advertising it.
- [ ] Focus Documents on assets, readiness, last attempt, reusable indexes, and upload actions.
- [ ] Run grounding/Documents tests; commit: `fix: expose only meeting-ready product capabilities`.

## Task 10: Full regression, runtime, and visual QA

- [ ] Bump all frontend asset cache keys to one release value.
- [ ] Run focused tests after every patch, then the complete relevant suite:

```bash
python -m pytest -q \
  tests/test_source_state_ui.py \
  tests/test_pipeline_run_contract.py \
  tests/test_dashboard_snapshot_state.py \
  tests/test_dashboard_metrics.py \
  tests/test_recommendations_ui.py \
  tests/test_pipeline_recommendations.py \
  tests/test_complete_pipeline_sources.py \
  tests/test_dashboard_source_catalog.py
```

- [ ] Run Python compile checks for modified scripts, `node --check` for all modified JavaScript, and `git diff --check`.
- [ ] Start the dashboard on a free local port, probe `/api/results`, `/api/result-sources`, `/api/options`, and preflight with official and uploaded/evidence-only source combinations.
- [ ] Capture and inspect desktop (approximately 1440 px) and mobile (approximately 390 px) screenshots for Overview, Metrics, Recommendations, Run Pipeline, Documents, and disabled Grounding Audit.
- [ ] Verify browser console and failed requests are clean; verify fresh-load BGE/Amazon winner parity against canonical API rows.
- [ ] Verify the official 180 matrix and baseline remain distinct and no NVIDIA row enters the official payload.
- [ ] Run final spec compliance and code-quality review.
- [ ] Commit final cache/QA adjustments: `chore: verify official meeting dashboard release`.
- [ ] Create a Git archive ZIP, run `unzip -t`, inspect critical files with `unzip -l`, calculate SHA-256, and leave `conftest.py` untracked/unmodified.
