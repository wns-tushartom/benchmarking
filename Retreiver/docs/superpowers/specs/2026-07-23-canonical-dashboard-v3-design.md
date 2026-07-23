# Canonical WNS Benchmark Dashboard V3 Design

**Date:** 2026-07-23  
**Branch:** `fix/canonical-dashboard-v3`  
**Canonical base:** `55ea159` (`fix: show selected source in run pipeline`)  
**Status:** Approved through the user's standing instruction to continue with the recommended canonical merge, plus explicit choices to place trade-off/heatmap UX on Recommendations and lock dataset-ground-truth pairing.

## 1. Problem

The product regressed because three independently valid lineages were combined by replacing whole files instead of merging contracts:

1. The latest Recommendations UI and hardened project modules lived on the `55ea159` lineage.
2. Complete-run gating, dataset-ground-truth lifecycle, metric trade-off/heatmap UX, and MinerU/project evaluation lived on the divergent `6ea6ff7` lineage.
3. VM-only operational fixes and benchmark artifacts were newer than GitHub/local packages.

The July 22 package preserved some new functionality but replaced the backend and frontend with incompatible generations. The result was a frontend requesting source/result contracts the backend no longer exposed. Dataset and linked-ground-truth selectors disappeared, `/api/result-sources` failed, metrics did not render, and the Recommendations page lost its newest analytical UX.

This design makes one local Git branch the canonical product source and treats VM state as an input that must be merged back, not an unofficial fork.

## 2. Goals

1. Preserve the complete hardened tree and latest Recommendations experience from `55ea159`.
2. Add the verified complete-run integrity, linked-ground-truth, setup lifecycle, heatmap/trade-off, MinerU, and project-evaluation behavior from the `6ea6ff7` lineage without deleting unrelated modules.
3. Restore and test every frontend/backend contract needed by dataset selection, linked ground truth, result-source selection, recommendations, and the 180-combination matrix.
4. Keep incomplete, failed, running, and not-run states visible without converting them to zero-quality results.
5. Prevent future ZIP/push regressions with a machine-checked release contract and VM-to-local parity workflow.
6. Verify every page on desktop and mobile using the real served application before packaging.

## 3. Non-goals

- Do not fabricate benchmark metrics locally when the real 180-combination artifacts exist only on the VM.
- Do not rerun paid or expensive combinations merely to populate a visual.
- Do not rebuild the dashboard in React or introduce a frontend build system.
- Do not silently downgrade MinerU PDF extraction to text-only extraction.
- Do not copy entire `web/app.js`, `web/index.html`, `web/styles.css`, or `scripts/serve_benchmark_dashboard.py` from another branch.
- Do not overwrite VM data or adapter files during deployment.

## 4. Canonical source strategy

### 4.1 Base

Start from `55ea159` because it contains:

- The latest dedicated Recommendations page.
- Quality, speed, and value recommendation categories.
- Top-ranked pipeline table and recommendation explanations.
- The full hardened project/module tree.
- Existing source selectors and Run Pipeline source restatement.
- Current Documents, NVIDIA, Repository, and Operations pages.

### 4.2 Intentional feature replay

Port behavior and tests, not whole files, from the divergent integrity lifecycle:

- `13bf6ea`, `8eb387c`, `5049f26`: complete-run-only ranking and deterministic latest-artifact selection.
- `438f2a7`: dataset-specific validated ground-truth binding and evidence-only gating.
- `a4ac3db`, `5e3a5fc`: persisted setup lifecycle, provenance, safe paths, and official/project selection separation.
- `cdb28ef`: complete-run trade-off explorer and configuration heatmap.
- `0365448`: MinerU-backed project upload and project-scoped evaluation.
- `6ea6ff7`: source selectors and full metric aggregation.
- `e6551c0` intent: `/api/result-sources`, project result/evidence endpoints, and source catalog in `/api/results` where not already present.

Every replayed behavior requires a red regression test against the canonical base before production edits.

### 4.3 VM parity

Before release, the VM must export or report all tracked source differences. Any verified VM source patch must be one of:

- merged into the canonical branch with a regression test,
- explicitly documented as an environment-only override, or
- rejected with a reason.

Runtime benchmark data remains on the VM. Source code must not remain VM-only after verification.

## 5. Frontend information architecture

The page inventory is fixed and release-tested:

1. Overview
2. Metrics
3. Recommendations
4. Run Pipeline
5. Top 5 Hits Evidence
6. Grounding Audit, explicitly disabled unless its backend is active
7. NVIDIA RAG
8. Repository
9. Documents
10. Operations

A release fails if a required page, navigation control, or primary panel is missing.

## 6. Global source and linked-ground-truth behavior

1. The global dataset selector is the primary source selector.
2. Selecting a dataset automatically selects its linked ground truth.
3. Ground truths not linked to the selected dataset cannot be selected.
4. If no valid linked ground truth exists, the UI enters Evidence-only mode.
5. Evidence-only mode may show retrieval evidence and operational state, but it must block quality metrics, rankings, winner claims, and scored recommendations.
6. Metrics, Recommendations, and Run Pipeline visibly restate the selected dataset and linked ground truth.
7. Source selection persists across page navigation and refreshes dependent results without stale cache reuse.

## 7. Recommendations page

The final Recommendations page is recommendation-first, with analytical detail below it.

### 7.1 Source context

A pinned source strip shows:

- selected dataset,
- linked ground truth or Evidence-only mode,
- selected completed result source,
- complete/evaluated combination count.

### 7.2 Recommendation strip

Three primary recommendation cards:

- Best Quality
- Fastest
- Best Value / no recorded API fee

Each card identifies the complete pipeline and includes a concise, data-backed explanation. Incomplete combinations never qualify.

### 7.3 Top-10 recommendation table

The table includes:

- rank,
- chunker,
- embedding,
- vector store,
- reranker,
- selected quality metric,
- latency,
- cost status,
- completion/source status.

The table is paginated and preserves current page only when the filtered result remains valid.

### 7.4 Practical trade-off explorer

The explorer appears below the recommendation strip and top-10 table.

- Default X-axis: average latency per query.
- Default Y-axis: Recall@5.
- User-selectable quality axes where real values exist.
- Configurable color grouping by vector store or embedding.
- Optional bubble size only for an actual available metric; no fake proxy.
- Clear directional guidance: faster left, better quality up.
- Winner labels remain visible; other point details appear on hover/focus.
- Only complete, ground-truth-evaluated rows become plotted quality points.

### 7.5 Configuration heatmap

- Rows: chunking strategies.
- Columns: vector databases.
- Controls: metric, embedding, reranker.
- Exact value shown inside measured cells.
- `not_run`, `running`, `incomplete`, and `failed` use distinct non-quality states.
- Color legend communicates direction and range.
- Mobile uses an explicit horizontally scrollable region with fixed row labels where practical.

## 8. Metrics page

Metrics remains the precision source:

- exact complete-run table,
- source context,
- incomplete combination diagnostics,
- no duplicated recommendation cards,
- no quality values for unscored rows.

Recommendations owns the visual decision-support experience; Metrics owns exact inspection.

## 9. Run Pipeline and Documents

### Run Pipeline

- Restates the active dataset and linked ground truth.
- Official 180-combination controls remain available for the official dataset.
- Uploaded projects remain isolated from official runs.
- Project scored evaluation is blocked until setup is complete and linked ground truth is valid.
- Missing combinations remain runnable where their adapters are configured.
- Amazon Rerank credential state must not be presented as a generic FAISS failure.

### Documents

- Upload accepts one dataset file and optional linked ground-truth file.
- PDFs use MinerU-backed layout-aware extraction.
- Tables, page provenance, and parser method are retained.
- MinerU failures are visible; no silent text-only downgrade.
- Project status shows extraction, chunking, embedding, indexing, and evaluation readiness.

## 10. Backend contracts

The canonical server must expose and test:

- `GET /api/results`
  - includes `source_catalog`, canonical matrix options, pipeline state, operational summaries, and selected source context.
- `GET /api/result-sources`
  - lists official and project result sources with exact mapping and no-store caching.
- Project run/result/evidence endpoints used by the current hardened frontend.
- Dataset upload, project setup, project evaluation, run, status, query, and evidence endpoints already present in the hardened tree.

Canonical pipeline key:

`chunker | embedding | vector_store | reranker`

Canonical run states:

- `not_run`
- `running`
- `complete`
- `incomplete`
- `failed`

Only `complete` rows with valid linked ground truth are eligible for quality rankings, recommendations, scatter points, and heatmap quality colors.

## 11. 180-combination truth contract

The official matrix remains:

- 5 chunking strategies
- 3 embedding models
- 4 vector stores
- 3 reranker modes
- 180 total combinations

The UI must distinguish:

- configured combinations,
- artifact-producing combinations,
- evaluated complete combinations,
- incomplete/failed/not-run combinations.

The local repository may legitimately show zero complete evaluations when VM artifacts are absent. The VM deployment is verified separately against its real result artifacts. A local empty result set must not erase selectors or change the configured matrix count.

## 12. Regression and release guardrails

Add a machine-readable release contract and verifier covering:

- required pages and DOM IDs,
- required source selectors,
- linked-ground-truth behavior,
- recommendation cards/table,
- trade-off chart and heatmap controls,
- required API routes,
- 180-combination formula,
- complete-run gating,
- MinerU project upload contract,
- no duplicate DOM IDs,
- no syntax errors,
- no forbidden whole-file lineage replacement during the merge.

Release process:

1. Clean canonical branch.
2. Focused and full relevant Python tests.
3. JavaScript syntax check.
4. Backend compilation.
5. Desktop and mobile browser QA across every page.
6. API contract probes.
7. VM source-diff inventory.
8. Build ZIP from the verified commit, not from a mutable folder copy.
9. Record commit SHA and ZIP SHA-256.
10. Deploy without overwriting VM runtime data.

## 13. Browser acceptance criteria

Desktop and mobile QA must verify:

- every page opens,
- no console errors,
- no failed API requests,
- no unintended horizontal page overflow,
- selected dataset and linked ground truth remain visible,
- Recommendations includes cards, top-10 table, trade-off explorer, and heatmap,
- Metrics renders exact values or an honest no-data state,
- Run Pipeline restates source and gating reason,
- Documents visibly reports parser/setup state,
- heatmap scroll behavior works on mobile,
- disabled features are explicitly disabled rather than blank.

Screenshots are captured for every page at desktop and mobile widths.

## 14. Delivery contract

The final handoff includes:

- canonical branch and commit,
- exact changed-file list,
- test output,
- browser QA output and screenshots,
- verified ZIP and SHA-256,
- narrow laptop add/commit/push command if needed,
- safe VM source-inventory and pull/deploy command,
- separate VM commands for diagnosing real 180-combination metric artifacts after the frontend is verified.

No release is described as complete until the served application has been exercised and the artifact hashes are recorded.
