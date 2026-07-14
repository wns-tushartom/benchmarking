# Official and Baseline Recommendations Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Recommendations reproduce the Metrics composite-score leaderboard for the official 180 rows, expose the 30 measured no-reranker rows as a separate result set, and display available official evidence-row counts.

**Architecture:** Keep the existing result-source selector. Add a secondary official-result-set selector that chooses either `benchmark_reference.summary` or measured canonical `none` rows from legacy evaluation summaries. Build each payload independently, attach the existing `metricScore` as `winner_score`, and join evidence-display counts by canonical pipeline key without mixing arrays.

**Tech Stack:** Vanilla JavaScript, HTML, Python pytest with Node VM frontend tests.

---

### Task 1: Lock the result-set contract with a failing frontend test

**Files:**
- Modify: `tests/test_recommendations_ui.py`

- [ ] **Step 1: Extend the existing official payload regression**

Assert that the official set contains only `benchmark_reference.summary`, has the composite winner score, and joins evidence counts. Add a second assertion that the baseline set contains only canonical `reranker: none` rows and never official reranked rows.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_recommendations_ui.py::test_official_recommendations_use_only_official_rows_and_recorded_evidence
```

Expected: FAIL because baseline result-set selection does not exist.

### Task 2: Add the isolated official result-set selector

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`

- [ ] **Step 1: Add the selector markup**

Add `recommendationOfficialSetField` and `recommendationOfficialSet` with exact options:

```html
<option value="official">Official reranked matrix — 180</option>
<option value="baseline">No-reranker baseline — 30</option>
```

- [ ] **Step 2: Build isolated payloads**

In `web/app.js`, make `officialRecommendationPayload(resultSet)`:

- `official`: use only `evaluation.benchmark_reference.summary`.
- `baseline`: use only canonical `reranker === 'none'` rows from `evaluation.summary` and `evaluation.reranked.summary`.
- Dedupe each set independently by canonical pipeline key.
- Add `winner_score: metricScore(row)`.
- Add `evidence_count` from `operational.benchmark_detail_evidence` by the same key, otherwise `null`.
- Set context counts from the selected row array; never label the baseline as official matrix coverage.

- [ ] **Step 3: Wire selector visibility and change handling**

Show the result-set control only for the official source. Hide it for uploaded projects. On change, rerender Recommendations without fetching or mixing source arrays.

- [ ] **Step 4: Run GREEN verification**

Run:

```bash
python3 -m pytest -q tests/test_recommendations_ui.py tests/test_pipeline_recommendations.py tests/test_dashboard_metrics.py
node --check web/app.js
```

Expected: all tests pass and Node syntax check exits zero.

### Task 3: Verify and package the laptop overlay

**Files:**
- Include: `Retreiver/web/app.js`
- Include: `Retreiver/web/index.html`
- Include: `Retreiver/tests/test_recommendations_ui.py`
- Include: `Retreiver/docs/superpowers/specs/2026-07-13-pipeline-recommendations-page-design.md`
- Include: `Retreiver/docs/superpowers/plans/2026-07-14-official-baseline-recommendations-hotfix.md`

- [ ] **Step 1: Run final focused regression and hygiene checks**

```bash
python3 -m pytest -q tests/test_recommendations_ui.py tests/test_pipeline_recommendations.py tests/test_dashboard_metrics.py
node --check web/app.js
node --check web/recommendations.js
git diff --check
```

- [ ] **Step 2: Build ZIP rooted at `Retreiver/`**

The ZIP must overwrite only the five listed files when extracted from the work-laptop repository root.

- [ ] **Step 3: Verify ZIP integrity and SHA-256**

```bash
unzip -t /home/fate/wns_official_baseline_recommendations_hotfix_20260714.zip
sha256sum /home/fate/wns_official_baseline_recommendations_hotfix_20260714.zip
```

- [ ] **Step 4: Apply, stage narrowly, commit, and push from the work laptop**

```bat
tar -xf "%USERPROFILE%\Downloads\wns_official_baseline_recommendations_hotfix_20260714.zip" -C .
git add -- Retreiver/web/app.js Retreiver/web/index.html Retreiver/tests/test_recommendations_ui.py Retreiver/docs/superpowers/specs/2026-07-13-pipeline-recommendations-page-design.md Retreiver/docs/superpowers/plans/2026-07-14-official-baseline-recommendations-hotfix.md
git diff --cached --check
git commit -m "fix: separate official and no-reranker recommendations"
git push origin feat/project-smiley-nvidia-rag
```

- [ ] **Step 5: Pull, restart, and verify on the VM**

Pull with `--ff-only`, restart `wns-benchmark-dashboard.service`, verify `/api/result-sources` remains `180/180`, and confirm the browser shows 180 official rows, 30 baseline rows, Metrics-aligned FAISS leaders, and available evidence counts.
