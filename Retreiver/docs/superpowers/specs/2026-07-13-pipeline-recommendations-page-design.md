# Source-Aware Pipeline Recommendations Page Design

**Date:** 2026-07-13
**Status:** Approved revised design
**Surface:** Dashboard result APIs, project matrix artifacts, `web/index.html`, `web/app.js`, `web/styles.css`
**Current page key:** `compare`

## Purpose

Turn the current comparison page into a source-aware decision surface. A user can load either the official WNS benchmark or one completed/partial matrix run from an uploaded project, then distinguish combinations using only metrics and evidence produced by that selected data source.

## Goals

1. Make the best quality, speed, and honest value options visually unmistakable when labels support scoring.
2. Let the user select exactly one official or uploaded-project result source at a time.
3. Show real metrics for combinations executed against newly uploaded data.
4. Keep project/run data strictly isolated from official and other project artifacts.
5. Treat unlabeled runs as evidence-only rather than fabricating quality scores.
6. Show official commercial-model unit prices and calculate run cost only from measured billable usage.
7. Preserve older uploaded runs without inventing metrics that were not recorded.
8. Keep supporting analysis available without allowing it to dominate the first viewport.

## Non-goals

- Combining, averaging, or ranking across different projects, datasets, or runs.
- Comparing an uploaded-project score directly with the official WNS score.
- Treating lexical-preview runs as matrix evaluations.
- Provider billing integration or invoice reconciliation.
- Estimating missing tokens, SearchUnits, latency, relevance, or infrastructure cost.
- Treating local infrastructure as free.
- Combining NVIDIA into the official WNS matrix.
- Changing the official 180-combination benchmark.
- Multi-user accounts or cross-user sharing.

## Core source invariant

The page has one active result source and one active result set:

```text
Official WNS benchmark → Official reranked matrix (180)
or
Official WNS benchmark → No-reranker baseline (30)
or
Uploaded project → one project_id → one matrix run_id
```

The 30 measured no-reranker rows are a separate legacy baseline, not part of the official 180-row matrix and not presented as a complete fourth reranker lane. Official, baseline, and uploaded rows are never appended to the same array, payload, table, chart, winner calculation, filter, or cache entry. Switching sources or official result sets clears the previous state before the new result payload is rendered.

A request-generation token or `AbortController` prevents an older, slower response from overwriting a newer source selection.

## Naming and information architecture

Rename the navigation label from **Accuracy, cost, latency** to **Recommendations**. Keep the internal page key `compare` to avoid unnecessary routing changes.

The page order is:

1. Result-source selector
2. Active-source context line
3. Adaptive recommendation strip
4. Top combinations table
5. Commercial pricing basis
6. Collapsed supporting analysis

The page heading is **Pipeline recommendations**.

## Source selector

Use a compact source bar above the recommendation strip.

### Control 1: Result source

- **Official WNS benchmark** — default
- **Uploaded project**

### Control 2: Official result set

Visible only in official-source mode:

- **Official reranked matrix — 180** — default; uses only filtered `benchmark_reference.summary` rows and the existing benchmark winner-score formula.
- **No-reranker baseline — 30** — uses only measured legacy rows whose canonical reranker is `none`.

The selector never combines these arrays. Its context line and table count state exactly which set is active. Evidence-display counts are joined by the exact canonical pipeline key from the already-loaded official benchmark evidence payload; missing counts remain **Not recorded**.

### Control 3: Project

Visible only in uploaded-project mode. Options contain project label and safe short ID. The control reads from the isolated project catalog and never accepts a raw path.

### Control 4: Matrix run

Visible only after a project is selected. Each option shows:

- creation/completion time
- state: completed or partial
- combination count
- scoring mode: retrieval labels or evidence only

Lexical-preview runs, malformed runs, and runs from another project are excluded.

Future matrix manifests persist `created_at` and `completed_at`. For older valid matrix runs without timestamps, the catalog may expose the manifest modification time as **Legacy artifact time**, not as a claimed execution timestamp.

### Context line

Always display the active source clearly:

```text
Viewing: Refund Policy Upload · run_ab12… · 24 combinations · retrieval labels
```

The official source shows:

```text
Viewing: Official WNS benchmark · 180 configured combinations · 180 evaluated rows
```

## Backend read APIs

### `GET /api/result-sources`

Returns:

- the official source descriptor
- available uploaded projects
- no run rows or evidence

### `GET /api/project-runs?project_id=<id>`

Returns matrix-run descriptors for exactly one validated project.

A run is eligible only when:

- the project/run paths pass existing lexical and resolved containment checks;
- neither project root, run root, canonical child, nor artifact is a symlink;
- `manifest.json` is a regular file;
- manifest `project_id` and `run_id` match the requested IDs;
- manifest contains a matrix request fingerprint and combination count;
- state is `completed` or `partial`;
- `summary.csv` exists as a regular file;
- the run is not `lexical_preview`.

### `GET /api/project-run-results?project_id=<id>&run_id=<id>`

Returns a normalized payload for exactly one validated matrix run.

Response fields:

```json
{
  "source_type": "uploaded_project",
  "project_id": "project_uuid",
  "project_label": "Refund Policy Upload",
  "run_id": "run_uuid",
  "run_state": "completed",
  "scoring_mode": "retrieval_labels",
  "created_at": "2026-07-13T10:00:00Z",
  "completed_at": "2026-07-13T10:12:00Z",
  "combination_count": 24,
  "succeeded": 24,
  "failed": 0,
  "metric_k": 10,
  "metric_names": ["recall_at_k", "mrr_at_k", "ndcg_at_k", "avg_query_latency_s"],
  "rows": [],
  "evidence_counts_by_combo": {}
}
```

The API never falls back to official benchmark rows when a project artifact is missing or invalid.

### `GET /api/project-run-evidence?project_id=<id>&run_id=<id>&combo_id=<id>&limit=<n>&offset=<n>`

Returns bounded, paginated evidence for exactly one completed combination in the selected run. The endpoint validates project, run, combination ownership, and evidence artifact containment before returning rows. The result-summary endpoint never bulk-loads all evidence.

## Normalized uploaded-run row

Each combination row uses explicit project fields rather than pretending to be an official benchmark row:

```json
{
  "project_id": "project_uuid",
  "run_id": "run_uuid",
  "combo_id": "combo_hash",
  "status": "completed",
  "chunker_id": "Heading_sections_l2",
  "embedding_id": "jina_v3",
  "vector_store_id": "Qdrant",
  "reranker_id": "Qwen3:4B Rerank",
  "query_count": 50,
  "labelled_queries": 50,
  "unlabelled_queries": 0,
  "recall_at_k": 0.84,
  "mrr_at_k": 0.71,
  "ndcg_at_k": 0.76,
  "avg_query_latency_s": 0.19,
  "retrieval_latency_s": 0.08,
  "rerank_latency_s": 0.11,
  "error_code": "",
  "commercial_model_ids": [],
  "measured_usage": {
    "embedding_input_tokens": null,
    "embedding_usage_scope": null,
    "rerank_search_units": null,
    "rerank_usage_scope": null
  }
}
```

Missing values remain `null`. The API does not map missing project metrics to zeros.

`commercial_model_ids` comes from exact canonical adapter IDs in the project matrix catalog. It is never inferred from a display-label substring.

## Metric contract for future labeled runs

Metrics are comparable only among combinations in the selected run.

### Recall@K

For each combination:

```text
applicable labeled queries with at least one relevant result
÷ applicable labeled queries
```

### MRR@K

Mean reciprocal rank of the first relevant result over applicable labeled queries. A labeled query with no relevant hit contributes zero.

### nDCG@K

Binary relevance gain using the selected run’s canonical corpus and versioned relevance rules. Ideal DCG uses the number of canonical chunks judged relevant for that question and chunker, capped at K. A query with no relevant result contributes zero.

### Denominators

Denominators remain per combination because a qualified `chunk_ref` label applies only to its canonical chunker. `answer` and `ground_truth` text remain reference-answer display fields and never become retrieval relevance automatically.

### Average query latency

The speed metric is:

```text
mean(retrieval latency + reranking latency) per query
```

It excludes:

- chunking
- document embedding
- vector indexing/upsert
- queue time
- worker startup
- cached setup

Full combination wall-clock time remains in receipts for operations but is not used to select the speed winner.

### Runner artifact evolution

Future matrix runs add these fields to `summary.csv` and `analysis.json`:

- `summary_schema_version`
- `recall_at_k`
- `mrr_at_k`
- `ndcg_at_k`
- `avg_query_latency_s`
- `retrieval_latency_s`
- `rerank_latency_s`

Future manifests add:

- `created_at`
- `completed_at`
- `summary_schema_version`

Metric calculations are written atomically with the existing project/run artifacts. Official artifact roots remain untouched.

## Backward compatibility

The result reader supports existing valid uploaded matrix runs.

- Existing Recall@K remains visible.
- Missing MRR, nDCG, and average query latency display **Not recorded**.
- Receipt wall-clock time is not substituted for average query latency.
- Existing evidence-only runs remain evidence-only.
- Existing failed rows remain visible with sanitized `error_code`.
- Older runs are not rewritten during read.
- Invalid, mismatched, or ambiguous legacy runs fail closed.

## Adaptive recommendation strip

Use one compact horizontal strip divided into three equal columns. Do not create three large standalone cards.

### Labeled source

#### Quality winner

- Accent: mint
- Official source rule: highest existing benchmark winner score
- Uploaded source rule: highest available quality ordering: nDCG@K, then MRR@K, then Recall@K
- Tie-breaks: lower average query latency, then stable pipeline key
- Display exact metric basis; do not label uploaded quality as the official benchmark score

#### Speed winner

- Accent: cyan
- Rule: lowest positive recorded average query latency among completed rows
- Tie-breaks: quality ordering, then stable pipeline key

#### Value winner or fallback

The interface must not compare commercial API charges with unmeasured VM infrastructure cost as if they were equivalent.

- If comparable measured total run cost exists for at least two candidates under one cost scope, label **Value winner** and use quality per measured USD.
- Otherwise label **Best no-API-fee option** and choose the highest-quality completed pipeline with no commercial model API dependency.
- Explain: **No external model API fee; VM infrastructure excluded.**
- Accent: amber

### Evidence-only source

No quality or value score is shown. The three columns become:

1. **Fastest completed combination** — lowest recorded average query latency
2. **Run health** — succeeded and failed combination counts
3. **Evidence coverage** — query count and available evidence-row count

If average query latency was not recorded in an older evidence-only run, the first column says **Latency not recorded**.

## Top combinations table

Show ten rows by default, scoped to the active source.

### Official or labeled uploaded source

| Column | Content |
|---|---|
| Rank | Rank under active source-specific sort |
| Pipeline | Chunker, embedding, vector DB, reranker |
| Status | Completed or sanitized failure stage |
| Recall | Official: Recall@5. Uploaded: Recall at the selected run's `metric_k` |
| MRR | Official: MRR. Uploaded: MRR at the selected run's `metric_k` or Not recorded |
| nDCG | Official: nDCG@5. Uploaded: nDCG at the selected run's `metric_k` or Not recorded |
| Avg sec/query | Positive recorded average query latency |
| Commercial cost | Measured estimate, published rate, or no-API-fee state |
| Why it stands out | Quality, Speed, Value, No API fee badges |

Uploaded metric headers include the actual run value, for example **Recall@10**, **MRR@10**, and **nDCG@10**. The page never labels uploaded `@K` metrics as official `@5` metrics.

### Evidence-only uploaded source

| Column | Content |
|---|---|
| Pipeline | Chunker, embedding, vector DB, reranker |
| Status | Completed or sanitized failure stage |
| Queries | Run query count |
| Avg sec/query | Recorded query latency or Not recorded |
| Commercial cost | Honest cost state |
| Evidence | View evidence action for completed combinations |

### Interaction

- Official/labeled default sort: Quality
- Evidence-only default sort: Speed when latency exists; otherwise Pipeline
- Compact sort controls adapt to available metrics
- Existing pipeline filters remain compressed into one row
- Winner rows receive subtle full-row tint/border treatment
- Do not use colored side stripes
- Maintain horizontal table scrolling on small screens
- Failed combinations never qualify as winners

## Evidence review

**View evidence** opens a scoped drawer/dialog for one `combo_id` from the active run.

Display:

- query ID and query text
- source name
- page number when available
- excerpt
- base retrieval score
- reranker score
- rank

The API returns evidence only from the selected project/run and only for the requested/loaded combination. Evidence from failed combinations is absent. All text is escaped before rendering.

Evidence is fetched on demand with bounded pagination. Closing or switching the active source clears the evidence drawer and aborts any stale evidence request.

## Commercial pricing basis

Rates verified against first-party documentation on 2026-07-13:

- OpenAI `text-embedding-3-large`: **USD 0.13 per 1 million input tokens**
- Amazon Rerank 1.0: **USD 0.001 per SearchUnit/query**
- One Amazon Rerank SearchUnit/query can contain up to 100 document chunks; larger requests consume multiple units

Sources:

- `https://developers.openai.com/api/docs/models/text-embedding-3-large`
- `https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-pricing.html`
- `https://aws.amazon.com/bedrock/pricing/`

### Cost display states

Measured usage always carries an attribution scope:

- OpenAI embedding usage may be shared by all combinations using one `(chunker_id, embedding_id)` cache entry. Its scope is `shared_embedding`.
- Amazon rerank SearchUnits are measured per combination because reranking executes independently for each combination. Their scope is `combination`.
- Shared usage is shown in the run pricing ledger but is never duplicated into each row or used to rank per-combination value.
- A per-combination `total_cost_usd` exists only when every commercial charge used by that row has `combination` scope.

1. **Measured usage available**
   - OpenAI: `embedding_input_tokens / 1,000,000 × 0.13`
   - Amazon: `rerank_search_units × 0.001`
   - Label: **Estimated run API cost from measured usage**

2. **Commercial model selected, usage unavailable**
   - Show official unit rate
   - Label: **Run usage not recorded**
   - Never show `$0.00`

3. **No commercial model API selected**
   - Label: **No external model API fee**
   - Supporting text: **VM infrastructure excluded**

4. **Unknown commercial model or stale pricing metadata**
   - Label: **Pricing unavailable**
   - Never infer from another model or a label substring

## Supporting analysis

Move supporting views under one collapsed `<details>` section titled **Explore trade-offs**.

### Labeled source

- top combination quality bars
- quality-versus-latency plot
- detailed pipeline comparison
- reranker lift
- best by component

### Evidence-only source

- latency by completed combination
- completion/failure breakdown
- evidence count by combination

Expanding the section must not change source selection, filters, sorting, or evidence state.

## Visual hierarchy

- Quality: mint badge/tint
- Speed: cyan badge/tint
- Value/no-API-fee: amber badge/tint
- Failure: rose status badge, never used as a decorative winner treatment
- Use explicit text labels in addition to color
- Keep text contrast at WCAG AA or better
- Pipeline labels may wrap but may not truncate critical component names
- Avoid decorative gradients, oversized metric numerals, and nested card clutter
- Desktop: three equal recommendation columns
- Mobile: source controls and recommendation rows stack; table scrolls horizontally

## Loading and error states

- Source switch: immediately clear old winners/table/charts and show loading skeleton/text
- No uploaded projects: **Upload a project and run a matrix first**
- Project with no matrix runs: **No completed or partial matrix runs for this project**
- Missing/malformed run: sanitized **Run results unavailable**; never fall back to official rows
- Partial run: show completed and failed rows; rank completed rows only
- Evidence-only run: explain why quality metrics are unavailable
- Labeled run with no applicable labels for a combination: display **Not applicable**, not zero
- Stale asynchronous response: ignore through request-generation token/abort logic
- API errors contain structured codes and no raw paths, exception text, credentials, or provider payloads

## Security and isolation

- Validate project and run IDs through `ProjectWorkspace`.
- Reject symlinked roots, children, manifests, summaries, analyses, receipts, and evidence artifacts.
- Read regular files with no-follow semantics and bounded sizes.
- Verify manifest project/run identity before reading child artifacts.
- Verify every normalized row matches the requested project/run IDs.
- Bound summary rows, evidence rows, field lengths, and JSON/CSV payload sizes.
- Escape all frontend text and use fixed templates for badges/classes.
- Do not expose physical vector namespaces, raw filesystem paths, credentials, or unsanitized provider errors.
- Official artifact protection checks remain adjacent to backend and browser verification.

## Verification

### Runner metric tests

- Recall@K uses only applicable labeled queries.
- MRR@K gives zero for misses and reciprocal rank for first relevant hit.
- nDCG@K uses canonical relevant-chunk count and is deterministic.
- Per-combination denominators differ correctly by chunker-qualified labels.
- Average query latency includes retrieval plus reranking only.
- Evidence-only runs contain no quality metrics or leaderboard.
- New metric artifacts are atomic and project/run-scoped.

### API/isolation tests

- Result-source catalog lists official and valid projects.
- Run catalog excludes lexical preview, queued/running, malformed, symlinked, and cross-project runs.
- Run-results endpoint rejects mismatched project/run IDs.
- Uploaded mode never returns official rows.
- Official mode never reads uploaded rows.
- Older Recall-only runs normalize missing metrics to `null`.
- Partial runs expose completed and failed rows without ranking failures.
- Evidence rows cannot cross projects, runs, or combinations.
- Run-results responses expose evidence counts only; evidence rows load through the bounded combination endpoint.
- Oversized/malformed artifacts fail with sanitized structured errors.

### Frontend tests

- Navigation and heading use Recommendations/Pipeline recommendations.
- Official source is the default.
- Uploaded source requires project and run selection.
- Source switch clears prior source content before fetch completion.
- Stale source responses cannot overwrite current selection.
- Labeled mode shows Quality, Speed, and Value/no-API-fee roles.
- Evidence-only mode shows Fastest, Run health, and Evidence coverage.
- Missing metrics render Not recorded/Not applicable, never zero.
- Failed combinations cannot receive winner badges.
- OpenAI and Amazon official rates render correctly.
- Missing usage never renders `$0.00`.
- Shared OpenAI embedding usage appears once in the run ledger and cannot create a per-combination value winner.
- Combination-scoped Amazon SearchUnits calculate at USD 0.001 per unit.
- Supporting analysis is collapsed by default.
- Official 180-combination and artifact-protection contracts remain unchanged.

### Browser QA

- Verify source APIs before browser checks.
- Capture official source, labeled uploaded run, and evidence-only uploaded run.
- Switch rapidly between two projects/runs and confirm no stale content appears.
- Confirm active project/run context is clearly visible above the recommendation strip and updates immediately after source changes.
- Open combination evidence and verify source/page/excerpt ownership.
- Capture desktop around 1440 px and mobile around 390 px.
- Check table scrolling, source-control stacking, dialogs, and chart clipping.
- Capture console errors, page errors, and failed network requests.

## Acceptance criteria

A user can:

1. choose the official benchmark or one uploaded project/run;
2. see exactly which data produced the current results;
3. distinguish combinations using real available metrics;
4. review evidence when labels are absent;
5. understand which metrics are missing instead of seeing fabricated zeros;
6. identify quality, speed, and cost posture without mixing datasets;
7. switch sources without stale or cross-project data leakage.

The feature is not complete until the revised source-aware implementation is verified locally, packaged, pushed from the work laptop, pulled on the VM, restarted, and browser-verified against real uploaded-run artifacts.
