# Benchmark integrity, lifecycle, and metrics UX

**Status:** Approved for implementation  
**Repository:** `/home/fate/wns-retreiver-product-hardening`  
**Baseline:** `8623208c38828723f7985d09dd6a97ba42cdcee3` (`master`)

## Objective

Make the WNS benchmark dashboard defensible as a configuration-selection product. It must never rank partial artifacts as winners, pair the wrong ground truth with a dataset, present evidence-only checks as evaluated results, or hide the setup lifecycle behind a query-time action.

## Scope

This implementation contains four connected slices:

1. Result integrity and complete-run gating.
2. Dataset and ground-truth binding with an evidence-only fallback.
3. Durable, visible dataset setup lifecycle with provenance and per-stage status.
4. Evaluation-only trade-off scatter plus a sliceable configuration heat-map, using real result state.

It extends the current standard-library Python dashboard server and static `web/` client. No frontend framework, CDN, mock metrics, or independent chart dependency will be introduced.

## Product contracts

### Canonical pipeline state

Every configured pipeline combination has a canonical key:

```text
chunker | embedding | vector_store | reranker
```

It has one of these states:

- `not_run`: configured but no artifact exists.
- `running`: an active dashboard job owns the combination.
- `complete`: all required evaluation fields are present and numerically valid.
- `incomplete`: an artifact exists, but is missing required evaluation fields.
- `failed`: the associated job or artifact reports a failure.

Only `complete` rows are eligible for score ranking, winner labels, trade-off plotting, and heat-map values. Other states remain visible as operational status, never as zero-valued performance.

A row carries run timestamp, artifact path/source, and evaluated query count. Where multiple artifacts report the same canonical key, the latest complete artifact wins. A newer incomplete retry does not replace a valid complete result.

### Dataset and ground truth

A dataset project manifest records its own optional ground-truth artifact and validation state.

- If a valid ground-truth mapping exists, evaluation actions and metrics are enabled.
- If it does not, the product is explicitly in `evidence_only` mode. Retrieval and reranking evidence can be displayed, but no score, recommendation, or winning configuration is calculated.
- The Run Pipeline screen receives its ground truth from the selected dataset record. It cannot substitute a different dataset's ground truth.

### Setup lifecycle

The Documents screen owns:

```text
upload → extract → chunk → embed → index
```

A setup job is created for the chosen setup matrix. Its status is persisted and returned from the existing job/status surface with per-stage timestamps, outputs, errors, and selected configuration provenance.

The Run Pipeline screen starts only after relevant setup completes. It owns:

```text
retrieve → rerank → evaluate
```

A selected matrix is either `full` or `partial`. A partial matrix may provide evidence and subset results, but it must visibly say that no global recommendation is established.

### Metrics UX

The first visual reference informs the interaction model, not the visual style verbatim:

- Compact control strip: X metric, Y metric, color-by component, bubble size.
- Honest KPI strip: best score, best latency, complete combinations, evaluated queries. No raw artifact-file count masquerading as a query count.
- Trade-off scatter: rendered only when the selected dataset has ground truth and at least one complete evaluated row.
- Heat-map: rows are chunkers, columns are vector stores, with embedding/reranker/metric as explicit slices. Configured but missing cells render `not run`, not `0`.
- Table remains the exact, accessible source of values and status.

The existing dashboard's dark compact design stays authoritative. The UI will not copy the reference's light theme or use fictional candidate/model totals.

## API and data shape

`/api/results` gains additive fields only:

- `pipeline_state`: canonical rows with key, components, state, source, run timestamp, query count, and metrics when complete.
- `dataset_catalog`: datasets with `dataset_id`, display name, ground-truth linkage/validation, setup state, provenance, and allowed mode.
- `setup_jobs`: bounded recent job summaries, each including selected matrix and stage status.

Existing response keys remain compatible with the current client.

## Failure and empty states

- Missing ground truth: explicit evidence-only banner. Metric cards, heat-map values, scatter points, and recommendations are unavailable.
- No complete evaluation rows: show configuration coverage/status and clear next action. Do not render empty axes as quality charts.
- Failed/incomplete setup: show stage, error, artifact/provenance, and retry/preflight route. Do not represent it as a benchmark score.
- Partial selection: show exact selected/official combination counts and state that recommendation coverage is partial.

## Tests and verification

Tests are written before production changes and must prove:

1. Duplicate pipeline rows select the latest complete result over a newer incomplete retry.
2. Incomplete rows cannot become ranked winners or plot/heat-map metric values.
3. Dataset selection can only resolve its own validated ground truth, and absent ground truth returns evidence-only mode.
4. Setup job state is persisted with selected-matrix provenance and gates evaluation execution.
5. Frontend renders complete/partial/not-run state and disables metric visuals in evidence-only mode.
6. Trade-off controls and heat-map use only completed evaluated rows and preserve exact row/table semantics.

Verification includes targeted Python/Node tests, real dashboard API probes, and browser screenshots at desktop and mobile widths using the served dashboard URL.

## Non-goals

- Automatic silver-ground-truth generation. This remains a separately labelled Phase-2 research track.
- Pretending parser/OCR, live model service, or vector-store health is better than the artifacts prove.
- Replacing the static client with React or adding an external charting dependency.
- Running expensive full benchmark jobs merely to make a UI screenshot look populated.
