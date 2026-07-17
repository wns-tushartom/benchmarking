# Product Completion and Five-Document Demo Design

## Goal

Make the dashboard genuinely executable for an uploaded customer corpus, not merely capable of listing uploaded sources and reading externally-created run artifacts. Add a safe five-document demo that proves every active retrieval pipeline stage in both scored and evidence-only modes.

This is an implementation addendum to the approved Official Meeting Product Hardening Design. Existing integrity rules remain authoritative.

## Audit findings driving this design

1. The live source catalog correctly exposes the default dataset, `groundtruth:none`, and validated repository ground truth using opaque IDs.
2. Invalid dataset and ground-truth IDs fail closed with HTTP 400. Evidence-only execution without queries also fails closed.
3. The main browser action always calls `/api/run/complete-pipeline`, which always launches `scripts/run_complete_pipeline.py`, including for uploaded datasets.
4. `ProjectMatrixRunner` is implemented and heavily tested, but no dashboard POST route creates its immutable `request.json` and launches `scripts/run_project_matrix.py`. Existing project-run API routes are read-only.
5. The legacy complete pipeline writes run metadata and stage artifacts into global repository directories, not `data/user_projects/<project_id>/runs/<run_id>/`. Using it for uploaded projects violates project isolation and prevents reliable project-result routing.
6. Evidence-only preflight accepts a selected reranker in the public command but `run_complete_pipeline.py` silently reduces rerankers to an empty list. This violates the approved requirement that evidence-only mode allow optional reranking and show reranker evidence without quality claims.
7. The current UI does suppress scored NVIDIA benchmarking and scored legacy actions when `groundtruth:none` is selected. This integrity boundary should be preserved.
8. Automated contract/UI suites pass, but interactive browser QA is environmentally limited: Selenium Firefox session creation times out and Snap Firefox ignores requested mobile viewport dimensions. This must be reported honestly, not counted as mobile proof.

## Product decision

### 1. Dispatch by dataset kind

- `dataset:wns-default` continues through the established official/legacy pipeline and official artifact lane.
- `project:<project_id>` must run through `ProjectMatrixRunner` and write only inside that project's immutable run directory.
- NVIDIA remains a separate lane and never contributes rows to the official or uploaded-project matrix.

The frontend keeps one primary action, but the server owns dispatch. The browser sends source IDs and canonical selections, never filesystem paths or collection names.

### 2. Add project-run preflight and launch endpoints

Add server-owned endpoints for uploaded projects:

- `POST /api/run/preflight-project-matrix`
- `POST /api/run/project-matrix`

The preflight endpoint resolves the selected project, validates readiness, validates or imports the query source, validates canonical matrix IDs, computes the exact Cartesian count, reports reusable/stale/required stages, and returns a confirmation requirement for a large matrix.

The launch endpoint repeats validation, creates a new project-owned run directory and immutable request envelope atomically, then launches `scripts/run_project_matrix.py --project-id <id> --run-id <id>` through the existing job system.

The response exposes only safe IDs, counts, status, and public diagnostics.

### 3. Query and ground-truth behavior

#### Ground truth selected

- Parse the selected validated CSV/XLSX through the hardened project-question parser.
- Store an immutable project question set whose content hash is recorded in the run request.
- Compute Recall@K, MRR@K, nDCG@K, latency, and eligible recommendations only from labelled queries.
- Do not claim that arbitrary reusable ground truth is semantically compatible with the selected dataset. Project-linked ground truth is marked `linked`; other valid sources are marked `compatibility_unverified` and require an explicit warning/acknowledgement before execution.

#### No ground truth selected

- Require at least one typed, pasted, or uploaded query.
- Run extraction/index reuse, retrieval, selected rerankers, and evidence publication.
- Show source document, page when available, excerpt, retrieval score, reranker score when available, latency, run health, and evidence coverage.
- Set Recall, MRR, nDCG, accuracy, winner score, quality/value recommendations, and scored leaderboard fields to not applicable; never zero-fill them.
- Label the result `Evidence-only — manual review`.
- A fastest/lowest-latency operational view may be shown, but it must not be called the best-quality combination.

### 4. Evidence-only consistency for the default dataset

Remove the silent reranker drop in `run_complete_pipeline.py`. If rerankers are selected in evidence-only mode:

1. retrieve candidates;
2. run the selected rerankers;
3. publish evidence and operational latency;
4. skip every scored evaluator and winner/recommendation calculation.

If reranking cannot be supported on a specific adapter, preflight must reject that explicit combination. It must never silently ignore the selection.

### 5. Five-document demo pack

Create an isolated, fictional corpus named **Northstar Services RAG Demo**. It must not contain real WNS policy claims.

Deliverables:

- `northstar-demo-documents.zip` containing exactly five readable PDFs:
  1. Refund and cancellation policy
  2. Security incident response procedure
  3. Employee travel and expense policy
  4. Customer support SLA tiers
  5. Data retention and deletion policy
- `northstar-demo-groundtruth.csv` with at least ten labelled questions, including exact single-document facts, one multi-document comparison, and one deliberately unanswerable question.
- `northstar-demo-queries.txt` with at least five unlabelled questions for evidence-only mode.
- `README.md` with a two-path demo script:
  - scored run with ground truth;
  - evidence-only run without ground truth.
- `EXPECTED_RESULTS.md` stating which document(s) should support each query, what is expected to be not applicable without labels, and what failures indicate.

The documents use distinctive facts and terminology so extraction/retrieval errors are diagnosable. The unanswerable question must not be treated as a retrieval-quality success merely because some text was returned.

### 6. UI behavior

- Dataset and ground-truth source context remains global.
- Selecting an uploaded project changes Check readiness and Run pipeline to the project-matrix endpoints.
- The selected combination count must exactly equal the server-validated Cartesian count.
- Evidence-only keeps reranker multi-select enabled and explains: `Reranking is allowed; quality scoring is unavailable without labels.`
- Scored-only controls remain disabled in evidence-only mode.
- After project completion, Metrics, Recommendations, and Top Hits Evidence load the exact new `project_id` + `run_id`; no official rows remain on screen.
- Upload file `accept` filters change by upload type and mirror backend-supported formats.
- Loading, empty, warning, failed, cancelled, partial, and completed states are explicit and preserve the last successful result separately from the latest attempt.

## Security and integrity requirements

- Opaque ID resolution only; no browser paths. Public legacy actions must stop accepting `groundtruth=<filesystem path>` and resolve only the selected catalog ID.
- Existing path/symlink containment and upload limits remain enforced.
- Exact request schema; unknown keys and duplicate selection IDs are rejected.
- Project/run ownership and request fingerprint are verified before execution and readback.
- Large matrices require a server-issued confirmation token.
- No official artifact directory may be written by an uploaded-project run.
- No source/project result may be displayed under a different active dataset or ground-truth selection.
- Errors expose safe public messages and request IDs, not provider credentials, local paths, raw child-process logs, or provider exception details.
- Deployment must enforce an authentication/authorization boundary before exposing process-launch and upload routes beyond localhost; binding a mutation-capable server publicly without such a boundary is not an acceptable production configuration.

## Additional release gates from the parallel audits

### Official artifact provenance

Official rows are rankable only when all of these are verified:

- an approved official result source and artifact root;
- an explicit completed status;
- the expected official dataset and ground-truth IDs;
- a valid matching manifest/config fingerprint and run identity;
- the complete required metric set and positive evaluated-query count;
- membership in the exact configured canonical combination-key set.

Blank, unknown, failed, copied, stale, wrong-ground-truth, or unmanifested rows remain diagnostic only. Missing status is not rewritten to `completed`. Official matrix completion compares exact expected keys, not only row count. Summary and detail/evidence artifacts must share the same run identity.

### Partial and failed project runs

- `partial` and `failed` runs remain inspectable but are never labelled as completed result sets.
- A partial scored run may show completed rows and failure diagnostics, but it receives no final Quality/Speed/Value winner recommendation.
- Every run writes an atomic terminal manifest: `completed`, `partial`, `failed`, or `cancelled`.
- Legacy run IDs become collision-safe UUID-backed IDs with exclusive run-root creation.

### Canonical schema contract

Catalog validation, query loading, and evaluation use one shared ground-truth schema parser and one alias table. A source that the catalog marks valid cannot later be silently interpreted as having no relevance labels. Unsupported or ambiguous schemas fail before execution.

### Source-consistent frontend

- The global source context governs Overview, Metrics, Recommendations, Evidence, Documents, Run Pipeline, and every source-sensitive KPI. A page that is intentionally official-only must say so and must not inherit an uploaded source label.
- Refresh re-applies the exact active dataset, ground-truth, result-set, and run identity before rendering. It cannot restore official winners under an incompatible selection.
- `Done · open Metrics` routes to the canonical `quality` page, selects the exact combination, and leaves exactly one tab/panel active. Unknown hashes fall back to Overview.
- Metrics paginates or virtualizes every matching row and states the visible range; no silent `slice(0, 60)` truncation remains.
- Dataset changes rebuild or annotate ground-truth choices by compatibility. `groundtruth:none` remains available where evidence-only is supported. Structurally valid but non-linked ground truth is visibly `compatibility unverified` rather than implied compatible.
- Matrix selections become touch/keyboard-safe controls where `All` is mutually exclusive with individual values. The server-validated count is authoritative.
- One immutable request snapshot is used for preflight, confirmation, and launch. Conflicting controls are disabled during submission/running to prevent duplicate expensive jobs.
- Loading, empty, incompatible, evidence-only/not-applicable, partial, failed, and completed states have distinct copy and accessible live-region behavior.

### Explicit unlabelled reranking mode

To remove contradictory semantics while preserving Karthik's requirement, evidence-only supports two explicit operational variants:

1. `Evidence-only retrieval` — retrieval without reranking.
2. `Evidence-only + reranking` — selected rerankers execute and reranker scores/latency are shown, but all relevance-quality metrics and winner claims remain not applicable.

Commercial rerankers show a cost/provider warning and require explicit selection; they are never silently invoked. Both variants remain under the broader UI label `Evidence-only — manual review` and retain `groundtruth:none` identity.

## Test-first implementation order

1. Add failing server tests proving uploaded-project preflight/launch dispatches to `run_project_matrix.py`, writes an immutable request under the correct project, and never invokes the legacy complete pipeline.
2. Add failing tests for query-source import, compatibility warnings, large-matrix confirmation, and safe errors.
3. Add failing execution tests proving evidence-only project runs execute rerankers while quality metrics remain null/not applicable.
4. Add failing legacy-run tests proving selected evidence-only rerankers are executed rather than silently ignored.
5. Add frontend tests for endpoint dispatch, source/run identity, evidence-only copy, upload `accept` filters, and stale-result clearing.
6. Generate the five-document fixture, validate all PDFs and CSV schemas, upload it through the real endpoint, and run both demo paths against configured adapters.
7. Run focused suites, full runnable suites, API probes, desktop/mobile browser QA, console/network checks, ZIP integrity, and checksum verification.

## Acceptance criteria

- An uploaded five-document ZIP becomes a ready isolated project without changing the official WNS corpus.
- One selected project matrix can be preflighted, launched, polled, and read back entirely from the webapp.
- Every selected combination has a truthful terminal state and immutable receipt.
- Ground-truth mode produces scored metrics only for applicable labelled queries.
- No-ground-truth mode runs retrieval plus selected rerankers, shows evidence, and exposes no quality score or winner claim.
- The exact new uploaded-project result appears consistently in Metrics, Recommendations, and evidence views.
- Changing active source cannot leave stale official/project rows visible.
- The five-document pack demonstrates extraction, chunking, embedding, vector-store ingestion/reuse, retrieval, reranking, evaluation, evidence browsing, Metrics, and Recommendations. Answer generation/Grounding Audit remains disabled because it is not an active verified component.
- All regressions, syntax checks, live API checks, browser checks, and packaging checks pass with evidence recorded. Any environment-limited check is labeled as blocked rather than passed.
