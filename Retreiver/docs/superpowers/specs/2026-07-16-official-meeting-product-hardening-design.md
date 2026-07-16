# Official Meeting Product Hardening Design

## Goal

Make the WNS Project Smiley dashboard reliable for the official stakeholder meeting: every page must show the same selected source and canonical benchmark evidence on first render, uploaded datasets must be reusable without repeating completed stages, and Run Pipeline must present an understandable modular RAG flow without exposing misleading or inactive controls.

## Inputs and authority

This design combines:

- Tushar's observed dashboard and pipeline defects.
- Karthik's recorded meeting feedback.
- Existing global-source and Metrics/Recommendations selector designs.
- Existing official 180-combination benchmark evidence and no-reranker baseline evidence.

When requirements conflict, benchmark integrity and verified artifacts win. The implementation must not fabricate metrics, imply inactive features are operational, merge NVIDIA into the official matrix, or destroy prior successful artifacts.

## 1. Canonical active source state

The dashboard will own one `activeSource` state object containing:

- `dataset_id`
- `groundtruth_id`
- `result_set_id`
- `source_type`
- `scoring_mode`

One global source bar beside **Refresh live data** will expose Dataset, Ground truth, and Result set. Overview, Metrics, Recommendations, Run Pipeline, Top Hits Evidence, Documents, and NVIDIA controls will read the selected dataset and ground truth from this state where applicable. Page-local source selectors will be removed or converted into read-only context; component filters such as reranker, vector DB, or chunker remain page-local because they filter rows rather than select the underlying source.

Dataset and ground truth remain independent. Selecting a dataset must not silently infer semantic compatibility with a ground-truth file. Uploaded-project result sets are restricted to runs whose stored dataset and ground-truth IDs match the active selection.

Selection changes are transactional: update state, resolve the matching payload, and render all dependent views from that same payload. Stale asynchronous responses are rejected by generation/identity checks.

## 2. Canonical official result payload

The official reranked matrix and no-reranker baseline will each be normalized through one payload builder before any Overview, Metrics, Recommendations, or KPI rendering.

- Official reranked matrix: only canonical, completed reranked rows from benchmark reference evidence; expected configured coverage is 180.
- No-reranker baseline: only canonical rows whose reranker is `none`; measured count is displayed honestly.
- No initial mixed summary/reference payload may render stakeholder winner cards.
- Ranking, deduplication, labels, completion rules, and score fallback are identical across all consuming views.
- First render and baseline → official or filter-toggle render must produce identical winner and top-combination outputs.

This fixes the stale Amazon/BGE winner and Recommendations mismatch at the data-contract boundary rather than adding refresh side effects.

## 3. Modular Run Pipeline stage model

The product flow is:

1. Extraction
2. Chunking
3. Embedding
4. Vector-store ingestion
5. Retrieval
6. Reranking
7. Answer generation
8. Evaluation

The main Run Pipeline card presents the stage sequence, selected source, stage readiness, and one primary action. Advanced controls are grouped by these stage names and use user-facing language. Raw port/library diagnostics move to Operations or an expandable **Technical details** area.

The existing **Chunk limit** control is removed from the UI. `chunk_limit` remains an internal compatibility parameter fixed to `0`, meaning all chunks. Users must not truncate ingestion accidentally.

Two independent depth controls are exposed:

- **Retrieval Top K**: number of candidates returned by vector retrieval.
- **Reranked Output K**: number of reranked candidates retained for downstream evidence/answer generation.

Validation requires `Retrieval Top K >= Reranked Output K >= 1`. Both values are persisted in run manifests and evidence artifacts. Existing scripts that accept only `top_k` continue to receive retrieval depth; reranker scripts gain/receive an explicit output depth without reusing an artifact-count limit.

Query limit remains available for meeting-sized evaluated runs, where `0` means all ground-truth queries.

## 4. Uploaded dataset reuse and stage provenance

Each uploaded project manifest will expose immutable source identity and reusable stage artifacts:

- extracted documents/workbook
- chunking sheets and chunk counts
- embedding identity/version
- vector-store identity, collection/table name, and ingestion fingerprint
- retrieval and reranking run IDs
- ground-truth ID and scoring mode

A stage fingerprint is derived from dataset content identity plus the configuration inputs that affect that stage. Before execution, the planner marks each stage as:

- `ready_reusable`
- `required`
- `stale`
- `blocked`

If the selected uploaded dataset already has a matching vector-store ingestion fingerprint and the target store is reachable, reranking/evaluation starts from retrieval and does not repeat extraction, chunking, embedding, or ingestion. A user can explicitly rebuild, but reuse is the default.

Reused artifacts remain project-isolated. The browser sends opaque IDs; the server resolves paths and collection metadata from allowlisted manifests only.

## 5. Last-success and latest-attempt state

Document readiness and run status are split into two records:

- `last_successful_snapshot`: the last verified corpus state safe for Overview/Documents KPIs.
- `latest_attempt`: the most recent run attempt, including failures and warnings.

A failed or partial extraction attempt may update `latest_attempt` but cannot replace `last_successful_snapshot` with zero clean/chunked documents. Successful publication is atomic: write a temporary manifest, validate its counts/artifacts, then replace the last-success pointer.

The UI displays the last successful readiness count and separately shows the latest failed attempt. It never hides a failure, but it also never converts a known-good corpus into a false zero state.

## 6. Query input and file upload

Evidence-only and evaluation query inputs support:

- typed/pasted text, one non-empty line per query
- `.txt`, one non-empty line per query
- `.csv`, using a supported query-column alias
- `.xlsx`, using a supported query-column alias

The upload parser normalizes whitespace, rejects empty files, enforces a safe query-count cap, reports accepted/rejected counts, and never treats document evidence uploads as query files. Parsed queries are sent as structured query values and their source filename/count are persisted in the run manifest.

Ground-truth uploads remain a separate validated source type.

## 7. Evidence-only integrity

When ground truth is `groundtruth:none`:

- extraction/index reuse, retrieval, optional reranking, and evidence display are allowed
- evidence rows show rank, source document, page when available, excerpt, retrieval score, and reranker score when available
- Recall, MRR, nDCG, accuracy, winner score, and quality recommendations are suppressed
- the UI labels the mode **Evidence-only — manual review**

Future LLM judging is not presented as human ground truth.

## 8. Answer-generation and grounding contract

A post-reranking answer-generation stage is introduced as a modular contract, not a fake active feature. Its run manifest records:

- provider family: Gemini, OpenAI, Anthropic, or open-source
- configured model ID
- exact reranked evidence IDs supplied
- prompt/template version
- generated answer
- citations
- latency and token/usage metadata when available
- provider/runtime status

Provider adapters are registered server-side and exposed only when configured. If no adapter is configured, Answer generation is shown as unavailable and the retrieval/evaluation pipeline continues normally.

**Grounding Audit remains grey and disabled for the official meeting unless real answer-generation and grounding artifacts are present.** No synthetic or placeholder audit is used to activate it. Existing grounding endpoints stay internal/advanced and must not be triggered from the inactive public tab.

## 9. Documents page cleanup

**Lexical Preview is removed from the stakeholder UI.** It is a token-overlap diagnostic that does not run embeddings, vector retrieval, reranking, or scored evaluation and is easily mistaken for the real pipeline. The backend helper may remain temporarily for compatibility tests or internal diagnostics, but it is not advertised as product behavior.

Documents focuses on:

- uploaded datasets and ground-truth assets
- readiness and last successful state
- latest attempt/failure
- document counts and reusable index status
- source selection and upload actions

## 10. Simplified preflight UX

The primary control becomes **Check readiness**. Its result shows:

- Ready / Needs attention
- selected dataset and ground truth
- reusable versus required stages
- concise actionable document errors
- pipeline combination/query counts

Raw service URLs, port checks, import/library details, command output, and full extraction-audit text are placed under **Technical details** and mirrored in Operations. The audited text-only override remains explicit; it downgrades the known extraction condition to a warning and never silently skips the audit.

## 11. NVIDIA isolation

NVIDIA remains a separate evidence/benchmark lane. It may read the global dataset and ground-truth selection but does not contribute rows to the official modular matrix. Its text-first and future multimodal collections remain isolated, and baseline/reranked NVIDIA evidence remains separately labeled.

## 12. Compatibility and migration

- Preserve existing artifact files and official benchmark evidence.
- Do not delete project data or vector collections.
- Keep server-side compatibility for `chunk_limit=0` and legacy endpoints while removing the UI control.
- Do not stage or modify unrelated local files such as the existing untracked `conftest.py`.
- Bump frontend cache keys after final UI changes.
- No destructive Git operations.

## 13. Acceptance criteria

### Canonical state

- A fresh load and an official → baseline → official toggle produce identical official winners and top combinations.
- Overview, Metrics, and Recommendations show the same BGE/Amazon/Qwen result dictated by the same canonical rows; no page can retain stale rows.
- One global Dataset/Ground truth/Result set selection controls all applicable pages.

### Pipeline execution

- No user-facing Chunk limit exists; command construction always passes `chunk_limit=0`.
- Retrieval Top K and Reranked Output K are independent, validated, and persisted.
- An uploaded project's valid matching index can run retrieval/reranking/evaluation without extraction or embedding.
- A changed dataset/config invalidates reuse truthfully.
- Failed extraction cannot overwrite the last successful readiness count.

### UX and integrity

- Advanced controls use the agreed stage names.
- Grounding Audit is visibly disabled when no real grounding artifacts exist.
- Lexical Preview is absent from the stakeholder UI.
- TXT/CSV/XLSX query input parses one query per row/line and reports errors.
- Evidence-only mode renders evidence and no scored quality claims.
- Main readiness output is concise; backend detail remains accessible under technical details/Operations.

### Verification

- New regression tests fail before each behavior fix and pass afterward.
- Existing relevant Python and JavaScript suites pass.
- Python compile, JavaScript syntax, and `git diff --check` pass.
- The live dashboard API is probed.
- Desktop and mobile screenshots are captured and inspected.
- Browser console and failed requests are checked.
- The final ZIP is integrity-tested, lists critical files, and has a SHA-256 checksum.

## Out of scope

- Claiming semantic compatibility between arbitrary datasets and ground-truth files.
- Activating Grounding Audit without real answer-generation artifacts.
- Merging NVIDIA results into the official 180-combination matrix.
- Deleting old uploaded projects or benchmark evidence.
- Implementing paid provider calls without configured credentials and explicit runtime availability.
