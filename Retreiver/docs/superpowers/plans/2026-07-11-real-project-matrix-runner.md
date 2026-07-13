# Real Uploaded-Project Matrix Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the user-selected chunker × embedding × vector-store × reranker matrix against exactly one uploaded project, producing honest per-combination evidence and ground-truth scores without touching or reading another project or the official WNS benchmark artifacts.

**Architecture:** Build a project-only runner beside the existing official benchmark runner. Inputs are resolved exclusively through `ProjectWorkspace` and an immutable run manifest; adapter namespaces include project and run identity; all outputs are written beneath `data/user_projects/<project_id>/runs/<run_id>/` and published atomically. Reuse the real provider/vector/reranker adapters, but do not reuse workbook paths, global smoke directories, or `data/modular_runs/latest`.

**Tech Stack:** Python 3.11+, stdlib CSV/ZIP/JSON, openpyxl for XLSX questions, existing benchmark adapter registry, FAISS/Qdrant/PGVector/Weaviate, Bedrock/remote rerankers, pytest, vanilla JavaScript dashboard, Node.js 18+ with lockfile-pinned `@playwright/test`.

**Prerequisites:** Complete Tasks 1–3 in `2026-07-11-project-isolation-hardening.md` first. The workspace implementation must include the `f178e34` invariants: canonicalize the base once; require genuine UUID4 IDs; reject project A → project B and outside symlinks; reject symlinks/junctions for every canonical child. The supplemental queue task below supersedes the base plan's reject-on-capacity registry because Karthik requires durable queued work and restart recovery.

---

## Non-negotiable contracts

- No accounts are introduced.
- Every upload has one UUID-backed `project_id`; every execution has one UUID-backed `run_id`.
- No project code may read `data/pdfs`, `data/retrieval_smoke`, `data/reranker_smoke`, `data/evaluation`, or `data/modular_runs/latest`.
- A real matrix row exists only if that exact chunker, embedding, store, and reranker executed.
- Without retrieval-valid labels, return evidence and latency only. Never emit accuracy, recall, MRR, nDCG, winner, or leaderboard claims.
- With partial retrieval labels, score only queries with labels applicable to that combination and expose exact labelled/unlabelled denominators. `answer|ground_truth` alone is reference-answer display data, not retrieval relevance.
- More than 50 combinations produces a warning. More than 100 requires an explicit confirmation nonce tied to the validated request fingerprint. The hard cap is the configured official option product, currently 180.
- The official WNS 180-combination config and artifacts are read-only and must retain their pre-change hashes.
- Project, run, question-set, documents, questions, and index directories are created exclusively (`O_CREAT|O_EXCL` / `mkdir(exist_ok=False)`) and reject pre-existing symlinks/junctions. Durable writes use a same-directory temp file, `flush`, `os.fsync(file_fd)`, atomic publication, then `os.fsync(parent_dir_fd)`; Linux paths use `dir_fd`/`O_NOFOLLOW` where available and always revalidate lexical-to-resolved identity immediately before publication.
- Product requests contain canonical production adapter IDs only. Tests may inject fake transports behind those IDs, but no local/fallback adapter ID may enter a product request or execution receipt.

## Run input contract

`runs/<run_id>/request.json`:

```json
{
  "schema_version": 1,
  "project_id": "refund-data_<uuidhex>",
  "run_id": "run_<uuidhex>",
  "top_k": 5,
  "questions_source": {"type": "typed", "query": "How do I refund a cancelled flight?"},
  "selections": {
    "chunkers": ["fixed_tok1200_ov150"],
    "embeddings": ["gte_multilingual_base"],
    "vector_stores": ["FAISS"],
    "rerankers": ["bge-reranker-base"]
  },
  "request_fingerprint": "<sha256>",
  "large_matrix_confirmation": null
}
```

For typed mode, `questions_source` has exactly `{"type":"typed","query":"..."}`: it carries no retrieval labels or reference answer, produces one canonical question with `source_row:1`, empty label arrays, and `reference_answer:null`, and is always evidence-only. For uploaded batches, `questions_source` is instead `{"type":"question_set","question_set_id":"questions_<uuidhex>","content_sha256":"..."}`. The API accepts IDs and selection names, never filesystem paths. The runner revalidates the manifest, question-set project membership, and content hash before execution.

## Run output contract

```text
data/user_projects/<project_id>/runs/<run_id>/
  request.json
  manifest.json
  status.json
  logs/run.log
  chunks/<chunker>.jsonl
  indexes/<adapter-namespace>/              # FAISS only; external namespaces recorded in manifest
  retrieval/<combo_id>.jsonl
  reranking/<combo_id>.jsonl
  evidence.json
  details.jsonl
  summary.csv
  analysis.json                            # only when labelled_query_count > 0 for at least one combination
```

All JSON/CSV/JSONL outputs use the durable-write sequence above: flush + file `fsync`, atomic publication, then parent-directory `fsync`. The worker publishes terminal `status.json` last; the runner never writes job status.

---

## Task 0: Freeze protected official artifacts

**Files:**
- Create: `configs/project_matrix_catalog.json`
- Create: `configs/protected_official_artifacts.sha256`
- Create: `configs/protected_runtime_artifact_roots.txt`
- Create: `scripts/verify_official_artifacts_unchanged.py`
- Create: `tests/test_official_artifact_protection.py`

- [ ] **Step 1: Enumerate and hash protected inputs before implementation**

Track `configs/protected_official_artifacts.sha256` with a deterministic repository-relative SHA-256 entry for required tracked `configs/benchmark.local.json`. The verifier exposes `--write-tracked-baseline <manifest> --tracked-path <path>`; this creation is exclusive, refuses overwrite unless `--replace-baseline` is explicitly supplied, and uses file + parent-directory `fsync`. Track `configs/protected_runtime_artifact_roots.txt` with exact roots `data/modular_runs`, `data/full_benchmark`, `data/evaluation`, `data/retrieval_smoke`, and `data/reranker_smoke`. The verifier fails when any required tracked path is missing or changed. Runtime snapshots recursively hash those exact roots and record an explicit `!MISSING <root>` marker when absent; comparison requires the same file set, hashes, and missing-root state. The script exposes exact commands `--write-runtime-baseline <file> --roots <roots-file>` and `--check-runtime-baseline <file> --roots <roots-file>`. Runtime baseline creation has the same exclusive/explicit-replacement/fsync rules. `configs/benchmark.local.json` is protected and must never be edited by this plan.

- [ ] **Step 2: Create a project-only canonical catalog**

`configs/project_matrix_catalog.json` references the same five chunker IDs, three embedding IDs, four vector-store IDs, and three reranker IDs; maps each to its production adapter/config; and contains no fallback adapter. Tests assert its Cartesian product is 180 and every ID resolves to the expected production class.

- [ ] **Step 3: Verify and commit**

```bash
mkdir -p artifacts/predeploy
python3 scripts/verify_official_artifacts_unchanged.py --write-tracked-baseline configs/protected_official_artifacts.sha256 --tracked-path configs/benchmark.local.json
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py --write-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
python3 -m pytest -q tests/test_official_artifact_protection.py
git add configs/project_matrix_catalog.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt scripts/verify_official_artifacts_unchanged.py tests/test_official_artifact_protection.py
git commit -m "test: freeze official benchmark artifacts"
```

---

## Task 1: Normalize project documents and make all five chunkers reusable

**Files:**
- Create: `source/services/project_chunking.py`
- Create: `source/services/project_documents.py`
- Modify: `source/services/project_workspace.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `scripts/apply_chunking_methods.py`
- Modify: `scripts/apply_candidate_chunking_methods.py`
- Create: `tests/test_project_chunking.py`
- Create: `tests/test_project_documents.py`

- [ ] **Step 1: Write failing tests for real chunk differentiation**

Create tests using crafted golden multi-section documents and a full API ingestion path. Upload TXT, PDF, and a safe ZIP containing TXT/PDF members; assert the handler publishes exactly `data/user_projects/<project_id>/extracted_text/documents.jsonl` plus a root `manifest.json` only after every source extracts successfully. Each immutable document row contains `schema_version`, `source_id`, `source_name`, `page_number`, `parser_method`, `text`, and `content_sha256`. TXT must decode as UTF-8/UTF-8-SIG; PDF uses the existing page extractor and rejects blank/unreadable pages; ZIP accepts only already-validated non-symlink TXT/PDF members, forbids nested archives, and preserves normalized member names. Any failed source sets project extraction status to `failed`, records sanitized per-source errors, and prevents publication of `documents.jsonl`; the runner must refuse incomplete projects.

Reject duplicate source IDs, hash mismatches, or text loaded from `search_index.json`/the duplicated workbook. Assert all five mappings call the expected function and parameters (`w4`, `w5`, `w6`, heading level 2, fixed 1200/150), every output contains only the supplied project sentinel, and chunk IDs are unique per method. Record algorithm version plus input/output hashes in each chunk artifact.

```python
def test_project_chunkers_execute_real_distinct_methods():
    docs = [ProjectDocument("policy.txt", SAMPLE_POLICY, {"page_number": "1"})]
    fixed = chunk_project_documents(docs, "fixed_tok1200_ov150")
    heading = chunk_project_documents(docs, "Heading_sections_l2")
    assert fixed and heading
    assert [c.paragraph for c in fixed] != [c.paragraph for c in heading]
    assert all("PROJECT_ALPHA_SENTINEL" in c.paragraph for c in fixed + heading)
    assert CHUNKER_SPECS["entity_heuristic_w4"].params == {"window_size": 4}
    assert CHUNKER_SPECS["entity_heuristic_w5"].params == {"window_size": 5}
    assert CHUNKER_SPECS["entity_heuristic_w6"].params == {"window_size": 6}
```

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m pytest -q tests/test_project_chunking.py
```

Expected: import failure for `source.services.project_chunking`.

- [ ] **Step 3: Extract dependency-light chunker functions**

Create `ProjectDocument`, an immutable documents writer/reader, a strict `OFFICIAL_PROJECT_CHUNKERS` mapping, and `chunk_project_documents(documents, method) -> list[benchmarking.core.schemas.Chunk]`. Add `extracted_text/documents.jsonl` to `ProjectWorkspace` as the sole canonical corpus artifact. Wire the existing upload handler so validation/staging completes first, extraction writes same-directory temporary rows, all source hashes/completeness are verified, and only then `os.replace` publishes `documents.jsonl` and an atomic project manifest with `extraction_status="complete"`, source/page counts, corpus SHA-256, and parser versions. On any source failure, publish only `extraction_status="failed"` plus sanitized failures and leave no canonical corpus. Move/refactor these existing algorithms without changing official behavior:

- `entity_heuristic_chunking(window_size=4|5|6)`
- `heading_sections_chunking(level=2)`
- `fixed_token_chunks(chunk_tokens=1200, overlap_tokens=150)`

Preserve source ID/name, page/parser metadata, algorithm/version, and input/output hashes on every chunk. Do not import pandas in the new service. Update both legacy scripts to import the shared functions; keep their CLI behavior unchanged. The project runner consumes `documents.jsonl`, never `search_index.json` or a generated workbook.

- [ ] **Step 4: Verify**

```bash
python3 -m pytest -q tests/test_project_documents.py tests/test_project_chunking.py tests/test_official_artifact_protection.py tests/test_modular_benchmark.py tests/test_benchmark_pipeline_core.py
```

- [ ] **Step 5: Commit**

```bash
git add source/services/project_chunking.py source/services/project_documents.py source/services/project_workspace.py scripts/serve_benchmark_dashboard.py scripts/apply_chunking_methods.py scripts/apply_candidate_chunking_methods.py tests/test_project_chunking.py tests/test_project_documents.py
git commit -m "refactor: expose real project chunking methods"
```

---

## Task 2: Parse typed and uploaded question sets without inventing ground truth

**Files:**
- Create: `source/services/project_questions.py`
- Create: `source/services/project_relevance.py`
- Create: `configs/project_relevance_stopwords_v1.txt`
- Create: `configs/project_relevance_stopwords_v1.sha256`
- Create: `tests/test_project_questions.py`
- Create: `tests/test_project_relevance.py`
- Modify: `source/services/project_workspace.py`

- [ ] **Step 1: Write failing parser tests**

Cover:

- typed single query;
- immutable uploaded TXT/CSV/XLSX question sets, each with a UUID4-backed `question_set_id`, exclusive directory, raw content hash, and normalized artifact;
- `.txt` with one non-empty question per line;
- CSV/XLSX using `question` or `query`;
- uploaded CSV/XLSX retrieval-label columns: `reference_context|context`, `source_id`, or chunker-qualified `chunk_ref`; `answer|ground_truth` alone is not retrieval ground truth and remains evidence-only; typed and TXT modes carry no labels;
- empty/duplicate questions;
- missing question column;
- maximum question count and 4,000-character query limit;
- a question set belonging to project B rejected when executing project A.

Define normalized `questions.jsonl` rows exactly as:

```json
{"schema_version":1,"question_id":"q_000001","query":"...","labels":{"reference_contexts":["..."],"source_ids":["source_<sha256>"],"chunk_refs":[{"chunker":"fixed_tok1200_ov150","chunk_id":"42"}]},"reference_answer":"...","source_row":2}
```

Column mapping is explicit: `question|query` → `query`; `reference_context|context` → `labels.reference_contexts`; `source_id` → exact source IDs; `chunk_ref` must always use `<canonical_chunker_id>:<chunk_id>`; `answer|ground_truth` → `reference_answer` only and never retrieval relevance. **Bare `chunk_id` values are always rejected**, regardless of the current run selection, because immutable question sets may be reused by future multi-chunker runs. Canonical JSONL never omits fields: missing labels are empty arrays, a missing/blank answer is JSON `null`, and `source_row` is a one-based physical source position (`1` for typed mode and TXT line 1; `2` for the first CSV/XLSX data row after its header). CSV/XLSX list-valued label cells use JSON arrays or semicolon-delimited values; whitespace-only entries are removed, original order is retained, and exact duplicates are removed. Typed and TXT questions always serialize `labels:{"reference_contexts":[],"source_ids":[],"chunk_refs":[]}` and `reference_answer:null`.

Relevance is deterministic per combination. `source_id` uses exact metadata equality. `chunk_ref` applies only to its named chunker and uses exact chunk-ID equality. `reference_context` uses tokenizer `wns_context_tokens_v1`: apply Unicode NFKC, `casefold()`, extract tokens with Python regex `[a-z0-9]+`, remove tokens appearing one-per-line in checked-in `configs/project_relevance_stopwords_v1.txt`, then convert each side to a **set** (duplicates do not increase recall). `configs/project_relevance_stopwords_v1.sha256` contains the checked-in SHA-256 pin for that exact file; verify the pin before every parse/score operation and fail closed on mismatch. Record tokenizer version and the verified pinned hash in every scored receipt. A context label is applicable only when its normalized context-token set contains at least three tokens. It matches when the full NFKC+casefold+collapsed-whitespace context is a substring of equivalently normalized hit text, or when set recall `len(context_tokens & hit_tokens) / len(context_tokens) >= 0.80`. A hit is relevant when any applicable exact source/chunk label or applicable context label matches. A query enters one combination's denominator only when it has at least one label applicable to that chunker/combination; otherwise it increments that combination's unlabelled count. `answer|ground_truth` alone is never applicable and remains unscored.

Add mixed-label, answer-only, ground-truth-only, source-ID, chunker-specific chunk-ref, and per-combination denominator tests.

```python
def test_answer_only_questions_remain_evidence_only(tmp_path):
    cases = parse_question_bytes("questions.csv", b"question,answer\nrefund policy,ask support\n")
    assert cases[0].reference_answer == "ask support"
    assert cases[0].labels.is_empty()
    assert metric_denominator(cases, chunker="fixed_tok1200_ov150") == 0
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_questions.py
```

- [ ] **Step 3: Implement normalized question cases**

Implement immutable `ProjectQuestion` and `RetrievalLabels` dataclasses matching the schema above; do not overload legacy `QueryCase.expected_text`. Preserve `answer|ground_truth` only as `reference_answer`. Implement `wns_context_tokens_v1` and the exact per-combination evaluator in `project_relevance.py`; load only the checked-in stopword file, verify it against `configs/project_relevance_stopwords_v1.sha256` before use, fail closed on mismatch, and never use environment/NLTK stopwords. Validate files by extension/signature through the upload service, cap rows and lengths from environment, preserve a source-row identifier, and atomically save each upload under `questions/<question_set_id>/` using exclusive creation. Store raw file, `manifest.json`, and normalized `questions.jsonl` with SHA-256. Never overwrite a question set or search outside that project.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest -q tests/test_project_questions.py tests/test_project_relevance.py tests/test_project_isolation.py
git add source/services/project_questions.py source/services/project_relevance.py source/services/project_workspace.py configs/project_relevance_stopwords_v1.txt configs/project_relevance_stopwords_v1.sha256 tests/test_project_questions.py tests/test_project_relevance.py
git commit -m "feat: parse project-scoped question sets"
```

---

## Task 3: Validate array selections and large-matrix confirmation

**Files:**
- Create: `source/services/project_matrix_contract.py`
- Create: `tests/test_project_matrix_contract.py`
- Read only: `configs/benchmark.local.json`
- Read: `configs/project_matrix_catalog.json`

- [ ] **Step 1: Write failing contract tests**

Assert selection arrays are non-empty, deduplicated, and limited to canonical production values from `configs/project_matrix_catalog.json`; unknown values, `none`, and local/fallback adapters fail closed. Assert exact Cartesian counts, warning at 51, confirmation requirement at 101, hard rejection above configured maximum, and confirmation nonce mismatch/expiry rejection.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_matrix_contract.py
```

- [ ] **Step 3: Implement immutable validated request**

Create frozen `ProjectMatrixRequest` and `ValidatedMatrix` dataclasses. Build a canonical JSON request fingerprint with SHA-256. Confirmation tokens must be HMAC-signed, short-lived, and bound to project ID, selections, top-K, immutable question-set ID/hash (or typed query), and fingerprint. Keep `PROJECT_MATRIX_CONFIRMATION_KEY_FILE` outside the repo, generate it once with mode 0600 during VM setup, and never log it.

- [ ] **Step 4: Prove official matrix unchanged**

```bash
python3 - <<'PY'
from pathlib import Path
from benchmarking.core.config import generate_matrix, load_benchmark_config
cfg = load_benchmark_config(Path('configs/benchmark.local.json'))
assert len(generate_matrix(cfg)) == 180
print('official_matrix=180')
PY
git diff --exit-code -- configs/benchmark.local.json
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
python3 -m pytest -q tests/test_project_matrix_contract.py tests/test_modular_benchmark.py
```

- [ ] **Step 5: Commit**

```bash
git add source/services/project_matrix_contract.py tests/test_project_matrix_contract.py
git commit -m "feat: validate uploaded project matrix requests"
```

---

## Task 4: Add explicit project/run namespaces to vector adapters

**Files:**
- Modify: `benchmarking/adapters/vector_faiss.py`
- Modify: `benchmarking/adapters/vector_qdrant.py`
- Modify: `benchmarking/adapters/vector_pgvector.py`
- Modify: `benchmarking/adapters/vector_weaviate.py`
- Create: `tests/test_project_vector_namespaces.py`

- [ ] **Step 1: Write failing namespace tests**

Use mocked clients/connections. For two project IDs and two run IDs, assert every physical namespace differs. Assert namespace values contain no path separators or SQL/GraphQL syntax. Assert FAISS resolves beneath that run's `indexes/` directory. Add project A → B, run A → B, `questions`, and `indexes` symlink/junction rejection tests plus exclusive run-directory creation tests. Assert omitting an explicit namespace preserves official benchmark behavior.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_vector_namespaces.py
```

- [ ] **Step 3: Implement explicit namespaces**

Add optional explicit namespace parameters while preserving current defaults:

- Qdrant: collection name derived from a sanitized hash-backed project/run/combo identity;
- PGVector: quoted, strict `[a-z0-9_]+` table name with bounded length;
- Weaviate: valid class name beginning with an uppercase letter and containing only alphanumerics;
- FAISS: `index_dir` resolved beneath `runs/<run_id>/indexes/`.

Expose `physical_namespace` and idempotent `drop_namespace()` on each adapter. The runner records namespaces in its manifest. Never use PID + second timestamps when an explicit project namespace is supplied. Run and index directories use exclusive creation and no-follow validation from the corrected `ProjectWorkspace` contract.

- [ ] **Step 4: Verify and commit**

```bash
python3 -m pytest -q tests/test_project_vector_namespaces.py tests/test_modular_benchmark.py
git add benchmarking/adapters/vector_*.py tests/test_project_vector_namespaces.py
git commit -m "feat: isolate project vector namespaces"
```

---

## Task 5: Execute a real project-scoped matrix

**Files:**
- Create: `source/services/project_matrix_runner.py`
- Create: `scripts/run_project_matrix.py`
- Create: `tests/test_project_matrix_runner.py`
- Modify: `source/services/project_workspace.py`

- [ ] **Step 1: Write failing end-to-end runner tests with canonical IDs and injected transports**

Create project Alpha and project Beta with unique sentinel text. Run a small matrix for Alpha whose request retains canonical production IDs. Inject deterministic fake provider transports and vector clients behind those canonical IDs; never place local/fallback IDs in the request. Separately assert every canonical ID resolves to its intended production adapter class. Assert:

- every evidence excerpt contains Alpha's sentinel and never Beta's;
- the number of summary rows equals the exact selected Cartesian product;
- each row records the adapters that actually executed;
- different chunkers reference their own chunk artifacts;
- run IDs and physical namespaces differ across runs;
- no global benchmark directory is created or modified;
- evidence-only runs omit score/leaderboard fields;
- labelled runs expose per-combination labelled/unlabelled denominators and metrics only over applicable retrieval labels;
- any failed adapter row is marked failed and is never cloned into another combination;
- execution receipts record canonical adapter ID, adapter class, config hash, embedding dimensions/vector count, physical namespace, provider request ID/response metadata where available, and prohibit fallback flags.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_matrix_runner.py
```

- [ ] **Step 3: Implement runner lifecycle**

`ProjectMatrixRunner.run(project_id, run_id)` must:

1. Resolve only through `ProjectWorkspace`.
2. Revalidate `request.json` and its fingerprint.
3. Load only that project's hash-verified `documents.jsonl` and typed/normalized questions; fail if extraction is incomplete or references `search_index.json`/a workbook.
4. Generate and atomically persist chunks per selected chunker.
5. Cache embeddings by `(chunker, embedding)` within this run only.
6. Create one explicit vector namespace per `(project, run, chunker, embedding, store)` and reuse retrieval across rerankers.
7. Execute the selected real reranker for every query.
8. Persist retrieval/reranking rows with source filename, page, paragraph, base score, rerank score, adapter IDs, query ID, and latency.
9. Score with `project_relevance.py` only: exact `source_id`, chunker-namespaced exact `chunk_ref`, or the documented normalized `reference_context` rule. `answer|ground_truth` alone is unscored. Record per-combination labelled and unlabelled denominators because chunk refs may apply to only one chunker.
10. Publish `evidence.json`, `details.jsonl`, `summary.csv`, and optional `analysis.json` atomically.
11. Persist per-combination progress and execution receipts; never write `status.json`. Exit with a machine-readable aggregate result so the worker alone can publish `completed`, `partial`, `failed`, or cancellation-related terminal status. Retain sanitized failure evidence and never copy another row's output.

The CLI accepts only `--project-id` and `--run-id`; no arbitrary input/output paths.

- [ ] **Step 4: Verify official artifacts are untouched**

Immediately before and after every runner test, execute both exact verifier commands below. The tracked check is manifest-driven; the explicit Git command covers every tracked protected input used by this plan. The runtime check compares the full file set, hashes, and missing-root markers captured at Task 0. Run the same pair around live laptop and VM smoke runs, using that machine's pre-smoke runtime baseline.

```bash
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
git diff --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
```

- [ ] **Step 5: Verify and commit**

```bash
python3 -m pytest -q tests/test_project_matrix_runner.py tests/test_project_isolation.py tests/test_modular_benchmark.py
python3 -m py_compile source/services/project_matrix_runner.py scripts/run_project_matrix.py
git add source/services/project_matrix_runner.py source/services/project_workspace.py scripts/run_project_matrix.py tests/test_project_matrix_runner.py
git commit -m "feat: run real isolated project matrices"
```

---

## Task 6: Implement the persistent FIFO worker, then wire project APIs

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Create: `source/services/project_job_queue.py`
- Create: `source/services/project_attempt_supervisor.py`
- Create: `scripts/run_project_job_worker.py`
- Create: `scripts/run_project_job_attempt.py`
- Create: `deploy/systemd/wns-project-runner@.service.in`
- Modify: `tests/test_dashboard_metrics.py`
- Create: `tests/test_project_job_queue.py`
- Create: `tests/test_project_attempt_supervisor.py`
- Create: `tests/test_project_run_api.py`

- [ ] **Step 1: Write failing API tests**

Cover immutable question-set upload, create-run, large-matrix warning/confirmation, project not found, invalid selection, bounded FIFO capacity, duplicate active fingerprint/idempotent retry, API-exclusive queued-record creation, worker-only post-enqueue mutation, atomic claim, single worker lease/heartbeat, deterministic attempt-unit naming, expired lease recovery, cgroup-wide cancellation, namespace cleanup/retention, per-combination progress, completed results, and evidence-only versus scored response shapes. Inject crashes after temp-file `fsync`, publication, parent-directory `fsync`, durable `launching` publication, systemd unit start, attempt identity receipt, and `running` publication. Kill the worker while its systemd-owned runner and a descendant writer remain alive; the restarted worker must stop/kill the entire cgroup, prove `populated=0`, write cleanup receipt, and only then retry. Assert no overlapping artifact/vector writers, unknown/ambiguous cgroup state becomes `interrupted`, and capacity counts `queued + launching + running`. Assert project/question-set IDs and unit names cannot be substituted after confirmation.

- [ ] **Step 2: Implement the exact queue state machine and API**

Queue control metadata lives at `data/project_job_queue/` and contains IDs/hashes only—never uploaded content. Each job record is `jobs/<run_id>.json`:

```json
{"schema_version":1,"sequence":42,"project_id":"...","run_id":"...","request_sha256":"...","state":"queued","attempt":0,"max_attempts":2,"lease_token":null,"lease_expires_at":null,"attempt_unit":null,"attempt_control_group":null,"cancel_requested_at":null,"created_at":"...","updated_at":"...","error_code":null}
```

The API is the **exclusive creator** of a job in `queued` state. While holding `locks/queue.lock` with `fcntl.flock(LOCK_EX)`, it verifies capacity/fingerprint, allocates the next monotonic value in `counter.json`, durably publishes the counter, exclusively creates `jobs/<run_id>.json`, and durably publishes initial `runs/<run_id>/status.json`; it never mutates that job record afterward. Exclusive publication writes and `fsync`s a same-directory temporary file, hard-links it to the absent final name (failing if the final name exists), `fsync`s the parent directory, then unlinks the temporary name and `fsync`s the directory again. Counter and later state replacements write temp, `fsync` the file, `os.replace`, then `fsync` the parent directory.

After creation, the **single worker is the sole mutator** of queue records and `status.json`; the runner writes only `progress.json` and combination/result artifacts. The worker holds `locks/worker.lock`, then takes `queue.lock` for every claim/mutation. Claim changes the lowest-sequence queued record to durable `launching`, increments `attempt`, creates random 256-bit attempt/lease tokens, and persists the deterministic unit `wns-project-runner@<run_uuid_hex>-a<attempt>.service` **before** asking systemd to start it. Unit names are constructed only from validated UUID4 run IDs and bounded integer attempts—never request strings.

`project_attempt_supervisor.py` starts/stops only that exact template through `systemctl`. The tracked template runs `scripts/run_project_job_attempt.py` as the unprivileged service user, loads the immutable run request by validated ID, and sets `Type=exec`, `KillMode=control-group`, `TimeoutStopSec=10`, `SendSIGKILL=yes`, `Delegate=no`, and `ProtectControlGroups=yes`. The root control worker never opens uploaded content or executes request-provided commands; it only validates IDs/hashes, mutates queue metadata, and controls canonical attempt-unit names. After start, it reads and durably records `Id`, `ActiveState`, `SubState`, `MainPID`, and `ControlGroup` from `systemctl show`, validates that `ControlGroup` belongs to the exact unit, then transitions `launching → running` with a 45-second lease. It heartbeats every 10 seconds while verifying attempt and lease tokens. A crash before start leaves a known inactive/not-found deterministic unit; a crash after start still leaves every attempt process owned by that known cgroup.

Expired-lease recovery never immediately requeues. For the persisted deterministic unit, issue `systemctl stop`; after 10 seconds issue `systemctl kill --kill-whom=all --signal=KILL` if needed. Resolve the unit's exact cgroup beneath `/sys/fs/cgroup` without following symlinks and poll `cgroup.events` until `populated 0`. An inactive/not-found unit is proof of cleanup only when its previously recorded cgroup path is absent; systemd may remove that path only after the cgroup is empty. A `launching` record with no control-group receipt is resolved by querying the already-persisted deterministic unit name and follows the same stop/emptiness proof. Missing permissions, mismatched unit/cgroup identity, unreadable `cgroup.events`, timeout, or any ambiguous state transitions terminally to `interrupted` and never retries. Durably write `attempts/<attempt>/cleanup.json` with unit identity, observations, signals, and emptiness proof **before** `running|launching → queued(retry)`. Tests include a descendant writer that survives its leader and ignores SIGTERM; no retry is allowed until the entire old cgroup reports `populated 0`.

The API requests cancellation by exclusively and durably creating `runs/<run_id>/cancel.request.json` using file + parent-directory `fsync`; it never edits job status. A duplicate create-run request with the same active fingerprint returns the already-created `run_id`, so a crash after job publication but before HTTP response cannot duplicate work. On startup, the worker reconciles each complete job record with `request.json` and repairs a missing initial queued `status.json`; an orphan temp/status without a final job is not runnable and is quarantined with a sanitized receipt. Cancellation uses the same deterministic-unit stop/kill/cgroup-empty proof and durably writes `cancel.ack.json` plus cleanup receipt before `cancelled`. Ambiguity becomes `interrupted`. Retry occurs only after proven cgroup cleanup and when `attempt < max_attempts`. Valid transitions are:

```text
queued -> launching | cancelled
launching -> running | queued(recovered-before-exec) | cancelled | interrupted
running -> completed | partial | failed | cancelled | queued(retry-after-proven-cleanup) | interrupted
partial/completed/failed/cancelled/interrupted -> terminal
```

`partial` means at least one combination succeeded and at least one failed; `failed` means none succeeded. On `failed`, `cancelled`, or `interrupted`, drop all external namespaces listed in receipts unless `PROJECT_MATRIX_RETAIN_FAILED=1`; always retain uploads, canonical documents, questions, logs, status, progress, and cleanup receipts. While holding `queue.lock`, capacity counts **every nonterminal record** in `queued`, `launching`, or `running`; enqueue rejects when adding one would exceed `PROJECT_MATRIX_QUEUE_CAPACITY`.

Implement these exact endpoints:

```text
POST /api/projects/<project_id>/matrix/validate
POST /api/projects/<project_id>/question-sets
POST /api/projects/<project_id>/runs
GET  /api/projects/<project_id>/runs/<run_id>
GET  /api/projects/<project_id>/runs/<run_id>/results
POST /api/projects/<project_id>/runs/<run_id>/cancel
```

Question-set upload creates an immutable UUID4 directory and returns only ID/hash. The run handler creates `run_id` exclusively, durably writes `request.json`, calls the API-owned `enqueue_new_job()` creator described above, and returns only after queued job + initial status have been `fsync`ed; it never passes user paths or raw shell commands. Status GET combines worker-owned `status.json` with runner-owned `progress.json` without mutating either. Results are unavailable until a terminal status and required files exist.

- [ ] **Step 3: Verify and commit**

```bash
python3 -m pytest -q tests/test_project_job_queue.py tests/test_project_attempt_supervisor.py tests/test_project_run_api.py tests/test_dashboard_metrics.py tests/test_project_isolation.py
python3 -m py_compile source/services/project_job_queue.py source/services/project_attempt_supervisor.py scripts/run_project_job_worker.py scripts/run_project_job_attempt.py
git add source/services/project_job_queue.py source/services/project_attempt_supervisor.py scripts/run_project_job_worker.py scripts/run_project_job_attempt.py scripts/serve_benchmark_dashboard.py deploy/systemd/wns-project-runner@.service.in tests/test_project_job_queue.py tests/test_project_attempt_supervisor.py tests/test_project_run_api.py tests/test_dashboard_metrics.py
git commit -m "feat: expose queued project matrix runs"
```

---

## Task 7: Build the project matrix UI without fake scores

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Add failing frontend assertions**

Assert active project is always visible, controls are disabled without one, typed and immutable TXT/CSV/XLSX question-set modes are explicit, exact combination count is shown, >50 warns, >100 requires confirmation, queued/running/per-combination status survives refresh/restart, cancellation is visible, evidence-only/answer-only results contain no leaderboard language, and retrieval-labelled results display exact labelled/unlabelled denominators.

- [ ] **Step 2: Implement project-run controls and result views**

Use the validation endpoint before queueing. Upload batches first to obtain an immutable question-set ID/hash. Display one card/row per real combination status and execution receipt. Evidence shows query, source, page, excerpt, retrieval score, rerank score, and latency. Show retrieval leaderboard/quality metrics only when retrieval-valid labelled count is greater than zero. Keep lexical preview as a clearly separate quick mode.

- [ ] **Step 3: Verify and commit**

```bash
python3 -m pytest -q tests/test_dashboard_metrics.py tests/test_project_run_api.py
node --check web/app.js
git add web/index.html web/app.js web/styles.css tests/test_dashboard_metrics.py
git commit -m "feat: add honest project matrix workspace"
```

---

## Task 8: Exact VM provisioning, provider smoke, and cross-project isolation proof

**Files:**
- Create: `scripts/verify_project_matrix_isolation.py`
- Create: `scripts/verify_project_provider_lanes.py`
- Create: `scripts/verify_project_confirmation_roundtrip.py`
- Create: `scripts/install_project_matrix_services.py`
- Create: `deploy/systemd/wns-project-worker.service.in`
- Create: `deploy/systemd/wns-dashboard.service.in`
- Create: `tests/browser/project_matrix_e2e.spec.mjs`
- Create: `tests/fixtures/project_matrix/refund_policy.txt`
- Create: `package.json`
- Create: `package-lock.json`
- Modify: `README.md`
- Modify: `PIPELINE_RUNBOOK.md`
- Modify: `APPLY_ZIP.md`

- [ ] **Step 1: Add deterministic local isolation verification**

The script creates two temporary projects with different sentinels, executes local two-combination runs, asserts zero cross-project evidence, checks atomic artifacts and distinct namespaces, then removes only its temporary projects.

- [ ] **Step 2: Provision and run bounded live provider coverage**

Render the three tracked systemd templates with absolute repository path and service user; install them as `/etc/systemd/system/wns-project-worker.service`, `wns-project-runner@.service`, and `wns-dashboard.service`. The worker is a root **control-only** service because it must start/stop the exact runner template; harden it with `NoNewPrivileges=yes`, `ProtectSystem=strict`, explicit queue/project `ReadWritePaths`, and no network access. It must reject every unit name not canonically derived from a validated UUID4 run ID and bounded attempt. The runner template and dashboard run as the unprivileged service user; the runner has `KillMode=control-group`, `SendSIGKILL=yes`, `Delegate=no`, and `ProtectControlGroups=yes` so descendants cannot escape attempt cgroup ownership. The installer creates `/var/lib/wns-project-matrix/confirmation.key` once using 32 random bytes, explicitly `chown`s it to the configured service user/group, and sets mode `0600`; it refuses to replace the key or proceed when owner, mode, or service-user readability is wrong. It writes `/etc/wns-project-matrix.env` mode `0600` with `PROJECT_MATRIX_CONFIRMATION_KEY_FILE`, queue limits, repo/data paths, and existing provider environment references. `verify_project_confirmation_roundtrip.py` runs as that service user, reads the configured key, issues and verifies one request-bound short-lived token, and proves tampering/expiry rejection. Provision tracked Python requirements, verify the model adapter and four vector services, and create writable `data/user_projects`.

Track this exact Node manifest and generate/commit its npm v3 lockfile with `npm install --package-lock-only`; `npm ci` must report no lock drift:

```json
{"name":"wns-project-matrix-qa","private":true,"devDependencies":{"@playwright/test":"1.61.1"}}
```

Run exactly:

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm
node -e 'const m=+process.versions.node.split(".")[0]; if(m<18) process.exit(1)'
npm ci
npx playwright install --with-deps chromium
sudo python3 scripts/install_project_matrix_services.py --repo-root "$PWD" --service-user "$USER"
sudo -u "$USER" test -r /var/lib/wns-project-matrix/confirmation.key
sudo -u "$USER" env PROJECT_MATRIX_CONFIRMATION_KEY_FILE=/var/lib/wns-project-matrix/confirmation.key python3 scripts/verify_project_confirmation_roundtrip.py
sudo systemctl daemon-reload
sudo systemctl enable --now wns-project-worker.service wns-dashboard.service
sudo systemctl restart wns-project-worker.service wns-dashboard.service
systemctl is-active --quiet wns-project-worker.service
systemctl is-active --quiet wns-dashboard.service
python3 scripts/check_services.py
curl -fsS http://127.0.0.1:5011/api/health
mkdir -p artifacts/predeploy
if [ -e artifacts/predeploy/runtime-artifacts.sha256 ]; then
  python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
else
  python3 scripts/verify_official_artifacts_unchanged.py --write-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
fi
python3 scripts/verify_project_matrix_isolation.py
SMOKE_RUN_DIR="$PWD/artifacts/provider-smoke"
mkdir -p "$SMOKE_RUN_DIR"
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
git diff --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
git diff --cached --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_project_provider_lanes.py \
  --create-smoke-project-from tests/fixtures/project_matrix/refund_policy.txt \
  --query "cancelled flight refund" --top-k 3 \
  --project-metadata "$SMOKE_RUN_DIR/project.json" \
  --output "$SMOKE_RUN_DIR/provider_receipts.json"
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
git diff --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
git diff --cached --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
SMOKE_PROJECT_ID="$(python3 -c 'import json; print(json.load(open("artifacts/provider-smoke/project.json"))["project_id"])')"
test -n "$SMOKE_PROJECT_ID"
```

The provider-lane verifier executes a bounded pairwise covering table: every one of 3 embeddings, 4 stores, and 3 rerankers appears in at least one real combination; all 5 chunkers run in a separate five-row chunking smoke. It proves:

- each selected chunker creates its own algorithm/version/input/output-hashed artifact;
- OpenAI/GTE/Jina embedding adapters produce real vectors and receipts;
- FAISS/Qdrant/PGVector/Weaviate use distinct project/run namespaces;
- BGE/Qwen/Amazon rerankers return real reranked evidence and provider receipts;
- failed providers produce failed rows rather than duplicated fallback evidence.

Use one typed query and low top-K for smoke cost control. Any missing receipt, fallback flag, namespace reuse, mixed sentinel, or adapter failure fails the release. Do not rerun the official 180 matrix.

- [ ] **Step 3: Full verification**

```bash
python3 -m pytest -q \
  tests/test_official_artifact_protection.py \
  tests/test_project_isolation.py \
  tests/test_project_documents.py \
  tests/test_project_chunking.py \
  tests/test_project_questions.py \
  tests/test_project_relevance.py \
  tests/test_project_matrix_contract.py \
  tests/test_project_vector_namespaces.py \
  tests/test_project_matrix_runner.py \
  tests/test_project_job_queue.py \
  tests/test_project_attempt_supervisor.py \
  tests/test_project_run_api.py \
  tests/test_dashboard_metrics.py \
  tests/test_modular_benchmark.py \
  tests/test_benchmark_pipeline_core.py
python3 scripts/verify_project_matrix_isolation.py
sudo -u "$USER" env PROJECT_MATRIX_CONFIRMATION_KEY_FILE=/var/lib/wns-project-matrix/confirmation.key python3 scripts/verify_project_confirmation_roundtrip.py
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
node --check web/app.js
npm ci
npx playwright --version
```

Expected: all tests pass; isolation verifier reports zero mixed evidence; official matrix remains 180.

- [ ] **Step 4: Browser QA and rollback drill**

Using tracked `tests/browser/project_matrix_e2e.spec.mjs`, upload two projects with the same filename but different sentinel content and prove only active-project evidence appears. Cover typed/TXT/CSV/XLSX question modes; answer-only evidence mode; mixed retrieval-labelled/unlabelled scoring denominators; 51 warning and 101 signed confirmation; queued/running/per-combination/completed refresh; worker restart/requeue; cancellation; provider failure with no cloned fallback; all-five chunker receipts; browser console and failed-network-request checks. Save screenshots, traces, console logs, network failures, and downloaded API JSON beneath `artifacts/playwright/project-matrix/`.

```bash
rm -rf artifacts/playwright/project-matrix
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
git diff --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
git diff --cached --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
PLAYWRIGHT_BASE_URL=http://127.0.0.1:5011 npx playwright test tests/browser/project_matrix_e2e.spec.mjs --output artifacts/playwright/project-matrix --trace on
python3 scripts/verify_official_artifacts_unchanged.py --check configs/protected_official_artifacts.sha256
git diff --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
git diff --cached --exit-code -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
python3 scripts/verify_official_artifacts_unchanged.py --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 --roots configs/protected_runtime_artifact_roots.txt
```

The command must exit zero and the artifact directory must contain screenshots, trace ZIPs, `console.jsonl`, `network-failures.jsonl`, and API captures. Then run the base plan's pre-deploy backup, service restart, health checks, and rollback drill. Verify rollback preserves `data/user_projects/`, question sets, run artifacts, queue records, and external vector volumes.

- [ ] **Step 5: Commit and package**

```bash
git add scripts/verify_project_matrix_isolation.py scripts/verify_project_provider_lanes.py scripts/verify_project_confirmation_roundtrip.py scripts/install_project_matrix_services.py deploy/systemd/wns-project-worker.service.in deploy/systemd/wns-dashboard.service.in tests/browser/project_matrix_e2e.spec.mjs tests/fixtures/project_matrix/refund_policy.txt package.json package-lock.json README.md PIPELINE_RUNBOOK.md APPLY_ZIP.md
git commit -m "test: verify real project matrix isolation"
```

Create the final handoff ZIP only after work-laptop push, VM pull, live provider smoke, browser QA, and rollback preservation all pass.

---

## Final acceptance checklist

- [ ] Two uploads with identical filenames create different project IDs and never share files, chunks, indexes, questions, runs, logs, results, or evidence.
- [ ] Two runs of one project create different run IDs and physical vector namespaces.
- [ ] Every displayed combination corresponds to adapters that actually executed.
- [ ] Five chunker selections run five actual chunking methods; they are not duplicated workbook sheets.
- [ ] Single typed query and TXT/CSV/XLSX question batches work.
- [ ] Every question upload receives an immutable UUID4 `question_set_id`; no upload overwrites or aliases another.
- [ ] Runner consumes hash-verified `documents.jsonl`, never `search_index.json` or duplicated workbook sheets.
- [ ] Without retrieval-valid labels, no quality score or leaderboard is emitted; `answer|ground_truth` remains reference-answer display data.
- [ ] Partial/full retrieval labels expose exact per-combination labelled/unlabelled denominators and metrics over applicable labels only.
- [ ] >50 warning and >100 bound confirmation work server-side and client-side.
- [ ] Persistent queue counts every nonterminal state, prevents concurrent heavy writes, durably survives crash boundaries, and never retries/cancels until the prior deterministic systemd attempt cgroup—including surviving descendants—proves `populated 0`; ambiguous cleanup becomes `interrupted`.
- [ ] Versioned relevance tokenizer, checked-in stopword hash, and per-combination denominators reproduce exactly across laptop and VM.
- [ ] Execution receipts prove canonical production adapter classes, dimensions/counts, namespaces, and provider metadata; fallback flags are forbidden.
- [ ] Tracked official hashes and the machine-specific runtime baseline both remain unchanged after tests and smoke.
- [ ] Lockfile-pinned Playwright Chromium QA runs with zero console/network failures and retains the required traces/screenshots/API captures.
- [ ] Work-laptop push → VM pull → service restart → health/API smoke → browser QA passes.
- [ ] Rollback preserves every uploaded project and database volume.
