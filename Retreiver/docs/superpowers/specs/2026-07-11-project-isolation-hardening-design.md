# Retreiver Project Isolation and VM Hardening Design

Date: 2026-07-11
Status: Approved direction, implementation pending

## Problem

Retreiver has evolved from a single-operator WNS benchmark dashboard into a tool where multiple people may upload and query their own documents. The current implementation creates project folders, but several paths still use global artifacts, in-memory job state, and a lexical preview duplicated across selected pipeline combinations. Concurrent or malformed requests can therefore mix, overwrite, or misrepresent project data.

The first real release does not need user accounts. It does need one absolute invariant: data uploaded for one project must never be read, indexed, queried, scored, overwritten, or displayed as part of another project unless an explicit merge feature is added later.

## Current State

- The official WNS benchmark is a read-only 180-combination matrix with existing reference artifacts.
- Uploads create `data/user_projects/<project_id>/` and basic manifests.
- Uploaded-project queries currently use one lexical token-overlap search and duplicate the same result across selected pipeline combinations.
- Several benchmark paths still write to shared folders such as `data/retrieval_smoke`, `data/reranker_smoke`, `data/evaluation`, and `data/modular_runs/latest`.
- Dashboard jobs are tracked only in process memory.
- Upload bodies are read without explicit request or ZIP expansion limits.
- The dashboard defaults to `0.0.0.0`; database ports are published broadly by Compose.
- A credential-shaped NVIDIA key is hard-coded in a test and must be removed from source. Rotation is an external credential action requiring WNS approval.
- Existing regression baseline: 41 relevant tests pass, Python compilation passes, and `web/app.js` syntax passes.

## Ideal State

Retreiver behaves as a project-scoped internal product:

1. Every upload creates a unique project.
2. Every project owns all derived data and execution artifacts.
3. Every execution creates a unique run.
4. All reads and writes require explicit project and run context where applicable.
5. Missing or invalid context fails closed; it never falls back to another project or the official WNS corpus.
6. The official benchmark remains unchanged and read-only.
7. Concurrent work cannot overwrite or consume another run's artifacts.
8. The UI describes what actually executed and never presents duplicated lexical results as real adapter comparisons.
9. VM deployment is repeatable, reversible, and verifiable through one command after the work-laptop Git push and VM pull.

## Out of Scope

- User accounts, passwords, SSO, roles, and per-user ownership.
- Public internet exposure.
- Automatic or manual project merging.
- Billing, usage metering, or customer tenancy administration.
- Replacing the current benchmark algorithms or changing the official 180-combination results.
- Rotating external credentials automatically.
- A full framework migration from the current standard-library dashboard server.

## Principles

- Project context is mandatory, never inferred.
- Run artifacts are immutable after completion.
- Temporary work is not visible as completed work.
- Official benchmark data is separate from uploaded-project data.
- Evidence-only results are labelled honestly.
- Reject unsafe work before consuming memory, disk, GPU, or provider cost.
- Use surgical changes around existing code and preserve proven benchmark behavior.
- Prefer one writer and deterministic queueing before adding concurrency.
- Deployment must include backup and rollback, not only startup.

## Architecture

### 1. Project storage boundary

Each upload creates:

```text
data/user_projects/<project_id>/
  manifest.json
  raw_uploads/
  extracted_text/
  chunks/
  questions/
  indexes/
    <run_id>/
  runs/
    <run_id>/
      manifest.json
      status.json
      logs/
      retrieval/
      reranking/
      evaluation/
      evidence.json
      summary.csv
```

`project_id` and `run_id` are server-generated UUID-backed slugs validated before path construction.

Project creation uses a temporary staging directory. The final project directory becomes visible only after upload validation, extraction, indexing metadata, and manifest writing succeed. Failure removes the staging directory.

No project operation may use these global paths:

- `data/pdfs`
- `data/retrieval_smoke`
- `data/reranker_smoke`
- `data/evaluation`
- `data/modular_runs/latest`

Those remain official/operator benchmark paths only.

### 2. Run isolation

Every project run receives a unique `run_id`. All stages receive explicit input and output paths derived from the same project/run manifest.

External vector-store namespaces include both project and run identity:

```text
<project_slug>__<run_id>__<adapter_slug>
```

A reranker consumes only the retrieval manifest from its own run. It never scans a shared retrieval directory.

Completed artifacts are written to temporary files and published with `os.replace`. A run status transitions through:

```text
queued -> running -> completed
                  -> failed
                  -> cancelled
```

The UI may display a completed run only after its manifest and required outputs are atomically published.

### 3. Honest uploaded-project query modes

The existing token-overlap query becomes **Lexical preview**:

- one search result set;
- no expanded adapter Cartesian product;
- no embedding, vector-store, or reranker claims;
- no accuracy, recall, MRR, or nDCG claims;
- source filename, page where available, lexical score, and evidence excerpt displayed.

A **Real pipeline run** appears only when selected adapters actually execute against project-scoped indexes and produce distinct run artifacts.

Without ground truth, real runs remain evidence-only. With a validated question set containing `question` and optional `ground_truth`, scored metrics are generated only for rows with ground truth.

### 4. Upload safety

Configurable conservative defaults:

- maximum request/upload bytes;
- maximum JSON body bytes;
- maximum ZIP entry count;
- maximum per-entry expanded bytes;
- maximum total expanded bytes;
- maximum compression ratio;
- maximum path depth.

Validation includes:

- allowed extension and file signature agreement;
- ZIP traversal rejection;
- symlink and special-file rejection;
- duplicate normalized destination rejection;
- destination containment after resolution;
- disk-headroom check before extraction;
- cleanup of incomplete projects on any failure.

The server returns structured 4xx errors for invalid or oversized input rather than raw exceptions.

### 5. Job control

All heavy work enters a persistent project-aware job registry stored on disk. Initial capacity is one active heavy job globally because the VM has shared GPU and database resources.

A second heavy request becomes queued or receives a clear capacity response. Duplicate submissions with the same project and request fingerprint are rejected while active.

Job state stores:

- job ID;
- project ID;
- run ID;
- sanitized command label, not raw secrets;
- status and timestamps;
- log path;
- exit code and failure code.

Dashboard restart reloads persisted job metadata. Unknown or orphaned running jobs become `interrupted`, never silently `completed`.

### 6. Product structure

The dashboard presents three explicit workspaces:

1. **Official benchmark**
   - read-only 180-combination WNS baseline;
   - coverage, leaderboard, latency, and evidence.

2. **Projects**
   - create project by upload;
   - select one project explicitly;
   - inspect its documents and processing state;
   - run lexical preview;
   - view project-scoped run history and evidence.

3. **Operator runs**
   - service readiness;
   - queued/running/completed jobs;
   - benchmark execution and technical logs;
   - official artifact operations.

The active project and run are always visible on project pages. Buttons remain disabled until a project is selected. Raw JSON is replaced with user-readable status and an optional technical detail section.

### 7. VM deployment and containment

- Dashboard default bind changes to `127.0.0.1`.
- Broad network binding requires an explicit environment variable or CLI argument.
- Database Compose ports bind to `127.0.0.1` by default.
- Postgres has no weak fallback password; startup requires an environment-provided value.
- Database containers gain restart policies and health checks.
- The hard-coded NVIDIA credential is removed and replaced with environment lookup.
- A user-level service definition or tracked launcher manages the dashboard without requiring source changes.
- Data directories and database volumes remain outside release replacement.

Deployment workflow:

```text
work laptop commit/push
-> VM pre-deploy backup
-> VM git pull
-> automated tests and config validation
-> canary dashboard health/API smoke
-> service restart
-> browser smoke
-> rollback to previous commit if any gate fails
```

The release includes one VM verification script and one rollback script. Rollback never deletes volumes or user projects.

## Error Handling

API errors use a consistent structure:

```json
{
  "error": {
    "code": "PROJECT_NOT_FOUND",
    "message": "The selected project does not exist.",
    "request_id": "..."
  }
}
```

Expected mappings:

- 400: malformed request;
- 404: unknown project, run, job, or artifact;
- 409: conflicting active run or duplicate submission;
- 413: request or upload too large;
- 415: unsupported or mismatched file type;
- 422: invalid project selection, query, matrix, or question schema;
- 429: queue or capacity limit reached;
- 500: sanitized internal failure;
- 504: stage timeout.

Client responses do not expose absolute paths, DSNs, credentials, raw command lines, authorization headers, or tracebacks.

## Testing Strategy

### Unit and contract tests

- strict project and run ID validation;
- project-root containment;
- required explicit project context;
- no official-data fallback;
- atomic manifest/result publication;
- upload byte limits and JSON body limits;
- ZIP traversal, symlink, duplicate, entry-count, expanded-size, and compression-ratio rejection;
- cleanup after failed upload;
- lexical preview has one result set and no fake adapter matrix;
- error schema and status codes;
- persistent job recovery and single-writer conflict behavior.

### Isolation integration tests

Create two projects with deliberately conflicting vocabulary and filenames. Verify:

- Project A queries return only A evidence.
- Project B queries return only B evidence.
- A project ID cannot read another project's run.
- Two same-second runs receive different IDs and directories.
- Reranking consumes only the retrieval manifest for its run.
- Official benchmark artifacts and checksums remain unchanged.

### Regression and UI tests

- Existing dashboard and modular benchmark suites remain green.
- Official matrix remains 180; FAISS/OSS validation remains 20.
- Python compile and JavaScript syntax checks pass.
- Browser flow: upload A, upload B, query each, inspect evidence, refresh, inspect run history.
- Browser console has no uncaught errors.
- Desktop and narrow screenshots show no overlap.

### VM verification

The VM verification command checks:

- source commit and sanitized configuration;
- dashboard liveness/readiness;
- model adapter, Qdrant, PGVector, and Weaviate readiness;
- project directory writability and disk headroom;
- official matrix counts;
- two-project no-mixing smoke;
- job queue behavior;
- existing benchmark reference checksums;
- browser/API accessibility through the intended VM access path.

## Ideal State Criteria

- [ ] ISC-1: Two uploaded projects with conflicting content never return each other's evidence.
- [ ] ISC-2: Every project execution has a unique project-scoped run directory and manifest.
- [ ] ISC-3: Missing or invalid project context returns an error and never falls back to official or other project data.
- [ ] ISC-4: Uploaded-project lexical preview contains no fake adapter-combination rows or scored benchmark claims.
- [ ] ISC-5: Concurrent heavy submissions cannot write to the same run or shared project artifact paths.
- [ ] ISC-6: Oversized, unsafe, or malformed uploads are rejected before unbounded resource use and leave no partial project.
- [ ] ISC-7: Official WNS benchmark evidence remains unchanged and validates at 180 combinations.
- [ ] ISC-8: Job status survives dashboard restart or is deterministically marked interrupted.
- [ ] ISC-9: Source contains no hard-coded NVIDIA credential.
- [ ] ISC-10: VM backup, apply, verification, and rollback commands execute without deleting user projects or database volumes.
- [ ] ISC-11: Existing regression suites, compile checks, JavaScript checks, API smoke, and browser smoke pass.
- [ ] ISC-12: A packaged handoff ZIP passes integrity and critical-file inspection.

## Verification Evidence

Each criterion is proven by one or more of:

- pytest output;
- `py_compile` and `node --check` output;
- matrix validation output;
- two-project isolation test artifacts;
- API responses and job manifests;
- browser screenshots and console capture;
- VM verification-script output;
- backup and rollback dry-run output;
- `zip -T`, SHA256, and `unzip -l` output.

## Decisions

- 2026-07-11: No user accounts in this release.
- 2026-07-11: Project visibility is acceptable; data mixing is not.
- 2026-07-11: No merge feature in this release.
- 2026-07-11: Preserve the official 180-combination benchmark unchanged.
- 2026-07-11: Keep the current server architecture and harden it surgically rather than perform a framework rewrite.
- 2026-07-11: Start with one active heavy job globally.
