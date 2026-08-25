# All-Method Portfolio and Adapter Control Implementation Plan

> **Required execution workflow:** test-driven implementation in the existing isolated `feature/retrieval-reranker-candidates` worktree. Preserve unrelated dirty files and never touch the protected accepted-result CSVs.

**Goal:** Build an isolated 2,160-combination candidate portfolio, deterministically partition it into 12 executable batches of 180, and add token-protected homepage controls for the VM's allowlisted ports `5000–5019`.

**Architecture:** A compatibility-aware portfolio planner creates canonical combination IDs, immutable batch plans, and receipts. The existing benchmark runner executes one declared batch into a candidate-only output path. A manifest-driven VM service manager owns fixed service profiles and safe process receipts. Dashboard APIs expose portfolio/service state; the frontend renders explicit controls without mixing candidate results into official sources.

**Tech stack:** Python 3.12, pytest, stdlib `http.server`, FastAPI/Uvicorn model adapters, Docker Compose DB services, vanilla JS/CSS, Playwright/Chromium for browser QA.

**Approved design:** `docs/superpowers/specs/2026-08-20-all-method-portfolio-and-adapter-control-design.md`

---

## Guardrails before every commit

- Work only in `/home/fate/.config/superpowers/worktrees/wns-retriever/retrieval-reranker-candidates/Retreiver`.
- Never reset, clean, restore, or broadly stage.
- Never modify or stage:
  - `data/modular_runs/latest/modular_details.csv`
  - `data/modular_runs/latest/modular_summary.csv`
- Use explicit `git add -- <paths>` and inspect `git diff --cached --name-only`.
- The official configuration must remain 180 combinations.
- Candidate outputs must remain under candidate-only roots.
- Write failing tests before production changes.

## Task 1: Define and validate the full portfolio catalog

**Files:**
- Create: `configs/benchmark.all-methods-portfolio.json`
- Create: `configs/all_methods_portfolio_batches.json` only as generated/verified evidence if the planner intentionally materializes it; otherwise keep plans in runtime candidate outputs.
- Modify: `benchmarking/core/config.py`
- Modify: `benchmarking/core/schemas.py`
- Test: `tests/test_all_methods_portfolio.py`

**Step 1: Write failing inventory and compatibility tests**

Assert:

- exact method lists from the approved design;
- `Cosine Similarity` is not duplicated beside `Dense Cosine` in the portfolio;
- store/index compatibility yields exactly five valid pairs;
- invalid HNSW/TurboVec and TurboQuant/network-store pairs are excluded with stable reason codes;
- total valid combinations is exactly 2,160;
- combination IDs are stable under dictionary key reordering;
- the protected config remains 180.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_all_methods_portfolio.py
```

Expected: FAIL because the portfolio config and compatibility support do not exist.

**Step 2: Implement the catalog and compatibility filter**

Add a versioned `compatibility` section mapping each vector store to valid index types. Extend matrix generation without changing behavior for configs that omit compatibility. Preserve explicit exclusions as records with `combination_id`, attempted axes, and reason code.

**Step 3: Prove inventory and backward compatibility**

Run the focused test plus:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/benchmark_cli.py matrix configs/benchmark.local.json > /tmp/official-matrix.json
python -c "import json; print(len(json.load(open('/tmp/official-matrix.json'))))"
```

Expected: `180`.

## Task 2: Implement deterministic 12×180 batch planning

**Files:**
- Create: `benchmarking/core/portfolio.py`
- Modify: `scripts/benchmark_cli.py`
- Test: `tests/test_portfolio_batch_planner.py`

**Step 1: Write failing planner tests**

Assert:

- exactly 12 batches;
- every batch has exactly 180 combinations and never exceeds 250;
- each batch contains one embedding and one approved reranker group;
- union of batch combination IDs equals all 2,160 valid portfolio IDs;
- no duplicate IDs across batches;
- deterministic order and portfolio hash across repeated loads;
- planner fails closed for cap `0`, cap `>250`, unknown reranker group, duplicate group member, missing coverage, and malformed config;
- configured, excluded, selected, completed, failed, and not-run are distinct states.

**Step 2: Implement pure planning primitives**

Create immutable data classes/functions for:

- canonical combination serialization and SHA-256 IDs;
- portfolio hash;
- reranker groups A/B;
- batch IDs derived from embedding/group/hash;
- plan validation and JSON serialization;
- run-state reconciliation from receipts without mutating the plan.

**Step 3: Add CLI planning commands**

Add:

```text
benchmark_cli.py portfolio-plan <config> [--output <candidate path>]
benchmark_cli.py portfolio-status <plan-or-root>
```

Planning may write only to the all-method candidate root. `--output` must pass the same symlink/path isolation rules as candidate execution.

**Step 4: Run focused tests and inspect generated plan**

Run the planner test and a temporary plan generation. Verify counts and JSON hashes programmatically.

## Task 3: Add TurboVec 4-bit as an honest embedded challenger

**Files:**
- Create: `benchmarking/adapters/vector_turbovec.py`
- Modify: `benchmarking/core/registry.py`
- Modify: `benchmarking/core/runner.py`
- Modify: `requirements.txt` or create a narrowly documented VM requirement file if installation constraints require it
- Test: `tests/test_turbovec_vector_adapter.py`

**Step 1: Write failing adapter contract tests**

Use a fake TurboVec module for deterministic unit coverage and, when the installed wheel is available, one real smoke. Assert:

- 4-bit only;
- normalized finite vectors;
- stable external IDs through `IdMapIndex`;
- deterministic search result mapping;
- dimension mismatch and non-finite values fail closed;
- duplicate IDs fail;
- persistence metadata includes content hashes;
- adapter is registered only for `TurboVec` + `TurboQuant4bit`;
- no network port or vector-database claims.

**Step 2: Implement the adapter**

Keep imports lazy so official and non-TurboVec runs do not require the package at import time. Return actionable readiness errors when the dependency is absent.

**Step 3: Run focused tests and an isolated wheel smoke**

Do not benchmark accepted data. Exercise add/search/write/load against synthetic vectors.

## Task 4: Execute and receipt one declared batch safely

**Files:**
- Modify: `benchmarking/core/runner.py`
- Modify: `scripts/benchmark_cli.py`
- Create: `scripts/publish_portfolio_results.py`
- Test: `tests/test_portfolio_batch_execution.py`
- Test: `tests/test_portfolio_results_publication.py`

**Step 1: Write failing execution/isolation tests**

Assert:

- only exact IDs declared in the selected batch can run;
- a batch above 250 is rejected before provider calls;
- `--max-runs` cannot silently truncate a declared batch;
- resume skips only valid completed receipts and retains failed/not-run distinctions;
- output roots are exactly `data/modular_runs/all-methods-portfolio/<portfolio-id>/<batch-id>`;
- aliases, nested official paths, `latest`, traversal, and symlink escapes fail;
- receipt hashes summary, details, config snapshot, and combination manifest;
- portfolio publication requires all 12 valid batch receipts and exact 2,160 coverage;
- `promotion_status` is `not_accepted`;
- official result scanners exclude the new lane.

**Step 2: Implement batch execution selection**

Add CLI commands:

```text
benchmark_cli.py run-batch <config> --plan <path> --batch-id <id> [--limit-queries N]
benchmark_cli.py run-next <config> --plan <path> [--limit-queries N]
```

The normal `run` command remains backward compatible. Batch execution records state atomically and never edits the immutable plan.

**Step 3: Implement batch and portfolio receipt verification**

`publish_portfolio_results.py` verifies every artifact hash and exact coverage before writing a candidate-only portfolio receipt/aggregate summary.

## Task 5: Define the 20-port allowlist and service-manager core

**Files:**
- Create: `configs/vm_adapter_ports.json`
- Create: `scripts/vm_adapter_manager.py`
- Modify: `scripts/wns_vm_adapter_service.py`
- Modify: `.env.example`
- Modify: `docker-compose.benchmark.yml`
- Modify: `scripts/start_vm_stack.sh`
- Test: `tests/test_vm_adapter_port_manifest.py`
- Test: `tests/test_vm_adapter_manager.py`
- Test: `tests/test_start_vm_stack_script.py`

**Step 1: Write failing manifest tests**

Assert:

- exactly ports 5000 through 5019, each once;
- fixed service IDs and roles;
- Qdrant gRPC is 5015 and HTTP is 5019;
- dashboard is 5011 and NVIDIA is 5016–5018;
- reserved/external/in-process entries cannot launch;
- no arbitrary command/host/port fields are accepted from API input;
- endpoint-env mappings are complete and contain no credentials.

**Step 2: Write failing lifecycle tests with fakes**

Mock sockets, subprocesses, process metadata, clocks, and health checks. Assert:

- idempotent start;
- collision with an unknown process returns blocked and does not kill it;
- stale PID receipt is handled safely;
- stop requires a dashboard-owned PID plus matching process-start time and command fingerprint;
- bounded startup polling;
- symlinked or wrong-owner runtime files are rejected;
- stdout/stderr logs stay under `data/adapter_runtime`;
- public status redacts tokens, environment values, absolute private paths, and command details;
- missing binary/model/credential returns `unconfigured` or `blocked`, not fake healthy;
- no shell execution (`shell=False`, fixed argv).

**Step 3: Implement profile-restricted model adapter service**

Add `WNS_ADAPTER_PROFILE` so each local model port advertises and serves only its allowed endpoint. Wrong endpoint/profile requests return a clear error. Keep existing unified behavior only as an explicit legacy profile, not the new default launcher.

**Step 4: Implement fixed launch profiles**

- local Python adapter profiles for Jina, GTE, BGE, Qwen, Nemotron rerank, and GTE ModernBERT;
- fixed Docker Compose service starts for PGVector, Weaviate, and Qdrant;
- fixed vLLM profiles for the three Nemotron embedding variants, enabled only when required binary/model configuration exists;
- reserved slots for Jupyter and spare;
- fixed health contracts for HTTP and TCP services.

**Step 5: Update endpoint generation and port remap**

Write `.env.vm.generated` without secrets. Replace Qdrant gRPC 5020 with 5015 in current operational files while preserving historical documents unless they are actively authoritative. Validate shell syntax.

## Task 6: Add token-protected dashboard APIs

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Test: `tests/test_dashboard_adapter_control.py`
- Test: `tests/test_dashboard_portfolio_api.py`

**Step 1: Write failing route tests**

Add contracts for:

```text
GET  /api/portfolio
GET  /api/portfolio/batches
GET  /api/adapters
POST /api/adapters/start
POST /api/adapters/retry
POST /api/adapters/stop
POST /api/adapters/start-required
POST /api/portfolio/run-batch
POST /api/portfolio/run-next
```

Assert:

- unknown keys/IDs rejected;
- mutation disabled unless `WNS_ENABLE_ADAPTER_CONTROL=1`;
- constant-time bearer-token validation against a server secret;
- no token in responses/logs;
- status GET has no-store headers;
- start-required derives services from the immutable batch plan, not browser-provided port lists;
- batch launch revalidates 250 cap and candidate output path;
- official source APIs continue excluding portfolio roots.

**Step 2: Implement small route bridge functions**

Keep pure validation separate from HTTP handlers. Use the manager and planner modules rather than duplicating lifecycle logic in the dashboard.

**Step 3: Run dashboard regression tests**

Include existing candidate/dashboard integrity suites after focused route tests pass.

## Task 7: Build the homepage and portfolio controls

**Skills before implementation:** load and follow `impeccable` and `ui-visual-qa`; read `DESIGN.md` and preserve the existing visual system.

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Test: `tests/test_adapter_control_frontend.py`
- Test: `tests/test_portfolio_frontend.py`
- Modify/Create: browser evidence receipts under `docs/evidence/`

**Step 1: Write failing static frontend contract tests**

Assert required IDs, labels, endpoint strings, state text, token session-storage behavior, candidate warnings, and absence of unsafe `innerHTML` for server error/log fields.

**Step 2: Implement Overview adapter control surface**

Render all 20 slots with explicit states. Add operator-token entry, refresh, per-service controls, and Start Required for selected batch. Disable actions for reserved/in-process/external/unconfigured slots.

**Step 3: Implement portfolio batch panel on Run page**

Show 2,160 planned, 12×180, max 250, selected batch, required services, state counts, preflight blockers, Run Batch, and Run Next. Label everything candidate/not accepted.

**Step 4: Preserve responsive behavior**

Use existing tokens/layout patterns. Desktop shows dense service table/cards; mobile stacks controls without page overflow. Do not alter official Metrics/Recommendations semantics.

**Step 5: Run JS syntax and frontend tests**

```bash
node --check web/app.js
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_adapter_control_frontend.py tests/test_portfolio_frontend.py
```

## Task 8: Documentation, VM runbook, and source-only packaging

**Files:**
- Modify: `VM_ADAPTER_PORT_MAP.md`
- Modify: `docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md`
- Create: `docs/VM_ALL_METHODS_PORTFOLIO_RUNBOOK.md`
- Modify/Create: source-only packager and its tests

Document one-command-at-a-time VM operations, token setup, service status/start behavior, batch smoke, resume, full sequential execution, receipts, and rollback that never touches accepted artifacts. Package source/tests/docs only: no data, runtime receipts, logs, tokens, env files, model weights, or accepted results.

## Task 9: Fresh verification and browser QA

**Step 1: Focused suite**

Run all new tests plus existing candidate integrity, dashboard, hybrid retrieval, matrix, VM adapter, publication, and overlay tests.

**Step 2: Static checks**

```bash
node --check web/app.js
bash -n scripts/start_vm_stack.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  benchmarking/core/portfolio.py \
  benchmarking/adapters/vector_turbovec.py \
  scripts/benchmark_cli.py \
  scripts/vm_adapter_manager.py \
  scripts/serve_benchmark_dashboard.py \
  scripts/wns_vm_adapter_service.py \
  scripts/publish_portfolio_results.py
python -m json.tool configs/benchmark.all-methods-portfolio.json >/dev/null
python -m json.tool configs/vm_adapter_ports.json >/dev/null
git diff --check
```

**Step 3: Contract verification**

Programmatically prove:

- official matrix = 180;
- portfolio = 2,160 valid;
- batches = 12;
- every batch = 180;
- union = 2,160 unique IDs;
- ports = exactly 5000–5019;
- candidate source roots absent from official result-source APIs;
- protected official CSV content hashes match the pre-work receipt.

**Step 4: Browser QA**

Start the dashboard on a local test port and use Chromium with `--no-sandbox`:

- desktop 1440×1000;
- mobile 390×844;
- inspect loading, offline, blocked, starting, healthy, and failed fixtures;
- verify token-required actions, 20 slots, batch selection, Start Required, Run Batch, candidate warnings;
- verify no page overflow, no console errors, and no unexpected network failures;
- capture screenshots and JSON evidence.

**Step 5: Package verification**

Build a new versioned source-only archive, test ZIP CRC, inspect member allowlist/exclusions, calculate SHA-256, and write a receipt. Do not deploy to the VM without a separate explicit deployment action and real endpoint/hardware validation.

## Task 10: Final narrow commits and review

Use explicit staging per logical slice. Before each commit inspect cached names. Suggested commits:

1. `feat: plan full candidate portfolio batches`
2. `feat: add turbovec candidate adapter`
3. `feat: control allowlisted VM adapter ports`
4. `feat: expose portfolio and adapter dashboard APIs`
5. `feat: add portfolio and adapter homepage controls`
6. `docs: add full portfolio VM runbook`
7. `test: verify full portfolio isolation and UI`

Before completion, request an independent code/spec review and address only verified issues. Report real test counts, browser evidence paths, package path/hash, and any VM-only blockers. Never claim the 2,160 combinations were executed unless a real VM run and receipts prove it.
