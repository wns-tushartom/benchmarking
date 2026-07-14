# Retreiver Project Isolation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Retreiver safe for multiple independently uploaded projects with no implicit data mixing, honest query behavior, bounded resource use, persistent single-writer jobs, and reversible VM deployment.

**Architecture:** Extract project storage/query rules and job persistence from the large dashboard server into focused service modules. Keep the official 180-combination benchmark path unchanged. User-project operations always require an explicit server-generated project ID, write only under that project, and expose lexical preview until real adapters produce project-scoped artifacts.

**Tech Stack:** Python 3.12 standard library, pandas/openpyxl where already present, pytest, vanilla JavaScript/CSS, Docker Compose, Bash, Python `http.server` dashboard.

---

## File map

**Create:**

- `source/services/project_workspace.py`: project/run ID validation, project layout, atomic project creation, bounded ZIP extraction, project manifests, lexical preview.
- `source/services/job_registry.py`: persistent job metadata, global single-writer lock, restart recovery, safe status serialization.
- `tests/test_project_isolation.py`: project boundary, upload safety, no-fallback, lexical-preview, and concurrent-project regression tests.
- `tests/test_job_registry.py`: persistent single-writer and restart-recovery tests.
- `scripts/verify_vm_product.py`: API/config/filesystem/two-project isolation verification.
- `scripts/install_user_service.sh`: install/restart a user-level dashboard service without deleting data.
- `scripts/backup_before_update.sh`: checksum-backed source/config/project backup.
- `scripts/rollback_last_update.sh`: restore prior source release without deleting projects or volumes.
- `deploy/systemd/wns-retreiver-dashboard.service`: user service template.
- `docs/VM_PRODUCT_RUNBOOK.md`: exact laptop-push, VM-pull, verify, and rollback workflow.

**Modify:**

- `scripts/serve_benchmark_dashboard.py`: import focused services, enforce request limits, structured errors, explicit project context, lexical-preview response, persistent jobs, localhost default.
- `web/index.html`: make project context visible; rename project query to Lexical preview; separate official/projects/operator wording.
- `web/app.js`: remove fake adapter matrix from project queries, render safe errors, show project/run/mode metadata, disable project actions without explicit selection.
- `web/styles.css`: project-context banner, error states, accessible focus, responsive project evidence.
- `tests/test_dashboard_metrics.py`: Handler/frontend regression checks for explicit project selection and honest query labels.
- `docker-compose.benchmark.yml`: loopback port bindings, required Postgres password, restart policies, complete health checks, capped logs.
- `.env.example`: hardening limits, bind host, job capacity, required database password.
- `tests/test_qa_text_ragas.py`: remove embedded NVIDIA credential and require `NVIDIA_API_KEY` from environment.
- `README.md`, `APPLY_ZIP.md`, `PIPELINE_RUNBOOK.md`: safe startup and verification commands.

## Task 1: Lock project-boundary behavior with failing tests

**Files:**
- Create: `tests/test_project_isolation.py`
- Create: `source/services/project_workspace.py`

- [ ] **Step 1: Write failing ID and containment tests**

```python
from pathlib import Path
import pytest

from source.services.project_workspace import ProjectWorkspace, validate_resource_id


def test_resource_ids_are_uuid_backed_and_reject_path_syntax(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = workspace.new_project_id("Refund Data")
    assert project_id.startswith("refund-data_")
    assert validate_resource_id(project_id) == project_id
    for value in ("../other", "/tmp/project", "a/b", "", ".", "project id"):
        with pytest.raises(ValueError):
            validate_resource_id(value)


def test_project_root_cannot_escape_base(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    with pytest.raises(ValueError):
        workspace.project_root("../official")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_project_isolation.py
```

Expected: collection fails because `source.services.project_workspace` does not exist.

- [ ] **Step 3: Implement minimal ID and layout service**

Create `source/services/project_workspace.py` with:

```python
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

_RESOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]{0,79}_[0-9a-f]{32}$")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9.-]+", "-", value.strip().lower()).strip(".-")
    return slug[:40] or "project"


def validate_resource_id(value: str) -> str:
    if not _RESOURCE_ID.fullmatch(str(value or "")):
        raise ValueError("Invalid resource identifier")
    return value


@dataclass(frozen=True)
class ProjectWorkspace:
    base: Path

    def new_project_id(self, label: str) -> str:
        return f"{safe_slug(label)}_{uuid.uuid4().hex}"

    def new_run_id(self) -> str:
        return f"run_{uuid.uuid4().hex}"

    def project_root(self, project_id: str) -> Path:
        clean = validate_resource_id(project_id)
        base = self.base.resolve()
        root = (base / clean).resolve()
        if root.parent != base:
            raise ValueError("Invalid project path")
        return root

    def layout(self, project_id: str) -> dict[str, Path]:
        root = self.project_root(project_id)
        return {
            "root": root,
            "raw_uploads": root / "raw_uploads",
            "extracted_text": root / "extracted_text",
            "chunks": root / "chunks",
            "questions": root / "questions",
            "indexes": root / "indexes",
            "runs": root / "runs",
        }
```

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
python3 -m pytest -q tests/test_project_isolation.py
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add source/services/project_workspace.py tests/test_project_isolation.py
git commit -m "feat: establish strict project workspace boundaries"
```

## Task 2: Add atomic bounded uploads and ZIP safety

**Files:**
- Modify: `source/services/project_workspace.py`
- Modify: `tests/test_project_isolation.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing upload safety tests**

Add tests that construct ZIPs in memory:

```python
import io
import json
import zipfile

from source.services.project_workspace import UploadLimits


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return out.getvalue()


def test_upload_creates_complete_project_atomically(tmp_path: Path):
    ws = ProjectWorkspace(tmp_path / "projects")
    result = ws.create_upload("Policy", "policy.txt", b"refund only in project A")
    root = ws.project_root(result["project_id"])
    assert (root / "manifest.json").exists()
    assert (root / "raw_uploads" / "policy.txt").read_bytes() == b"refund only in project A"
    assert not list((tmp_path / "projects").glob(".staging-*"))


def test_zip_traversal_and_bomb_limits_leave_no_project(tmp_path: Path):
    limits = UploadLimits(max_upload_bytes=4096, max_zip_entries=2, max_zip_expanded_bytes=32)
    ws = ProjectWorkspace(tmp_path / "projects", limits=limits)
    for payload in (
        zip_bytes({"../escape.txt": b"x"}),
        zip_bytes({"a.txt": b"x", "b.txt": b"y", "c.txt": b"z"}),
        zip_bytes({"large.txt": b"x" * 64}),
    ):
        with pytest.raises(ValueError):
            ws.create_upload("Unsafe", "unsafe.zip", payload)
    assert not list((tmp_path / "projects").glob("unsafe-*"))
    assert not list((tmp_path / "projects").glob(".staging-*"))
```

Also test duplicate normalized paths, symlink mode bits, unsupported signatures/extensions, and upload byte cap.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python3 -m pytest -q tests/test_project_isolation.py
```

Expected: failures for missing `UploadLimits` and `create_upload`.

- [ ] **Step 3: Implement bounded atomic upload service**

Add:

```python
@dataclass(frozen=True)
class UploadLimits:
    max_upload_bytes: int = 100 * 1024 * 1024
    max_json_bytes: int = 1024 * 1024
    max_zip_entries: int = 500
    max_zip_entry_bytes: int = 100 * 1024 * 1024
    max_zip_expanded_bytes: int = 500 * 1024 * 1024
    max_zip_ratio: int = 100
    max_zip_depth: int = 8
```

Implement `create_upload()` using a `.staging-<uuid>` directory, strict extension/signature checks, `ZipInfo.external_attr` symlink rejection, normalized destination de-duplication, resolved containment checks, cumulative expanded-byte checks, atomic manifest writes, and final `os.replace(staging, final_root)`. On exception, remove only the staging directory and re-raise.

Expose limits through `.env.example`:

```text
DASHBOARD_MAX_UPLOAD_MB=100
DASHBOARD_MAX_JSON_KB=1024
DASHBOARD_MAX_ZIP_FILES=500
DASHBOARD_MAX_ZIP_EXPANDED_MB=500
DASHBOARD_MAX_ZIP_RATIO=100
```

- [ ] **Step 4: Replace dashboard upload implementation**

In `serve_benchmark_dashboard.py`:

- reject missing/invalid `Content-Length`;
- reject over-limit requests with 413 before multipart parsing;
- call `ProjectWorkspace.create_upload()`;
- map unsupported type to 415 and validation errors to 422;
- stop returning raw exception strings.

Use a structured helper:

```python
def api_error(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}
```

- [ ] **Step 5: Verify focused and existing tests**

```bash
python3 -m pytest -q tests/test_project_isolation.py tests/test_dashboard_metrics.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add source/services/project_workspace.py scripts/serve_benchmark_dashboard.py tests/test_project_isolation.py .env.example
git commit -m "feat: create bounded atomic project uploads"
```

## Task 3: Fail closed and replace fake matrix output with lexical preview

**Files:**
- Modify: `source/services/project_workspace.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_project_isolation.py`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing no-mixing tests**

```python
def test_two_projects_never_cross_return_evidence(tmp_path: Path):
    ws = ProjectWorkspace(tmp_path / "projects")
    a = ws.create_upload("Alpha", "same.txt", b"alpha-only cancellation remedy")
    b = ws.create_upload("Beta", "same.txt", b"beta-only baggage remedy")

    alpha = ws.lexical_preview(a["project_id"], "alpha cancellation", top_k=5)
    beta = ws.lexical_preview(b["project_id"], "beta baggage", top_k=5)

    assert alpha["mode"] == "lexical_preview"
    assert beta["mode"] == "lexical_preview"
    assert all("beta-only" not in row["paragraph"] for row in alpha["hits"])
    assert all("alpha-only" not in row["paragraph"] for row in beta["hits"])
    assert "rows" not in alpha
    assert "combo_count" not in alpha
    assert alpha["run_id"] != beta["run_id"]
    alpha_run = ws.project_root(a["project_id"]) / "runs" / alpha["run_id"]
    beta_run = ws.project_root(b["project_id"]) / "runs" / beta["run_id"]
    assert (alpha_run / "manifest.json").exists()
    assert (alpha_run / "evidence.json").exists()
    assert (beta_run / "manifest.json").exists()
    assert (beta_run / "evidence.json").exists()


def test_query_requires_existing_explicit_project(tmp_path: Path):
    ws = ProjectWorkspace(tmp_path / "projects")
    with pytest.raises(ValueError):
        ws.lexical_preview("", "refund")
    missing = ws.new_project_id("missing")
    with pytest.raises(FileNotFoundError):
        ws.lexical_preview(missing, "refund")
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_isolation.py
```

Expected: failures because `lexical_preview` does not exist.

- [ ] **Step 3: Implement one honest lexical preview**

Move token-overlap search into `ProjectWorkspace.lexical_preview()`. Load only `<project>/search_index.json`, cap query length at 4000 characters and `top_k` to 1..50, create a new UUID-backed `run_id`, atomically persist `runs/<run_id>/manifest.json` and `runs/<run_id>/evidence.json`, and return:

```python
{
    "ok": True,
    "mode": "lexical_preview",
    "project_id": project_id,
    "run_id": run_id,
    "created_at": created_at,
    "query": query,
    "top_k": top_k,
    "hits": hits,
    "message": "Lexical preview only. Embedding, vector database, and reranker selections were not executed.",
}
```

Do not accept or return adapter selections. The run manifest records only the selected project, query metadata, mode, status, and output filenames. It must not copy document text from another location.

- [ ] **Step 4: Update API and frontend**

- `/api/project-query` requires `project_id`, `query`, and bounded `top_k` only.
- Remove `selections` from the project-query request.
- Rename UI labels to `Lexical preview` and `Search this project`.
- Display active project, mode, timestamp, source, page, lexical score, and all requested top-K hits.
- Remove combination-count language from project query results.
- Keep pipeline selectors only in the operator run workspace.

Add frontend assertions:

```python
def test_uploaded_project_query_is_labelled_lexical_preview():
    root = Path(__file__).resolve().parents[1]
    index = (root / "web" / "index.html").read_text()
    app = (root / "web" / "app.js").read_text()
    assert "Lexical preview" in index
    assert "projectSelections()" not in app
    assert 'mode === "lexical_preview"' in app or "lexical_preview" in app
```

- [ ] **Step 5: Verify tests and JS syntax**

```bash
python3 -m pytest -q tests/test_project_isolation.py tests/test_dashboard_metrics.py
node --check web/app.js
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add source/services/project_workspace.py scripts/serve_benchmark_dashboard.py web tests
git commit -m "fix: make uploaded project queries isolated and honest"
```

## Task 4: Add persistent single-writer job control

**Files:**
- Create: `source/services/job_registry.py`
- Create: `tests/test_job_registry.py`
- Modify: `scripts/serve_benchmark_dashboard.py`

- [ ] **Step 1: Write failing persistence and conflict tests**

```python
from pathlib import Path
import pytest

from source.services.job_registry import JobConflict, JobRegistry


def test_registry_persists_jobs_and_recovers_running_as_interrupted(tmp_path: Path):
    registry = JobRegistry(tmp_path / "jobs", max_active=1)
    job = registry.create(project_id="official", run_id="run_" + "a" * 32, kind="official_pipeline")
    registry.mark_running(job["job_id"], pid=12345)

    recovered = JobRegistry(tmp_path / "jobs", max_active=1)
    status = recovered.get(job["job_id"])
    assert status["status"] == "interrupted"


def test_registry_allows_only_one_active_heavy_job(tmp_path: Path):
    registry = JobRegistry(tmp_path / "jobs", max_active=1)
    first = registry.create(project_id="official", run_id="run_" + "a" * 32, kind="official_pipeline")
    registry.mark_running(first["job_id"], pid=123)
    with pytest.raises(JobConflict):
        registry.create(project_id="official", run_id="run_" + "b" * 32, kind="official_pipeline")
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_job_registry.py
```

Expected: module import failure.

- [ ] **Step 3: Implement registry with atomic JSON files and lock**

Use one `registry.lock` acquired with `fcntl.flock`, one JSON document per job, temp-file plus `os.replace`, and status values `queued`, `running`, `completed`, `failed`, `cancelled`, `interrupted`. Do not persist raw environment variables or secrets.

- [ ] **Step 4: Integrate with dashboard launch/status**

Replace process-global `JOBS` as the source of truth. Keep live `Popen` handles in memory only for active local children, while API status comes from persisted metadata. Return HTTP 409 on `JobConflict`. Launch subprocesses in a process group and update completion status from a watcher thread.

- [ ] **Step 5: Verify focused and dashboard tests**

```bash
python3 -m pytest -q tests/test_job_registry.py tests/test_dashboard_metrics.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add source/services/job_registry.py scripts/serve_benchmark_dashboard.py tests/test_job_registry.py tests/test_dashboard_metrics.py
git commit -m "feat: persist and serialize dashboard jobs"
```

## Task 5: Contain VM networking and remove embedded credential

**Files:**
- Modify: `docker-compose.benchmark.yml`
- Modify: `.env.example`
- Modify: `tests/test_qa_text_ragas.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing static configuration tests**

```python
def test_vm_services_bind_loopback_and_have_restart_health_controls():
    compose = Path("docker-compose.benchmark.yml").read_text()
    assert '127.0.0.1:5019:6333' in compose
    assert '127.0.0.1:5003:5432' in compose
    assert '127.0.0.1:5004:8080' in compose
    assert compose.count("restart: unless-stopped") >= 3
    assert "wns_password" not in compose


def test_source_has_no_embedded_nvidia_api_key():
    text = Path("tests/test_qa_text_ragas.py").read_text()
    assert "nvapi-" not in text
    assert 'os.environ["NVIDIA_API_KEY"]' in text or "os.getenv(\"NVIDIA_API_KEY\")" in text
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_dashboard_metrics.py -k 'vm_services or embedded_nvidia'
```

Expected: failures against current Compose and test credential.

- [ ] **Step 3: Implement safe defaults**

- Dashboard default host becomes `127.0.0.1`; `DASHBOARD_BIND_HOST` or CLI argument can explicitly override it.
- Compose port mappings use `${WNS_BIND_HOST:-127.0.0.1}:host:container`.
- `WNS_POSTGRES_PASSWORD` uses required interpolation `${WNS_POSTGRES_PASSWORD:?set WNS_POSTGRES_PASSWORD}`.
- Add `restart: unless-stopped`, Qdrant/Weaviate health checks, and Docker log rotation.
- Replace the embedded NVIDIA value with `os.environ["NVIDIA_API_KEY"]` at execution time.

- [ ] **Step 4: Verify configuration and tests**

```bash
WNS_POSTGRES_PASSWORD=test-only docker compose -f docker-compose.benchmark.yml config >/tmp/wns-compose.yml
python3 -m pytest -q tests/test_dashboard_metrics.py
python3 -m py_compile tests/test_qa_text_ragas.py scripts/serve_benchmark_dashboard.py
```

Expected: all pass and Compose config exits 0.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.benchmark.yml .env.example tests/test_qa_text_ragas.py scripts/serve_benchmark_dashboard.py tests/test_dashboard_metrics.py
git commit -m "security: contain VM services and remove embedded credential"
```

## Task 6: Add reversible VM deployment and verification

**Files:**
- Create: `scripts/verify_vm_product.py`
- Create: `scripts/backup_before_update.sh`
- Create: `scripts/rollback_last_update.sh`
- Create: `scripts/install_user_service.sh`
- Create: `deploy/systemd/wns-retreiver-dashboard.service`
- Create: `docs/VM_PRODUCT_RUNBOOK.md`
- Modify: `README.md`
- Modify: `APPLY_ZIP.md`
- Modify: `PIPELINE_RUNBOOK.md`

- [ ] **Step 1: Write verifier tests before implementation**

Add subprocess tests that run `verify_vm_product.py --offline --json` against temporary projects and assert:

- official config count is 180;
- FAISS/OSS config count is 20;
- two-project isolation passes;
- source credential scan passes;
- required project/run directories are writable;
- output JSON contains `ok`, `checks`, `failed`, and `created_at`.

- [ ] **Step 2: Verify RED**

```bash
python3 -m pytest -q tests/test_project_isolation.py -k vm_verifier
```

Expected: missing script failure.

- [ ] **Step 3: Implement verifier**

`verify_vm_product.py` supports:

```text
--offline       filesystem, source, config, and two-project checks
--base-url URL  dashboard liveness, results, upload, and query checks
--json          machine-readable output
--keep-smoke-projects
```

It creates two temporary smoke projects with conflicting sentinel strings, queries each, proves no cross-hit, and removes only those smoke projects unless `--keep-smoke-projects` is set.

- [ ] **Step 4: Implement backup and rollback scripts**

`backup_before_update.sh`:

- resolves repo root;
- requires a clean-enough Retreiver source state or records dirty paths;
- archives source/config/docs plus `data/user_projects` manifests and run metadata;
- does not copy database volumes;
- writes SHA256 and a release manifest;
- never deletes source data.

`rollback_last_update.sh`:

- accepts an explicit backup archive;
- verifies SHA256;
- restores source files to a staging directory;
- preserves `data/user_projects`, database volumes, `.env*`, and generated benchmark artifacts;
- switches source only after offline verifier succeeds.

- [ ] **Step 5: Implement user service and runbook**

Service template:

```ini
[Unit]
Description=WNS Retreiver Dashboard
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/benchmarking/Retreiver
EnvironmentFile=-%h/benchmarking/Retreiver/.env.vm.generated
ExecStart=%h/benchmarking/Retreiver/.venv-vm/bin/python scripts/serve_benchmark_dashboard.py 5011 127.0.0.1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Runbook must use the real flow:

```bash
cd ~/benchmarking/Retreiver
bash scripts/backup_before_update.sh
git pull --ff-only
source .venv-vm/bin/activate
python scripts/verify_vm_product.py --offline --json
bash scripts/install_user_service.sh
python scripts/verify_vm_product.py --base-url http://127.0.0.1:5011 --json
```

- [ ] **Step 6: Verify scripts safely**

```bash
bash -n scripts/backup_before_update.sh scripts/rollback_last_update.sh scripts/install_user_service.sh
python3 -m py_compile scripts/verify_vm_product.py
python3 scripts/verify_vm_product.py --offline --json
```

Expected: syntax checks pass and offline verifier returns `"ok": true`.

- [ ] **Step 7: Commit**

```bash
git add scripts/verify_vm_product.py scripts/backup_before_update.sh scripts/rollback_last_update.sh scripts/install_user_service.sh deploy docs README.md APPLY_ZIP.md PIPELINE_RUNBOOK.md tests/test_project_isolation.py
git commit -m "ops: add reversible VM deployment and verification"
```

## Task 7: Full verification, browser QA, review, and handoff ZIP

**Files:**
- Modify only files required by failures found during verification.
- Create outside repository: final overlay ZIP and checksum.

- [ ] **Step 1: Run complete automated verification**

```bash
python3 -m pytest -q
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/verify_vm_product.py source/services/project_workspace.py source/services/job_registry.py
node --check web/app.js
python3 scripts/benchmark_cli.py validate configs/benchmark.local.json
python3 scripts/benchmark_cli.py validate configs/benchmark.faiss-noaws.json
WNS_POSTGRES_PASSWORD=test-only docker compose -f docker-compose.benchmark.yml config >/tmp/wns-compose-final.yml
python3 scripts/verify_vm_product.py --offline --json
```

Expected: zero failures; official matrix 180; FAISS/OSS matrix 20; isolation smoke passes.

- [ ] **Step 2: Run dashboard and API smoke**

Start on a temporary port and verify:

- `/api/results` returns 180 official combinations;
- two uploads create different project IDs;
- project A query never returns B sentinel;
- project B query never returns A sentinel;
- missing project ID returns structured 4xx;
- project query response mode is `lexical_preview` with no `combo_count`.

- [ ] **Step 3: Run real browser QA**

Using Playwright Chromium:

- capture 1440×1000 and 390×844 screenshots;
- upload project A and B through the UI;
- select and query each project;
- verify project context remains visible;
- verify no fake adapter matrix appears;
- verify no uncaught console or page errors;
- inspect screenshots for overlap and clipped controls.

- [ ] **Step 4: Run independent code review**

Review the diff for:

- any operation without explicit project context;
- any user project path outside `data/user_projects/<project_id>`;
- any shared user-run output path;
- raw exception/secret leakage;
- hard-coded credentials;
- accidental official benchmark modifications.

Fix review findings with failing regression tests first.

- [ ] **Step 5: Build and verify handoff ZIP**

Package only changed/new Retreiver files relative to the baseline commit. Include `APPLY_ZIP.md`, VM scripts, tests, and design/plan docs. Then:

```bash
zip -T "$HOME/wns_retreiver_project_isolation_final_20260711.zip"
sha256sum "$HOME/wns_retreiver_project_isolation_final_20260711.zip"
unzip -l "$HOME/wns_retreiver_project_isolation_final_20260711.zip"
```

Expected: integrity OK and critical files visible.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "release: harden Retreiver project isolation and VM operations"
```

If no post-review changes exist, do not create an empty commit.

## Completion gate

Do not claim completion until all are true:

- [ ] Two-project isolation passes through helper, API, and browser paths.
- [ ] Uploaded-project response contains no fake adapter matrix.
- [ ] Official benchmark still validates at 180 and remains unchanged.
- [ ] All tests and syntax checks pass.
- [ ] Job registry persists and serializes heavy work.
- [ ] Upload limits and ZIP safety tests pass.
- [ ] Embedded credential is absent from source.
- [ ] Compose is contained and validates.
- [ ] Backup, verifier, service install, and rollback scripts pass syntax/dry-run checks.
- [ ] Browser screenshots are inspected and console is clean.
- [ ] ZIP integrity, SHA256, and contents are verified.
- [ ] Real VM execution is either verified from live command output or reported as the remaining external blocker.
