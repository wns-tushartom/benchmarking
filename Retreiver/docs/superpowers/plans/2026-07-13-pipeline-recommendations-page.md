# Source-Aware Pipeline Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Recommendations page that loads either the official WNS benchmark or exactly one uploaded-project matrix run and shows honest source-specific metrics, evidence, latency, and commercial-cost posture without mixing datasets.

**Architecture:** Extend project matrix artifacts with deterministic ranking metrics and scoped usage, then read them through a bounded fail-closed `ProjectRunResultService`. Keep recommendation and cost selection in a pure CommonJS-compatible JavaScript module; `web/app.js` only orchestrates source requests and rendering. Official and uploaded result arrays remain separate at every layer.

**Tech Stack:** Python 3, stdlib HTTP server, existing `ProjectWorkspace`, CSV/JSON/JSONL run artifacts, vanilla JavaScript, CommonJS/Node VM tests, pytest, Playwright Chromium for browser QA.

---

## Scope map

| Unit | Responsibility |
|---|---|
| `source/services/project_run_artifacts.py` | Bounded no-follow reads for run-owned regular files and paginated evidence JSON |
| `source/services/project_run_metrics.py` | Pure Recall@K, MRR@K, nDCG@K, and average-query-latency calculations |
| `source/services/project_relevance.py` | Reusable canonical chunk relevance predicate |
| `source/services/project_matrix_runner.py` | Persist schema-v2 metric, latency, timestamp, evidence-count, and scoped-usage artifacts |
| `benchmarking/adapters/remote_embeddings.py` | Capture provider-returned OpenAI input-token usage without estimating it |
| `source/services/project_run_results.py` | Discover valid project runs and normalize one selected run without cross-source fallback |
| `scripts/serve_benchmark_dashboard.py` | Expose source, run, result, and paginated evidence GET APIs |
| `web/recommendations.js` | Pure source-aware row normalization, winner logic, cost states, sorting, and badges |
| `web/index.html` | Source controls, adaptive recommendation strip/table, pricing ledger, trade-off section, evidence dialog |
| `web/app.js` | Request cancellation/generation guards, source orchestration, escaped rendering, evidence pagination |
| `web/styles.css` | Compact hierarchy, winner treatments, responsive selectors/table/dialog |

## Non-negotiable constraints

- Keep `configs/benchmark.local.json` and all protected official artifacts unchanged.
- The official matrix remains exactly 180 combinations.
- NVIDIA remains a separate lane and never enters official/project recommendation rows.
- One active source only: official, or one `project_id` + one `run_id`.
- Lexical-preview runs never appear in matrix-result selectors.
- Evidence-only runs never receive quality/value scores.
- Missing metrics and usage remain `null`/Not recorded, never zero.
- Shared OpenAI embedding usage appears once in the run ledger and cannot become a per-row cost.
- Amazon rerank SearchUnits are combination-scoped and calculated from actual rerank calls.
- Do not commit `artifacts/`, logs, data, screenshots, traces, or generated run artifacts.
- Deployment is incomplete until work-laptop push, VM pull, restart, API smoke, browser QA, and artifact-protection checks pass.

---

### Task 0: Lock the baseline and protected-artifact gate

**Files:**
- Verify only: `configs/benchmark.local.json`
- Verify only: `configs/protected_official_artifacts.sha256`
- Verify only: `artifacts/predeploy/runtime-artifacts.sha256`

- [ ] **Step 1: Confirm the branch and worktree state**

Run:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
```

Expected:

```text
feat/project-isolation-hardening
?? artifacts/
```

The commit must be `ab70906` or a descendant containing only the approved design/plan commits. Do not stage `artifacts/`.

- [ ] **Step 2: Record protected hashes before implementation**

Run:

```bash
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py \
  --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 \
  --roots configs/protected_runtime_artifact_roots.txt
git diff -- configs/benchmark.local.json configs/protected_official_artifacts.sha256 configs/protected_runtime_artifact_roots.txt
```

Expected: both verification commands pass and the protected-file diff is empty.

- [ ] **Step 3: Run the existing focused baseline**

Run:

```bash
python3 -m pytest -q \
  tests/test_project_relevance.py \
  tests/test_project_matrix_runner.py \
  tests/test_dashboard_metrics.py \
  tests/test_official_artifact_protection.py
node --check web/app.js
```

Expected: all tests pass; only the known Python `cgi` deprecation warning is acceptable.

---

### Task 1: Add bounded no-follow project-run artifact reads

**Files:**
- Create: `source/services/project_run_artifacts.py`
- Create: `tests/test_project_run_artifacts.py`

- [ ] **Step 1: Write failing regular-file and pagination tests**

Create `tests/test_project_run_artifacts.py` with these core cases:

```python
import json
import os
from pathlib import Path

import pytest

from source.services.project_run_artifacts import (
    ProjectRunArtifactError,
    read_project_artifact,
    read_project_json_rows_page,
)


def test_reads_only_bounded_regular_descendants(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "summary.csv").write_bytes(b"a,b\n1,2\n")
    assert read_project_artifact(root, "summary.csv", max_bytes=64) == b"a,b\n1,2\n"
    with pytest.raises(ProjectRunArtifactError, match="size limit"):
        read_project_artifact(root, "summary.csv", max_bytes=3)
    with pytest.raises(ProjectRunArtifactError, match="path is invalid"):
        read_project_artifact(root, "../outside", max_bytes=64)


def test_rejects_symlinked_root_parent_and_leaf(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "summary.csv").write_text("ok", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ProjectRunArtifactError):
        read_project_artifact(alias, "summary.csv", max_bytes=64)
    (real / "leaf.csv").symlink_to(real / "summary.csv")
    with pytest.raises(ProjectRunArtifactError):
        read_project_artifact(real, "leaf.csv", max_bytes=64)


def test_evidence_json_page_is_bounded_and_filtered(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    rows = [
        {"combo_id": "combo_a", "rank": 1},
        {"combo_id": "combo_b", "rank": 1},
        {"combo_id": "combo_a", "rank": 2},
    ]
    (root / "evidence.json").write_text(
        json.dumps({
            "schema_version": 1,
            "project_id": "project_a",
            "run_id": "run_a",
            "mode": "evidence_only",
            "rows": rows,
        }),
        encoding="utf-8",
    )
    page = read_project_json_rows_page(
        root,
        "evidence.json",
        combo_id="combo_a",
        offset=1,
        limit=1,
        max_bytes=4096,
    )
    assert page["rows"] == [{"combo_id": "combo_a", "rank": 2}]
    assert page["next_offset"] is None
    assert page["project_id"] == "project_a"
```

Also test malformed UTF-8, malformed JSON, an invalid evidence envelope, non-dict rows, an oversized JSON artifact, directories used as files, and an inode-changing leaf through a monkeypatched `os.fstat`.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
python3 -m pytest -q tests/test_project_run_artifacts.py
```

Expected: collection fails because `source.services.project_run_artifacts` does not exist.

- [ ] **Step 3: Implement the bounded reader**

Create `source/services/project_run_artifacts.py` with this public contract:

```python
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


class ProjectRunArtifactError(RuntimeError):
    """A project run artifact is missing, unsafe, oversized, or malformed."""


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    relative = PurePosixPath(relative_path)
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ProjectRunArtifactError("artifact path is invalid")
    return parts


def _open_artifact(root: Path, relative_path: str) -> tuple[int, int, os.stat_result]:
    root = Path(root)
    root_meta = root.lstat()
    if stat.S_ISLNK(root_meta.st_mode) or not stat.S_ISDIR(root_meta.st_mode):
        raise ProjectRunArtifactError("artifact root is unavailable")
    if root.resolve(strict=True) != root:
        raise ProjectRunArtifactError("artifact root is unavailable")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, directory_flags)
    try:
        opened_root = os.fstat(current_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (root_meta.st_dev, root_meta.st_ino):
            raise ProjectRunArtifactError("artifact root changed during loading")
        parts = _relative_parts(relative_path)
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
            if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
                raise ProjectRunArtifactError("artifact parent is unavailable")
        leaf_meta = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        if stat.S_ISLNK(leaf_meta.st_mode) or not stat.S_ISREG(leaf_meta.st_mode):
            raise ProjectRunArtifactError("artifact must be a regular file")
        leaf_fd = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current_fd,
        )
        opened_leaf = os.fstat(leaf_fd)
        if (opened_leaf.st_dev, opened_leaf.st_ino) != (leaf_meta.st_dev, leaf_meta.st_ino):
            os.close(leaf_fd)
            raise ProjectRunArtifactError("artifact changed during loading")
        return current_fd, leaf_fd, opened_leaf
    except Exception:
        os.close(current_fd)
        raise


def read_project_artifact(root: Path, relative_path: str, *, max_bytes: int) -> bytes:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    parent_fd, leaf_fd, metadata = _open_artifact(root, relative_path)
    try:
        if metadata.st_size > max_bytes:
            raise ProjectRunArtifactError("artifact exceeds size limit")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(leaf_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_bytes:
            raise ProjectRunArtifactError("artifact exceeds size limit")
        return content
    except OSError:
        raise ProjectRunArtifactError("artifact is unavailable") from None
    finally:
        os.close(leaf_fd)
        os.close(parent_fd)
```

Implement `read_project_json_rows_page(root: Path, relative_path: str, *, combo_id: str, offset: int, limit: int, max_bytes: int) -> dict[str, Any]` by calling `read_project_artifact`, decoding one bounded UTF-8 JSON object, requiring exact envelope keys `schema_version`, `project_id`, `run_id`, `mode`, and `rows`, validating every row is a dict, filtering exact `combo_id`, applying matched-row `offset`, enforcing `1 <= limit <= 100`, and returning envelope metadata plus `rows` and `next_offset`. Reject malformed content with `ProjectRunArtifactError("evidence artifact is invalid")`. Never return raw exception text.

- [ ] **Step 4: Run the artifact-reader tests**

Run:

```bash
python3 -m pytest -q tests/test_project_run_artifacts.py
python3 -m py_compile source/services/project_run_artifacts.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 5: Commit the reader**

```bash
git add -- source/services/project_run_artifacts.py tests/test_project_run_artifacts.py
git diff --cached --check
git commit -m "feat: add bounded project run artifact reads"
```

---

### Task 2: Add deterministic project-run metrics

**Files:**
- Create: `source/services/project_run_metrics.py`
- Create: `tests/test_project_run_metrics.py`
- Modify: `source/services/project_relevance.py`
- Modify: `tests/test_project_relevance.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_project_run_metrics.py`:

```python
import math

from source.services.project_run_metrics import QueryMetricInput, summarize_query_metrics


def test_summarizes_recall_mrr_ndcg_and_query_latency():
    result = summarize_query_metrics(
        [
            QueryMetricInput("q1", True, (2,), 1, 0.10, 0.20),
            QueryMetricInput("q2", True, (), 1, 0.15, 0.25),
        ],
        k=3,
    )
    assert result["recall_at_k"] == 0.5
    assert result["mrr_at_k"] == 0.25
    assert math.isclose(result["ndcg_at_k"], (1 / math.log2(3)) / 2)
    assert math.isclose(result["retrieval_latency_s"], 0.125)
    assert math.isclose(result["rerank_latency_s"], 0.225)
    assert math.isclose(result["avg_query_latency_s"], 0.35)


def test_evidence_only_metrics_keep_latency_but_not_quality():
    result = summarize_query_metrics(
        [QueryMetricInput("q1", False, (), 0, 0.04, 0.06)],
        k=10,
    )
    assert result["recall_at_k"] is None
    assert result["mrr_at_k"] is None
    assert result["ndcg_at_k"] is None
    assert result["avg_query_latency_s"] == 0.10
```

Add tests that reject duplicate query IDs, non-positive K, non-finite/negative latency, relevant ranks outside `1..K`, and `relevant_corpus_count < len(relevant_ranks)`.

- [ ] **Step 2: Add a failing canonical chunk-relevance test**

In `tests/test_project_relevance.py`, assert `chunk_is_relevant(question, chunk, chunker_id)` returns the same decision as `hit_is_relevant(question, SearchHit(chunk, 0.0), chunker_id)` for source ID, qualified chunk reference, reference-context match, and miss cases.

Run:

```bash
python3 -m pytest -q tests/test_project_run_metrics.py tests/test_project_relevance.py
```

Expected: failures for missing `project_run_metrics` and `chunk_is_relevant`.

- [ ] **Step 3: Refactor canonical relevance without changing semantics**

In `source/services/project_relevance.py`, import `Chunk` and extract:

```python
def chunk_is_relevant(question: ProjectQuestion, chunk: Chunk, chunker_id: str) -> bool:
    load_pinned_stopwords()
    metadata = chunk.metadata or {}
    source_id = metadata.get("source_id")
    if isinstance(source_id, str) and source_id in question.labels.source_ids:
        return True
    chunk_id = str(chunk.id)
    if any(
        reference.chunker == chunker_id and reference.chunk_id == chunk_id
        for reference in question.labels.chunk_refs
    ):
        return True
    normalized_chunk = _normalize_text(chunk.paragraph)
    chunk_tokens = wns_context_tokens_v1(chunk.paragraph)
    for normalized_context, context_tokens in _eligible_contexts(question):
        if normalized_context in normalized_chunk:
            return True
        if len(context_tokens & chunk_tokens) / len(context_tokens) >= _CONTEXT_RECALL_THRESHOLD:
            return True
    return False


def hit_is_relevant(question: ProjectQuestion, hit: SearchHit, chunker_id: str) -> bool:
    return chunk_is_relevant(question, hit.chunk, chunker_id)
```

- [ ] **Step 4: Implement the pure metric module**

Create `source/services/project_run_metrics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Iterable


@dataclass(frozen=True)
class QueryMetricInput:
    query_id: str
    applicable: bool
    relevant_ranks: tuple[int, ...]
    relevant_corpus_count: int
    retrieval_latency_s: float
    rerank_latency_s: float


def _dcg(ranks: tuple[int, ...]) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in ranks)


def summarize_query_metrics(rows: Iterable[QueryMetricInput], *, k: int) -> dict[str, float | int | None]:
    items = tuple(rows)
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if len({row.query_id for row in items}) != len(items):
        raise ValueError("query metric IDs must be unique")
    for row in items:
        if not row.query_id or row.relevant_corpus_count < 0:
            raise ValueError("query metric input is invalid")
        if any(rank < 1 or rank > k for rank in row.relevant_ranks):
            raise ValueError("relevant rank is outside K")
        if len(set(row.relevant_ranks)) != len(row.relevant_ranks):
            raise ValueError("relevant ranks must be unique")
        if row.relevant_corpus_count < len(row.relevant_ranks):
            raise ValueError("relevant corpus count is inconsistent")
        if any(not math.isfinite(value) or value < 0 for value in (row.retrieval_latency_s, row.rerank_latency_s)):
            raise ValueError("query latency is invalid")

    applicable = [row for row in items if row.applicable]
    recall = mrr = ndcg = None
    if applicable:
        recall = fmean(1.0 if row.relevant_ranks else 0.0 for row in applicable)
        mrr = fmean(1.0 / min(row.relevant_ranks) if row.relevant_ranks else 0.0 for row in applicable)
        ndcg_values = []
        for row in applicable:
            ideal_count = min(row.relevant_corpus_count, k)
            ideal = _dcg(tuple(range(1, ideal_count + 1)))
            ndcg_values.append(_dcg(row.relevant_ranks) / ideal if ideal > 0 else 0.0)
        ndcg = fmean(ndcg_values)

    retrieval = fmean(row.retrieval_latency_s for row in items) if items else None
    rerank = fmean(row.rerank_latency_s for row in items) if items else None
    return {
        "labelled_queries": len(applicable),
        "unlabelled_queries": len(items) - len(applicable),
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "retrieval_latency_s": retrieval,
        "rerank_latency_s": rerank,
        "avg_query_latency_s": (
            retrieval + rerank if retrieval is not None and rerank is not None else None
        ),
    }
```

- [ ] **Step 5: Run and commit the metric layer**

```bash
python3 -m pytest -q tests/test_project_run_metrics.py tests/test_project_relevance.py
python3 -m py_compile source/services/project_run_metrics.py source/services/project_relevance.py
git add -- source/services/project_run_metrics.py source/services/project_relevance.py tests/test_project_run_metrics.py tests/test_project_relevance.py
git diff --cached --check
git commit -m "feat: calculate project run ranking metrics"
```

Expected: all tests pass and the commit contains only the four listed files.

---

### Task 3: Persist schema-v2 metrics, latency, timestamps, evidence counts, and scoped usage

**Files:**
- Modify: `source/services/project_matrix_runner.py`
- Modify: `benchmarking/adapters/remote_embeddings.py`
- Modify: `tests/test_project_matrix_runner.py`

- [ ] **Step 1: Write failing runner assertions**

Extend `test_real_matrix_is_project_scoped_honest_and_failure_isolated` and add a labeled-run test asserting:

```python
summary = _summary_rows(run_root / "summary.csv")
completed = next(row for row in summary if row["status"] == "completed")
assert completed["summary_schema_version"] == "2"
assert completed["mrr_at_k"]
assert completed["ndcg_at_k"]
assert completed["avg_query_latency_s"]
assert completed["evidence_count"]

manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["summary_schema_version"] == 2
assert manifest["created_at"].endswith("Z")
assert manifest["completed_at"].endswith("Z")
assert manifest["artifacts"]["summary"] == "summary.csv"
```

For an OpenAI fixture response containing `{"usage": {"prompt_tokens": 123}}`, assert the embedding receipt records:

```python
assert receipt["adapters"]["embedding"]["provider_metadata"]["embedding_input_tokens"] == 123
assert receipt["usage"]["embedding_usage_scope"] == "shared_embedding"
assert receipt["usage"]["embedding_usage_key"] == f"{chunker_id}|openai_text-embedding-3-large"
```

For Amazon Rerank with one query and at most 100 hits:

```python
assert receipt["usage"]["rerank_search_units"] == 1
assert receipt["usage"]["rerank_usage_scope"] == "combination"
```

For typed/evidence-only questions, assert all quality fields are empty while measured average query latency and evidence count remain present.

- [ ] **Step 2: Run the focused runner test and observe failure**

```bash
python3 -m pytest -q tests/test_project_matrix_runner.py
```

Expected: failures for missing schema-v2 fields, MRR/nDCG/latency, timestamps, evidence count, and usage scope.

- [ ] **Step 3: Capture provider-returned embedding input tokens without estimating**

In `benchmarking/adapters/remote_embeddings.py`, change `_response_metadata` and each `last_response_metadata` annotation from `dict[str, str]` to `dict[str, Any]`, keep request IDs/model metadata, and add a strict usage helper:

```python
def _input_tokens(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("input_tokens", usage.get("prompt_tokens"))
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
```

Initialize `self.input_tokens = 0` and `self.input_tokens_complete = True` in `OpenAIEmbeddingAdapter`. After each successful provider response, add only a validated `_input_tokens(response)`. If **any** successful batch response omits or invalidates input-token usage, set `input_tokens_complete = False` permanently for that adapter instance and omit `embedding_input_tokens` from final metadata; never publish a partial total as complete measured usage. Set:

```python
metadata = _response_metadata(response)
measured = _input_tokens(response)
if measured is None:
    self.input_tokens_complete = False
elif self.input_tokens_complete:
    self.input_tokens += measured
if self.input_tokens_complete:
    metadata["embedding_input_tokens"] = self.input_tokens
self.last_response_metadata = metadata
```

Add a multi-batch regression test where the first response reports usage and the second omits it; final metadata must not contain `embedding_input_tokens`. Do not infer tokens from text length or `total_tokens`.

- [ ] **Step 4: Extend runner fields and caches**

In `source/services/project_matrix_runner.py`:

1. Import `datetime`, `timezone`, `chunk_is_relevant`, `QueryMetricInput`, and `summarize_query_metrics`.
2. Add to `_SUMMARY_FIELDS`:

```python
"summary_schema_version",
"mrr_at_k",
"ndcg_at_k",
"retrieval_latency_s",
"rerank_latency_s",
"avg_query_latency_s",
"evidence_count",
"embedding_input_tokens",
"embedding_usage_scope",
"embedding_usage_key",
"rerank_search_units",
"rerank_usage_scope",
```

3. Capture `created_at` once before execution and `completed_at` immediately before final manifest publication:

```python
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
```

4. Extend retrieval cache entries with `retrieval_latency_by_question`.
5. Precompute canonical relevant-chunk counts once per `(chunker_id, question_id)`:

```python
relevant_chunk_counts = {
    (chunker_id, question.question_id): sum(
        1 for chunk in chunks_by_id.get(chunker_id, ())
        if chunk_is_relevant(question, chunk, chunker_id)
    )
    for chunker_id in request.chunkers
    for question in questions
    if label_applies(question, chunker_id)
}
```

6. For each query, collect the actual relevant ranks, retrieval latency, and rerank latency into `QueryMetricInput`.
7. Call `summarize_query_metrics(query_metric_inputs, k=request.top_k)` once per completed combination.
8. Set `evidence_count = len(combo_evidence_rows)`.
9. Count Amazon SearchUnits from actual calls only:

```python
rerank_search_units += math.ceil(len(base_hits) / 100) if base_hits else 0
```

10. Record OpenAI usage as `shared_embedding` with key `f"{chunker_id}|{embedding_id}"`; record Amazon units as `combination`.
11. Keep failed-row quality/latency fields `None` unless a complete query metric exists; never publish partial quality.
12. Add `summary_schema_version`, `scoring_mode`, `metric_k`, `created_at`, and `completed_at` to final `manifest.json`.
13. Add schema version/timestamps plus Recall, MRR, nDCG, and latency fields to each eligible `analysis.json` row.
14. Keep the existing `evidence.json` envelope at `schema_version: 1`; schema v2 applies to summary/analysis and manifest metadata only.

- [ ] **Step 5: Verify project-run artifacts remain isolated and atomic**

Run:

```bash
python3 -m pytest -q tests/test_project_matrix_runner.py tests/test_project_run_metrics.py tests/test_project_vector_namespaces.py
python3 -m py_compile source/services/project_matrix_runner.py benchmarking/adapters/remote_embeddings.py
git diff --check
```

Expected: all tests pass; no `data/modular_runs` or official artifact writes occur.

- [ ] **Step 6: Commit runner schema v2**

```bash
git add -- source/services/project_matrix_runner.py benchmarking/adapters/remote_embeddings.py tests/test_project_matrix_runner.py
git diff --cached --check
git commit -m "feat: persist source-aware project run metrics"
```

---

### Task 4: Add the fail-closed run result service and HTTP APIs

**Files:**
- Create: `source/services/project_run_results.py`
- Create: `tests/test_project_run_results.py`
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/test_project_run_results.py` with fixtures that build two uploaded projects, one schema-v2 labeled run, one legacy Recall-only run, one lexical-preview run, and one partial run.

Assert the public service contract:

```python
service = ProjectRunResultService(ProjectWorkspace(projects_root))

sources = service.result_sources(official_configured=180, official_evaluated=180)
assert sources["official"] == {"source_type": "official", "configured": 180, "evaluated": 180}
assert {row["project_id"] for row in sources["projects"]} == {alpha_id, beta_id}

runs = service.project_runs(alpha_id)
assert {row["run_id"] for row in runs} == {labeled_run_id, legacy_run_id, partial_run_id}
assert all(row["run_id"] != lexical_run_id for row in runs)

result = service.project_run_results(alpha_id, labeled_run_id)
assert result["source_type"] == "uploaded_project"
assert result["metric_k"] == 10
assert all(row["project_id"] == alpha_id for row in result["rows"])
assert result["evidence_counts_by_combo"]
```

Assert legacy normalization keeps Recall and returns `None` for missing MRR/nDCG/latency. Assert a `beta_id` + `alpha_run_id` request fails. Add symlinked manifest/summary/evidence, malformed CSV/JSON, oversized file, row-count overflow, mismatched row identity, duplicate combo ID, unknown canonical adapter ID, and raw-error sanitization tests.

- [ ] **Step 2: Write failing evidence-page tests**

Assert:

```python
page = service.project_run_evidence(alpha_id, labeled_run_id, combo_id, limit=2, offset=0)
assert len(page["rows"]) == 2
assert all(row["combo_id"] == combo_id for row in page["rows"])
assert "paragraph" not in page["rows"]
assert set(page["rows"][0]) <= {
    "combo_id", "query_id", "query", "source_name", "page_number",
    "excerpt", "base_score", "rerank_score", "latency_s", "rank",
}
```

Requesting a failed or foreign `combo_id` must return a safe not-found error.

Run:

```bash
python3 -m pytest -q tests/test_project_run_results.py
```

Expected: import failure because the service does not exist.

- [ ] **Step 3: Implement `ProjectRunResultService`**

Create `source/services/project_run_results.py` with this complete public error type:

```python
class ProjectRunResultsError(RuntimeError):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status = status
```

Define `ProjectRunResultService` with these exact public signatures:

| Method | Return |
|---|---|
| `result_sources(*, official_configured: int, official_evaluated: int)` | `dict[str, Any]` |
| `project_runs(project_id: str)` | `list[dict[str, Any]]` |
| `project_run_results(project_id: str, run_id: str)` | `dict[str, Any]` |
| `project_run_evidence(project_id: str, run_id: str, combo_id: str, *, limit: int, offset: int)` | `dict[str, Any]` |

The constructor is:

```python
class ProjectRunResultService:
    MAX_PROJECT_MANIFEST_BYTES = 2 * 1024 * 1024
    MAX_RUN_MANIFEST_BYTES = 8 * 1024 * 1024
    MAX_REQUEST_BYTES = 2 * 1024 * 1024
    MAX_SUMMARY_BYTES = 16 * 1024 * 1024
    MAX_SUMMARY_ROWS = 10_000
    MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
    MAX_EVIDENCE_PAGE = 100

    def __init__(self, workspace: ProjectWorkspace, *, catalog_path: Path = _DEFAULT_CATALOG):
        if not isinstance(workspace, ProjectWorkspace):
            raise TypeError("workspace must be a ProjectWorkspace")
        self.workspace = workspace
        self.catalog_path = Path(catalog_path).resolve()
```

Implement the four public methods with these exact rules:

- Resolve project/run only through `ProjectWorkspace.layout()` and `run_layout()`.
- Read all artifacts through `read_project_artifact`/`read_project_json_rows_page`.
- Project methods never inspect `FULL_DIR`, `MODULAR_DIR`, `EVAL_DIR`, official configs, or official benchmark rows; missing project artifacts fail without fallback.
- Parse `request.json`, validate its fingerprint with `validate_project_matrix_request`, and require requested identity.
- Require matrix manifest keys `request_fingerprint`, `combination_count`, `state`, and `artifacts`; exclude any manifest with `mode == "lexical_preview"`.
- Admit only direct project/run directory names validated by the existing workspace ID rules; skip malformed/symlinked catalog entries without exposing their paths.
- Read the project label from its validated manifest and sort projects by safe artifact time descending.
- Allow only `completed`/`partial` run states. Sort runs by `completed_at`, then `created_at`; older valid runs use manifest modification time labeled `Legacy artifact time`.
- Require the manifest evidence path to equal `evidence.json`; do not accept an arbitrary relative artifact name from the manifest.
- Parse CSV with `csv.DictReader`; reject duplicate/missing headers, more than 10,000 rows, duplicate combo IDs, and row project/run mismatches.
- Validate chunker/embedding/store/reranker IDs against `project_matrix_catalog.json`.
- Derive `commercial_model_ids` from exact catalog entries whose `license == "commercial"`.
- Normalize empty legacy metrics to `None`; parse finite nonnegative floats only.
- Parse `metric_k` from validated request `top_k`.
- Build `measured_usage` from schema-v2 fields and receipts; deduplicate `shared_embedding` usage by usage key for the run ledger.
- Do not duplicate shared embedding cost into rows.
- Use summary `evidence_count` when present. For legacy runs, load the bounded `evidence.json` artifact once and count valid rows by combo; if safely unavailable, return `None`, never zero.
- Evidence responses whitelist fields and filter exact combo ownership.
- Wrap filesystem/parse errors as `ProjectRunResultsError` with one of: `invalid_request`/400, `not_found`/404, `run_unavailable`/409, `artifact_too_large`/413.

- [ ] **Step 4: Add HTTP endpoint tests**

In `tests/test_dashboard_metrics.py`, import `quote` from `urllib.parse`, start `ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)` against a temporary `USER_PROJECTS_DIR`, then use `urllib.request.urlopen` to assert:

- `/api/result-sources`
- `f"/api/project-runs?project_id={quote(project_id)}"`
- `f"/api/project-run-results?project_id={quote(project_id)}&run_id={quote(run_id)}"`
- `f"/api/project-run-evidence?project_id={quote(project_id)}&run_id={quote(run_id)}&combo_id={quote(combo_id)}&limit=2&offset=0"`

return JSON with `Cache-Control: no-store`. Assert malformed, missing, cross-project, and oversized requests return structured JSON containing only `error.code` and `error.message`, never paths or exception text.

- [ ] **Step 5: Wire the GET endpoints**

In `scripts/serve_benchmark_dashboard.py`, import `ProjectRunResultService` and `ProjectRunResultsError`. Add helpers before `Handler`:

```python
def project_result_service() -> ProjectRunResultService:
    return ProjectRunResultService(ProjectWorkspace(USER_PROJECTS_DIR))


def exact_query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name, [])
    if len(values) != 1 or not values[0]:
        raise ProjectRunResultsError("invalid_request", "Invalid result request", 400)
    return values[0]
```

Handle the four endpoints before `/api/results`. Use `parse_qs(parsed.query, keep_blank_values=True)`, reject unknown query keys, parse `limit`/`offset` as bounded integers, and map service errors:

```python
except ProjectRunResultsError as exc:
    self.send_json({"error": {"code": exc.code, "message": exc.public_message}}, exc.status)
```

For `/api/result-sources`, calculate official evaluated count from `read_benchmark_reference()["summary"]` rather than trusting a constant.

- [ ] **Step 6: Run and commit the result service**

```bash
python3 -m pytest -q tests/test_project_run_artifacts.py tests/test_project_run_results.py tests/test_dashboard_metrics.py
python3 -m py_compile source/services/project_run_results.py scripts/serve_benchmark_dashboard.py
git diff --check
git add -- source/services/project_run_results.py tests/test_project_run_results.py scripts/serve_benchmark_dashboard.py tests/test_dashboard_metrics.py
git diff --cached --check
git commit -m "feat: expose isolated project run results"
```

Expected: all service/API tests pass and official artifact verification remains green.

---

### Task 5: Implement pure source-aware recommendation and pricing logic

**Files:**
- Create: `web/recommendations.js`
- Create: `tests/test_pipeline_recommendations.py`

- [ ] **Step 1: Write failing Node-backed unit tests**

Create `tests/test_pipeline_recommendations.py` with a helper that executes `web/recommendations.js` through `node -e`. Use the first-party rates verified on 2026-07-13:

- OpenAI model card: `https://developers.openai.com/api/docs/models/text-embedding-3-large` — `$0.13 / 1M tokens`.
- AWS public Price List API: `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json` — us-west-2 SKU `Z7M6S4MRBXNXJRB4`, dimension `USW2-AmazonRerank-v1-searchunits`, effective `2026-07-01T00:00:00Z`, `$1 / 1,000 Search Units` (`$0.001/SearchUnit`).

Cover:

```javascript
const labelled = {
  source_type: 'uploaded_project', metric_k: 10, scoring_mode: 'retrieval_labels',
  rows: [
    {combo_id:'quality',status:'completed',ndcg_at_k:0.9,mrr_at_k:0.8,recall_at_k:0.95,avg_query_latency_s:0.4,commercial_model_ids:[],measured_usage:{}},
    {combo_id:'speed',status:'completed',ndcg_at_k:0.7,mrr_at_k:0.7,recall_at_k:0.8,avg_query_latency_s:0.1,commercial_model_ids:[],measured_usage:{}},
    {combo_id:'failed',status:'failed',ndcg_at_k:1,avg_query_latency_s:0.01,commercial_model_ids:[],measured_usage:{}},
  ],
};
const result = R.recommendationsForSource(labelled);
if (result.mode !== 'labelled' || result.roles.quality.combo_id !== 'quality' || result.roles.speed.combo_id !== 'speed') process.exit(1);
```

Also assert:

- evidence-only returns roles `fastest`, `run_health`, `evidence_coverage` and no quality/value;
- failed rows never win;
- missing latency produces `latency_not_recorded`;
- OpenAI rate is 0.13 per million tokens;
- Amazon rate is 0.001 per SearchUnit;
- shared embedding usage appears once in ledger and yields no row `total_cost_usd`;
- combination-scoped Amazon units calculate row cost;
- unknown canonical commercial ID is `pricing_unavailable`;
- missing measured usage is `usage_missing`, never zero;
- official rows use existing winner score and official metric labels;
- project quality ordering is nDCG, then MRR, then Recall, then latency, then stable key.

- [ ] **Step 2: Verify the unit test fails**

```bash
python3 -m pytest -q tests/test_pipeline_recommendations.py
```

Expected: failure because `web/recommendations.js` does not exist.

- [ ] **Step 3: Implement `web/recommendations.js`**

Use a UMD wrapper so browser and Node tests share exactly one implementation:

```javascript
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PipelineRecommendations = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const PRICING = Object.freeze({
    'openai_text-embedding-3-large': Object.freeze({unit:'million_input_tokens', usd_rate:0.13}),
    'Amazon Rerank v1': Object.freeze({unit:'search_unit', usd_rate:0.001}),
  });

  const finite = value => Number.isFinite(Number(value)) ? Number(value) : null;
  const positive = value => { const number = finite(value); return number !== null && number > 0 ? number : null; };
  const stableKey = row => [row.chunker_id || row.sheet, row.embedding_id || row.embedding, row.vector_store_id || row.store, row.reranker_id || row.reranker || 'none'].join('|');
  const completed = (rows, sourceType) => (rows || []).filter(row => {
    if (!row) return false;
    if (sourceType === 'official') return !row.status || row.status === 'completed';
    return row.status === 'completed';
  });

  function projectQualityCompare(a, b) {
    return (finite(b.ndcg_at_k) ?? -1) - (finite(a.ndcg_at_k) ?? -1)
      || (finite(b.mrr_at_k) ?? -1) - (finite(a.mrr_at_k) ?? -1)
      || (finite(b.recall_at_k) ?? -1) - (finite(a.recall_at_k) ?? -1)
      || (positive(a.avg_query_latency_s) ?? Infinity) - (positive(b.avg_query_latency_s) ?? Infinity)
      || stableKey(a).localeCompare(stableKey(b));
  }

  function officialQualityCompare(a, b) {
    return (finite(b.winner_score) ?? -1) - (finite(a.winner_score) ?? -1)
      || (finite(b.recall_at_5) ?? -1) - (finite(a.recall_at_5) ?? -1)
      || (positive(a.avg_latency_seconds) ?? Infinity) - (positive(b.avg_latency_seconds) ?? Infinity)
      || stableKey(a).localeCompare(stableKey(b));
  }
```

Complete the module with:

- `pricingForRow(row)` returning `measured_usage`, `usage_missing`, `no_api_fee`, or `pricing_unavailable`;
- no per-row OpenAI cost when `embedding_usage_scope == 'shared_embedding'`;
- Amazon row cost only when `rerank_usage_scope == 'combination'`;
- `pricingLedger(source)` deduplicating shared usage by `embedding_usage_key`;
- `recommendationsForSource(source)` returning adaptive roles and sorted completed/failed rows;
- `metricLabels(source)` returning official `Recall@5/MRR/nDCG@5` or project labels using `metric_k`;
- `rowBadges(row, roles)` returning fixed text keys only;
- exports frozen with `Object.freeze`.

No function may inspect arbitrary model-name substrings to classify a paid model.

- [ ] **Step 4: Run and commit pure frontend logic**

```bash
python3 -m pytest -q tests/test_pipeline_recommendations.py
node --check web/recommendations.js
git add -- web/recommendations.js tests/test_pipeline_recommendations.py
git diff --cached --check
git commit -m "feat: add source-aware recommendation logic"
```

---

### Task 6: Build the source selector, adaptive table, evidence dialog, and responsive hierarchy

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing static and Node VM tests**

Extend `tests/test_dashboard_metrics.py` to require:

```python
assert 'data-page="compare" type="button">Recommendations<' in index
assert 'id="recommendationSource"' in index
assert 'id="recommendationProject"' in index
assert 'id="recommendationRun"' in index
assert 'id="recommendationContext"' in index
assert 'id="recommendationStrip"' in index
assert 'id="recommendationTable"' in index
assert 'id="pricingLedger"' in index
assert 'id="tradeoffDetails"' in index
assert 'id="projectEvidenceDialog"' in index
assert 'Average retrieval plus reranking latency per query' in index
assert 'src="/recommendations.js?v=' in index
assert '/api/result-sources' in app
assert '/api/project-runs?' in app
assert '/api/project-run-results?' in app
assert '/api/project-run-evidence?' in app
assert 'AbortController' in app
```

Add a Node VM test with delayed fake fetch responses:

1. select project A/run A;
2. immediately select project B/run B;
3. resolve B first and A last;
4. assert only B context/rows render.

Add separate render fixtures for official, labeled project, evidence-only project, and partial run. Assert all dynamic text is escaped and missing values render `Not recorded`/`Not applicable`.

- [ ] **Step 2: Verify frontend tests fail**

```bash
python3 -m pytest -q tests/test_dashboard_metrics.py tests/test_pipeline_recommendations.py
```

Expected: failures for missing source-aware DOM and orchestration.

- [ ] **Step 3: Replace the compare-page markup**

In `web/index.html`:

1. Rename the tab label to `Recommendations` while keeping `data-page="compare"`.
2. Update the metric glossary so `Avg sec/query` reads `Average retrieval plus reranking latency per query` and explicitly excludes chunking, embedding, vector upsert/indexing, queue, and startup time.
3. Load `/recommendations.js?v=20260713-source-aware` immediately before `/app.js?v=20260713-source-aware`.
4. Replace the current compare panel with:

```html
<section class="page" data-page-panel="compare">
  <section class="panel recommendation-panel">
    <div class="section-head">
      <div><h2>Pipeline recommendations</h2><p>Choose one result source, then compare only combinations run against that data.</p></div>
      <span id="recommendationStatus" class="mini-stat">Official benchmark</span>
    </div>
    <div class="result-source-bar">
      <label>Result source<select id="recommendationSource"><option value="official">Official WNS benchmark</option><option value="uploaded_project">Uploaded project</option></select></label>
      <label id="recommendationProjectField" hidden>Project<select id="recommendationProject"></select></label>
      <label id="recommendationRunField" hidden>Matrix run<select id="recommendationRun"></select></label>
    </div>
    <p id="recommendationContext" class="result-context">Viewing: Official WNS benchmark</p>
    <div id="recommendationStrip" class="recommendation-strip" aria-live="polite"></div>
  </section>

  <section class="panel">
    <div class="section-head"><div><h2>Top combinations</h2><p id="recommendationModeNote">Quality, speed, and honest cost posture for the active source.</p></div><div id="recommendationSort" class="compact-sort"></div></div>
    <div id="recommendationFilters" class="filter-row"></div>
    <div class="table-wrap"><table id="recommendationTable"></table></div>
  </section>

  <section class="panel pricing-panel">
    <div class="section-head"><div><h2>Commercial pricing basis</h2><p>Published API rates and measured usage scope. VM infrastructure is excluded.</p></div></div>
    <div id="pricingLedger" class="pricing-ledger"></div>
  </section>

  <details id="tradeoffDetails" class="panel tradeoff-details">
    <summary>Explore trade-offs</summary>
    <div class="viz-grid"><div id="top10ScoreChart" class="svg-chart"></div><div id="qualityLatencyChart" class="svg-chart"></div></div>
    <div id="comparisonGrid" class="comparison-grid"></div>
    <section class="grid two"><article><h3>Reranker lift</h3><div id="rerankerLiftChart" class="bar-list"></div></article><article><h3>Best by component</h3><div id="stageComparisonChart" class="bar-list"></div></article></section>
  </details>

  <dialog id="projectEvidenceDialog" class="evidence-dialog">
    <form method="dialog"><button type="submit" aria-label="Close evidence">Close</button></form>
    <h2 id="projectEvidenceTitle">Combination evidence</h2>
    <p id="projectEvidenceStatus"></p>
    <div id="projectEvidenceRows" class="evidence-dialog-rows"></div>
    <button id="projectEvidenceMore" type="button" hidden>Load more</button>
  </dialog>
</section>
```

- [ ] **Step 4: Add source orchestration with stale-request protection**

In `web/app.js`, define:

```javascript
const recommendationState = {
  sourceType: 'official',
  projects: [],
  runs: [],
  active: null,
  generation: 0,
  controller: null,
  evidenceController: null,
  evidenceOffset: 0,
  evidenceComboId: null,
};

function beginRecommendationRequest() {
  recommendationState.generation += 1;
  recommendationState.controller?.abort();
  recommendationState.controller = new AbortController();
  clearRecommendationView('Loading selected result source…');
  return {generation: recommendationState.generation, signal: recommendationState.controller.signal};
}

function requestIsCurrent(generation) {
  return generation === recommendationState.generation;
}
```

Implement:

- `loadResultSources()` from `/api/result-sources`;
- `loadProjectRuns(projectId)` from `/api/project-runs`;
- `loadProjectRun(projectId, runId)` from `/api/project-run-results`;
- `showOfficialRecommendations()` using only the current official evaluation payload;
- `renderRecommendationSource(payload)` using `PipelineRecommendations.recommendationsForSource`;
- `clearRecommendationView(message)` clearing winners, table, charts, filters, and evidence;
- exact source/project/run context text;
- adaptive sort/filter controls based on `metric_names`;
- no call to `evaluatedRows()` in uploaded-project mode;
- no fallback to `state.operational.evaluation` after uploaded-source failure.

Every awaited request must check `requestIsCurrent(generation)` before mutating DOM.

- [ ] **Step 5: Render adaptive winners, table, ledger, and evidence**

Use only escaped dynamic values. Fixed badge classes/text may come from the pure module.

Implement:

```javascript
function metricCell(value, {percent = false} = {}) {
  if (value === null || value === undefined || value === '') return 'Not recorded';
  const number = Number(value);
  if (!Number.isFinite(number)) return 'Not recorded';
  return percent ? `${(number * 100).toFixed(1)}%` : number.toFixed(3);
}
```

- Labeled strip: Quality, Speed, Value/no-API-fee.
- Evidence-only strip: Fastest, Run health, Evidence coverage.
- Rank completed rows only; append failed rows after completed rows with rose status badge.
- Show actual uploaded metric K in headers.
- For labels that do not apply to one combination, render `Not applicable`.
- Render shared OpenAI usage once in the ledger and combination-scoped Amazon cost per row.
- Render `Run usage not recorded`, `No external model API fee · VM infrastructure excluded`, and `Pricing unavailable` exactly.
- `View evidence` requests `/api/project-run-evidence` with `limit=25`; Load more uses returned `next_offset`.
- Source switch/close aborts the evidence request and clears dialog contents.

- [ ] **Step 6: Add compact responsive styles**

Append focused classes in `web/styles.css`:

```css
.result-source-bar{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:14px 0}
.result-context{margin:8px 0 14px;color:var(--soft);font-weight:800}
.recommendation-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border:1px solid rgba(125,211,252,.14);border-radius:18px;overflow:hidden;background:rgba(3,11,20,.34)}
.recommendation-item{padding:16px;min-width:0;border-right:1px solid rgba(125,211,252,.12)}
.recommendation-item:last-child{border-right:0}.recommendation-item.quality{box-shadow:inset 0 3px var(--mint)}.recommendation-item.speed{box-shadow:inset 0 3px var(--accent)}.recommendation-item.value{box-shadow:inset 0 3px var(--amber)}
.recommendation-row-quality{background:rgba(167,243,208,.055)}.recommendation-row-speed{background:rgba(125,211,252,.055)}.recommendation-row-value{outline:1px solid rgba(251,191,36,.20);outline-offset:-1px}
.pricing-ledger{display:grid;gap:8px}.pricing-entry{display:grid;grid-template-columns:minmax(180px,1fr) auto;gap:12px;padding:10px 0;border-bottom:1px solid rgba(125,211,252,.10)}
.tradeoff-details>summary{cursor:pointer;color:var(--accent);font-weight:900}.tradeoff-details[open]>summary{margin-bottom:16px}
.evidence-dialog{width:min(980px,94vw);max-height:88vh;overflow:auto;border:1px solid rgba(125,211,252,.22);border-radius:18px;background:#07111f;color:var(--text)}
.evidence-dialog-rows{display:grid;gap:10px}.evidence-dialog-row{padding:12px;border:1px solid rgba(125,211,252,.12);border-radius:14px;background:rgba(3,11,20,.34)}
@media(max-width:760px){.result-source-bar,.recommendation-strip{grid-template-columns:1fr}.recommendation-item{border-right:0;border-bottom:1px solid rgba(125,211,252,.12)}.recommendation-item:last-child{border-bottom:0}.pricing-entry{grid-template-columns:1fr}}
```

Do not add decorative gradients or colored side stripes.

- [ ] **Step 7: Run frontend tests and syntax checks**

```bash
python3 -m pytest -q tests/test_pipeline_recommendations.py tests/test_dashboard_metrics.py
node --check web/recommendations.js
node --check web/app.js
git diff --check
```

Expected: all tests pass and both JavaScript files parse.

- [ ] **Step 8: Commit the source-aware UI**

```bash
git add -- web/index.html web/app.js web/styles.css tests/test_dashboard_metrics.py
git diff --cached --check
git commit -m "feat: select uploaded runs in recommendations"
```

---

### Task 7: Run regression, security review, and real browser QA

**Files:**
- Modify only if failures expose a scoped defect in files already listed above
- Do not commit: `artifacts/visual_qa/**`

- [ ] **Step 1: Run complete focused regression**

```bash
python3 -m pytest -q \
  tests/test_project_run_artifacts.py \
  tests/test_project_run_metrics.py \
  tests/test_project_relevance.py \
  tests/test_project_matrix_runner.py \
  tests/test_project_run_results.py \
  tests/test_project_matrix_contract.py \
  tests/test_project_isolation.py \
  tests/test_project_vector_namespaces.py \
  tests/test_pipeline_recommendations.py \
  tests/test_dashboard_metrics.py \
  tests/test_official_artifact_protection.py
python3 -m py_compile \
  source/services/project_run_artifacts.py \
  source/services/project_run_metrics.py \
  source/services/project_relevance.py \
  source/services/project_matrix_runner.py \
  source/services/project_run_results.py \
  benchmarking/adapters/remote_embeddings.py \
  scripts/serve_benchmark_dashboard.py
node --check web/recommendations.js
node --check web/app.js
git diff --check
```

Expected: all tests and syntax checks pass; only known non-blocking deprecation warnings are acceptable.

- [ ] **Step 2: Re-run official artifact protection immediately before live QA**

```bash
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py \
  --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 \
  --roots configs/protected_runtime_artifact_roots.txt
```

Expected: both checks pass.

- [ ] **Step 3: Start the dashboard privately and verify readiness**

Run in a tracked background process:

```bash
python3 scripts/serve_benchmark_dashboard.py 5011 127.0.0.1
```

Verify from another shell:

```bash
curl -fsS http://127.0.0.1:5011/api/result-sources | python3 -m json.tool
curl -fsS 'http://127.0.0.1:5011/api/results?retrieval_limit=0&reranker_limit=0&detail_evidence_limit=0' >/tmp/wns_results_smoke.json
```

Expected: result sources return JSON; official source reports configured 180; dashboard result smoke succeeds.

- [ ] **Step 4: Create disposable labeled and evidence-only QA fixtures through test helpers**

Use a temporary `DASHBOARD_USER_PROJECTS_DIR` or a test-only copied workspace, never official data roots. Produce:

- one schema-v2 labeled run with at least three completed combinations and one failure;
- one evidence-only run with at least two completed combinations;
- evidence rows owned by distinct combo IDs.

Do not commit these fixtures.

- [ ] **Step 5: Run Playwright Chromium QA**

At `http://127.0.0.1:5011/#compare`, verify:

1. Official source loads first and shows 180 evaluated rows.
2. Uploaded mode reveals project/run selectors.
3. Labeled run shows Quality, Speed, and Value/no-API-fee roles with actual `@K` labels.
4. Evidence-only run shows Fastest, Run health, Evidence coverage and no quality score.
5. Partial run displays failed rows after completed rows; failed rows never win.
6. Evidence dialog returns only the selected combination and paginates.
7. Rapid A→B source switching cannot render stale A data.
8. Missing metrics show Not recorded/Not applicable.
9. No page errors, console errors, or failed API calls occur.
10. Desktop 1440×1000 and mobile 390×844 have no clipped controls; table scrolls horizontally.

Save screenshots, traces, console logs, and network failures under `artifacts/visual_qa/pipeline-recommendations-source-aware/`. Keep them untracked.

- [ ] **Step 6: Re-run artifact protection immediately after live QA**

```bash
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py \
  --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 \
  --roots configs/protected_runtime_artifact_roots.txt
git status --short
```

Expected: protection passes; only intended source changes and untracked `artifacts/` appear.

- [ ] **Step 7: Request independent reviews**

Dispatch two fresh reviewers:

- **Specification/security review:** source isolation, no-follow reads, artifact bounds, identity checks, evidence pagination, stale response handling, metric honesty, usage attribution.
- **Code-quality/UI review:** module boundaries, duplication, readability, accessibility, responsive hierarchy, test quality.

Fix every Critical/Important finding, rerun its focused test, then repeat Steps 1, 2, and 6.

- [ ] **Step 8: Commit review fixes narrowly**

Stage only reviewed source/test paths. Use a scoped message such as:

```bash
git commit -m "fix: harden source-aware recommendations"
```

Do not stage `artifacts/`.

---

### Task 8: Build the handoff, push from the work laptop, and verify on the VM

**Files included in handoff:**

```text
benchmarking/adapters/remote_embeddings.py
source/services/project_run_artifacts.py
source/services/project_run_metrics.py
source/services/project_relevance.py
source/services/project_matrix_runner.py
source/services/project_run_results.py
scripts/serve_benchmark_dashboard.py
web/recommendations.js
web/index.html
web/app.js
web/styles.css
tests/test_project_run_artifacts.py
tests/test_project_run_metrics.py
tests/test_project_relevance.py
tests/test_project_matrix_runner.py
tests/test_project_run_results.py
tests/test_pipeline_recommendations.py
tests/test_dashboard_metrics.py
docs/superpowers/specs/2026-07-13-pipeline-recommendations-page-design.md
docs/superpowers/plans/2026-07-13-pipeline-recommendations-page.md
```

- [ ] **Step 1: Confirm a clean source diff and final commit**

```bash
git status --short
git diff --check
git log --oneline ab70906..HEAD
```

Expected: no tracked unstaged changes; only untracked `artifacts/`; all implementation commits are visible.

- [ ] **Step 2: Build a narrow timestamped overlay with real metadata**

Run this Python from the repository root through `execute_code` or a temporary file:

```python
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

repo = Path.cwd().resolve()
base_commit = "ab70906"
source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
package_root = Path("/tmp") / f"wns_source_aware_recommendations_{stamp}"
zip_path = Path.home() / f"wns_source_aware_recommendations_handoff_{stamp}.zip"
files = [line for line in """benchmarking/adapters/remote_embeddings.py
source/services/project_run_artifacts.py
source/services/project_run_metrics.py
source/services/project_relevance.py
source/services/project_matrix_runner.py
source/services/project_run_results.py
scripts/serve_benchmark_dashboard.py
web/recommendations.js
web/index.html
web/app.js
web/styles.css
tests/test_project_run_artifacts.py
tests/test_project_run_metrics.py
tests/test_project_relevance.py
tests/test_project_matrix_runner.py
tests/test_project_run_results.py
tests/test_pipeline_recommendations.py
tests/test_dashboard_metrics.py
docs/superpowers/specs/2026-07-13-pipeline-recommendations-page-design.md
docs/superpowers/plans/2026-07-13-pipeline-recommendations-page.md""".splitlines() if line]
if package_root.exists():
    shutil.rmtree(package_root)
for relative in files:
    source = repo / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = package_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
(package_root / "CHANGED_FILES.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
(package_root / "COMMITS.txt").write_text(
    subprocess.check_output(["git", "log", "--oneline", f"{base_commit}..{source_commit}"], cwd=repo, text=True),
    encoding="utf-8",
)
manifest = {
    "feature": "source-aware-pipeline-recommendations",
    "base_commit": base_commit,
    "source_commit": source_commit,
    "official_matrix_count": 180,
    "pricing_checked_at": "2026-07-13",
    "files": files,
}
(package_root / "HANDOFF_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(package_root / "APPLY_ON_WORK_LAPTOP.md").write_text(
    "# Apply on work laptop\n\nBack up and copy only paths in CHANGED_FILES.txt. Run focused tests and artifact protection. Stage only listed paths, commit, and push. Pull on the VM only after the laptop push succeeds.\n",
    encoding="utf-8",
)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(package_root))
print(zip_path)
print(hashlib.sha256(zip_path.read_bytes()).hexdigest())
```

- [ ] **Step 3: Verify the overlay against a clean checkout**

```bash
ZIP="$(python3 -c "from pathlib import Path; print(max(Path.home().glob('wns_source_aware_recommendations_handoff_*.zip'), key=lambda p: p.stat().st_mtime))")"
unzip -t "$ZIP"
sha256sum "$ZIP"
TMP="$(mktemp -d)"
git worktree add --detach "$TMP/clean" ab70906
mkdir -p "$TMP/overlay"
unzip -q "$ZIP" -d "$TMP/overlay"
while IFS= read -r file; do install -D "$TMP/overlay/$file" "$TMP/clean/$file"; done < "$TMP/overlay/CHANGED_FILES.txt"
cd "$TMP/clean"
python3 -m pytest -q \
  tests/test_project_run_artifacts.py \
  tests/test_project_run_metrics.py \
  tests/test_project_relevance.py \
  tests/test_project_matrix_runner.py \
  tests/test_project_run_results.py \
  tests/test_pipeline_recommendations.py \
  tests/test_dashboard_metrics.py \
  tests/test_official_artifact_protection.py
node --check web/recommendations.js
node --check web/app.js
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
git diff --check
cd -
git worktree remove "$TMP/clean"
```

Expected: ZIP integrity, SHA-256, tests, syntax, artifact protection, and diff checks all pass.

- [ ] **Step 4: Apply, test, commit, and push from the work laptop**

On the work laptop:

1. Back up each destination path listed in `CHANGED_FILES.txt`.
2. Copy only the allowlisted overlay files.
3. Run the exact clean-checkout test commands from Step 3.
4. Inspect `git status --short`.
5. Stage only the allowlisted source/test/spec/plan files—never `git add .`.
6. Commit with `feat: add source-aware pipeline recommendations`.
7. Push the work-laptop branch.

Expected: the remote contains the laptop commit before any VM pull.

- [ ] **Step 5: Pull and restart on the VM**

On the VM:

```bash
git status --short
git pull --ff-only
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py \
  --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 \
  --roots configs/protected_runtime_artifact_roots.txt
```

If tracked VM changes block the pull, stop and use a selective stash for only those tracked paths. Do not reset, clean, or delete artifacts.

Restart only the dashboard as a private user-managed transient service. The script loads repository env files itself:

```bash
UNIT=wns-benchmark-dashboard
systemctl --user stop "$UNIT.service" 2>/dev/null || true
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
mapfile -t OLD_DASHBOARD_PIDS < <(pgrep -f '[p]ython.*scripts/serve_benchmark_dashboard.py 5011' || true)
if ((${#OLD_DASHBOARD_PIDS[@]})); then
  kill -TERM "${OLD_DASHBOARD_PIDS[@]}"
  for _ in {1..30}; do
    pgrep -f '[p]ython.*scripts/serve_benchmark_dashboard.py 5011' >/dev/null || break
    sleep 1
  done
fi
if pgrep -f '[p]ython.*scripts/serve_benchmark_dashboard.py 5011' >/dev/null; then
  echo 'dashboard process did not stop cleanly' >&2
  exit 1
fi
PYTHON_BIN="$PWD/.venv-vm/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3)"
systemd-run --user \
  --unit="$UNIT" \
  --collect \
  --property=Restart=on-failure \
  --property=RestartSec=3 \
  --working-directory="$PWD" \
  "$PYTHON_BIN" scripts/serve_benchmark_dashboard.py 5011 127.0.0.1
systemctl --user is-active --quiet "$UNIT.service"
for _ in {1..30}; do
  curl -fsS http://127.0.0.1:5011/api/result-sources >/dev/null && break
  sleep 1
done
curl -fsS http://127.0.0.1:5011/api/result-sources >/dev/null
```

If `systemd-run --user` is unavailable, stop and report the VM supervision blocker rather than falling back to `nohup`. Do not restart provider/vector services unless their independent health checks fail.

- [ ] **Step 6: Run VM API and browser smoke**

```bash
SOURCE_JSON="$(mktemp)"
RUN_JSON="$(mktemp)"
trap 'rm -f "$SOURCE_JSON" "$RUN_JSON"' EXIT
curl -fsS http://127.0.0.1:5011/api/result-sources | tee "$SOURCE_JSON" | python3 -m json.tool
PROJECT_ID="$(python3 - "$SOURCE_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
projects = payload.get('projects') or []
print(projects[0].get('project_id', '') if projects else '')
PY
)"
if [ -z "$PROJECT_ID" ]; then
  echo 'No valid uploaded project is available for VM project-run smoke' >&2
  exit 2
fi
curl -fsS -G --data-urlencode "project_id=$PROJECT_ID" \
  http://127.0.0.1:5011/api/project-runs | tee "$RUN_JSON" | python3 -m json.tool
RUN_ID="$(python3 - "$RUN_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
runs = payload.get('runs') or []
print(runs[0].get('run_id', '') if runs else '')
PY
)"
if [ -z "$RUN_ID" ]; then
  echo 'No completed/partial matrix run is available for VM project-run smoke' >&2
  exit 3
fi
curl -fsS -G \
  --data-urlencode "project_id=$PROJECT_ID" \
  --data-urlencode "run_id=$RUN_ID" \
  http://127.0.0.1:5011/api/project-run-results | python3 -m json.tool
rm -f "$SOURCE_JSON" "$RUN_JSON"
```

The script derives real IDs from the VM APIs. If it exits 2 or 3, record that exact data-availability blocker; do not substitute official rows or fabricate an uploaded run.

Browser-check the official source plus one real VM uploaded run. Confirm no cross-project rows, no stale source state, correct evidence ownership, and no console/network errors.

- [ ] **Step 7: Re-run post-deploy protection and record delivery evidence**

```bash
python3 scripts/verify_official_artifacts_unchanged.py \
  --check configs/protected_official_artifacts.sha256
python3 scripts/verify_official_artifacts_unchanged.py \
  --check-runtime-baseline artifacts/predeploy/runtime-artifacts.sha256 \
  --roots configs/protected_runtime_artifact_roots.txt
git rev-parse HEAD
git status --short
```

Record in the final delivery:

- work-laptop commit and VM HEAD
- focused test counts
- source API responses/counts
- browser QA result
- ZIP absolute path and SHA-256
- explicit statement that official 180 rows and protected artifacts remained unchanged
- any real blocker if no valid uploaded matrix run exists on the VM

Do not call the feature deployed until all seven steps pass.
