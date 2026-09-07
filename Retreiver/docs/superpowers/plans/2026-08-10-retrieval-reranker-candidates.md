# Retrieval and Reranker Candidate Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a separately namespaced 50-combination candidate lane for GTE dense cosine vs BM25+GTE+RRF and five reranker options, visible in the dashboard and deployable to the VM without altering accepted official-180 data or configuration.

**Architecture:** Keep `configs/benchmark.local.json` and `configs/project_matrix_catalog.json` as read-only official contracts. Add a candidate modular config for the controlled benchmark matrix, real hybrid retrieval support in the modular runner, and project-request schema v2 so uploaded project runs can select Dense, BM25, or Hybrid. The VM adapter exposes dedicated route/model identities for Nemotron and GTE ModernBERT, while a common remote HTTP response contract returns index-aligned scores.

**Tech Stack:** Python 3, pytest, FastAPI, SentenceTransformers, Transformers, PyTorch, rank-bm25, FAISS, static JavaScript dashboard, Playwright/Chromium, Git ZIP overlay.

---

## File structure

| Path | Responsibility |
|---|---|
| `configs/benchmark.retrieval-reranker-candidates.json` | Fixed 50-combination candidate modular matrix and immutable retrieval/model settings. |
| `benchmarking/retrieval.py` | BM25, dense cosine, and deterministic RRF retrieval primitives with provenance. |
| `benchmarking/core/runner.py` | Execute the modular candidate config with actual selected retrieval behavior and persist provenance. |
| `benchmarking/core/registry.py` | Register retrieval adapters/primitives. |
| `requirements.txt` | Declare `rank-bm25`. |
| `scripts/wns_vm_adapter_service.py` | Model-aware Nemotron/GTE ModernBERT loaders, routes, health data, batching and truncation. |
| `.env.example`, `scripts/check_env.py`, `scripts/check_services.py`, `scripts/start_vm_stack.sh` | New endpoint variables, smoke checks, and VM env export. |
| `configs/project_matrix_candidate_catalog.json` | Candidate-only project matrix catalog; never extend the official catalog. |
| `source/services/project_matrix_contract.py` | Schema-v2 retrieval selection/fingerprint and lexical-aware combination policy. |
| `source/services/project_matrix_runner.py` | Execute BM25/dense/hybrid project runs and persist retrieval method/provenance. |
| `source/services/project_run_results.py` | Admit and normalize schema-v3 candidate summaries/evidence while preserving legacy dense runs. |
| `scripts/serve_benchmark_dashboard.py` | Candidate options, selected-reranker endpoint preflight, project result routing, cache token. |
| `web/index.html`, `web/app.js` | Candidate controls, schema-v2 request payload, method-aware counts and Metrics/Evidence views. |
| `tests/test_hybrid_retriever.py` | Pure deterministic BM25/dense/RRF behavior and provenance. |
| `tests/test_modular_candidate_matrix.py` | Candidate config cardinality, official-180 immutability, modular runner behavior. |
| `tests/test_vm_adapter_candidates.py` | Nemotron exact prompt, model loaders/routes/health and GTE ModernBERT pair contract. |
| `tests/test_project_matrix_contract.py`, `tests/test_project_matrix_runner.py`, `tests/test_project_run_results.py` | Schema-v2/candidate policy, lexical execution, artifact/reader compatibility. |
| `tests/test_project_matrix_frontend.py`, `tests/test_dashboard_metrics.py`, `tests/test_start_vm_stack_script.py` | Frontend payload/copy and deployment environment regressions. |
| `scripts/package_retrieval_reranker_candidate_overlay.py` | Build a narrow source-only deployment ZIP and produce an integrity receipt. |
| `docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md` | Work-laptop-only VM smoke and run procedure. |

### Task 1: Establish candidate configuration and protect official invariants

**Files:**
- Create: `configs/benchmark.retrieval-reranker-candidates.json`
- Create: `configs/project_matrix_candidate_catalog.json`
- Create: `tests/test_modular_candidate_matrix.py`
- Modify: `requirements.txt`
- Test: `tests/test_official_artifact_protection.py`

- [ ] **Step 1: Write a failing cardinality and isolation test**

```python
from benchmarking.core.config import generate_matrix, load_benchmark_config


def test_candidate_matrix_is_exactly_fifty_and_official_matrix_stays_180() -> None:
    official = load_benchmark_config(REPO_ROOT / "configs/benchmark.local.json")
    candidate = load_benchmark_config(
        REPO_ROOT / "configs/benchmark.retrieval-reranker-candidates.json"
    )
    assert len(generate_matrix(official)) == 180
    rows = generate_matrix(candidate)
    assert len(rows) == 50
    assert {row["retrieval_method"] for row in rows} == {
        "GTE Dense Cosine", "BM25 + GTE Dense + RRF"
    }
    assert {row["reranker"] for row in rows} == {
        "none", "bge-reranker-base", "Qwen3:4B Rerank",
        "nemotron_rerank_1b", "gte_modernbert_base",
    }


def test_candidate_catalog_is_a_distinct_file_and_does_not_reclassify_official_models() -> None:
    official = json.loads((REPO_ROOT / "configs/project_matrix_catalog.json").read_text())
    candidate = json.loads((REPO_ROOT / "configs/project_matrix_candidate_catalog.json").read_text())
    assert "nemotron_rerank_1b" not in official["matrix"]["rerankers"]
    assert "gte_modernbert_base" not in official["matrix"]["rerankers"]
    assert candidate["candidate_lane"] == "retrieval_reranker_candidates"
```

- [ ] **Step 2: Run the test to verify it fails because candidate files do not exist**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_modular_candidate_matrix.py
```

Expected: FAIL with missing candidate configuration files.

- [ ] **Step 3: Add minimal candidate config/catalog files**

Use the existing technique shape, with only GTE/FAISS in the candidate config:

```json
{
  "experiment": {
    "name": "wns-retrieval-reranker-candidates",
    "mode": "vm_remote_required",
    "top_k": 10,
    "candidate_depth": 50,
    "fusion_depth": 20,
    "rrf_k": 60,
    "output_lane": "retrieval-reranker-candidates"
  },
  "matrix": {
    "chunkers": ["entity_heuristic_w6", "entity_heuristic_w5", "entity_heuristic_w4", "Heading_sections_l2", "fixed_tok1200_ov150"],
    "embeddings": ["gte_multilingual_base"],
    "vector_stores": ["FAISS"],
    "index_types": ["HNSW"],
    "retrieval_methods": ["GTE Dense Cosine", "BM25 + GTE Dense + RRF"],
    "rerankers": ["none", "bge-reranker-base", "Qwen3:4B Rerank", "nemotron_rerank_1b", "gte_modernbert_base"],
    "evaluators": ["overlap_relevance"]
  }
}
```

Add `rank-bm25>=0.2.2` to `requirements.txt`. The candidate project catalog must carry `candidate_lane: "retrieval_reranker_candidates"`, `schema_version: 2`, a `retrieval_methods` matrix dimension, and dedicated remote HTTP reranker definitions with immutable `model_id` values and endpoint environment variables.

- [ ] **Step 4: Run test and official protection regression**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_modular_candidate_matrix.py tests/test_official_artifact_protection.py
```

Expected: PASS. The official config still generates exactly 180 rows.

- [ ] **Step 5: Commit only config, requirement, and test paths**

```bash
git add Retreiver/configs/benchmark.retrieval-reranker-candidates.json \
  Retreiver/configs/project_matrix_candidate_catalog.json Retreiver/requirements.txt \
  Retreiver/tests/test_modular_candidate_matrix.py
git -c gc.auto=0 commit -m "feat: define isolated retrieval reranker candidate lane"
```

### Task 2: Build deterministic retrieval primitives with evidence provenance

**Files:**
- Create: `benchmarking/retrieval.py`
- Modify: `benchmarking/core/registry.py`
- Create: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write failing BM25/RRF tests**

```python
from benchmarking.retrieval import BM25Retriever, HybridRRFRetriever, rrf_score


def test_bm25_returns_lexical_hit_without_embedding_or_vector_store() -> None:
    retriever = BM25Retriever(chunks(CHUNKS))
    hit = retriever.search("PAX-REF-992 refund", top_k=1)[0]
    assert hit.chunk.id == "policy-1"
    assert hit.provenance["bm25_rank"] == 1
    assert hit.provenance["bm25_score"] > 0


def test_hybrid_rrf_fuses_ranks_and_preserves_all_branch_fields() -> None:
    result = HybridRRFRetriever.rrf_fuse(
        dense=[("a", 0.9), ("b", 0.8)],
        bm25=[("b", 5.0), ("c", 4.0)],
        rrf_k=60,
    )
    assert [item.chunk_id for item in result] == ["b", "a", "c"]
    b = result[0].provenance
    assert b == {
        "dense_rank": 2, "dense_score": 0.8,
        "bm25_rank": 1, "bm25_score": 5.0,
        "rrf_score": rrf_score(2, 60) + rrf_score(1, 60),
    }
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_hybrid_retriever.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.retrieval'`.

- [ ] **Step 3: Implement the smallest reusable retrieval API**

Create `benchmarking/retrieval.py` with:

```python
def rrf_score(rank: int, rrf_k: int) -> float:
    return 1.0 / (rrf_k + rank)

class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None: ...
    def search(self, query: str, top_k: int) -> list[RetrievedHit]: ...

class DenseCosineRetriever:
    def __init__(self, store: VectorStoreAdapter) -> None: ...
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedHit]: ...

class HybridRRFRetriever:
    def search(self, query: str, query_vector: list[float], top_k: int) -> list[RetrievedHit]: ...
```

Tokenize BM25 deterministically using `text.casefold().split()`. Preserve zero/missing branch fields as absent rather than fabricated numeric values. Break RRF ties by best branch rank, then stable chunk ID so artifact ordering is repeatable.

- [ ] **Step 4: Run the primitive tests and compilation**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_hybrid_retriever.py && python -m py_compile benchmarking/retrieval.py
```

Expected: PASS.

- [ ] **Step 5: Commit retrieval primitives and tests**

```bash
git add Retreiver/benchmarking/retrieval.py Retreiver/benchmarking/core/registry.py Retreiver/tests/test_hybrid_retriever.py
git -c gc.auto=0 commit -m "feat: add BM25 dense RRF retrieval primitives"
```

### Task 3: Execute actual candidate retrieval behavior in the modular benchmark

**Files:**
- Modify: `benchmarking/core/runner.py`
- Modify: `benchmarking/adapters/remote_rerankers.py`
- Modify: `tests/test_modular_candidate_matrix.py`
- Modify: `tests/test_modular_benchmark.py`

- [ ] **Step 1: Add failing behavior tests for method dispatch and provenance**

```python
def test_candidate_runner_uses_hybrid_rrf_not_a_label_only_row(tmp_path, monkeypatch) -> None:
    output = run_experiment(CANDIDATE_CONFIG, ROOT, tmp_path / "candidate", limit_queries=2)
    details = read_csv(tmp_path / "candidate" / "modular_details.csv")
    hybrid = [row for row in details if row["retrieval_method"] == "BM25 + GTE Dense + RRF"]
    assert hybrid
    assert all(row["bm25_rank"] for row in hybrid)
    assert all(row["dense_rank"] for row in hybrid)
    assert all(row["rrf_score"] for row in hybrid)


def test_identity_reranker_preserves_retrieval_order(tmp_path, monkeypatch) -> None:
    ...
    assert detail_rows_for("none")[0]["top_ids"] == retrieval_rows[0]["top_ids"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_modular_candidate_matrix.py -k 'hybrid or identity'
```

Expected: FAIL because runner still calls `store.search` directly and does not produce provenance fields.

- [ ] **Step 3: Replace runner direct search with a retrieval-method dispatch helper**

Add a helper that takes `row["retrieval_method"]`, chunk list, vector store, GTE query vector, query text, and the fixed experiment depth settings. It must:

```python
if method == "GTE Dense Cosine":
    return DenseCosineRetriever(store).search(query_vector, candidate_depth)
if method == "BM25 + GTE Dense + RRF":
    return HybridRRFRetriever(
        dense=DenseCosineRetriever(store),
        bm25=BM25Retriever(chunks),
        rrf_k=rrf_k,
    ).search(query, query_vector, fusion_depth)
raise ValueError(f"unsupported retrieval method: {method}")
```

Treat `reranker == "none"` as identity, never instantiate an unconfigured remote reranker. Add method settings/model config hashes to the manifest. Put the hit provenance in modular detail rows as JSON plus normalized fields `dense_rank`, `dense_score`, `bm25_rank`, `bm25_score`, `rrf_score`.

- [ ] **Step 4: Run focused runner checks**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_hybrid_retriever.py tests/test_modular_candidate_matrix.py tests/test_modular_benchmark.py
```

Expected: Feature tests PASS. If the known corpus workbook is absent, report that pre-existing environment failure distinctly and keep fixture-backed candidate tests green.

- [ ] **Step 5: Commit modular execution behavior**

```bash
git add Retreiver/benchmarking/core/runner.py Retreiver/benchmarking/adapters/remote_rerankers.py \
  Retreiver/tests/test_modular_candidate_matrix.py Retreiver/tests/test_modular_benchmark.py
git -c gc.auto=0 commit -m "feat: run candidate dense and hybrid retrieval methods"
```

### Task 4: Add dedicated VM reranker endpoints and environment checks

**Files:**
- Modify: `scripts/wns_vm_adapter_service.py`
- Modify: `.env.example`
- Modify: `scripts/check_env.py`
- Modify: `scripts/check_services.py`
- Modify: `scripts/start_vm_stack.sh`
- Modify: `tests/test_start_vm_stack_script.py`
- Create: `tests/test_vm_adapter_candidates.py`

- [ ] **Step 1: Write failing endpoint-contract tests**

```python
from scripts.wns_vm_adapter_service import format_nemotron_pair, health, rerank


def test_nemotron_uses_documented_question_passage_formatter() -> None:
    assert format_nemotron_pair("Where is refund policy?", "Refunds are allowed.") == (
        "question:Where is refund policy? \n \n passage:Refunds are allowed."
    )


def test_health_advertises_candidate_model_ids_and_routes() -> None:
    payload = health()
    assert payload["models"]["nemotron_reranker"] == "nvidia/llama-nemotron-rerank-1b-v2"
    assert payload["models"]["gte_modernbert_reranker"] == "Alibaba-NLP/gte-reranker-modernbert-base"
    assert "/rerank/nemotron" in payload["endpoints"]
    assert "/rerank/gte-modernbert" in payload["endpoints"]


def test_nemotron_response_preserves_input_score_order(monkeypatch) -> None:
    monkeypatch.setattr(service, "nemotron_model", lambda: FakeModel([0.1, 0.9]))
    payload = rerank("nemotron", RerankRequest(query="q", documents=["a", "b"]))
    assert payload["scores"] == [0.1, 0.9]
    assert [item["index"] for item in payload["results"]] == [1, 0]
```

- [ ] **Step 2: Run RED test**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_vm_adapter_candidates.py tests/test_start_vm_stack_script.py
```

Expected: FAIL because candidate routes, model IDs, formatter, and variables do not exist.

- [ ] **Step 3: Implement dedicated model loaders and routes**

Add constants:

```python
NEMOTRON_RERANK_MODEL = os.getenv("NEMOTRON_RERANKER_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
GTE_MODERNBERT_RERANK_MODEL = os.getenv("GTE_MODERNBERT_RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
RERANK_MAX_LENGTH = int(os.getenv("WNS_RERANK_MAX_LENGTH", "8192"))
RERANK_BATCH_SIZE = int(os.getenv("WNS_RERANK_BATCH_SIZE", "8"))
```

Implement `format_nemotron_pair(query, document)` exactly as model-card documented. Implement `nemotron_model()` using `AutoTokenizer` and `AutoModelForSequenceClassification`, explicit padding/truncation/max length, CUDA BF16 only when available, and raw aligned logits. Implement `gte_modernbert_model()` with `CrossEncoder(..., automodel_args={"torch_dtype": "auto"})` and its ordinary `(query, document)` pairs. Register `POST /rerank/nemotron` and `POST /rerank/gte-modernbert`; include selector/upstream model/device/dtype/maximum/batch/load state in `/health`.

Add `NEMOTRON_RERANK_URL=http://VM_HOST:5000/rerank/nemotron` and `GTE_MODERNBERT_RERANK_URL=http://VM_HOST:5000/rerank/gte-modernbert` to the example env and stack exporter. Extend `check_services.py` to issue a two-document request to each endpoint.

- [ ] **Step 4: Run adapter and shell-contract tests**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_vm_adapter_candidates.py tests/test_start_vm_stack_script.py && python -m py_compile scripts/wns_vm_adapter_service.py
```

Expected: PASS without downloading model weights.

- [ ] **Step 5: Commit VM adapter changes**

```bash
git add Retreiver/scripts/wns_vm_adapter_service.py Retreiver/.env.example \
  Retreiver/scripts/check_env.py Retreiver/scripts/check_services.py Retreiver/scripts/start_vm_stack.sh \
  Retreiver/tests/test_vm_adapter_candidates.py Retreiver/tests/test_start_vm_stack_script.py
git -c gc.auto=0 commit -m "feat: serve Nemotron and GTE ModernBERT rerankers"
```

### Task 5: Version project runs for method-aware candidate execution

**Files:**
- Modify: `source/services/project_matrix_contract.py`
- Modify: `source/services/project_matrix_runner.py`
- Modify: `source/services/project_run_results.py`
- Modify: `tests/test_project_matrix_contract.py`
- Modify: `tests/test_project_matrix_runner.py`
- Modify: `tests/test_project_run_results.py`

- [ ] **Step 1: Write failing request contract tests**

```python
def test_schema_v2_candidate_request_has_retrieval_methods_in_fingerprint() -> None:
    dense = validate_project_matrix_request(payload(retrieval_methods=["GTE Dense Cosine"]), catalog_path=CANDIDATE_CATALOG)
    hybrid = validate_project_matrix_request(payload(retrieval_methods=["BM25 + GTE Dense + RRF"]), catalog_path=CANDIDATE_CATALOG)
    assert dense.request_fingerprint != hybrid.request_fingerprint


def test_bm25_count_is_lexical_aware() -> None:
    matrix = validate_project_matrix_request(
        payload(chunkers=["entity_heuristic_w6"], embeddings=["gte_multilingual_base"], vector_stores=["FAISS"], rerankers=[], retrieval_methods=["BM25"]),
        catalog_path=CANDIDATE_CATALOG,
    )
    assert matrix.combination_count == 1


def test_legacy_schema_v1_stays_implicit_dense_cosine() -> None:
    matrix = validate_project_matrix_request(v1_payload(), catalog_path=OFFICIAL_CATALOG)
    assert matrix.request.retrieval_methods == ("GTE Dense Cosine",)
```

- [ ] **Step 2: Run RED contract tests**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_project_matrix_contract.py -k 'schema_v2 or lexical or legacy'
```

Expected: FAIL because schema version 1 rejects retrieval methods and no lexical-aware count exists.

- [ ] **Step 3: Implement schema-v2 without invalidating schema-v1**

Accept schema versions 1 and 2. Schema 1 gets exactly `("GTE Dense Cosine",)` without changing its existing fingerprint shape. Schema 2 validates `retrieval_methods` against the selected catalog and includes it in request fingerprints, confirmation tokens, request envelopes and semantic matrix identity. Count BM25 selections as `chunkers × rerankers` and dense/hybrid selections as `chunkers × embeddings × vector_stores × rerankers`; empty rerankers continue to represent one identity/no-reranker lane.

In the project runner, use common retrieval interfaces. `BM25` must not construct/probe embedding/vector dependencies and must record `embedding_id="N/A"`, `vector_store_id="N/A"`. Dense and Hybrid must use the configured GTE+FAISS branch. Hybrid must use the fixed candidate depth, RRF k, and fusion depth from candidate config. Include `retrieval_method_id` in combo IDs, request receipts, summary v3 rows, details, and evidence. Preserve retrieval provenance as a nested evidence object plus explicit fields.

In the result reader, admit v3 rows and evidence fields, normalize legacy absent method to `GTE Dense Cosine`, validate `N/A` axes only for BM25, and include retrieval method in dedupe identity.

- [ ] **Step 4: Run runner and reader tests**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q \
  tests/test_project_matrix_contract.py tests/test_project_matrix_runner.py tests/test_project_run_results.py
```

Expected: PASS, including the test that forces embedding/vector construction to fail while BM25 completes.

- [ ] **Step 5: Commit project execution and reader changes**

```bash
git add Retreiver/source/services/project_matrix_contract.py Retreiver/source/services/project_matrix_runner.py \
  Retreiver/source/services/project_run_results.py Retreiver/tests/test_project_matrix_contract.py \
  Retreiver/tests/test_project_matrix_runner.py Retreiver/tests/test_project_run_results.py
git -c gc.auto=0 commit -m "feat: support method-aware candidate project runs"
```

### Task 6: Connect candidate controls, readiness, metrics, and evidence in the dashboard

**Files:**
- Modify: `scripts/serve_benchmark_dashboard.py`
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `tests/test_project_matrix_frontend.py`
- Modify: `tests/test_project_matrix_server_bridge.py`
- Modify: `tests/test_dashboard_metrics.py`

- [ ] **Step 1: Write failing UI/server regression tests**

```python
def test_project_payload_uses_schema_v2_and_selected_retrieval_methods() -> None:
    assert "schema_version: 2" in APP
    assert "retrieval_methods: projectSelectedRetrievalMethods()" in APP
    assert "runRetrievalMethod" in APP


def test_dashboard_candidate_preflight_checks_selected_reranker_endpoint(monkeypatch) -> None:
    response = dashboard.preflight_project_matrix(candidate_payload("nemotron_rerank_1b"), root=root, workspace=workspace)
    assert any(check["name"] == "nemotron_reranker" for check in response["service_checks"])
    assert response["ok"] is False


def test_metrics_render_method_column_and_candidate_lane_not_official_count() -> None:
    assert "Retrieval method" in APP
    assert "Candidate retrieval and reranking" in INDEX
    assert "Official accepted benchmark" in INDEX
```

- [ ] **Step 2: Run RED frontend/server tests**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_project_matrix_frontend.py tests/test_project_matrix_server_bridge.py tests/test_dashboard_metrics.py
```

Expected: FAIL because existing payload is schema v1 and no retrieval method control/preflight routes exist.

- [ ] **Step 3: Implement the UI and preflight contract**

Add a Candidate retrieval and reranking control area to `web/index.html` containing a method multiselect, fixed settings display, endpoint readiness display, and candidate-lane note. Do not alter the official accepted 180 section.

In `web/app.js`:

```javascript
function projectSelectedRetrievalMethods() {
  return projectSelectedValues('runRetrievalMethod', benchmarkOptions.candidate_retrieval_methods || []);
}

function projectMatrixPayload(confirmationToken = null) {
  return {
    schema_version: 2,
    // existing immutable IDs and selections
    selections: {
      chunkers: projectSelectedValues('runSheet', benchmarkOptions.chunkers || []),
      embeddings: projectSelectedValues('runEmbedding', benchmarkOptions.candidate_embeddings || []),
      vector_stores: projectSelectedValues('runStore', benchmarkOptions.candidate_vector_stores || []),
      retrieval_methods: projectSelectedRetrievalMethods(),
      rerankers: projectSelectedRerankers(),
    },
  };
}
```

Use the exact lexical-aware count rules shared by the server response rather than a blind browser Cartesian product. Render retrieval method in Metrics and include hit provenance in Evidence. Bump the dashboard asset cache token and ensure `index.html` references the changed `app.js?v=<release-token>`.

In the server, expose candidate option metadata separately from official options; use the candidate catalog only when a schema-v2 candidate request identifies the lane. Preflight must issue a POST two-document request to selected reranker endpoint, fail closed if unavailable, and report exact endpoint/model identity. Add model endpoint suffix normalization for both new routes.

- [ ] **Step 4: Run frontend/server tests and JavaScript syntax check**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q \
  tests/test_project_matrix_frontend.py tests/test_project_matrix_server_bridge.py tests/test_dashboard_metrics.py \
  && node --check web/app.js
```

Expected: PASS.

- [ ] **Step 5: Commit dashboard integration**

```bash
git add Retreiver/scripts/serve_benchmark_dashboard.py Retreiver/web/index.html Retreiver/web/app.js \
  Retreiver/tests/test_project_matrix_frontend.py Retreiver/tests/test_project_matrix_server_bridge.py Retreiver/tests/test_dashboard_metrics.py
git -c gc.auto=0 commit -m "feat: expose candidate retrieval and reranker controls"
```

### Task 7: Package a source-only VM overlay and prove release integrity

**Files:**
- Create: `scripts/package_retrieval_reranker_candidate_overlay.py`
- Create: `docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md`
- Create: `tests/test_retrieval_reranker_candidate_overlay.py`

- [ ] **Step 1: Write a failing overlay containment test**

```python
def test_candidate_overlay_contains_required_files_and_no_data_or_secrets(tmp_path) -> None:
    archive, receipt = build_overlay(REPO_ROOT, tmp_path)
    names = zipfile.ZipFile(archive).namelist()
    assert "Retreiver/configs/benchmark.retrieval-reranker-candidates.json" in names
    assert "Retreiver/scripts/wns_vm_adapter_service.py" in names
    assert not any(name.startswith("Retreiver/data/") for name in names)
    assert not any(Path(name).name in {".env", ".env.vm.generated"} for name in names)
    assert receipt["zip_crc_ok"] is True
```

- [ ] **Step 2: Run RED test**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_retrieval_reranker_candidate_overlay.py
```

Expected: FAIL because the packager does not exist.

- [ ] **Step 3: Implement deterministic package builder and work-laptop VM runbook**

Use `zipfile.ZipFile` with a fixed explicit allowlist. Write a receipt with archive SHA-256, CRC result, source commit, and included paths. Never add data, user project files, indexes, logs, caches, `.env`, or model weights.

The runbook uses one-command-at-a-time work-laptop VM steps: verify commit/status, extract overlay, run `curl /health`, run direct two-document endpoint smokes, run candidate config with `--limit-queries 3`, inspect artifact/manifest/evidence provenance, then run full 50 only after all gates pass. It must state `transformers==4.57.6` is retained and never recommend Transformers 5+ for GTE remote-code compatibility.

- [ ] **Step 4: Build and verify the overlay**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q tests/test_retrieval_reranker_candidate_overlay.py \
  && python scripts/package_retrieval_reranker_candidate_overlay.py --output-dir ../wns-deliverables \
  && python -m zipfile -t ../wns-deliverables/Retreiver-retrieval-reranker-candidates-overlay.zip
```

Expected: PASS; receipt SHA-256 is emitted; no `Retreiver/data/` archive members.

- [ ] **Step 5: Commit package tooling and runbook**

```bash
git add Retreiver/scripts/package_retrieval_reranker_candidate_overlay.py \
  Retreiver/docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md \
  Retreiver/tests/test_retrieval_reranker_candidate_overlay.py
git -c gc.auto=0 commit -m "docs: package candidate reranker VM overlay"
```

### Task 8: Execute verification and readiness gates before VM handoff

**Files:**
- Verify only; do not modify generated data paths.

- [ ] **Step 1: Run syntax and focused test suite**

Run:

```bash
cd Retreiver && PYTHONPATH=. python -m pytest -q \
  tests/test_hybrid_retriever.py tests/test_modular_candidate_matrix.py tests/test_vm_adapter_candidates.py \
  tests/test_project_matrix_contract.py tests/test_project_matrix_runner.py tests/test_project_run_results.py \
  tests/test_project_matrix_frontend.py tests/test_project_matrix_server_bridge.py tests/test_dashboard_metrics.py \
  tests/test_start_vm_stack_script.py tests/test_retrieval_reranker_candidate_overlay.py \
  && python -m py_compile benchmarking/retrieval.py benchmarking/core/runner.py \
  source/services/project_matrix_contract.py source/services/project_matrix_runner.py \
  source/services/project_run_results.py scripts/wns_vm_adapter_service.py scripts/serve_benchmark_dashboard.py \
  && node --check web/app.js
```

Expected: zero feature regressions. If tests require unavailable protected corpus workbook, isolate that as a known environment failure and retain independent fixture test evidence.

- [ ] **Step 2: Run browser QA against a disposable local fixture**

Run the existing dashboard test server against controlled local stub endpoints. Use Playwright Chromium to select Dense and Hybrid plus both new rerankers; assert exact run selection in Metrics/Recommendations/Evidence; inspect desktop and 390px screenshots; assert browser console has no errors.

Expected: screenshots show the candidate lane separately and exact method labels/provenance remain visible.

- [ ] **Step 3: Verify a committed-source archive independently of dirty generated files**

Run:

```bash
cd .. && git archive --format=tar HEAD Retreiver | tar -x -C /tmp/retriever-candidate-archive && \
cd /tmp/retriever-candidate-archive/Retreiver && \
PYTHONPATH=. python -m pytest -q tests/test_hybrid_retriever.py tests/test_vm_adapter_candidates.py tests/test_project_matrix_frontend.py && \
node --check web/app.js
```

Expected: PASS without requiring `git clean`, reset, staging, copying, or modifying the pre-existing `data/modular_runs/latest/*` files.

- [ ] **Step 4: Make final narrow commit/checkpoint**

```bash
git status --short
git log --oneline --decorate -8
git diff --check HEAD~1..HEAD
```

Expected: only intended source/config/tests/docs commits; no generated data staged.

## VM execution gate (outside this development environment)

The full candidate benchmark is not run locally. It may be run only by Tushar from the work laptop’s authorized VM shell after these observed gates are all true:

1. Adapter `/health` names both new endpoints and immutable upstream model IDs.
2. One direct two-document request succeeds against each new reranker endpoint with order-aligned scores.
3. GTE `/embed/gte` succeeds using the retained `transformers==4.57.6` environment.
4. Candidate 3-query smoke produces two retrieval-method result groups and method-specific evidence provenance.
5. Dashboard options/preflight show both retrieval methods and all five rerankers; exact launched run renders in Metrics, Recommendations, and Evidence.
6. Overlay hash/CRC receipt matches and archive excludes `data/`.
7. The user explicitly observes/approves the smoke before the full 50-combination VM run.
