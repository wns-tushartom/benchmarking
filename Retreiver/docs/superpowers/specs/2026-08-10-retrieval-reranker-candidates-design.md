# Retrieval and Reranker Candidate Lane Design

## Goal

Add an isolated, VM-backed candidate benchmark lane that compares GTE dense cosine retrieval with BM25 + GTE dense + RRF retrieval, and evaluates the existing BGE/Qwen rerankers plus NVIDIA Nemotron 1B and Alibaba GTE ModernBERT rerankers. The accepted 180-combination benchmark and its artifacts remain unchanged.

## Scope boundary

The canonical 180 matrix remains exactly:

```text
5 chunkers × 3 embeddings × 4 vector stores × 3 rerankers = 180
```

`configs/benchmark.local.json`, `configs/project_matrix_catalog.json`, accepted artifacts, and their integrity tests remain the official lane. The implementation must not add the new models or retrieval methods to that matrix, mutate accepted results, or write candidate output under the accepted-result paths.

The new candidate configuration is:

```text
5 chunkers
× 2 retrieval methods
  - Dense Cosine: GTE multilingual base + FAISS cosine retrieval
  - BM25 + GTE Dense + RRF: BM25 branch + the same GTE/FAISS dense branch, rank-fused
× 5 rerankers
  - none (identity baseline)
  - bge-reranker-base
  - Qwen3:4B Rerank
  - nemotron_rerank_1b
  - gte_modernbert_base
= 50 combinations
```

Candidate outputs are written beneath `data/modular_runs/retrieval-reranker-candidates/` and include a configuration hash, source commit, model IDs, retrieval settings, and result/evidence provenance.

## User-visible naming and model contracts

| Product ID | UI label | Verified upstream model ID | Contract |
|---|---|---|---|
| `nemotron_rerank_1b` | Nemotron Rerank 1B | `nvidia/llama-nemotron-rerank-1b-v2` | NVIDIA Llama cross-encoder; `question:<query>\n\npassage:<document>` prompt format; raw logit; 8,192-token maximum |
| `gte_modernbert_base` | GTE ModernBERT Base Reranker | `Alibaba-NLP/gte-reranker-modernbert-base` | English cross-encoder; 8,192-token maximum; standard query/document pairs |

`Alibaba-NLP/gte-modernbert-base` is an embedding model, not the selected reranker, and must not be registered under `gte_modernbert_base`.

## Retrieval behavior

### Dense Cosine

1. Embed corpus chunks and queries through the existing VM GTE endpoint.
2. Build/reuse the FAISS cosine index.
3. Retrieve the first `candidate_depth` candidates.
4. Apply the selected reranker, or preserve the ranking for `none`.
5. Return final `top_k` results.

### BM25 + GTE Dense + RRF

1. Build BM25 once per chunker from the same canonical chunk text used by the dense branch.
2. Independently retrieve `candidate_depth` candidates from BM25 and GTE/FAISS cosine.
3. Fuse the two ranked lists by Reciprocal Rank Fusion using the fixed formula:

```text
rrf_score = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
```

4. Preserve the top `fusion_depth` fused candidates.
5. Apply the selected reranker, or preserve RRF ranking for `none`.
6. Return final `top_k` results.

The initial candidate configuration fixes `candidate_depth=50`, `fusion_depth=20`, `rrf_k=60`, and `top_k=10`. These values are recorded in each manifest and must not be silently tuned during the run.

Each evidence row persists method-specific provenance:

```text
Dense: dense_rank, dense_score
Hybrid: dense_rank, dense_score, bm25_rank, bm25_score, rrf_score
```

## Runtime and adapter architecture

The existing VM model adapter service gains two explicit endpoints:

```text
POST /rerank/nemotron
POST /rerank/gte-modernbert
```

`/health` lists the exact model IDs, endpoint names, device, dtype, configured max length, batch size, and loaded-model state. The dashboard preflight treats endpoint availability as a real dependency and refuses a candidate run when its selected reranker is unavailable.

Nemotron uses an explicit `AutoTokenizer` + `AutoModelForSequenceClassification` loader. It applies NVIDIA's required `question:` / `passage:` template, sets a pad token when needed, uses BF16 on CUDA or FP32 otherwise, batches pairs, truncates deterministically at the configured maximum, and returns aligned raw logits.

GTE ModernBERT uses a `CrossEncoder` loader for `Alibaba-NLP/gte-reranker-modernbert-base`, with the same explicit maximum-token and batch settings. Scores remain aligned to the submitted document order. The existing generic HTTP reranker adapter continues to consume the aligned `scores` response shape.

The environment/configuration adds:

```text
NEMOTRON_RERANK_URL
GTE_MODERNBERT_RERANK_URL
NEMOTRON_RERANKER_MODEL
GTE_MODERNBERT_RERANKER_MODEL
WNS_RERANK_MAX_LENGTH
WNS_RERANK_BATCH_SIZE
```

No local/dummy reranker fallback is permitted. The VM service is the only allowed execution path for these two models.

## Frontend behavior

The official 180 controls remain visibly labelled as the accepted benchmark and retain their current fixed dimensions.

A separate **Candidate retrieval and reranking** section exposes only the candidate lane:

- retrieval method selector: `GTE Dense Cosine` or `BM25 + GTE Dense + RRF`;
- reranker selector: the five candidate choices;
- fixed, displayed `candidate depth`, `RRF k`, `fusion depth`, and final `top K` values;
- calculated selected-combination count from the candidate matrix;
- endpoint/readiness state before launch;
- method and reranker labels in Metrics, Recommendations, and Evidence.

The frontend identifies the candidate lane and its exact run ID in every result view. Candidate data must never be merged into the official-180 display or count.

## Scripts and deployment

A config-driven candidate CLI run uses the separate configuration and result root. Supporting setup/check scripts receive the two endpoint environment variables and direct smoke probes. The VM handoff package is a narrow overlay containing source, tests, config, and scripts only; it contains no `data/`, existing result artifacts, secrets, or model weights.

VM deployment occurs only from the work laptop. After the work-laptop branch is pushed, the VM procedure must be one compact command at a time:

1. verify the source commit and existing data preservation;
2. apply the narrow overlay/pull;
3. restart only the model adapter service;
4. run adapter health and one request per new reranker;
5. run the candidate config with a small labelled smoke subset;
6. inspect evidence provenance and candidate result publication;
7. run the full 50 combinations only after those gates pass.

## Test and acceptance plan

Tests are written before production code and cover:

1. Candidate configuration produces exactly 50 combinations and official configuration remains exactly 180.
2. BM25 returns lexical matches without creating an embedding/vector-store dependency.
3. Dense and hybrid retrieval return deterministic results; hybrid exposes both branch provenance and mathematically correct RRF scores.
4. Identity reranking preserves retrieval order; remote reranker registry/config maps the two new UI IDs to their endpoints and upstream model IDs.
5. Nemotron input formatting, truncation, batching, response alignment, device/dtype fallback, and health metadata.
6. GTE ModernBERT response alignment, truncation, batching, and health metadata.
7. Candidate preflight fails closed when the selected endpoint is unhealthy.
8. The frontend submits the candidate lane identity, shows selected method/reranker, reports the correct count, and does not alter official-180 labels/counts.
9. Browser QA selects each new reranker and both retrieval methods, then verifies the launched run appears in Metrics, Recommendations, and Evidence with correct provenance.
10. Overlay archive contains no `data/` entries, passes archive integrity checks, and has a recorded SHA-256.

A full candidate result is accepted only when all 50 combinations have terminal status, evidence has method-specific provenance, metric fields are complete for labelled queries, and the run manifest records the fixed retrieval settings and model IDs.
