# Modular Benchmarking Roadmap

## Current state

The current WNS benchmark tool is useful but still closer to a project-specific benchmark script than a reusable benchmarking platform.

Strengths:
- Runs full combination matrices.
- Produces summary/detail CSVs and dashboard visuals.
- Handles chunking, embeddings, DB labels, retrieval, reranking, recall metrics.
- Has local/offline fallback mode for fast QA.

Limitations:
- Techniques/models are hardcoded in Python constants.
- Real providers are not proper adapters yet.
- Vector DB labels in fallback mode do not represent real DB behavior.
- Metrics are recall-heavy and do not explain failure modes deeply.
- Dashboard shows winners, but not enough “why this won”.
- No experiment registry, baseline comparison, or reproducibility manifest.

## Ideal state

A reusable benchmark harness where we can add any new chunker, embedding model, vector DB, retriever, reranker, query expansion method, or LLM answer generator through config and plugin adapters.

The system should answer:

1. Which configuration wins?
2. Why does it win?
3. What does it cost?
4. Is the result statistically meaningful?
5. Which queries fail, and what pattern do failures share?
6. Is the result reproducible?
7. Can we plug in the next model or technique in under 30 minutes?

## Recommended architecture

```text
configs/
  benchmark.yaml
  techniques/
    chunkers.yaml
    embeddings.yaml
    vector_dbs.yaml
    retrievers.yaml
    rerankers.yaml
    evaluators.yaml

benchmarking/
  core/
    experiment.py
    matrix.py
    runner.py
    registry.py
    cache.py
    schemas.py
  adapters/
    chunkers/
    embeddings/
    vector_dbs/
    retrievers/
    rerankers/
    evaluators/
  analysis/
    ranking.py
    significance.py
    failure_analysis.py
    explainability.py
    reporting.py
  storage/
    sqlite_store.py
    parquet_store.py
  api/
    server.py
  ui/
    dashboard
```

## Config-first matrix

Instead of hardcoded constants:

```yaml
experiment:
  name: wns-rag-benchmark-v2
  dataset: data/query.csv
  corpus: data/chunking_methods_output_v2.xlsx
  top_k: [3, 5, 10, 20]
  repetitions: 3
  random_seed: 42

matrix:
  chunkers:
    - semantic_split
    - entity_heuristic_w6
  embeddings:
    - openai_text_embedding_3_large
    - jina_v3
    - bge_m3
  vector_dbs:
    - qdrant_hnsw
  retrievers:
    - dense_cosine
    - hybrid_bm25_dense
  rerankers:
    - bge_reranker_base
    - qwen3_rerank
  evaluators:
    - exact_or_overlap_recall
    - llm_judge_groundedness
```

Adding a new model should be config plus adapter, not editing runner logic.

## Adapter interface

Every technique should implement a small interface.

### Chunker

```python
class Chunker:
    name: str
    def chunk(self, documents: list[Document]) -> list[Chunk]: ...
```

### Embedder

```python
class Embedder:
    name: str
    dimensions: int
    cost_per_1k_tokens: float | None
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...
```

### Vector DB

```python
class VectorStore:
    name: str
    def reset_collection(self, schema: Schema) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> Metrics: ...
    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]: ...
```

### Retriever

```python
class Retriever:
    name: str
    def retrieve(self, query: QueryCase, store: VectorStore, top_k: int) -> list[SearchHit]: ...
```

### Reranker

```python
class Reranker:
    name: str
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...
```

### Evaluator

```python
class Evaluator:
    name: str
    def evaluate(self, query: QueryCase, hits: list[SearchHit]) -> dict[str, float | str]: ...
```

## What we are missing for “go-to benchmark” quality

### 1. Dataset discipline

Need:
- versioned datasets
- query IDs stable forever
- answer/ground-truth fields validated
- document IDs and chunk parent IDs tracked
- train/dev/test split if tuning thresholds
- query categories, for example refunds, rebooking, passport, voucher, PNR, airline-specific

Why: without categories, you only know the winner globally. You do not know where it fails.

### 2. Real provider adapters

Need production adapters for:
- OpenAI embeddings
- Jina embeddings
- BGE/GTE local embeddings
- Qdrant
- PGVector
- Weaviate
- Milvus optional
- Amazon Rerank
- Qwen reranker
- BGE reranker

Each adapter needs:
- health check
- config schema
- credentials check
- timeout/retry policy
- rate-limit handling
- cost tracking
- cache key

### 3. Caching and reproducibility

Need:
- embedding cache keyed by model + text hash
- rerank cache keyed by reranker + query + candidate IDs
- run manifest with git SHA, config hash, dataset hash, timestamp, env readiness
- deterministic run IDs
- ability to resume failed runs

Why: full real benchmark will cost money and time. Re-running everything from scratch is wasteful.

### 4. More metrics

Current metrics are too recall-heavy.

Add:
- Recall@1, @3, @5, @10, @20
- MRR
- nDCG@k
- precision@k
- hit position distribution
- answer coverage score
- groundedness score
- latency p50, p95, p99 by stage
- indexing time
- embedding time
- reranking time
- total cost per 1,000 queries
- chunk count and avg chunk tokens
- storage size
- failed query count
- timeout/error rate

### 5. “Why did it win?” analysis

Need automated explanations:
- winner is better on which query categories
- winner loses on which query categories
- average rank of correct chunk
- examples where config A finds answer and config B misses
- overlap between top-k retrieved sets
- chunk size vs recall curve
- reranker lift over base retrieval
- embedding model lift while holding chunker/DB/reranker constant

This is the difference between a leaderboard and a decision tool.

### 6. Statistical confidence

Need:
- bootstrap confidence intervals
- paired comparison between top configs
- significance flags: “clear winner”, “tie”, “not enough difference”

Why: if config A is 0.9367 and B is 0.9348, that may not be meaningful.

### 7. Baselines and regression gates

Need locked baselines:
- current production config
- fastest config
- highest recall config
- cheapest config

Add regression checks:
- fail if Recall@5 drops by more than X
- fail if p95 latency rises above Y
- fail if cost rises above Z
- fail if category-specific recall drops for critical categories

### 8. Better dashboard

Needed dashboard upgrades:
- experiment selector
- compare two configs side by side
- per-query drilldown
- failure explorer
- category filters
- cost/latency/quality Pareto frontier
- confidence intervals
- “reranker lift” chart
- “chunking tradeoff” chart: chunks vs recall vs latency
- stage timing waterfall
- export PDF/HTML report
- mark recommended config with rationale

### 9. Plugin system

Need a simple contract:

```bash
benchmark adapters list
benchmark adapters validate configs/techniques/embeddings.yaml
benchmark run configs/benchmark.yaml
benchmark report runs/<run_id>
benchmark dashboard
```

Adding a model should look like:

```yaml
- id: voyage_3_large
  type: embedding
  adapter: benchmarking.adapters.embeddings.voyage:VoyageEmbedding
  params:
    model: voyage-3-large
    dimensions: 1024
```

### 10. CI and nightly benchmark

Need:
- tiny smoke benchmark in CI
- weekly full benchmark on stable dataset
- result diff posted to dashboard/report
- alert if production baseline regresses

## Build phases

### Phase 1: Config and registry refactor

Goal: remove hardcoded constants.

Deliverables:
- `configs/benchmark.yaml`
- `benchmarking/core/registry.py`
- adapter interfaces
- matrix generated from config
- current local fallback adapters migrated into plugin format

### Phase 2: Evaluation depth

Goal: make ranking trustworthy.

Deliverables:
- MRR, nDCG, precision, recall@20
- per-query detail schema
- query category support
- failure analysis report
- paired config comparison

### Phase 3: Real provider adapters

Goal: make benchmark production-valid.

Deliverables:
- Qdrant real adapter
- OpenAI embedding adapter
- Jina embedding adapter
- local BGE/GTE adapter
- Qwen/BGE rerank adapters
- cost and latency stage tracing

### Phase 4: Dashboard decision cockpit

Goal: explain winners clearly.

Deliverables:
- experiment selector
- side-by-side compare
- Pareto frontier
- failure explorer
- category breakdown
- exportable final report

### Phase 5: Regression and automation

Goal: make it the go-to internal benchmark.

Deliverables:
- benchmark CLI
- run manifest and result store
- CI smoke run
- weekly scheduled benchmark
- regression gates

## Definition of done for “go-to benchmark”

- [ ] A new chunking method can be added with one config entry and one adapter file.
- [ ] A new embedding/reranker model can be added without editing runner logic.
- [ ] Every run has dataset hash, config hash, code version, and provider readiness status.
- [ ] Results include quality, latency, cost, failure rate, and stage timings.
- [ ] Dashboard explains winner, runner-up, tradeoffs, and failure examples.
- [ ] Reports can be exported for managers/client review.
- [ ] Production baseline regressions can be detected automatically.
- [ ] Real provider mode and local fallback mode are clearly separated.

## Strong opinion

Do not jump straight into adding every provider. First refactor to config + registry + adapter interfaces. Otherwise each new model will make the scripts messier. The next serious step is architectural modularity, then real adapters, then dashboard explainability.
