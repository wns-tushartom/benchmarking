# WNS Parallel Execution Plan

**Updated from Raghu meeting audio:** 2026-05-26

**Core decision:** Backend benchmark pipeline remains the source of truth, but UI and backend can move in parallel. UI should control, monitor, preview, and report batch runs. It should not become the heavy execution engine that loads/unloads models interactively.

---

## North Star

Build a real RAG benchmark system that can run the selected WNS matrix, save reproducible retrieval/reranking artifacts, and present results clearly in a dashboard/report.

Current target matrix:

- 5 chunking methods
- 3 embedding models
- 3 vector DBs
- 1 index type: HNSW
- 1 retrieval method: cosine similarity
- 3 rerankers
- Total: 135 combinations

---

## Architecture Decision

```text
Local/UI layer
  - configure run
  - show VM health
  - launch or instruct backend jobs
  - show progress/status
  - preview saved artifacts
  - compare final metrics

Backend batch layer on VM
  - chunk documents
  - embed chunks
  - store vectors in Qdrant/PGVector/Weaviate
  - retrieve top candidates
  - save retrieval artifacts
  - rerank saved retrieval artifacts
  - save final outputs
  - compute metrics once Karthik finalizes them
```

Important rule:

```text
UI is the cockpit. Backend batch pipeline is the engine.
```

---

## Track A: Backend Batch Pipeline

### A1. VM readiness

- Confirm VM is switched on.
- Confirm Docker works.
- Confirm ports 5000–5005 are free or acceptable.
- Pull latest GitHub.
- Run `setup_all_on_vm.sh --host <VM_IP> --force-env`.
- Verify:
  - model service health
  - Qdrant
  - PGVector
  - Weaviate

### A2. Data and chunking

- Confirm final PDFs/documents for first benchmark run.
- Confirm benchmark input question set.
- Generate/verify chunk workbook.
- Keep active chunkers:
  - `entity_heuristic_w6`
  - `entity_heuristic_w5`
  - `entity_heuristic_w4`
  - `Heading_sections_l2`
  - `fixed_tok1200_ov150`

### A3. Embedding and vector storage

For each embedding model:

- Load embedding model/API adapter.
- Embed all chunks.
- Store vectors in each DB:
  - Qdrant
  - PGVector
  - Weaviate
- Use separate collection/table/class per combination.
- Save manifest with:
  - chunker
  - embedding model
  - vector DB
  - dimensions
  - timestamp
  - row counts
  - status/error

### A4. Retrieval artifact generation

For each question and each chunker/embedding/vector DB combination:

- Retrieve top candidates, default target: top 25, confirm with Karthik.
- Save retrieval artifacts before reranking.
- Save enough fields for reranking and audit:
  - query ID
  - query text
  - chunk ID
  - PDF name
  - paragraph/chunk text
  - retrieval score
  - rank
  - combination metadata

### A5. Reranking artifact generation

For each saved retrieval artifact:

- Run reranker:
  - Amazon Rerank v1, if AWS credentials exist
  - Qwen3:4B Rerank
  - bge-reranker-base
- Do not recompute retrieval when only reranker changes.
- Save reranker output separately.
- Include reranker latency and errors.

### A6. Caching

- Cache embedding outputs by chunker + embedding model.
- Cache vector-store build status by chunker + embedding + DB.
- Cache retrieval outputs by query set + chunker + embedding + DB + top_k.
- Cache rerank outputs by retrieval artifact + reranker.

---

## Track B: UI / Dashboard in Parallel

### B1. VM connection panel

UI should show:

- VM host
- model service health
- Qdrant health
- PGVector health
- Weaviate health
- current `.env`/port summary, with secrets hidden

### B2. Run setup/status screen

UI should show batch pipeline stages:

- VM setup
- chunking
- embedding
- vector DB indexing
- retrieval
- reranking
- metrics/report

Each stage should show:

- pending/running/done/failed
- timestamp
- error message if failed
- artifact path if done

### B3. Matrix selector

UI can allow selecting:

- chunkers
- embedding models
- vector DBs
- rerankers
- top_k
- question subset/smoke vs full run

But selection should generate a backend batch job/config, not run heavy model operations inside frontend request lifecycle.

### B4. Artifact browser

UI should display saved outputs:

- retrieval CSV/Excel
- rerank CSV/Excel
- summary metrics
- failed combinations
- latency/cost summary

### B5. Results and comparison

After Karthik finalizes metrics, UI should show:

- best configuration
- per-query successes/failures
- recall/MRR/NDCG or chosen metric
- latency comparison
- commercial vs open-source comparison
- export to PPT/report

---

## Track C: Metrics and Team Alignment

Ask Karthik/team:

1. Should retrieval top_k before reranking be 25?
2. What exact final metric should decide the winner?
3. What does “relevant” mean for this dataset?
4. Is ground truth paragraph/context enough, or do we need manual labels?
5. Should final output be Excel, dashboard, PPT, or all three?
6. Should commercial combinations run now, or should OpenAI/AWS wait until keys are available?

Until answered, use provisional metrics only:

- Recall@k
- MRR
- NDCG@10
- Precision@5
- latency
- error rate

---

## Workstreams We Can Run Together

### Backend workstream

1. Finalize real VM setup.
2. Run smoke pipeline with 2–5 questions.
3. Save retrieval artifacts.
4. Run reranker from saved retrieval artifacts.
5. Confirm artifact schema.
6. Scale to full matrix.

### UI workstream

1. Build VM health/status panel.
2. Add matrix selector.
3. Add artifact browser.
4. Add run progress screen.
5. Add provisional results screen.
6. Replace provisional metrics once Karthik confirms final metric.

### Reporting workstream

1. Maintain task-wise timeline.
2. Update daily progress tracker.
3. Capture blockers.
4. Prepare final report/PPT only after metrics stabilize.

---

## Immediate Next Sprint

### Completed VM model checkpoint

- GitHub repo pulled on VM.
- Python environment created.
- Model service runs on GPU at port `5000`.
- Verified endpoints:
  - `/embed/gte`
  - `/embed/jina`
  - `/rerank/bge`
  - `/rerank/qwen`
- Benchmark config validates with 135 combinations.
- Frontend now exposes reranker model cards/options.

### Current blocker

- Docker permission is missing for user `U481019`, so Qdrant, PGVector, and Weaviate cannot start yet.
- Required admin action: `sudo usermod -aG docker U481019`, then logout/login.

### Next backend steps

- Get Docker access.
- Start vector DBs through `setup_all_on_vm.sh`.
- Run selected smoke benchmark:
  - one chunker
  - one embedding
  - one vector DB
  - BGE reranker
  - 2–5 questions
- Save retrieval artifacts before reranking.
- Run rerankers from saved retrieval artifacts.

### Next UI steps

- Add VM/vector DB health panel.
- Add artifact browser for retrieval and rerank outputs.
- Show selected reranker status and output preview.

### Day 3+

- Run BGE/Qwen reranking from saved retrieval outputs.
- Confirm rerank output schema.
- UI: show retrieval vs reranked rows.

### Day 4

- Ask Karthik metric questions.
- Add provisional metrics report.
- UI: summary comparison screen.

### Day 5

- Expand to larger subset.
- Fix failures.
- Prepare weekly progress update with task-wise status.

---

## Principle

We can work on UI and backend together, but we should keep the roles clean:

```text
Backend produces truthful artifacts.
UI makes those artifacts usable, visible, and impressive.
```

That is how we move fast without fooling ourselves.
