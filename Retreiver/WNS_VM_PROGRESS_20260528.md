# WNS VM Progress Update — 2026-05-28

## Current verified state

### Source documents

KnowRA document package has been uploaded to the VM through Jupyter on allowed port `5009`.

VM path:

```text
~/benchmarking/Retreiver/data/pdfs/knowra/
```

Verified:

```text
225 PDFs
649M
no nested folder after flattening
```

Sample files verified:

```text
Content - ETG and Partner Products 20250310-072912.pdf
Content - Retail Loss - General Information & Dashboards 20250310-072601.pdf
Guidelines for Providing Ticket Numbers to Airline Support to Avoid ADMs (Exceptional scenarios).pdf
Content - Refunds 20250310-074217.pdf
Content - 01. Schedule Change Description 20250310-065704.pdf
```

### VM services

Docker permission for user `U481019` has been fixed. User groups now include `docker`.

Benchmark services are running and manually verified:

```text
Model adapter: http://127.0.0.1:5000/health OK, device cuda
Qdrant:       http://127.0.0.1:5001/ OK
PGVector:     127.0.0.1:5003 OK / healthy container
Weaviate:     http://127.0.0.1:5004/v1/meta OK
```

Docker containers verified:

```text
wns-pgvector  -> 0.0.0.0:5003->5432
wns-qdrant    -> 0.0.0.0:5001->6333, 5002->6334
wns-weaviate  -> 0.0.0.0:5004->8080, 5005->50051
qdrant_ui     -> old existing container on 6333-6334, not used by benchmark
```

Setup script output verified:

```text
Qdrant HTTP OK
PGVector/Postgres OK
Weaviate HTTP OK
benchmark config OK, matrix_count 135
Jina embedding smoke OK
GTE embedding smoke OK
BGE rerank smoke OK
Qwen smoke skipped by default because heavy preload
```

### Benchmark config

Validation passed on VM:

```json
{
  "ok": true,
  "experiment": "wns-rag-benchmark-selectable",
  "matrix_count": 135
}
```

Matrix remains:

```text
5 chunkers x 3 embeddings x 3 vector DBs x 1 HNSW x 1 cosine retrieval x 3 rerankers = 135 combos
```

Active matrix chunkers:

```text
entity_heuristic_w6
entity_heuristic_w5
entity_heuristic_w4
Heading_sections_l2
fixed_tok1200_ov150
```

Candidate/available chunker:

```text
semantic_split
```

## Extraction result

Existing extraction script required top-level `data/pdfs/*.pdf`, so KnowRA PDFs were symlinked from:

```text
data/pdfs/knowra/*.pdf
```

to:

```text
data/pdfs/*.pdf
```

Extraction ran with venv Python because system Python is externally managed.

Clean extraction result:

```text
225 PDF symlinks
6403 raw extracted rows
69 null paragraph rows removed
5754 clean paragraphs
222 unique PDFs with usable extracted text
```

Three PDFs did not produce usable text through PyPDF2:

```text
Automatic segment removal and ghost line additions.pdf
Expedia contact White Label.pdf
Offline sales.pdf
```

Missing list should be kept at:

```text
data/knowra_missing_text_pdfs.txt
```

Benchmark input verified:

```text
Columns: id, pdf_name, paragraph
Total rows: 5754
Paragraph length min: 40 chars
Paragraph length max: 5372 chars
Paragraph length mean: 502.8 chars
Unique PDFs: 222
```

## Chunking result

Generated and verified chunking workbook:

```text
data/chunking_methods_output_v2.xlsx
```

Sheets and counts:

```text
original_input:        5754
entity_heuristic_w6:   16873
entity_heuristic_w5:   16965
entity_heuristic_w4:   17099
Heading_sections_l2:   7617
semantic_split:        19338
fixed_tok1200_ov150:   5580
```

Important: current `verify_all_chunking_methods.py` expansion ratio output is stale/wrong because it still compares against the old 61-row baseline. Correct ratios vs 5754 clean paragraphs are:

```text
entity_heuristic_w6:   2.93x
entity_heuristic_w5:   2.95x
entity_heuristic_w4:   2.97x
Heading_sections_l2:   1.32x
semantic_split:        3.36x
fixed_tok1200_ov150:   0.97x
```

## Query ground-truth status

Chunk recall comparison could not run because both files are missing on VM:

```text
data/qa_text_test.csv
data/query.csv
```

Error:

```text
FileNotFoundError: No usable query ground truth found in data/qa_text_test.csv or data/query.csv
```

Until a query/ground-truth file is restored or created, recall metrics cannot be computed. DB ingestion can still be tested without query ground truth.

## Code/runtime fixes discovered on VM

### Qdrant client API compatibility

Problem:

```text
qdrant-client 1.18.0 warns it is incompatible with Qdrant server 1.12.6
QdrantClient.search no longer exists
```

Fix applied on VM and should be preserved in repo:

- `benchmarking/adapters/vector_qdrant.py` now supports both:
  - old `client.search(...)`
  - new `client.query_points(...)`

### Qdrant upsert timeout

Problem:

Full 5580-vector upsert timed out when sending all points in one request.

Fix applied on VM and should be preserved in repo:

- Qdrant client timeout increased from `60` to `300`
- Qdrant upsert now batches points in chunks of `256`

### GTE CUDA OOM

Problem:

Full GTE embedding with batch size `32` caused CUDA OOM:

```text
GPU total: 22.07 GiB
free: ~173 MiB
PyTorch allocated: ~20.79 GiB
```

Fix:

- Restarted model adapter service to clear memory.
- Started with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- Reduced GTE embedding batch size to `4` for full KnowRA ingestion.

## DB ingestion tests

### 50-chunk smoke test

Input:

```text
sheet: fixed_tok1200_ov150
chunks: 50
embedding: GTE multilingual base
vector dim: 768
```

Result:

```text
Qdrant:   upsert OK, search_hits 3
PGVector: upsert OK, search_hits 3
Weaviate: upsert OK, search_hits 3
```

### Full fixed_tok1200_ov150 ingestion

Input:

```text
sheet: fixed_tok1200_ov150
chunks: 5580
embedding: GTE multilingual base
vector dim: 768
embedding batch size: 4
```

Result:

```text
Embedding time: 51.06 sec
Embedding cache saved: data/embedding_cache/fixed_tok1200_ov150_gte_vectors.npy
```

DB results:

```text
Qdrant:
  upsert_latency_s: 4.681980115001352
  vector_count: 5580
  search_hits: 5

PGVector:
  upsert_latency_s: 5.815058075000707
  vector_count: 5580
  search_hits: 5

Weaviate:
  upsert_latency_s: 7.56662101099937
  vector_count: 5580
  search_hits: 5
```

Top search hits from vector DBs are consistent:

```text
07. New - Handling NACO request for CC2C orders.pdf
Naco_ Scenario of refund of old ticket and then issuing a new ticket.pdf
Content - FL Phone Scripts 20250310-073137.pdf
```

This proves full path works for one chunker:

```text
KnowRA PDFs -> extracted text -> chunks -> GTE embeddings -> Qdrant/PGVector/Weaviate -> vector search
```

## Full Heading_sections_l2 ingestion

Completed successfully after the first fixed-window chunker.

Input:

```text
sheet: Heading_sections_l2
chunks: 7617
embedding: GTE multilingual base
batch size: 4
cache target: data/embedding_cache/Heading_sections_l2_gte_vectors.npy
```

DB results:

```text
Qdrant:
  upsert_latency_s: 6.337221447000047
  vector_count: 7617
  search_hits: 5

PGVector:
  upsert_latency_s: 8.127521294998587
  vector_count: 7617
  search_hits: 5

Weaviate:
  upsert_latency_s: 9.787720871001511
  vector_count: 7617
  search_hits: 5
```

Top search hits from vector DBs are consistent:

```text
07. New - Handling NACO request for CC2C orders.pdf
Naco_ Scenario of refund of old ticket and then issuing a new ticket.pdf
Content - Refunds 20250310-074217.pdf
```

This proves two full KnowRA chunkers now work end-to-end through all three vector DBs:

```text
fixed_tok1200_ov150 -> GTE -> Qdrant/PGVector/Weaviate -> search
Heading_sections_l2 -> GTE -> Qdrant/PGVector/Weaviate -> search
```

## GTE all chunkers DB ingestion complete

All generated KnowRA chunking methods have now been embedded with GTE and inserted/searched successfully in all three vector DBs.

Completed chunkers:

```text
fixed_tok1200_ov150: 5580 chunks
Heading_sections_l2: 7617 chunks
entity_heuristic_w6: 16873 chunks
entity_heuristic_w5: 16965 chunks
entity_heuristic_w4: 17099 chunks
semantic_split: 19338 chunks
```

Latest long-run summary:

```text
run: data/db_ingestion_runs/20260528_112335/summary.csv
failures: 0
```

Latest completed results:

```text
entity_heuristic_w5:
  chunks: 16965
  Qdrant:   OK, hits=5, 14.000788s
  PGVector: OK, hits=5, 19.132576s
  Weaviate: OK, hits=5, 20.81994s

entity_heuristic_w4:
  chunks: 17099
  embedding_seconds: 104.14
  Qdrant:   OK, hits=5, 13.873128s
  PGVector: OK, hits=5, 19.501576s
  Weaviate: OK, hits=5, 21.018817s

semantic_split:
  chunks: 19338
  embedding_seconds: 108.97
  Qdrant:   OK, hits=5, 15.811039s
  PGVector: OK, hits=5, 24.62386s
  Weaviate: OK, hits=5, 23.697815s
```

GTE embedding caches saved under:

```text
data/embedding_cache/
```

The database setup/ingestion phase for GTE is now proven across scale. Remaining benchmark work needs query ground truth and reranker evaluation.

## Next steps

1. Restore/create query ground-truth file:
   - `data/qa_text_test.csv` with `question|ground_truth|context`, or
   - `data/query.csv` with `query|answer`
2. Once query file exists, run actual retrieval metrics:
   - Recall@5
   - Recall@10
   - MRR
   - nDCG
3. Jina embedding ingestion completed for the pasted long run:
   - detailed timing notes saved in `JINA_TIMINGS_20260528.md`.
   - final pasted run: `data/db_ingestion_runs/20260528_114128/summary.csv`, failures `0`.
   - captured: `entity_heuristic_w6`, `entity_heuristic_w5`, `entity_heuristic_w4`, and `semantic_split` all passed Qdrant/PGVector/Weaviate with hits=5.
4. BGE reranker smoke completed:
   - result saved in `RERANKER_SMOKE_20260528.md`.
   - query: `refund old ticket and issue new ticket`.
   - setup: `fixed_tok1200_ov150` + `gte_multilingual_base` + `Qdrant`, retrieved top 20, BGE reranked top 5.
   - rerank_seconds: `0.103`.
   - endpoint worked and reordered vector hits into more directly refund/reissue-focused results.
5. Qwen reranker smoke completed:
   - result saved in `RERANKER_SMOKE_20260528.md`.
   - query: `refund old ticket and issue new ticket`.
   - setup: `fixed_tok1200_ov150` + `gte_multilingual_base` + `Qdrant`, retrieved top 20, Qwen reranked top 5.
   - retrieval_seconds: `0.046`; rerank_seconds: `1.592`.
   - endpoint worked and produced a refund-focused but different top-5 order than BGE.
6. Remaining reranker smoke:
   - Amazon only if AWS/Bedrock credentials are available.

## User/process preferences learned in this WNS session

- WNS group files/updates should be sent in this group only, not DM.
- Only VM ports `5000-5010` are available.
- Jupyter was run on port `5009` for upload.
- Backend batch artifacts are the source of truth; UI is the cockpit/status/reporting layer.
