# WNS Benchmark Final Report

## Executive recommendation

Use **`semantic_split`** as the best full-pipeline candidate from the completed local/offline benchmark matrix.

Best local/offline configuration:

```text
run_0127
chunking: semantic_split
embedding: openai_text-embedding-3-large
vector DB: Qdrant
retrieval: HNSW + Cosine Similarity
reranker: Amazon Rerank v1
Recall@5: 0.936759
Recall@10: 0.962451
Avg query latency: 22.524 ms
```

Practical production starting point:

```text
semantic_split + openai_text-embedding-3-large + Qdrant HNSW + Amazon Rerank v1
```

If Amazon Rerank is not ready because AWS credentials/runtime are incomplete, start with:

```text
semantic_split + gte_multilingual_base or openai_text-embedding-3-large + Qdrant HNSW + Qwen3/BGE reranker
```

## Run scope

- Query cases: **506**
- Configurations tested: **162**
- Matrix: **6 chunking methods × 3 embedding labels × 3 vector DB labels × 3 rerankers**
- Output mode: **local_offline_fallback**
- Runtime: deterministic local embeddings/vector/rerank adapters for comparable pipeline validation

Important: these numbers are useful for **relative method selection and pipeline QA**. They are not final production latency until real provider adapters are connected.

## Chunking conclusion

There are two views:

1. **Standalone chunking recall script** with one fixed embedding/DB/reranker setup:
   - `entity_heuristic_w6` has the highest standalone Recall@5/Recall@10.
2. **Full matrix** across embeddings, DB labels, and rerankers:
   - `semantic_split` is the best average chunking method and owns the top overall configurations.

Final decision for the benchmark pipeline: **choose `semantic_split` as the third/final comparison method and recommended production candidate**.

Standalone chunking recall ranking from all-query run:

- `entity_heuristic_w6`: R@5 **0.922925**, R@10 **0.970356**, chunks 214, avg 22.712 ms
- `entity_heuristic_w5`: R@5 **0.920949**, R@10 **0.968379**, chunks 222, avg 22.993 ms
- `entity_heuristic_w4`: R@5 **0.918972**, R@10 **0.964427**, chunks 227, avg 24.043 ms
- `semantic_split`: R@5 **0.913043**, R@10 **0.954545**, chunks 122, avg 13.151 ms
- `Heading_sections_l2`: R@5 **0.883399**, R@10 **0.946640**, chunks 127, avg 13.696 ms
- `fixed_tok1200_ov150`: R@5 **0.883399**, R@10 **0.926877**, chunks 56, avg 6.096 ms

## Top 10 configurations from full matrix

1. `run_0127` — `semantic_split` / `openai_text-embedding-3-large` / `Qdrant` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.962451**, avg **22.524 ms**
2. `run_0130` — `semantic_split` / `openai_text-embedding-3-large` / `PGVector` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.962451**, avg **22.524 ms**
3. `run_0133` — `semantic_split` / `openai_text-embedding-3-large` / `Weaviate` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.962451**, avg **22.524 ms**
4. `run_0118` — `semantic_split` / `gte_multilingual_base` / `Qdrant` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.958498**, avg **12.536 ms**
5. `run_0121` — `semantic_split` / `gte_multilingual_base` / `PGVector` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.958498**, avg **12.536 ms**
6. `run_0124` — `semantic_split` / `gte_multilingual_base` / `Weaviate` / `Amazon Rerank v1` — R@5 **0.936759**, R@10 **0.958498**, avg **12.536 ms**
7. `run_0119` — `semantic_split` / `gte_multilingual_base` / `Qdrant` / `Qwen3:4B Rerank` — R@5 **0.934783**, R@10 **0.958498**, avg **12.536 ms**
8. `run_0120` — `semantic_split` / `gte_multilingual_base` / `Qdrant` / `bge-reranker-base` — R@5 **0.934783**, R@10 **0.958498**, avg **12.536 ms**
9. `run_0122` — `semantic_split` / `gte_multilingual_base` / `PGVector` / `Qwen3:4B Rerank` — R@5 **0.934783**, R@10 **0.958498**, avg **12.536 ms**
10. `run_0123` — `semantic_split` / `gte_multilingual_base` / `PGVector` / `bge-reranker-base` — R@5 **0.934783**, R@10 **0.958498**, avg **12.536 ms**

## Average by chunking method

- `semantic_split`: avg R@5 **0.928195**, avg R@10 **0.958498**, avg latency **16.051 ms**, chunks 122
- `entity_heuristic_w6`: avg R@5 **0.926658**, avg R@10 **0.971673**, avg latency **29.195 ms**, chunks 214
- `entity_heuristic_w5`: avg R@5 **0.925780**, avg R@10 **0.969038**, avg latency **26.002 ms**, chunks 222
- `entity_heuristic_w4`: avg R@5 **0.923803**, avg R@10 **0.967721**, avg latency **26.944 ms**, chunks 227
- `Heading_sections_l2`: avg R@5 **0.893061**, avg R@10 **0.944664**, avg latency **15.071 ms**, chunks 127
- `fixed_tok1200_ov150`: avg R@5 **0.891304**, avg R@10 **0.934124**, avg latency **6.552 ms**, chunks 56

Interpretation:

- `semantic_split` gives the best Recall@5 average and much lower chunk volume than entity heuristics.
- `entity_heuristic_w6` gives the best Recall@10 average but uses 214 chunks and is slower.
- `fixed_tok1200_ov150` is fastest but loses too much recall.

## Average by embedding label

- `openai_text-embedding-3-large`: avg R@5 **0.918643**, avg R@10 **0.962780**, avg latency **23.701 ms**
- `gte_multilingual_base`: avg R@5 **0.918094**, avg R@10 **0.954875**, avg latency **18.595 ms**
- `jina_v3`: avg R@5 **0.907664**, avg R@10 **0.955204**, avg latency **17.612 ms**

Interpretation:

- `openai_text-embedding-3-large` wins on recall.
- `gte_multilingual_base` is close and faster in the local fallback run.
- Real provider latency/cost must be checked before final production choice.

## Average by reranker

- `Amazon Rerank v1`: best Recall@5 in this fallback benchmark.
- `Qwen3:4B Rerank` and `bge-reranker-base`: very close behind for top semantic_split configs.

Production note: Amazon Rerank cannot be treated as ready until AWS credentials/region/runtime are configured.

## Vector DB result

In local/offline fallback mode, `Qdrant`, `PGVector`, and `Weaviate` labels use the same in-process vector adapter, so their recall/latency numbers are identical. This validates matrix wiring, but **does not rank real DB engines**.

Production DB readiness currently favors Qdrant because the local Qdrant service is reachable.

## Artifacts

- `data/chunking_recall_summary.csv`
- `data/chunking_recall_results.csv`
- `data/full_benchmark/benchmark_summary.csv`
- `data/full_benchmark/benchmark_details.csv`
- `data/full_benchmark/benchmark_report.json`
- `data/full_benchmark/final_analysis.json`

## Next production benchmark slice

Run this first in provider mode once adapters are wired:

```text
semantic_split + openai_text-embedding-3-large + Qdrant HNSW + available reranker
semantic_split + gte_multilingual_base + Qdrant HNSW + available reranker
entity_heuristic_w6 + openai_text-embedding-3-large + Qdrant HNSW + available reranker
```

That gives a small real-world sanity check before spending time/cost on the full provider matrix.
