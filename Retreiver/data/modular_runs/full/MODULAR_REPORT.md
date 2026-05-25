# Modular WNS Benchmark Report

## Best configuration

- Chunker: `entity_heuristic_w6`
- Embedding: `openai_text-embedding-3-large`
- Vector store: `Weaviate`
- Reranker: `Amazon Rerank v1`
- Recall@5: **0.940711**
- MRR: **0.848848**
- nDCG@10: **0.864869**
- Avg latency: **28.132782 ms**

## Why it won
- Highest Recall@5: 0.940711
- Recall@5 delta versus runner-up: 0.000000
- Confidence interval status: statistical tie likely
- Chunk count: 214, avg latency: 28.132782 ms

## Top 5

1. `entity_heuristic_w6` / `openai_text-embedding-3-large` / `Weaviate` / `Amazon Rerank v1`: R@5=0.940711, MRR=0.848848, latency=28.132782 ms
2. `entity_heuristic_w6` / `openai_text-embedding-3-large` / `PGVector` / `Amazon Rerank v1`: R@5=0.940711, MRR=0.848848, latency=28.393542 ms
3. `entity_heuristic_w6` / `openai_text-embedding-3-large` / `Qdrant` / `Amazon Rerank v1`: R@5=0.940711, MRR=0.848848, latency=29.166848 ms
4. `entity_heuristic_w6` / `openai_text-embedding-3-large` / `Qdrant` / `bge-reranker-base`: R@5=0.936759, MRR=0.849971, latency=27.96372 ms
5. `entity_heuristic_w6` / `openai_text-embedding-3-large` / `Weaviate` / `bge-reranker-base`: R@5=0.936759, MRR=0.849971, latency=28.435408 ms

## Pareto front

- `mod_0025`: R@5=0.940711, latency=28.132782 ms, `entity_heuristic_w6` + `openai_text-embedding-3-large`
- `mod_0052`: R@5=0.936759, latency=15.980815 ms, `semantic_split` + `openai_text-embedding-3-large`
- `mod_0040`: R@5=0.934783, latency=12.298157 ms, `semantic_split` + `gte_multilingual_base`
- `mod_0041`: R@5=0.932806, latency=12.19418 ms, `semantic_split` + `gte_multilingual_base`
- `mod_0031`: R@5=0.920949, latency=12.156211 ms, `semantic_split` + `jina_v3`
- `mod_0079`: R@5=0.916996, latency=8.141895 ms, `fixed_tok1200_ov150` + `openai_text-embedding-3-large`
- `mod_0066`: R@5=0.895257, latency=6.586377 ms, `fixed_tok1200_ov150` + `gte_multilingual_base`
- `mod_0055`: R@5=0.893281, latency=6.470624 ms, `fixed_tok1200_ov150` + `jina_v3`
