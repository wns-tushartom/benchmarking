# Modular WNS Benchmark Report

## Best configuration

- Chunker: `fixed_tok1200_ov150`
- Embedding: `jina_v3`
- Vector store: `Qdrant`
- Index type: `HNSW`
- Retrieval method: `Cosine Similarity`
- Reranker: `bge-reranker-base`
- Recall@5: **0.8**
- MRR: **0.8**
- nDCG@10: **0.773903**
- Avg latency: **6.379435 ms**

## Why it won
- Highest Recall@5: 0.8
- Chunk count: 56, avg latency: 6.379435 ms

## Top 5

1. `fixed_tok1200_ov150` / `jina_v3` / `Qdrant` / `HNSW` / `Cosine Similarity` / `bge-reranker-base`: R@5=0.8, MRR=0.8, latency=6.379435 ms

## Pareto front

- `mod_0001`: R@5=0.8, latency=6.379435 ms, `fixed_tok1200_ov150` + `jina_v3`
