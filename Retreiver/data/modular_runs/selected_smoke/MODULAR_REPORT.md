# Modular WNS Benchmark Report

## Best configuration

- Chunker: `entity_heuristic_w6`
- Embedding: `jina_v3`
- Vector store: `Qdrant`
- Index type: `HNSW`
- Retrieval method: `Cosine Similarity`
- Reranker: `bge-reranker-base`
- Recall@5: **1.0**
- MRR: **0.866667**
- nDCG@10: **0.862412**
- Avg latency: **20.982435 ms**

## Why it won
- Highest Recall@5: 1.0
- Chunk count: 214, avg latency: 20.982435 ms

## Top 5

1. `entity_heuristic_w6` / `jina_v3` / `Qdrant` / `HNSW` / `Cosine Similarity` / `bge-reranker-base`: R@5=1.0, MRR=0.866667, latency=20.982435 ms

## Pareto front

- `mod_0001`: R@5=1.0, latency=20.982435 ms, `entity_heuristic_w6` + `jina_v3`
