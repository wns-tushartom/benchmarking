# WNS selectable benchmark combinations

The dashboard and modular CLI now support selecting any combination across the exact WNS matrix.

## Official selectable dimensions

### Chunking

- `entity_heuristic_w6`
- `entity_heuristic_w5`
- `entity_heuristic_w4`
- `Heading_sections_l2`
- `fixed_tok1200_ov150`

Optional candidate methods still exist in config for experiments, but they are not part of the official default selectable matrix:

- `semantic_split`

### Embeddings

- `jina_v3`, open source
- `gte_multilingual_base`, open source
- `openai_text-embedding-3-large`, commercial

### Vector database

- `Qdrant`
- `PGVector`
- `Weaviate`

### Index

- `HNSW`

### Retrieval

- `Cosine Similarity`

### Rerank

- `Amazon Rerank v1`, commercial
- `Qwen3:4B Rerank`, open source
- `bge-reranker-base`, open source

## Matrix size

```text
5 chunking x 3 embeddings x 4 vector stores x 1 index x 1 retrieval x 3 rerankers = 180 combinations
```

## Dashboard usage

Start the dashboard:

```bash
python3 scripts/serve_benchmark_dashboard.py 8765
```

Open:

```text
http://127.0.0.1:8765
```

Use the selectors:

- Chunking
- Embedding
- Vector database
- Index
- Retrieval
- Rerank

Then click:

```text
Run selected combo
```

If all selectors are set to `All`, the modular runner executes the selected matrix, which is currently the full 180 combinations unless `Max runs` is set.

## CLI usage

Run all 180 local fallback combinations:

```bash
python3 scripts/benchmark_cli.py run --limit-queries 0 --output-dir data/modular_runs/latest
```

Run one selected combination:

```bash
python3 scripts/benchmark_cli.py run \
  --limit-queries 0 \
  --output-dir data/modular_runs/latest \
  --chunker entity_heuristic_w6 \
  --embedding jina_v3 \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity" \
  --reranker "bge-reranker-base"
```

Run a partial matrix, for example all rerankers for one chunker/embedding/DB:

```bash
python3 scripts/benchmark_cli.py run \
  --limit-queries 0 \
  --chunker entity_heuristic_w6 \
  --embedding openai_text-embedding-3-large \
  --vector-store Qdrant \
  --index-type HNSW \
  --retrieval-method "Cosine Similarity"
```

Validate matrix count:

```bash
python3 scripts/benchmark_cli.py validate
```

Expected:

```json
{
  "ok": true,
  "experiment": "wns-rag-benchmark-selectable",
  "matrix_count": 180
}
```

## Output files

Selected runs write to:

```text
data/modular_runs/latest/manifest.json
data/modular_runs/latest/modular_summary.csv
data/modular_runs/latest/modular_details.csv
data/modular_runs/latest/analysis.json
data/modular_runs/latest/MODULAR_REPORT.md
```

The manifest records the selected dimensions under:

```json
"experiment": {
  "selection": {
    "chunker": "...",
    "embedding": "...",
    "vector_store": "...",
    "index_type": "...",
    "retrieval_method": "...",
    "reranker": "..."
  }
}
```
