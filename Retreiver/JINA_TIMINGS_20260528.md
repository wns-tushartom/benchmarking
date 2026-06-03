# WNS Jina v3 ingestion timings — 2026-05-28

Source: VM terminal output pasted in Telegram during Jina embedding/upsert run.

Embedding model:

```text
jina_v3
vector dim: 1024
stores: Qdrant, PGVector, Weaviate
```

## Captured timings

### entity_heuristic_w6

```text
chunks: 16873
embedding: jina_v3
```

DB timings:

```text
Qdrant:   OK, hits=5, seconds=17.500103
PGVector: OK, hits=5, seconds=27.461024
Weaviate: OK, hits=5, seconds=28.473279
```

Embedding timing was not included in the pasted excerpt. It should be recovered from the run summary/log if needed.

### entity_heuristic_w5

```text
chunks: 16965
embedding_seconds: 349.60
cache: data/embedding_cache/entity_heuristic_w5_jina_v3_vectors.npy
```

DB timings:

```text
Qdrant:   OK, hits=5, seconds=17.513055
PGVector: OK, hits=5, seconds=28.047891
Weaviate: OK, hits=5, seconds=28.653112
```

### entity_heuristic_w4

```text
chunks: 17099
embedding_seconds: 356.54
cache: data/embedding_cache/entity_heuristic_w4_jina_v3_vectors.npy
```

DB timings:

```text
Qdrant:   OK, hits=5, seconds=17.775777
PGVector: OK, hits=5, seconds=28.151388
Weaviate: OK, hits=5, seconds=29.154214
```

### semantic_split

```text
chunks: 19338
embedding_seconds: 397.91
cache: data/embedding_cache/semantic_split_jina_v3_vectors.npy
```

DB timings:

```text
Qdrant:   OK, hits=5, seconds=20.029745
PGVector: OK, hits=5, seconds=36.200676
Weaviate: OK, hits=5, seconds=32.701489
```

Final run summary:

```text
summary: data/db_ingestion_runs/20260528_114128/summary.csv
failures: 0
```

## Summary CSV captured

Run path:

```text
data/db_ingestion_runs/20260528_114128/summary.csv
```

Final status:

```text
failures: 0
```

Rows captured from summary CSV:

```text
fixed_tok1200_ov150 | chunks=5580 | Jina dim=1024
  run: data/db_ingestion_runs/20260528_113808/summary.csv
  Qdrant:   upsert=6.106117s, total=6.130337s, hits=5
  PGVector: upsert=6.648683s, total=6.663988s, hits=5
  Weaviate: upsert=10.029141s, total=10.039091s, hits=5

Heading_sections_l2 | chunks=7617 | Jina dim=1024
  Qdrant:   upsert=8.050364s, total=8.081434s, hits=5
  PGVector: upsert=9.196549s, total=9.212328s, hits=5
  Weaviate: upsert=13.178852s, total=13.188690s, hits=5

entity_heuristic_w6 | chunks=16873 | Jina dim=1024
  Qdrant:   upsert=17.438484s, total=17.500103s, hits=5
  PGVector: upsert=27.214139s, total=27.461024s, hits=5
  Weaviate: upsert=28.461036s, total=28.473279s, hits=5

entity_heuristic_w5 | chunks=16965 | Jina dim=1024
  Qdrant:   upsert=17.449540s, total=17.513055s, hits=5
  PGVector: upsert=27.800849s, total=28.047891s, hits=5
  Weaviate: upsert=28.640812s, total=28.653112s, hits=5

entity_heuristic_w4 | chunks=17099 | Jina dim=1024
  Qdrant:   upsert=17.712144s, total=17.775777s, hits=5
  PGVector: upsert=27.901324s, total=28.151388s, hits=5
  Weaviate: upsert=29.141266s, total=29.154214s, hits=5

semantic_split | chunks=19338 | Jina dim=1024
  Qdrant:   upsert=19.954506s, total=20.029745s, hits=5
  PGVector: upsert=35.917616s, total=36.200676s, hits=5
  Weaviate: upsert=32.688577s, total=32.701489s, hits=5
```

Still missing from the timing file if needed:

```text
Jina embedding_seconds for fixed_tok1200_ov150
Jina embedding_seconds for Heading_sections_l2
Jina embedding_seconds for entity_heuristic_w6
```

The DB upsert/search timings for all six Jina chunkers are now captured.

## Benchmarking note

These timings are useful as ingestion/runtime benchmark evidence, but they are not retrieval-quality metrics. They should be reported separately from Recall/MRR/nDCG.
