# WNS Benchmarking Workspace

Created as a single working directory for the WNS RAG/retrieval benchmarking work.

## Workspace layout

```text
/home/fate/.openclaw/workspace/wns-benchmarking/
├── Retreiver/
│   ├── NOTES.md
│   ├── BENCHMARK_PLAN.md
│   ├── CHUNKING_METHODS.md
│   ├── main.py
│   ├── requirements.txt
│   ├── config/QA_Text/config.yaml
│   ├── data/
│   │   ├── pdfs/
│   │   ├── benchmark_input.csv
│   │   ├── chunking_methods_output.xlsx
│   │   ├── chunking_methods_output_v2.xlsx
│   │   ├── baseline_benchmark_results.csv
│   │   ├── qa_text_test.csv
│   │   └── query.csv
│   ├── scripts/
│   │   ├── prepare_benchmark_input.py
│   │   ├── apply_chunking_methods.py
│   │   ├── apply_candidate_chunking_methods.py
│   │   ├── run_baseline_benchmark.py
│   │   ├── compare_chunking_recall.py
│   │   ├── verify_benchmark_input.py
│   │   ├── verify_chunking_output.py
│   │   └── verify_all_chunking_methods.py
│   ├── source/
│   └── tests/
├── setup_all_on_vm.sh
├── setup_nvidia_rag_pipeline_on_vm.sh
├── NVIDIA_RAG_PIPELINE.md
└── references/
    └── smiley_vector_db_subset/
        ├── readme.txt
        ├── requirements.txt
        ├── HNSW/
        │   ├── analyze_results.py
        │   ├── config/config.py
        │   └── src/
        │       ├── MD_upsert_vectors_HNSW.py
        │       └── MD_benchmark_search_HNSW.py
        └── HNSW_Single_db_optimizations/
            ├── config/config.py
            ├── Qdrant/
            ├── PGVector/
            ├── Weaviate/
            └── Milvus/
```

## Current verified assets

### NORA/source data

PDFs are in:

```text
Retreiver/data/pdfs/
```

### Benchmark input schema

Created file:

```text
Retreiver/data/benchmark_input.csv
```

Required columns:

```text
id,pdf_name,paragraph
```

### Chunking workbook

Latest workbook:

```text
Retreiver/data/chunking_methods_output_v2.xlsx
```

Sheets:

- `original_input` — 61 rows
- `entity_heuristic_w6` — 214 rows
- `entity_heuristic_w5` — 222 rows
- `entity_heuristic_w4` — 227 rows
- `Heading_sections_l2` — 127 rows
- `semantic_split` — 122 rows
- `fixed_tok1200_ov150` — 56 rows

All sheets use exact columns:

```text
id,pdf_name,paragraph
```

## Official benchmark dimensions from meeting

### Chunking

- `entity_heuristic_w6`
- `entity_heuristic_w5`
- `entity_heuristic_w4`
- `Heading_sections_l2`

Candidate methods to compare by recall:

- `semantic_split`
- `fixed_tok1200_ov150`

### Embeddings

- `jina_v3`
- `gte_multilingual_base`
- `openai_text-embedding-3-large`

### Vector databases

- `Qdrant`
- `PGVector`
- `Weaviate`

Keep `Milvus` as baseline/reference because current Retreiver implementation already uses Milvus.

### Index

- `HNSW`

Current Retreiver code uses `IVF_FLAT`; HNSW work should adapt from the Smiley reference repo.

### Retrieval

- `Cosine Similarity`

### Reranking

- `Amazon Rerank v1`
- `Qwen3:4B Rerank`
- `bge-reranker-base`

### NVIDIA RAG Blueprint lane

A separate end-to-end NVIDIA Blueprint lane is available for Project Smiley. It keeps the existing benchmark matrix intact and adds NVIDIA ingestion/retrieval/reranking/generation evidence under `Retreiver/data/nvidia_rag/`.

Key files:

- `setup_nvidia_rag_pipeline_on_vm.sh`
- `Retreiver/scripts/check_nvidia_rag_pipeline.py`
- `Retreiver/scripts/ingest_nvidia_rag_documents.py`
- `Retreiver/scripts/run_nvidia_rag_pipeline_smoke.py`
- `NVIDIA_RAG_PIPELINE.md`

Use ports `5006` for NVIDIA rag-server, `5007` for NVIDIA ingestor, and `5008` for the NVIDIA reference frontend.

## Reference repo subset

The `references/smiley_vector_db_subset/` folder contains only the useful parts from `Smiley_Vector_DB.zip`:

- HNSW upsert/search scripts
- Qdrant / PGVector / Weaviate / Milvus HNSW optimization scripts
- HNSW tuning config
- Docker compose files for target DBs

Do not blindly copy the whole reference repo into production code. Use it as a pattern source.

## Immediate next step

Start with a small, measurable MVP:

1. Verify the current workspace files.
2. Inspect query ground-truth files: `qa_text_test.csv` and `query.csv`.
3. Create/finish `scripts/compare_chunking_recall.py`.
4. Compare `semantic_split` vs `fixed_tok1200_ov150` using the same embedding/vector DB stack.
5. Pick the better candidate by `Recall@5` / `Recall@10`.
6. Then build HNSW latency MVP for Qdrant, PGVector, Weaviate using the reference scripts.

## Useful Windows commands for Tushar / Copilot

From repo root:

```powershell
cd "C:\Users\U481019\OneDrive - WNS\Documents\benchmarking\Dev\General_Components\QA_Text\Retreiver"
python scripts\verify_benchmark_input.py
python scripts\verify_chunking_output.py
python scripts\verify_all_chunking_methods.py
```

Generate candidate chunking workbook:

```powershell
python scripts\apply_candidate_chunking_methods.py
```

Baseline benchmark:

```powershell
python scripts\run_baseline_benchmark.py --limit 10
```

## Notes

- Folder spelling remains `Retreiver` because repo path uses that spelling.
- Documentation should use correct spelling: `Retriever`.
- Do not commit API keys or endpoint URLs; use `.env` or environment variables.
