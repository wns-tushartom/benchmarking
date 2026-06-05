# NVIDIA RAG Blueprint integration for Project Smiley

This adds NVIDIA's RAG Blueprint as a separate evidence lane inside the Project Smiley Integrated RAG Pipeline workspace.

## What this lane is for

Use it to prove an end-to-end NVIDIA pipeline next to the existing benchmark stack:

- NVIDIA ingestion through `ingestor-server` and NV-Ingest
- NVIDIA embeddings and reranker through NVIDIA hosted endpoints or self-hosted NIMs
- Elasticsearch/LanceDB/Milvus-compatible NVIDIA retrieval flow
- RAG generation through `/v1/generate`
- Agentic RAG via the blueprint's `agentic` request flag
- UI evidence and JSON artifacts under `data/nvidia_rag/`

This does not replace the existing Qdrant/PGVector/Weaviate matrix. It is a separate baseline/lane for end-to-end NVIDIA Blueprint behavior.

## Port map

Project Smiley keeps the WNS allowed port range:

- `5000`: existing WNS model adapter service
- `5001`: Qdrant HTTP
- `5002`: Qdrant gRPC
- `5003`: PGVector/Postgres
- `5004`: Weaviate HTTP
- `5005`: Weaviate gRPC
- `5006`: NVIDIA RAG server, external URL `http://VM_HOST:5006/v1`
- `5007`: NVIDIA ingestor server, external URL `http://VM_HOST:5007/v1`
- `5008`: NVIDIA reference frontend

Internal NVIDIA container ports remain unchanged: rag-server `8081`, ingestor `8082`, frontend `3000`.

## One-click setup

From the top-level WNS workspace on the VM:

```bash
export NGC_API_KEY="nvapi-..."
bash setup_all_on_vm.sh --host <VM_IP> --with-nvidia-rag --nvidia-rag-zip rag-main.zip --skip-model-smoke
```

Or if `rag-main` is already extracted:

```bash
export NGC_API_KEY="nvapi-..."
bash setup_all_on_vm.sh --host <VM_IP> --with-nvidia-rag --nvidia-rag-root /path/to/rag-main --skip-model-smoke
```

The NVIDIA-only setup can also be run directly:

```bash
export NGC_API_KEY="nvapi-..."
bash setup_nvidia_rag_pipeline_on_vm.sh --host <VM_IP> --rag-zip rag-main.zip --force-env
```

## Generated files

- `Retreiver/.env.project-smiley-nvidia`
- `Retreiver/data/nvidia_rag/health.json`
- `Retreiver/data/nvidia_rag/smoke_latest.json`
- `Retreiver/data/nvidia_rag/ingestion_latest.json`
- `NVIDIA_RAG_SETUP_RESULT.txt`

## Health check

```bash
cd Retreiver
python3 scripts/check_nvidia_rag_pipeline.py --env-file .env.project-smiley-nvidia
```

## Ingest a small sample first

```bash
cd Retreiver
python3 scripts/ingest_nvidia_rag_documents.py \
  --env-file .env.project-smiley-nvidia \
  --path data/pdfs \
  --collection multimodal_data \
  --limit 5 \
  --batch-size 2 \
  --create-collection
```

Add `--poll` only when you want the command to wait for ingestion tasks to finish.

## Retrieval smoke

```bash
cd Retreiver
python3 scripts/run_nvidia_rag_pipeline_smoke.py \
  --env-file .env.project-smiley-nvidia \
  --mode search \
  --query "refund old ticket and issue new ticket" \
  --collection multimodal_data \
  --top-k 10 \
  --reranker-top-k 5
```

For generation:

```bash
python3 scripts/run_nvidia_rag_pipeline_smoke.py \
  --env-file .env.project-smiley-nvidia \
  --mode generate \
  --query "refund old ticket and issue new ticket"
```

## Dashboard controls

Start the dashboard as usual and open the **NVIDIA RAG** tab. It can:

- check NVIDIA health
- run a retrieval smoke query
- upload a small batch of PDFs to the NVIDIA ingestor
- show NVIDIA JSON artifacts next to existing benchmark evidence

## Remaining real-world dependencies

The script can set up files, ports, compose env, and service commands. It cannot manufacture NVIDIA credentials or provider access. Real startup requires:

- valid `NGC_API_KEY`
- Docker access on the VM
- NGC image pull access
- enough disk for NVIDIA containers and vector DB data
- NVIDIA hosted endpoint access, unless you switch the blueprint to self-hosted NIMs
