# All-Method Portfolio and VM Adapter Control Design

**Date:** 2026-08-20
**Status:** Approved by Tushar

## Goal

Expose every listed retrieval-pipeline method as an isolated 2,160-combination candidate portfolio while enforcing Karthik's maximum of 250 combinations per executable batch, and add safe homepage controls for the VM's 20 allowed ports.

## Non-negotiable boundaries

- Preserve `configs/benchmark.local.json` and the accepted 180-result artifacts.
- Never write candidate portfolio rows into `data/modular_runs/latest`.
- Never present portfolio results as accepted official Metrics, Recommendations, or Evidence.
- No executable batch may contain more than 250 combinations.
- The dashboard may start only predefined services; it may not accept arbitrary commands, paths, models, hosts, or ports.
- The dashboard may stop only a process it started and whose PID/command identity still matches its receipt.
- Model services bind to `127.0.0.1`; the dashboard remains the operator-facing control surface.

## Method inventory

### Chunkers — 6

1. `entity_heuristic_w6`
2. `entity_heuristic_w5`
3. `entity_heuristic_w4`
4. `Heading_sections_l2`
5. `fixed_tok1200_ov150`
6. `semantic_split`

### Embeddings — 6

1. `jina_v3`
2. `gte_multilingual_base`
3. `openai_text-embedding-3-large`
4. `nemotron_3_embed_1b_bf16`
5. `nemotron_3_embed_1b_nvfp4`
6. `nemotron_3_embed_8b_bf16`

### Stores/indexes — 5 compatible pairs

1. Qdrant + HNSW
2. PGVector + HNSW
3. Weaviate + HNSW
4. FAISS + HNSW
5. TurboVec + TurboQuant 4-bit

The matrix generator must use explicit store/index compatibility. It must not cross TurboVec with HNSW or the network/in-process HNSW stores with TurboQuant.

### Retrieval — 2

1. `Dense Cosine`
2. `BM25 + Dense + RRF`

`Cosine Similarity` from the protected official config and `Dense Cosine` are canonical equivalents. The portfolio uses only `Dense Cosine` and does not create duplicate labels for the same operation.

### Reranker states — 6

1. `none`
2. `Amazon Rerank v1`
3. `bge-reranker-base`
4. `Qwen3:4B Rerank`
5. `nemotron_rerank_1b`
6. `gte_modernbert_base`

## Portfolio size and deterministic batching

The compatibility-filtered product is:

`6 chunkers × 6 embeddings × 5 compatible store/index pairs × 2 retrieval methods × 6 reranker states = 2,160 combinations`.

The planner creates 12 immutable batches of 180 combinations:

- one embedding per batch;
- all six chunkers;
- all five compatible store/index pairs;
- both retrieval methods;
- one of two reranker groups:
  - Group A: `none`, `bge-reranker-base`, `Qwen3:4B Rerank`
  - Group B: `Amazon Rerank v1`, `nemotron_rerank_1b`, `gte_modernbert_base`

Every batch is identified by portfolio hash, embedding ID, reranker-group ID, and deterministic batch ID. The planner verifies exact coverage, no duplicate combination IDs, and a batch size between 1 and 250.

## Execution and artifacts

Add a dedicated portfolio configuration and CLI operations to plan, inspect, run, resume, and receipt one batch. A full-portfolio action queues batches sequentially; it does not create one oversized run.

Outputs live only under:

`data/modular_runs/all-methods-portfolio/<portfolio-id>/<batch-id>/`

Each batch writes its own config snapshot, combination manifest, provider-readiness receipt, summary, details, and completion receipt. The portfolio root writes a plan and, only after all 12 batches validate, a portfolio receipt and aggregate candidate summary. Aggregation never copies data into official paths.

The dashboard source scanner treats `all-methods-portfolio` as excluded from official evidence. Candidate portfolio results have a separate reader and explicit `not_accepted` status.

## TurboVec

Add `TurboVecVectorStoreAdapter` using `IdMapIndex` and 4-bit quantization. Normalize corpus and query vectors, validate finite/dimension-consistent input, retain stable chunk IDs, and persist index plus chunk metadata with hashes. TurboVec remains a candidate embedded index, never a vector-database service or accepted replacement.

## VM port manifest

Create one allowlisted manifest for ports `5000–5019`:

| Port | Service |
|---:|---|
| 5000 | Jina embedding adapter |
| 5001 | GTE embedding adapter |
| 5002 | BGE reranker adapter |
| 5003 | PGVector/Postgres |
| 5004 | Weaviate HTTP |
| 5005 | Weaviate gRPC |
| 5006 | Qwen reranker adapter |
| 5007 | Nemotron reranker adapter |
| 5008 | GTE ModernBERT reranker adapter |
| 5009 | Reserved Jupyter/upload bridge |
| 5010 | Nemotron Embed 1B BF16 vLLM endpoint |
| 5011 | Benchmark dashboard |
| 5012 | Nemotron Embed 1B NVFP4 vLLM endpoint |
| 5013 | Nemotron Embed 8B BF16 vLLM endpoint |
| 5014 | Reserved spare |
| 5015 | Qdrant gRPC |
| 5016 | NVIDIA RAG server |
| 5017 | NVIDIA ingestor |
| 5018 | NVIDIA frontend |
| 5019 | Qdrant HTTP |

OpenAI and Amazon are external services. FAISS, TurboVec, BM25, and the identity reranker are in-process and need no port.

## Service manager

Implement a manifest-driven manager with:

- health/status for every slot;
- fixed launch command templates;
- PID, start-time, command-fingerprint, log-path, and health receipts under `data/adapter_runtime`;
- collision detection before launch;
- start, retry, and stop-owned-process operations;
- Docker Compose starts for Qdrant, PGVector, and Weaviate;
- adapter-profile starts for local Python model services;
- configured vLLM starts for Nemotron embedding services;
- explicit `reserved`, `external`, `in_process`, `unconfigured`, `starting`, `healthy`, `unhealthy`, and `blocked` states;
- no shell interpolation and no arbitrary user command execution.

Mutating API actions require both `WNS_ENABLE_ADAPTER_CONTROL=1` and an operator bearer token supplied through the homepage and held in session storage. Status remains readable without the token. Startup failures return bounded, redacted log excerpts.

## Homepage behavior

Expand the Overview runtime-services panel into a 20-slot adapter-port control surface. Show service name, method role, port, status, endpoint/profile, and latest error. Provide per-service Start/Retry/Stop controls only when allowed. Add `Start required adapters` for the selected portfolio batch. Never show a Start button for external, in-process, reserved, or unconfigured services.

The Run page shows:

- 2,160 planned candidate combinations;
- 12 batches of 180;
- hard 250 cap;
- selected batch and required adapters;
- preflight blockers;
- sequential batch progress;
- candidate/not-accepted labeling.

Loading, empty, blocked, starting, healthy, and failed states must be explicit. Desktop and mobile must have no horizontal page overflow.

## Error handling and safety

- Fail closed on unknown methods, services, ports, batch IDs, config hashes, output paths, symlinked runtime files, duplicate PIDs, occupied ports, missing binaries, missing credentials, model identity mismatch, vector dimension mismatch, or batch count above 250.
- Never kill a process solely because it occupies a configured port.
- Never display tokens, credentials, absolute private paths, or full environment values.
- A failed adapter or batch remains diagnosable and resumable without contaminating completed batch receipts.

## Verification

- Unit tests for inventory, compatibility filtering, exact 2,160 coverage, deterministic 12×180 batching, cap rejection, duplicate detection, output isolation, TurboVec adapter behavior, port-manifest validation, process ownership, collision handling, authorization, and API redaction.
- Red-green TDD for every production behavior.
- Focused and broad regression suites.
- Python compilation, JavaScript syntax check, shell syntax check, JSON validation, and `git diff --check`.
- Browser QA at desktop and mobile viewports, including offline/starting/healthy/error states and no console/network errors.
- Official matrix remains 180 and protected official artifact hashes remain unchanged.
