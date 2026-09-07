# Candidate Retrieval/Reranker VM Runbook

**Scope:** the isolated `retrieval-reranker-candidates` lane only. Never write to `data/modular_runs/latest` or alter the accepted 180-combination configuration.

Run every command below from the **work laptop VM shell**, one at a time. Stop and inspect the output before moving on. The overlay contains source/config/tests only: no data, result artifacts, model weights, or secrets.

## 1. Verify the deployed checkout and preserve existing generated data

```bash
cd ~/benchmarking/Retreiver && git merge-base --is-ancestor 853cb7c HEAD && echo BASE_OK || echo BASE_MISMATCH
```

Continue only on `BASE_OK`. Then record, but do not stage/reset/clean, any existing worktree changes:

```bash
cd ~/benchmarking/Retreiver && git status --short
```

## 2. Verify and apply the overlay

From the directory containing `Retreiver-retrieval-reranker-candidates-overlay.zip`:

```bash
python3 -m zipfile -t Retreiver-retrieval-reranker-candidates-overlay.zip
```

```bash
cd ~/benchmarking && unzip -o ~/Downloads/Retreiver-retrieval-reranker-candidates-overlay.zip
```

```bash
cd ~/benchmarking/Retreiver && python3 -m pip install -r requirements.txt
```

`transformers==4.57.6` is intentional. Do **not** upgrade to Transformers 5+: GTE remote code is not compatible with it.

## 3. Confirm the running adapter and candidate model identities

First locate the existing supervisor/process; do not start a duplicate service:

```bash
ps auxww | grep -E '[w]ns_vm_adapter_service|[u]vicorn.*5000'
```

Restart **that existing service manager** after the overlay is applied, then prove the service is healthy:

```bash
curl -fsS http://127.0.0.1:5000/health
```

The health response must identify both exact models:

- `nvidia/llama-nemotron-rerank-1b-v2`
- `Alibaba-NLP/gte-reranker-modernbert-base`

Confirm the candidate endpoint environment variables are set in the service environment/configuration:

```bash
printf 'NEMOTRON_RERANK_URL=%s\nGTE_MODERNBERT_RERANK_URL=%s\nGTE_EMBEDDING_URL=%s\n' "$NEMOTRON_RERANK_URL" "$GTE_MODERNBERT_RERANK_URL" "$GTE_EMBEDDING_URL"
```

## 4. Smoke the two new rerankers directly

```bash
curl -fsS -X POST http://127.0.0.1:5000/rerank/nemotron -H 'Content-Type: application/json' -d '{"query":"How do I change my booking?","documents":["You can modify an eligible booking online.","Baggage rules apply at check-in."],"top_k":2}'
```

```bash
curl -fsS -X POST http://127.0.0.1:5000/rerank/gte-modernbert -H 'Content-Type: application/json' -d '{"query":"How do I change my booking?","documents":["You can modify an eligible booking online.","Baggage rules apply at check-in."],"top_k":2}'
```

Each response must report its selected exact model and two scored results. A missing endpoint, model-load failure, or substituted model is a hard stop—do not run the matrix.

## 5. Validate config, then run a 3-query isolated smoke

```bash
cd ~/benchmarking/Retreiver && PYTHONPATH=. python scripts/benchmark_cli.py validate configs/benchmark.retrieval-reranker-candidates.json
```

```bash
cd ~/benchmarking/Retreiver && RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)" && PYTHONPATH=. python scripts/benchmark_cli.py run configs/benchmark.retrieval-reranker-candidates.json --output-dir "data/modular_runs/retrieval-reranker-candidates/$RUN_ID" --limit-queries 3 --max-runs 1 --chunker entity_heuristic_w6 --embedding gte_multilingual_base --vector-store FAISS --retrieval-method 'BM25 + GTE Dense + RRF' --reranker nemotron_rerank_1b
```

Inspect only the run directory printed by the command. Require a completed manifest and retrieval provenance with `bm25_rank`, `bm25_score`, `dense_rank`, `dense_score`, and `rrf_score`. Do not copy it into `latest`.

## 6. Run the full candidate matrix only after all gates pass

```bash
cd ~/benchmarking/Retreiver && RUN_ID="candidate-$(date +%Y%m%d-%H%M%S)" && PYTHONPATH=. python scripts/benchmark_cli.py run configs/benchmark.retrieval-reranker-candidates.json --output-dir "data/modular_runs/retrieval-reranker-candidates/$RUN_ID"
```

Retain the resulting immutable candidate directory and its manifest. Report it as a **candidate comparison**, never as the official 180 benchmark.
