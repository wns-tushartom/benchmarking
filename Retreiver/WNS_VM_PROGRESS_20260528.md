# WNS VM Progress Update — 2026-05-28

## Verified today

The VM model adapter side is now working on GPU.

Verified endpoint status:

- `/health`: OK, device `cuda`
- `/embed/gte`: OK
  - model: `Alibaba-NLP/gte-multilingual-base`
  - dimensions: 768
- `/embed/jina`: OK
  - model: `jinaai/jina-embeddings-v3`
  - dimensions: 1024
- `/rerank/bge`: OK
  - model: `BAAI/bge-reranker-base`
- `/rerank/qwen`: OK
  - model: `Qwen/Qwen3-Reranker-4B`

Benchmark config validation:

- experiment: `wns-rag-benchmark-selectable`
- matrix count: 135 combinations

## Fixes applied on VM and now folded into repo files

### PyTorch / CUDA

Problem:

- VM driver reports CUDA 12.8 through `nvidia-smi`.
- Initial PyTorch wheel was `2.12.0+cu130`.
- CUDA failed because PyTorch CUDA 13.0 expected a newer driver.

Fix:

- Install PyTorch CUDA 12.8 wheels:
  - `torch==2.11.0+cu128` or latest available from PyTorch cu128 index
  - `torchvision` cu128
  - `torchaudio` cu128

### Transformers / sentence-transformers

Problem:

- `transformers>=5` caused `Alibaba-NLP/gte-multilingual-base` encode failure with corrupted `position_ids` / RoPE indexing.

Fix:

- Pin `transformers==4.57.6`
- Pin `sentence-transformers==4.1.0`
- Pin `setuptools<82`

### Jina

Problem:

- Jina remote-code cache was incomplete and missing files like `mlp.py` / `rotary.py`.
- Jina also needed `einops`.

Fix:

- Clear Jina Hugging Face module cache when needed.
- Add `einops` to requirements.

### Qwen reranker

Problem:

- Qwen loaded on GPU, but API failed because batch prediction needs a padding token.

Fix:

- Set `tokenizer.pad_token = tokenizer.eos_token`.
- Set model config `pad_token_id`.
- Predict Qwen rerank pairs one at a time.

## Current blocker

Docker permission is still the main blocker for full benchmark execution.

Docker is installed, but user `U481019` cannot access `/var/run/docker.sock`.

Need admin/Karthik to run:

```bash
sudo usermod -aG docker U481019
```

Then logout/login and verify:

```bash
groups
docker ps
```

Until Docker access is fixed, these cannot start:

- Qdrant
- PGVector
- Weaviate

## Current working path while waiting for Docker access

We can continue with:

- frontend VM/model health UI
- matrix selector
- reranker cards/options in UI
- artifact schema and saved retrieval/rerank output design
- small model endpoint smoke tests
- progress and timeline documentation

We cannot run full 135-combination vector DB benchmark until Docker access is granted.

## Next steps after Docker access

Run:

```bash
cd ~/benchmarking
VM_IP=$(hostname -I | awk '{print $1}')
bash setup_all_on_vm.sh --host "$VM_IP" --force-env
```

Verify:

```bash
docker ps
curl -s http://127.0.0.1:5000/health
curl -s http://127.0.0.1:5001/
curl -s http://127.0.0.1:5004/v1/meta
```

Then run a small selected smoke benchmark before full matrix:

- 1 chunker
- 1 embedding
- 1 vector DB
- BGE reranker
- 2–5 questions

## Raghu-aligned execution rule

Backend batch pipeline remains the source of truth.

UI should run in parallel, but it should act as the cockpit:

- health/status
- matrix selection
- run progress
- artifact browsing
- comparison/reporting

Heavy model/vector DB work should remain in backend jobs that save retrieval and rerank artifacts.
