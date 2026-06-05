# WNS NVIDIA RAG package apply commands

## Apply ZIP locally

Unzip this package at the WNS benchmarking workspace root that contains `Retreiver/`.

```bash
cd /path/to/wns-benchmarking
unzip -o wns-nvidia-rag-project-smiley-20260604.zip
```

Windows CMD equivalent:

```cmd
cd C:\path\to\General_Components\QA_Text
tar -xf wns-nvidia-rag-project-smiley-20260604.zip -C .
```

## Verify after applying

```bash
cd Retreiver
python3 scripts/benchmark_cli.py validate
python3 -m pytest tests/test_modular_benchmark.py -q
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/check_nvidia_rag_pipeline.py scripts/run_nvidia_rag_pipeline_smoke.py scripts/ingest_nvidia_rag_documents.py
node --check web/app.js
```

## Push to GitHub

```bash
git status
git add README_WNS_BENCHMARKING_WORKSPACE.md NVIDIA_RAG_PIPELINE.md WNS_NVIDIA_RAG_APPLY_COMMANDS.md setup_all_on_vm.sh setup_nvidia_rag_pipeline_on_vm.sh Retreiver/.env.example Retreiver/VM_ADAPTER_PORT_MAP.md Retreiver/docker-compose.benchmark.yml Retreiver/configs/benchmark.local.json Retreiver/requirements-benchmark.txt Retreiver/benchmarking Retreiver/scripts Retreiver/tests Retreiver/web Retreiver/DESIGN.md
git commit -m "Add NVIDIA RAG Blueprint lane to Project Smiley benchmark"
git push
```

## Pull and run on VM

```bash
git pull
export NGC_API_KEY="nvapi-..."
bash setup_all_on_vm.sh --host <VM_IP> --with-nvidia-rag --nvidia-rag-zip rag-main.zip --skip-model-smoke
cd Retreiver
python3 scripts/check_nvidia_rag_pipeline.py --env-file .env.project-smiley-nvidia
python3 scripts/serve_benchmark_dashboard.py 5009 0.0.0.0
```

Open:

```text
http://<VM_IP>:5009
```

## NVIDIA smoke from VM

```bash
cd Retreiver
python3 scripts/run_nvidia_rag_pipeline_smoke.py \
  --env-file .env.project-smiley-nvidia \
  --mode search \
  --collection multimodal_data \
  --query "refund old ticket and issue new ticket" \
  --top-k 10 \
  --reranker-top-k 5
```
