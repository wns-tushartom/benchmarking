# WNS NVIDIA RAG package apply commands

## Apply ZIP locally

Unzip this package at the WNS benchmarking workspace root that contains `Retreiver/`.

```bash
cd /path/to/wns-benchmarking
unzip -o wns-nvidia-rag-project-smiley-20260605.zip
```

Windows CMD equivalent:

```cmd
cd C:\path\to\General_Components\QA_Text
tar -xf wns-nvidia-rag-project-smiley-20260605.zip -C .
```

## Verify after applying

```bash
cd Retreiver
python3 scripts/benchmark_cli.py validate
python3 -m pytest tests/test_modular_benchmark.py -q
python3 -m py_compile scripts/serve_benchmark_dashboard.py scripts/check_nvidia_rag_pipeline.py scripts/run_nvidia_rag_pipeline_smoke.py scripts/ingest_nvidia_rag_documents.py scripts/run_nvidia_rag_benchmark.py scripts/prepare_benchmark_input_mineru.py scripts/run_chunking_pipeline.py scripts/run_complete_pipeline.py scripts/analyze_reranker_lift.py scripts/wns_vm_adapter_service.py
node --check web/app.js
```

## Push to GitHub

```bash
git status
git add README_WNS_BENCHMARKING_WORKSPACE.md NVIDIA_RAG_PIPELINE.md QWEN_RERANKER_DIAGNOSIS.md WNS_NVIDIA_RAG_APPLY_COMMANDS.md setup_all_on_vm.sh setup_nvidia_rag_pipeline_on_vm.sh Retreiver/.env.example Retreiver/VM_ADAPTER_PORT_MAP.md Retreiver/docker-compose.benchmark.yml Retreiver/configs/benchmark.local.json Retreiver/requirements-benchmark.txt Retreiver/requirements-mineru.txt Retreiver/benchmarking Retreiver/scripts Retreiver/tests Retreiver/web Retreiver/DESIGN.md
git commit -m "Add NVIDIA RAG Blueprint lane to Project Smiley benchmark"
git push
```

## Pull and run on VM

```bash
git pull
export NGC_API_KEY="nvapi-..."
bash setup_all_on_vm.sh --host 10.31.236.170 --with-nvidia-rag --nvidia-rag-zip rag-main.zip --skip-model-smoke
cd Retreiver
python3 scripts/check_nvidia_rag_pipeline.py --env-file .env.project-smiley-nvidia
python3 scripts/serve_benchmark_dashboard.py 5009 0.0.0.0
```

Open:

```text
http://10.31.236.170:5009
```

## Re-run PDF extraction and chunking

Recommended for client-facing benchmark consistency: rerun all PDFs, then regenerate every chunking sheet.

```bash
cd ~/benchmarking/Retreiver

.venv-vm/bin/python -m pip install -r requirements-benchmark.txt
.venv-vm/bin/python -m pip install -r requirements-mineru.txt

.venv-vm/bin/python scripts/run_chunking_pipeline.py --mode all
```

If only new PDFs were added to `data/pdfs` and you want a faster extraction pass, append only missing PDFs but still regenerate all chunking sheets:

```bash
cd ~/benchmarking/Retreiver

.venv-vm/bin/python scripts/run_chunking_pipeline.py --mode missing
```

Use `--mode chunk-only` only when `data/benchmark_input.csv` is already correct and you just need to rebuild `data/chunking_methods_output_v2.xlsx`.

## Meaningful full-pipeline benchmark from VM

Use this instead of separate manual test buttons. It checks requirements first. If anything required is missing, it prints the exact missing item and exits before running.

Selected one-combination run:

```bash
cd ~/benchmarking/Retreiver

.venv-vm/bin/python scripts/run_complete_pipeline.py \
  --sheets fixed_tok1200_ov150 \
  --embeddings jina_v3 \
  --stores Qdrant \
  --rerankers qwen3_4b_rerank \
  --groundtruth data/groundtruth/groundtruth_500.csv \
  --top-k 10 \
  --fresh-run
```

All live-supported chunker, embedding, DB, and OSS reranker combinations:

```bash
cd ~/benchmarking/Retreiver

.venv-vm/bin/python scripts/run_complete_pipeline.py \
  --sheets all \
  --embeddings all \
  --stores all \
  --rerankers all \
  --groundtruth data/groundtruth/groundtruth_500.csv \
  --top-k 10 \
  --fresh-run
```

Qwen-vs-no-reranker outputs:

```text
data/reranker_analysis/reranker_lift_summary.csv
data/reranker_analysis/reranker_lift_details.csv
data/reranker_analysis/qwen_vs_none_report.json
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

## NVIDIA current-data benchmark from VM

```bash
cd Retreiver
python3 scripts/ingest_nvidia_rag_documents.py \
  --env-file .env.project-smiley-nvidia \
  --path data/pdfs \
  --collection multimodal_data \
  --limit 5 \
  --batch-size 2 \
  --create-collection

python3 scripts/run_nvidia_rag_benchmark.py \
  --env-file .env.project-smiley-nvidia \
  --groundtruth data/groundtruth/groundtruth_500.csv \
  --collection multimodal_data \
  --limit 25 \
  --top-k 10 \
  --reranker-top-k 5
```
