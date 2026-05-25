@echo off
setlocal EnableExtensions
set VM_HOST=%~1
if "%VM_HOST%"=="" (
  echo Usage: scripts\windows\Configure-LocalFrontendForVM.cmd VM_HOST_OR_IP
  exit /b 2
)
if not exist scripts\benchmark_cli.py (
  echo Run this from the Retreiver project root.
  exit /b 2
)
(
  echo WNS_PUBLIC_HOST=%VM_HOST%
  echo JINA_EMBEDDING_MODE=vm_remote
  echo JINA_EMBEDDING_URL=http://%VM_HOST%:5000/embed/jina
  echo JINA_API_KEY=
  echo JINA_EMBEDDING_MODEL=jina-embeddings-v3
  echo GTE_EMBEDDING_MODE=vm_remote
  echo GTE_EMBEDDING_URL=http://%VM_HOST%:5000/embed/gte
  echo HF_TOKEN=
  echo GTE_EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base
  echo QDRANT_URL=http://%VM_HOST%:5001
  echo PGVECTOR_DSN=postgresql://wns:wns_password@%VM_HOST%:5003/wns_benchmark
  echo DATABASE_URL=postgresql://wns:wns_password@%VM_HOST%:5003/wns_benchmark
  echo WEAVIATE_URL=http://%VM_HOST%:5004
  echo QWEN_RERANK_URL=
  echo AWS_REGION=
  echo AWS_DEFAULT_REGION=
  echo OPENAI_API_KEY=
) > .env

echo Wrote .env for VM %VM_HOST%.
echo Now run: scripts\windows\Start-WnsDashboard-VM.cmd
endlocal
