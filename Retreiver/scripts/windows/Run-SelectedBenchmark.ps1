<#
Run one selected benchmark combination on Windows.
Usage example:
  powershell -ExecutionPolicy Bypass -File .\scripts\windows\Run-SelectedBenchmark.ps1 -Chunker fixed_tok1200_ov150 -Embedding jina_v3 -VectorStore Qdrant -Reranker bge-reranker-base -LimitQueries 20
#>

param(
  [ValidateSet("entity_heuristic_w6", "entity_heuristic_w5", "entity_heuristic_w4", "Heading_sections_l2", "fixed_tok1200_ov150")]
  [string]$Chunker = "entity_heuristic_w6",

  [ValidateSet("jina_v3", "gte_multilingual_base", "openai_text-embedding-3-large")]
  [string]$Embedding = "jina_v3",

  [ValidateSet("Qdrant", "PGVector", "Weaviate")]
  [string]$VectorStore = "Qdrant",

  [ValidateSet("HNSW")]
  [string]$IndexType = "HNSW",

  [ValidateSet("Cosine Similarity")]
  [string]$RetrievalMethod = "Cosine Similarity",

  [ValidateSet("Amazon Rerank v1", "Qwen3:4B Rerank", "bge-reranker-base")]
  [string]$Reranker = "bge-reranker-base",

  [int]$LimitQueries = 20,
  [string]$OutputDir = "data/modular_runs/windows_selected",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  if (Get-Command py -ErrorAction SilentlyContinue) { $VenvPython = "py" }
  elseif (Get-Command python -ErrorAction SilentlyContinue) { $VenvPython = "python" }
  else { throw "Python not found. Run Setup-WnsBenchmark.ps1 first." }
}

if (Test-Path ".env") {
  & (Join-Path $PSScriptRoot "Load-WnsEnv.ps1")
}

& $VenvPython scripts\benchmark_cli.py run `
  --limit-queries $LimitQueries `
  --output-dir $OutputDir `
  --chunker $Chunker `
  --embedding $Embedding `
  --vector-store $VectorStore `
  --index-type $IndexType `
  --retrieval-method $RetrievalMethod `
  --reranker $Reranker
