<#
.SYNOPSIS
  Windows setup for the WNS benchmark harness.

.DESCRIPTION
  Creates .env from .env.example, creates/updates Python virtualenv, installs benchmark dependencies,
  optionally starts Docker vector DB services, validates env, and runs a smoke benchmark.

.USAGE
  From the Retreiver project root:
    powershell -ExecutionPolicy Bypass -File .\scripts\windows\Setup-WnsBenchmark.ps1

  Skip Docker services:
    powershell -ExecutionPolicy Bypass -File .\scripts\windows\Setup-WnsBenchmark.ps1 -SkipDocker

  Skip package install:
    powershell -ExecutionPolicy Bypass -File .\scripts\windows\Setup-WnsBenchmark.ps1 -SkipInstall
#>

param(
  [switch]$SkipDocker,
  [switch]$SkipInstall,
  [switch]$SkipSmoke,
  [string]$PythonCommand = "py",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

function Write-Step($Message) {
  Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-Ok($Message) {
  Write-Host "OK: $Message" -ForegroundColor Green
}

function Write-WarnMsg($Message) {
  Write-Host "WARN: $Message" -ForegroundColor Yellow
}

function Test-Command($Name) {
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot
Write-Step "WNS benchmark setup in $ProjectRoot"

if (-not (Test-Command $PythonCommand)) {
  if (Test-Command "python") {
    $PythonCommand = "python"
  } else {
    throw "Python not found. Install Python 3.10+ from python.org or Microsoft Store, then rerun."
  }
}
& $PythonCommand --version

Write-Step "Creating .env if missing"
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Ok "Created .env from .env.example. Fill API keys in this file."
} else {
  Write-Ok ".env already exists, leaving it untouched."
}

Write-Step "Creating Python virtual environment"
if (-not (Test-Path $VenvDir)) {
  & $PythonCommand -m venv $VenvDir
  Write-Ok "Created $VenvDir"
} else {
  Write-Ok "$VenvDir already exists"
}

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtualenv Python not found at $VenvPython"
}

if (-not $SkipInstall) {
  Write-Step "Installing Python benchmark dependencies"
  & $VenvPython -m pip install --upgrade pip
  & $VenvPython -m pip install -r requirements-benchmark.txt
  Write-Ok "Installed requirements-benchmark.txt"
} else {
  Write-WarnMsg "Skipping Python dependency install."
}

if (-not $SkipDocker) {
  Write-Step "Checking Docker"
  if (-not (Test-Command "docker")) {
    Write-WarnMsg "Docker CLI not found. Install Docker Desktop, start it, then rerun without -SkipDocker."
  } else {
    docker version | Out-Host
    Write-Step "Starting vector DB services with Docker Compose"
    docker compose -f docker-compose.benchmark.yml up -d qdrant postgres-pgvector weaviate
    Write-Ok "Docker services requested: qdrant, postgres-pgvector, weaviate"
  }
} else {
  Write-WarnMsg "Skipping Docker services."
}

Write-Step "Loading .env into current PowerShell process"
& (Join-Path $PSScriptRoot "Load-WnsEnv.ps1")

Write-Step "Checking env and service readiness"
& $VenvPython scripts\check_env.py
& $VenvPython scripts\check_services.py

Write-Step "Validating benchmark config"
& $VenvPython scripts\benchmark_cli.py validate

if (-not $SkipSmoke) {
  Write-Step "Running selected local fallback smoke benchmark"
  & $VenvPython scripts\benchmark_cli.py run `
    --limit-queries 5 `
    --chunker entity_heuristic_w6 `
    --embedding jina_v3 `
    --vector-store Qdrant `
    --index-type HNSW `
    --retrieval-method "Cosine Similarity" `
    --reranker bge-reranker-base `
    --output-dir data/modular_runs/windows_smoke
  Write-Ok "Smoke benchmark completed."
} else {
  Write-WarnMsg "Skipping smoke benchmark."
}

Write-Step "Next steps"
Write-Host "1. Prefer CMD scripts on WNS laptop because PowerShell is blocked by group policy."
Write-Host "2. Fill .env with JINA_EMBEDDING_URL and GTE_EMBEDDING_URL for WNS VM embedding endpoints."
Write-Host "3. JINA_API_KEY is optional only for hosted/authenticated Jina API; self-hosted VM Jina does not need it."
Write-Host "4. Fill OPENAI_API_KEY only if using OpenAI embedding."
Write-Host "5. Put QWEN_RERANK_URL as the VM/remote endpoint. Do not run Qwen3:4B locally on this laptop."
Write-Host "6. Start dashboard with: scripts\windows\Start-WnsDashboard.cmd"
Write-Ok "Windows setup finished."
