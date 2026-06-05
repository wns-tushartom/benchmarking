@echo off
setlocal EnableExtensions

REM Windows CMD setup for WNS benchmark harness. No PowerShell required.
REM Run from any directory:
REM   scripts\windows\Setup-WnsBenchmark.cmd
REM Options:
REM   --skip-docker   Do not start Docker services
REM   --skip-install  Do not install Python requirements
REM   --skip-smoke    Do not run smoke benchmark

set "SKIP_DOCKER=0"
set "SKIP_INSTALL=0"
set "SKIP_SMOKE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--skip-docker" set "SKIP_DOCKER=1"
if /I "%~1"=="--skip-install" set "SKIP_INSTALL=1"
if /I "%~1"=="--skip-smoke" set "SKIP_SMOKE=1"
shift
goto parse_args
:args_done

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." || exit /b 1
set "PROJECT_ROOT=%CD%"

echo.
echo ==^> WNS benchmark CMD setup in %PROJECT_ROOT%

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY_CMD=py"
) else (
  where python >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PY_CMD=python"
  ) else (
    echo ERROR: Python not found. Install Python 3.10+ and rerun.
    popd
    exit /b 1
  )
)
%PY_CMD% --version
if errorlevel 1 exit /b 1

echo.
echo ==^> Creating .env if missing
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo OK: Created .env from .env.example. Fill API keys in this file.
) else (
  echo OK: .env already exists, leaving it untouched.
)

echo.
echo ==^> Creating Python virtual environment
if not exist ".venv\Scripts\python.exe" (
  %PY_CMD% -m venv .venv
  if errorlevel 1 exit /b 1
  echo OK: Created .venv
) else (
  echo OK: .venv already exists
)
set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if "%SKIP_INSTALL%"=="0" (
  echo.
  echo ==^> Installing Python benchmark dependencies
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  "%VENV_PY%" -m pip install -r requirements-benchmark.txt
  if errorlevel 1 exit /b 1
  echo OK: Installed requirements-benchmark.txt
) else (
  echo WARN: Skipping Python dependency install.
)

if "%SKIP_DOCKER%"=="0" (
  echo.
  echo ==^> Checking Docker
  where docker >nul 2>nul
  if errorlevel 1 (
    echo WARN: Docker CLI not found. Install/start Docker Desktop or rerun with --skip-docker.
  ) else (
    docker version
    echo.
    echo ==^> Starting vector DB services with Docker Compose
    docker compose -f docker-compose.benchmark.yml up -d qdrant postgres-pgvector weaviate
    if errorlevel 1 echo WARN: Docker compose failed. Open Docker Desktop and rerun, or use --skip-docker.
  )
) else (
  echo WARN: Skipping Docker services.
)

echo.
echo ==^> Loading .env into this CMD process
call "%SCRIPT_DIR%Load-WnsEnv.cmd"

echo.
echo ==^> Checking env and services
"%VENV_PY%" scripts\check_env.py
"%VENV_PY%" scripts\check_services.py

echo.
echo ==^> Validating benchmark config
"%VENV_PY%" scripts\benchmark_cli.py validate
if errorlevel 1 exit /b 1

if "%SKIP_SMOKE%"=="0" (
  echo.
  echo ==^> Running selected local fallback smoke benchmark
  "%VENV_PY%" scripts\benchmark_cli.py run --limit-queries 5 --chunker entity_heuristic_w6 --embedding jina_v3 --vector-store Qdrant --index-type HNSW --retrieval-method "Cosine Similarity" --reranker bge-reranker-base --output-dir data/modular_runs/windows_cmd_smoke
  if errorlevel 1 exit /b 1
  echo OK: Smoke benchmark completed.
) else (
  echo WARN: Skipping smoke benchmark.
)

echo.
echo ==^> Next steps
echo 1. Fill .env with JINA_EMBEDDING_URL and GTE_EMBEDDING_URL for the WNS VM embedding endpoints.
echo 2. JINA_API_KEY is optional only for hosted/authenticated Jina API. Self-hosted VM Jina does not need it.
echo 3. Fill OPENAI_API_KEY only if using OpenAI embedding.
echo 4. Add AWS keys only when testing Amazon Rerank v1.
echo 5. Put QWEN_RERANK_URL as the VM/remote endpoint. Do not run Qwen3:4B locally on this laptop.
echo 6. Do not run Jina/GTE embeddings on laptop CPU for real runs; use the WNS VM.
echo 7. Start dashboard with: scripts\windows\Start-WnsDashboard.cmd
echo.
echo OK: CMD setup finished.
popd
endlocal
