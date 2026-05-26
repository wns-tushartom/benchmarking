@echo off
setlocal EnableExtensions

REM Run a selected benchmark combination from CMD.
REM Usage:
REM   scripts\windows\Run-SelectedBenchmark.cmd fixed_tok1200_ov150 jina_v3 Qdrant none 20
REM Args:
REM   1 chunker, default entity_heuristic_w6
REM   2 embedding, default jina_v3
REM   3 vector_store, default Qdrant
REM   4 reranker, default none
REM   5 limit_queries, default 20

set "CHUNKER=%~1"
set "EMBEDDING=%~2"
set "VECTOR_STORE=%~3"
set "RERANKER=%~4"
set "LIMIT=%~5"

if "%CHUNKER%"=="" set "CHUNKER=entity_heuristic_w6"
if "%EMBEDDING%"=="" set "EMBEDDING=jina_v3"
if "%VECTOR_STORE%"=="" set "VECTOR_STORE=Qdrant"
if "%RERANKER%"=="" set "RERANKER=none"
if "%LIMIT%"=="" set "LIMIT=20"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." || exit /b 1
set "PROJECT_ROOT=%CD%"

if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PY_CMD=py"
  ) else (
    set "PY_CMD=python"
  )
)

if exist ".env" call "%SCRIPT_DIR%Load-WnsEnv.cmd"

"%PY_CMD%" scripts\benchmark_cli.py run --limit-queries %LIMIT% --output-dir data/modular_runs/windows_cmd_selected --chunker "%CHUNKER%" --embedding "%EMBEDDING%" --vector-store "%VECTOR_STORE%" --index-type HNSW --retrieval-method "Cosine Similarity" --reranker "%RERANKER%"

popd
endlocal
