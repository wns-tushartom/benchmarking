@echo off
setlocal EnableExtensions

REM Start WNS benchmark dashboard. No PowerShell required.
REM Optional first arg is port, default 8765.

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"

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

echo Starting WNS dashboard at http://127.0.0.1:%PORT%
"%PY_CMD%" scripts\serve_benchmark_dashboard.py %PORT%

popd
endlocal
