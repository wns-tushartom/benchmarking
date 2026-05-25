@echo off
setlocal EnableExtensions

REM Check WNS env, services, and benchmark config. No PowerShell required.

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

call "%SCRIPT_DIR%Load-WnsEnv.cmd"

"%PY_CMD%" scripts\check_env.py
"%PY_CMD%" scripts\check_services.py
"%PY_CMD%" scripts\benchmark_cli.py validate

popd
endlocal
