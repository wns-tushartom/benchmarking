@echo off
setlocal EnableExtensions
set VM_HOST=%~1
if "%VM_HOST%"=="" (
  echo Usage: scripts\windows\Connect-Frontend-To-VM.cmd VM_HOST_OR_IP
  echo Example: scripts\windows\Connect-Frontend-To-VM.cmd 10.20.30.40
  exit /b 2
)
if not exist scripts\benchmark_cli.py (
  echo Run this from the Retreiver project root.
  exit /b 2
)

call scripts\windows\Configure-LocalFrontendForVM.cmd %VM_HOST%
if errorlevel 1 exit /b %errorlevel%

python scripts\test_vm_connection.py
if errorlevel 1 (
  echo.
  echo VM connection check failed. Confirm VM setup completed and ports 5000, 5001, 5003, 5004 are reachable.
  exit /b %errorlevel%
)

call scripts\windows\Start-WnsDashboard-VM.cmd
endlocal
