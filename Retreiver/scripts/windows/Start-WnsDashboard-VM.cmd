@echo off
setlocal EnableExtensions
if not exist scripts\serve_benchmark_dashboard.py (
  echo Run this from the Retreiver project root.
  exit /b 2
)
if exist .env (
  for /f "usebackq tokens=1,* delims==" %%A in (`type .env ^| findstr /v /r "^#"`) do set "%%A=%%B"
)
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements-benchmark.txt
python scripts\check_services.py
start "WNS Benchmark Dashboard" http://127.0.0.1:8765
python scripts\serve_benchmark_dashboard.py 8765
endlocal
