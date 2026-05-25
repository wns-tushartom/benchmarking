<#
Print missing WNS env values and service readiness on Windows.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\windows\Check-WnsSetup.ps1
#>

param(
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

& $VenvPython scripts\check_env.py
& $VenvPython scripts\check_services.py
& $VenvPython scripts\benchmark_cli.py validate
