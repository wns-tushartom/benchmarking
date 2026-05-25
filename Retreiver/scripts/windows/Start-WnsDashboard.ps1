<#
Start the WNS benchmark dashboard on Windows.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\windows\Start-WnsDashboard.ps1
#>

param(
  [int]$Port = 8765,
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $VenvPython = "py"
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $VenvPython = "python"
  } else {
    throw "Python not found. Run Setup-WnsBenchmark.ps1 first."
  }
}

if (Test-Path ".env") {
  & (Join-Path $PSScriptRoot "Load-WnsEnv.ps1")
}

Write-Host "Starting WNS dashboard at http://127.0.0.1:$Port" -ForegroundColor Cyan
& $VenvPython scripts\serve_benchmark_dashboard.py $Port
