<#
Load .env values into current PowerShell process.
Usage:
  . .\scripts\windows\Load-WnsEnv.ps1
#>

param(
  [string]$EnvPath = ".env"
)

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$FullEnvPath = Join-Path $ProjectRoot $EnvPath

if (-not (Test-Path $FullEnvPath)) {
  throw "Env file not found: $FullEnvPath. Copy .env.example to .env first."
}

Get-Content $FullEnvPath | ForEach-Object {
  $line = $_.Trim()
  if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
  $parts = $line.Split("=", 2)
  $key = $parts[0].Trim()
  $value = $parts[1].Trim().Trim('"').Trim("'")
  [Environment]::SetEnvironmentVariable($key, $value, "Process")
}

Write-Host "Loaded env from $FullEnvPath into current PowerShell process." -ForegroundColor Green
