@echo off
REM Load .env values into current CMD process.
REM IMPORTANT: use CALL so variables remain in your current script/session:
REM   call scripts\windows\Load-WnsEnv.cmd

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul || exit /b 1

if not exist ".env" (
  echo ERROR: .env not found. Copy .env.example to .env first.
  popd >nul
  exit /b 1
)

for /f "usebackq tokens=1,* delims== eol=#" %%A in (".env") do (
  if not "%%A"=="" (
    set "%%A=%%B"
  )
)

echo Loaded .env into current CMD process.
popd >nul
exit /b 0
