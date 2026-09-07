@echo off
git add --pathspec-from-file=release/STAGE_PATHS.txt
if errorlevel 1 exit /b 1
git diff --cached --name-only
