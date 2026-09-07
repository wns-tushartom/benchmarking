# Retreiver frontend handoff — 20260907-handoff-v1
Source snapshot with intentional uncommitted changes, not a pushed commit. Read this before older runbooks.

## Scope
Updated heatmap/metric explorer, stable colors and Pareto views; strict 500-query ranking admission; archived candidate API states and launch protection; selected-plan counts. 180 accepted results remain the baseline. No new runs or model starts were performed.
All runtime data, indexes, model caches, real environment files and .git are excluded. The authoritative accepted manifest is included separately under release/, with its pinned original bytes. No historical short-query raw results are included.

## Work laptop (Windows CMD)
Extract ZIP into a separate staging directory. Do not delete or replace your existing repository. From your existing Retreiver directory:

    git status --short
    git remote -v
    git branch --show-current
    robocopy "C:\STAGING\Retreiver" "." /E /XD data .git /XF .env .env.vm.generated

Replace the staging path. Robocopy exit codes 0–7 are nonfatal; 8+ means failure. Never use /MIR. Existing data is untouched. Review source differences; if this conflicts with work-laptop changes, stop rather than discard them.
Use Python 3.12 (cgi is removed in 3.13) and Node 22. For a dashboard-test environment:

    py -3.12 -m venv .venv-dashboard
    .venv-dashboard\Scripts\python -m pip install pytest rank-bm25
    .venv-dashboard\Scripts\python release\verify.py

If the accepted manifest is absent, copy release/ACCEPTED_180_MANIFEST.json into data/accepted/ACCEPTED_180_MANIFEST.json. If present, compare SHA-256 first; never overwrite a different existing file. The included manifest SHA is f9d585134798016a23cbd560464914b364808b1233a076a702fc28d1b64f09eb.

Stage the explicit source allowlist from release/STAGE.cmd ONLY after reviewing differences. This includes the full source snapshot, not runtime data. Check no previously staged unrelated files remain:

    release\STAGE.cmd
    git diff --cached --name-only
    git diff --cached --check
    git -c gc.auto=0 commit -m "Finish benchmark frontend and full-depth candidate safeguards"
    git rev-parse HEAD
    git push REMOTE HEAD:feature/meeting-400-new-methods

Replace REMOTE with the verified remote name; do not guess. Keep the printed commit hash.

## Work VM
Preserve its existing data and environment. Inspect before pulling; if dirty or divergent, stop and reconcile, do not reset/clean.

    git status --short
    git remote -v
    git fetch REMOTE
    git pull --ff-only REMOTE feature/meeting-400-new-methods
    git rev-parse HEAD

HEAD must exactly equal the laptop commit. Verify with Python 3.12 and Node using release/verify.py. Restore the accepted manifest using the same absent-only rule. Restart the existing dashboard using its normal service mechanism; do not start a second process on its port. For an isolated manual dashboard test:

    python3.12 scripts/serve_benchmark_dashboard.py 17185

Use private access only. Verify /api/data, /api/result-sources, /api/portfolio; check accepted180 and desktop/mobile Metrics/Recommendations. No full-depth candidate completions should be inferred from planned or archived rows.

## Known execution limitation (not a frontend deployment failure)
The current Meeting-400 plan deliberately returns no selected batch. Consequently Start required adapters / Run selected batch remain disabled, and run-next rejects an unfinished plan with no selection. Individual unrun batch buttons are a separate path, not authorization to bypass preflight or a user block. Prepare/reconcile the new full-depth plan before expecting automatic next-batch selection. This ZIP is the verified frontend/source handoff, not a certification that the historical queue is ready to run. Use this supplied ZIP, not the older repository overlay-packaging scripts, which do not inventory the new visualization helper and full current source.

## Before any benchmark launch
This release does NOT make the VM run-ready automatically. Verify data/groundtruth_500.csv, data/chunking_methods_output_225.xlsx and corpus receipt match each other; exact embedding/reranker models and dimensions; real endpoint inference; store readiness; memory capacity; immutable candidate-only output destination; 500 queries per combination. Keep any BLOCKED_BY_USER.json intact. An archived batch cannot be rerun in-place; prepare a separate full-depth run namespace first. Do not run old all-batch shell queues from historical docs. Never write candidates into data/modular_runs/latest.

## Verification scope
112 focused Python tests plus three Node regression suites; Chromium desktop1440/mobile390 heatmap interaction, accepted .964 Recall@5, latency switching, API portfolio states and no page errors. Not a full repository test claim; legacy QA/model dependencies are outside this gate. VM deployment and inference remain operator-side gates.
