# VM All-Methods Candidate Portfolio Runbook

## Scope

This runbook operates the isolated, **not accepted** all-method candidate portfolio:

- 2,160 configured combinations.
- 2,160 explicit incompatible exclusions.
- 12 immutable batches of 180 combinations.
- Maximum executable batch size: 250.
- Candidate output root: `data/modular_runs/all-methods-portfolio/`.

It must never write to, copy into, or replace:

```text
data/modular_runs/latest/
```

The accepted 180-result benchmark remains the control. Portfolio completion is not acceptance or promotion.

## Hard stop conditions

Stop immediately if any of these occurs:

- the VM repository does not contain baseline commit `853cb7c` in its ancestry;
- `configs/benchmark.local.json` or accepted-result artifacts have semantic changes;
- the generated plan is not exactly 2,160 configured combinations and 12 batches;
- any batch exceeds 250 combinations;
- the portfolio status or completion receipt fails integrity validation;
- a provider returns a missing or mismatched model identity, vector dimension, or non-finite output;
- an approved adapter port is occupied by an unrelated process;
- the output path resolves outside `data/modular_runs/all-methods-portfolio/`.

Do not reset, clean, restore, or broadly stage a dirty repository to force this work through.

## 1. Enter the isolated candidate checkout

Run:

```bash
cd ~/benchmarking/Retreiver
```

Verify the baseline gate:

```bash
git merge-base --is-ancestor 853cb7c HEAD && echo BASE_OK || echo BASE_MISMATCH
```

Proceed only on `BASE_OK`.

Inspect the current branch and dirty state:

```bash
git status --short --branch
```

## 2. Apply the source-only package

Apply the package only to this isolated candidate checkout. Do not extract it over an accepted production checkout.

Verify the package checksum against its adjacent receipt before extraction:

```bash
sha256sum Retreiver-all-methods-portfolio-overlay.zip
```

Test ZIP integrity:

```bash
python -m zipfile -t Retreiver-all-methods-portfolio-overlay.zip
```

Inspect members before extraction:

```bash
python -m zipfile -l Retreiver-all-methods-portfolio-overlay.zip
```

The archive must contain source, tests, configs, dashboard assets, and runbooks only. It must not contain `data/`, accepted CSV/XLSX files, `.env`, logs, model weights, runtime receipts, or browser evidence.

## 3. Install pinned benchmark dependencies

Use the VM benchmark virtual environment. Preserve `transformers==4.57.6`; do not upgrade to Transformers 5.x.

Install the pinned benchmark requirements:

```bash
python -m pip install -r requirements-benchmark.txt
```

Verify TurboVec is exactly 1.0.0:

```bash
python -c "import turbovec; assert turbovec.__version__ == '1.0.0'; print(turbovec.__version__)"
```

## 4. Validate fixed adapter slots

Validate the manifest syntax:

```bash
python -m json.tool configs/vm_adapter_ports.json >/dev/null
```

Run the manager/profile tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_vm_adapter_manager.py tests/test_vm_adapter_profiles.py tests/test_start_vm_stack_script.py
```

The manifest must expose exactly `127.0.0.1:5000–5019`. Browser requests may not supply commands, paths, models, hosts, ports, or arbitrary service lists.

## 5. Configure provider endpoints and secrets

Set endpoint variables according to the VM deployment inventory. Do not paste secrets into this runbook, Git, receipts, logs, or browser-visible state.

Read the Hugging Face token without echoing it:

```bash
read -rsp 'HF token: ' HF_TOKEN && export HF_TOKEN && printf '\nHF_TOKEN loaded\n'
```

Read the dashboard operator token without echoing it:

```bash
read -rsp 'Dashboard operator token: ' WNS_ADAPTER_CONTROL_TOKEN && export WNS_ADAPTER_CONTROL_TOKEN && printf '\nOperator token loaded\n'
```

Enable the fixed control surface:

```bash
export WNS_ENABLE_ADAPTER_CONTROL=1
```

Keep model services bound to `127.0.0.1`. Never expose raw model ports publicly.

## 6. Generate and verify the immutable plan

Generate the plan:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/benchmark_cli.py portfolio-plan configs/benchmark.all-methods-portfolio.json
```

For this source configuration, the expected identity is:

```text
portfolio_d60ae6468c22bc0880610698067e2f7940a6a461bec911265c6d30e9924f5783
```

Set the exact plan path:

```bash
export PORTFOLIO_PLAN=data/modular_runs/all-methods-portfolio/portfolio_d60ae6468c22bc0880610698067e2f7940a6a461bec911265c6d30e9924f5783/portfolio_plan.json
```

Inspect status without executing work:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/benchmark_cli.py portfolio-status "$PORTFOLIO_PLAN"
```

Expected planning state before execution:

- 12 batches;
- 180 combinations per batch;
- 2,160 configured combinations;
- 2,160 explicit exclusions;
- promotion status `not_accepted`;
- the first incomplete batch is selected.

## 7. Start the dashboard

Run the dashboard in its own supervised terminal or service:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/serve_benchmark_dashboard.py 5011
```

Open the operator dashboard through the approved private VM access path. The homepage must show:

- 20 fixed adapter slots;
- 12 immutable portfolio batches;
- configured and excluded counts separately;
- `Promotion: not accepted`;
- read-only controls unless the control gate and operator token are present.

The operator token is held only in the password field and is not persisted by the browser.

## 8. Service readiness

Use **Start required adapters** for the selected immutable batch. The server derives required services from the plan; the browser does not send service lists.

Refresh status until each required service is healthy. A failed or occupied service must remain blocked until its underlying VM condition is corrected. Never stop or kill a process merely because it occupies an approved port.

## 9. Execute or resume one immutable batch

Preferred dashboard action: **Run selected batch**.

Equivalent CLI command for the receipt-selected next batch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/benchmark_cli.py run-next configs/benchmark.all-methods-portfolio.json --plan "$PORTFOLIO_PLAN"
```

Do not add `--limit-queries` to production portfolio execution. Do not supply matrix-axis filters or `max_runs`.

After the command exits, verify status:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/benchmark_cli.py portfolio-status "$PORTFOLIO_PLAN"
```

Repeat the `run-next` command one batch at a time. A completed batch is skipped only when its completion receipt and artifact hashes validate. Failed or tampered batches fail closed; they are not silently counted as completed.

## 10. Verify and publish candidate-only aggregate results

Do not publish until all 12 batch completion receipts validate.

Verify and create the candidate portfolio receipt:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/publish_portfolio_results.py "$PORTFOLIO_PLAN"
```

The published portfolio remains:

```text
promotion_status: not_accepted
```

Publishing must not copy, merge, or link portfolio results into `data/modular_runs/latest/` or official dashboard Metrics, Recommendations, or Evidence.

## 11. Safe stop and rollback

The dashboard may stop only Python/vLLM processes that it started and can still prove through its PID, start-tick, command fingerprint, owner, and Linux pidfd checks. Reserved, companion, external, and unrelated processes are never terminated by the dashboard.

Rollback means stopping dashboard-owned candidate services and discarding only the isolated candidate checkout after preserving required receipts. It does **not** mean resetting, cleaning, restoring, or deleting accepted benchmark work.

Before removing any isolated checkout or output, obtain explicit operator approval and verify the exact path. No destructive rollback command is included intentionally.

## 12. Evidence to retain

Retain outside accepted-result paths:

- source package and package receipt;
- immutable `portfolio_plan.json`;
- per-batch config snapshots and combination manifests;
- provider-readiness and completion receipts;
- per-batch artifact digest maps;
- final candidate portfolio receipt;
- service logs with secrets redacted.

These artifacts prove candidate execution and integrity. They do not prove acceptance.
