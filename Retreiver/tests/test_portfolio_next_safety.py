import pytest
import scripts.serve_benchmark_dashboard as dashboard
from test_dashboard_portfolio_control import CONFIG, not_run_status
from benchmarking.core.config import load_benchmark_config
from benchmarking.core.portfolio import build_portfolio_plan

@pytest.mark.parametrize('kind', ['unrun', 'missing', 'duplicate', 'complete'])
def test_no_selection_requires_exact_completed_coverage(monkeypatch, kind):
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    status = not_run_status(plan)
    status['selected_batch_id'] = None
    if kind != 'unrun':
        for row in status['batches']:
            row['state'] = 'completed'
    if kind == 'missing':
        status['batches'].pop()
    if kind == 'duplicate':
        status['batches'][-1] = status['batches'][0].copy()
    monkeypatch.setattr(dashboard, 'load_dashboard_portfolio_plan', lambda: plan)
    monkeypatch.setattr(dashboard, 'dashboard_portfolio_status', lambda _: (config, status))
    monkeypatch.setattr(dashboard, 'launch_job', lambda _: pytest.fail('must not launch'))
    if kind == 'complete':
        assert dashboard.launch_portfolio_action('run-next', {})['state'] == 'completed'
    else:
        with pytest.raises(dashboard.DashboardControlError) as exc:
            dashboard.launch_portfolio_action('run-next', {})
        assert exc.value.code == 'no_batch_selected'


@pytest.mark.parametrize('action', ['run-batch', 'run-next'])
def test_archived_batch_cannot_be_relaunched(monkeypatch, action):
    config = load_benchmark_config(CONFIG)
    plan = build_portfolio_plan(config)
    status = not_run_status(plan)
    batch_id = plan.batches[0].batch_id
    status['batches'][0]['state'] = 'archived_query_depth'
    status['selected_batch_id'] = batch_id
    launches = []
    monkeypatch.setattr(dashboard, 'load_dashboard_portfolio_plan', lambda: plan)
    monkeypatch.setattr(dashboard, 'dashboard_portfolio_status', lambda _: (config, status))
    monkeypatch.setattr(dashboard, 'launch_job', lambda command: launches.append(command) or {})
    with pytest.raises(dashboard.DashboardControlError) as exc:
        dashboard.launch_portfolio_action(action, {'batch_id': batch_id} if action == 'run-batch' else {})
    assert exc.value.code == 'batch_archived'
    assert not launches
