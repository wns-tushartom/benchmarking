from types import SimpleNamespace
import scripts.serve_benchmark_dashboard as dashboard


def test_operations_preserves_archived_query_depth(monkeypatch):
    plan = SimpleNamespace(batches=[SimpleNamespace(batch_id='short'), SimpleNamespace(batch_id='new')], portfolio_id='p', portfolio_hash='hash', max_combinations_per_batch=250)
    config = {'experiment': {'output_lane': 'meeting-400-candidates'}}
    monkeypatch.setattr(dashboard, 'load_benchmark_config', lambda _: config)
    monkeypatch.setattr(dashboard, 'read_meeting_candidate_results', lambda **_: {'batches': [{'batch_id': 'short', 'state': 'archived_query_depth'}]})
    _, status = dashboard.dashboard_portfolio_status(plan)
    assert status['batches'][0]['state'] == 'archived_query_depth'
    assert status['batches'][1]['state'] == 'not_run'
    assert status['state_counts']['archived_query_depth'] == 1
    assert status['state_counts']['not_run'] == 1
    assert status['state_counts']['completed'] == 0
    assert status['selected_batch_id'] is None


def test_archived_status_survives_public_portfolio_payload(monkeypatch):
    from pathlib import Path
    from benchmarking.core.config import load_benchmark_config
    from benchmarking.core.portfolio import build_portfolio_plan
    from scripts.dashboard_portfolio_api import portfolio_dashboard_payload

    config_path = Path(__file__).resolve().parents[1] / 'configs/benchmark.meeting-400-candidates.json'
    config = load_benchmark_config(config_path)
    plan = build_portfolio_plan(config)
    monkeypatch.setattr(dashboard, 'PORTFOLIO_CONFIG_PATH', config_path)
    monkeypatch.setattr(dashboard, 'read_meeting_candidate_results', lambda **_: {
        'batches': [{'batch_id': plan.batches[0].batch_id, 'state': 'archived_query_depth'}]
    })
    _, status = dashboard.dashboard_portfolio_status(plan)
    payload = portfolio_dashboard_payload(config, plan, status, include_combinations=True)
    counts = payload['combination_state_counts']
    archived_count = len(plan.batches[0].combination_ids)
    assert counts['archived_query_depth'] == archived_count
    assert counts['completed'] == 0
    assert counts['not_run'] == len(plan.combination_ids) - archived_count
    assert sum(counts[key] for key in ('completed', 'failed', 'not_run', 'archived_query_depth')) == len(plan.combination_ids)
    archived_rows = [row for row in payload['configured_combinations'] if row['execution_state'] == 'archived_query_depth']
    assert {row['combination_id'] for row in archived_rows} == set(plan.batches[0].combination_ids)
    assert payload['selected_batch_id'] is None
