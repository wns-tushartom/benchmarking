"""Compose real immutable receipt, full-depth and approved-data checks."""
import json
from pathlib import Path
from benchmarking.core.portfolio import PortfolioPlan, build_portfolio_plan
from scripts.publish_portfolio_results import verify_completed_batch
from full_depth_guard import verify_full_depth
from queue_completion_bindings import verify_bindings


def verify_completion(root, portfolio_root, batch_id, config, approved_hashes, *, metrics):
    root = Path(root).absolute()
    portfolio_root = Path(portfolio_root).absolute()
    lane = config['experiment'].get('output_lane')
    if not isinstance(lane, str) or not lane or Path(lane).name != lane or lane in ('.', '..', 'latest'):
        raise ValueError('invalid candidate output lane')
    if root.resolve() != root or portfolio_root.resolve() != portfolio_root:
        raise ValueError('path alias rejected')
    plan_path = portfolio_root/'portfolio_plan.json'
    if plan_path.is_symlink():
        raise ValueError('plan path alias rejected')
    plan = PortfolioPlan.from_json(plan_path.read_text())
    if portfolio_root != root/'data/modular_runs'/lane/plan.portfolio_id:
        raise ValueError('noncanonical portfolio path')
    rebuilt = build_portfolio_plan(config, max_combinations_per_batch=plan.max_combinations_per_batch)
    if rebuilt != plan:
        raise ValueError('plan does not match current configuration')
    batch = next((b for b in plan.batches if b.batch_id == batch_id), None)
    if batch is None:
        raise ValueError('batch not in immutable plan')
    verify_completed_batch(portfolio_root, plan, batch, candidate_lane=lane)
    out = portfolio_root/batch_id
    binding = verify_bindings(root, config, json.loads((out/'config_snapshot.json').read_text()),
                              json.loads((out/'manifest.json').read_text()), approved_hashes)
    depth = verify_full_depth(out/'modular_summary.csv', batch.combination_ids,
                              id_column='combination_id', metrics=metrics)
    # Rehash after reading to detect accidental concurrent modification.
    # Controller must still hold its producer exclusion lock throughout.
    verify_completed_batch(portfolio_root, plan, batch, candidate_lane=lane)
    return {'receipt_verified': True, 'promotion_status': 'not_accepted',
            'portfolio_id': plan.portfolio_id, 'batch_id': batch_id, **depth,
            'data_bindings_verified': binding['data_bindings_verified']}
