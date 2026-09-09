"""Compose read-only identity validation and provider probes for one batch.

Caller must supply an independently approved immutable plan and data hashes.
Run in a bounded child on the VM. This is not a producer-lock or corpus-schema
verification and cannot by itself authorize batch execution.
"""
from benchmarking.core.config import generate_matrix_catalog, config_hash
from benchmarking.core.portfolio import PortfolioPlan, build_portfolio_plan
from queue_completion_bindings import verify_data_identity
from queue_vector_preflight import probe_vector_stores
from queue_provider_preflight import probe_models
from queue_corpus_admission import verify_corpus


def preflight_batch(*, root, config, approved_plan, batch_id, approved_hashes,
                    approved_config_hash, allow_openai=False, vector_probe=None, model_probe=None):
    if config_hash(config) != approved_config_hash:
        raise ValueError('approved plan configuration hash mismatch')
    plan = PortfolioPlan.from_dict(approved_plan)
    if build_portfolio_plan(config, max_combinations_per_batch=plan.max_combinations_per_batch) != plan:
        raise ValueError('approved plan differs from configuration')
    batch = next((b for b in plan.batches if b.batch_id == batch_id), None)
    if batch is None:
        raise ValueError('batch missing from approved plan')
    catalog = generate_matrix_catalog(config)['configured']
    rows = {r['combination_id']: r for r in catalog}
    if len(rows) != len(catalog) or any(cid not in rows for cid in batch.combination_ids):
        raise ValueError('plan combination coverage mismatch')
    selected = [rows[cid] for cid in batch.combination_ids]
    data_hash = verify_data_identity(root, config, approved_hashes)
    corpus_result = verify_corpus(root, config, selected)
    vector_result = (vector_probe or probe_vector_stores)(config, selected)
    if vector_result.get('vector_stores_verified') is not True:
        raise RuntimeError('vector preflight did not verify')
    model_result = (model_probe or probe_models)(config, selected, allow_openai=allow_openai)
    if model_result.get('models_verified') is not True:
        raise RuntimeError('model preflight did not verify')
    if verify_data_identity(root, config, approved_hashes) != data_hash:
        raise ValueError('data identity changed during preflight')
    return {'preflight_verified': True, 'batch_ready': False,
            'portfolio_id': plan.portfolio_id, 'batch_id': batch_id,
            'combination_ids': list(batch.combination_ids),
            'config_hash': config_hash(config), 'dataset_hash': data_hash,
            'models': model_result, 'vector_stores': vector_result,
            'corpus': corpus_result,
            'pending_gates': ['producer_lock', 'launch_revalidation'],
            'promotion_status': 'not_accepted'}
