"""Read-only full-portfolio discovery. Pending is never inferred ready."""
from collections import Counter
from benchmarking.core.config import generate_matrix_catalog, technique
from benchmarking.core.portfolio import canonical_combination_id


def discover(config, *, allow_openai=False):
    catalog = generate_matrix_catalog(config)
    rows = []
    for original in catalog['configured']:
        row = dict(original)
        row['combination_id'] = canonical_combination_id(row)
        adapters = {kind: technique(config, kind + 's', row[kind]).get('adapter')
                    for kind in ('embedding', 'reranker', 'vector_store')}
        if (adapters['embedding'] == 'openai' and not allow_openai) or adapters['reranker'] == 'amazon_bedrock':
            state, reason = 'blocked_authorization', 'paid_provider_not_authorized'
        elif adapters['vector_store'] == 'turbovec':
            state, reason = 'pending_selection', 'turbovec_not_selected'
        else:
            state, reason = 'pending_readiness', 'real_provider_inference_required'
        row.update(queue_state=state, queue_reason=reason)
        rows.append(row)
    ids = [r['combination_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate canonical combinations in configured portfolio')
    return {'configured_count': len(rows), 'excluded_count': len(catalog['excluded']),
            'states': dict(sorted(Counter(r['queue_state'] for r in rows).items())),
            'combinations': rows, 'excluded': catalog['excluded'],
            'execution_started': False}
