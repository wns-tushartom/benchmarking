"""Allowlisted public projection of recorded status, not live verification.

This pure function performs no filesystem, network, provider, or process operations.
Freshness and receipt validation must never be inferred from this projection.
"""
_STATES = frozenset({'ready', 'completed', 'blocked', 'running', 'pending_not_run',
    'pending_partial_output', 'pending_readiness', 'waiting_external_producer',
    'stopped_by_request', 'failed', 'pending', 'unknown'})


# Display labels only, never execution eligibility or provider authorization.
_METHOD_LABELS = {
    'chunker': frozenset(('entity_heuristic_w6', 'entity_heuristic_w5', 'entity_heuristic_w4',
                         'Heading_sections_l2', 'fixed_tok1200_ov150', 'semantic_split')),
    'embedding': frozenset(('jina_v3', 'gte_multilingual_base', 'openai_text-embedding-3-large',
                           'nemotron_3_embed_1b_bf16', 'nemotron_3_embed_1b_nvfp4', 'nemotron_3_embed_8b_bf16')),
    'vector_store': frozenset(('Qdrant', 'PGVector', 'Weaviate', 'FAISS', 'TurboVec')),
    'index_type': frozenset(('HNSW', 'TurboQuant4bit')),
    'retrieval_method': frozenset(('Dense Cosine', 'BM25 + Dense + RRF')),
    'reranker': frozenset(('none', 'Amazon Rerank v1', 'bge-reranker-base', 'Qwen3:4B Rerank',
                          'nemotron_rerank_1b', 'gte_modernbert_base')),
}


def public_methods(raw, totals, *, projected=False):
    """All-or-nothing reconciliation; unknown identities receive neutral labels."""
    if not isinstance(raw, dict) or set(raw) != set(_METHOD_LABELS):
        return None
    if any(type(value) is not int or value < 0 for value in totals.values()):
        return None
    result = {}
    for dimension, allowed in _METHOD_LABELS.items():
        rows = raw[dimension]
        if not isinstance(rows, list) or len(rows) > 10000:
            return None
        seen, entries = set(), []
        for row in rows:
            if not isinstance(row, dict):
                return None
            name = row.get('method')
            if not isinstance(name, str) or not name or len(name) > 1024 or name in seen:
                return None
            seen.add(name)
            counts = {key: row.get(key) for key in totals}
            if any(type(value) is not int or value < 0 for value in counts.values()):
                return None
            if counts['configured'] != counts['assigned'] + counts['unassigned'] + counts['excluded']:
                return None
            entries.append(dict(method=name, **counts))
        if any(sum(row[key] for row in entries) != total for key, total in totals.items()):
            return None
        # Public rows already have deterministic order. Sorting neutral labels
        # lexically would move label 10 before label 2 and reassign their counts.
        # Still revalidate every field and sanitize every name in either mode.
        if not projected:
            entries.sort(key=lambda row: row['method'])
        unknown = 0
        for row in entries:
            if row['method'] not in allowed:
                unknown += 1
                row['method'] = 'Unlisted method ' + str(unknown)
        result[dimension] = entries
    return result


def public_status(raw):
    has_jobs = isinstance(raw, dict) and isinstance(raw.get('jobs'), list)
    control_only = (isinstance(raw, dict) and 'jobs' not in raw
                    and isinstance(raw.get('queue_state'), str)
                    and raw['queue_state'] in {'blocked', 'stopped_by_request'})
    available = has_jobs or control_only
    snapshot = raw if available else {}
    jobs = []
    for ordinal, job in enumerate(snapshot.get('jobs', []), 1):
        state = job.get('state') if isinstance(job, dict) else None
        jobs.append({'ordinal': ordinal, 'state': state if isinstance(state, str) and state in _STATES else 'unknown'})
    inventory = snapshot.get('inventory')
    configured = inventory.get('master_configured') if isinstance(inventory, dict) else None
    if type(configured) is not int or configured < 0:
        configured = None
    inventory_counts = {}
    for public_key, source_key in [('selected_combinations', 'nonpaid_selected'),
                                   ('authorization_excluded_combinations', 'paid_excluded'),
                                   ('assigned_combinations', 'owned_by_jobs'),
                                   ('unassigned_combinations', 'pending_unassigned')]:
        value = inventory.get(source_key) if isinstance(inventory, dict) else None
        inventory_counts[public_key] = value if type(value) is int and value >= 0 else None
    selected = inventory_counts['selected_combinations']
    excluded = inventory_counts['authorization_excluded_combinations']
    assigned = inventory_counts['assigned_combinations']
    unassigned = inventory_counts['unassigned_combinations']
    values = (configured, selected, excluded, assigned, unassigned)
    inconsistent = any(value is not None and configured is not None and value > configured
                       for value in values)
    if selected is not None:
        inconsistent |= any(value is not None and value > selected for value in (assigned, unassigned))
    for total, left, right in ((configured, selected, excluded), (selected, assigned, unassigned)):
        if all(value is not None for value in (total, left, right)):
            inconsistent |= total != left + right
    consistency = 'consistent' if all(value is not None for value in values) else 'incomplete'
    if inconsistent:
        consistency = 'inconsistent'
        configured = None
        inventory_counts = dict.fromkeys(inventory_counts)
    queue_state = snapshot.get('queue_state', snapshot.get('state'))
    queue_states = {'empty', 'completed', 'pending', 'running', 'blocked',
                    'waiting_external_producer', 'stopped_by_request'}
    if not isinstance(queue_state, str) or queue_state not in queue_states:
        queue_state = 'unknown'
    if queue_state == 'completed' and (not jobs or any(job['state'] != 'completed' for job in jobs)
                                       or consistency != 'consistent' or unassigned != 0):
        # Recorded success must agree with jobs and fully assigned inventory.
        # This is consistency checking, never independent receipt validation.
        queue_state = 'unknown'
    methods = public_methods(snapshot.get('method_inventory'), {
        'configured': configured, 'assigned': inventory_counts['assigned_combinations'],
        'unassigned': inventory_counts['unassigned_combinations'],
        'excluded': inventory_counts['authorization_excluded_combinations'],
    }) if consistency == 'consistent' else None
    return {
        'method_inventory_available': methods is not None,
        'method_inventory': methods,
        'inventory_consistency': consistency,
        'available': available,
        'queue_state': queue_state,
        'counts': {
            **inventory_counts,
            'configured_combinations': configured,
            'listed_jobs': len(jobs) if has_jobs else None,
            'recorded_completed_jobs': sum(j['state'] == 'completed' for j in jobs) if has_jobs else None,
        },
        'jobs': jobs,
        'promotion_status': 'not_accepted',
        'receipt_revalidated': False,
    }
