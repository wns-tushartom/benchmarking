"""Deterministic configured-method counts. No completion or readiness claims.

Internal projection only: method names still require public-display admission.
"""
DIMENSIONS = ('chunker', 'embedding', 'vector_store', 'index_type',
              'retrieval_method', 'reranker')


def method_inventory(rows, excluded_ids, assigned_ids):
    rows = list(rows)
    identifiers = [row.get('combination_id') for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError('invalid combination identity')
    ids = set(identifiers)
    excluded, assigned = set(excluded_ids), set(assigned_ids)
    if len(ids) != len(rows) or (excluded | assigned) - ids or excluded & assigned:
        raise ValueError('ambiguous method inventory classifications')
    result = {}
    for dimension in DIMENSIONS:
        counts = {}
        for row in rows:
            method = row.get(dimension)
            if method is None:
                continue
            if not isinstance(method, str) or not method:
                raise ValueError('invalid method identity')
            entry = counts.setdefault(method, dict(method=method, configured=0,
                                                  assigned=0, unassigned=0, excluded=0))
            entry['configured'] += 1
            identity = row['combination_id']
            classification = 'excluded' if identity in excluded else 'assigned' if identity in assigned else 'unassigned'
            entry[classification] += 1
        result[dimension] = [counts[name] for name in sorted(counts)]
    return result
