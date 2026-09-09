"""Read-only semantic HTML fragment for the sanitized public queue snapshot.

No scripts, process controls, filesystem reads, or receipt validation. Intended
for composition inside the existing dashboard, which owns visual styling.
"""
from html import escape

_STATES = frozenset({'ready', 'empty', 'completed', 'blocked', 'running', 'pending_not_run',
    'pending_partial_output', 'pending_readiness', 'waiting_external_producer',
    'stopped_by_request', 'failed', 'pending', 'unknown'})


def _state(value):
    return value.replace('_', ' ') if isinstance(value, str) and value in _STATES else 'unknown'


def _count(value):
    return str(value) if type(value) is int and value >= 0 else 'Unknown'


def render_panel(snapshot):
    """Render an allowlisted projection; never interpolate arbitrary diagnostics."""
    data = snapshot if isinstance(snapshot, dict) else {}
    available = data.get('available') is True
    counts = data.get('counts') if available and isinstance(data.get('counts'), dict) else {}
    freshness = data.get('freshness')
    if not isinstance(freshness, str) or freshness not in {'fresh', 'stale', 'clock_mismatch', 'unavailable'}:
        freshness = 'unavailable'
    parts = ['<section id="queue-status-panel" aria-labelledby="queue-status-title">',
             '<h2 id="queue-status-title">Candidate queue</h2>',
             '<p>Read-only snapshot · Candidate evidence · not_accepted</p>',
             '<p>Receipts are not revalidated by this view. Recorded running state and file freshness are not proof of process liveness.</p>',
             '<p>Start and STOP remain CLI-only.</p>']
    if not available:
        parts.append('<p role="status">Status unavailable</p>')
    consistency = data.get('inventory_consistency') if available else None
    if consistency == 'inconsistent':
        parts.append('<p role="alert">Inventory counts conflict. Combination counts are Unknown until the snapshot is reconciled.</p>')
    elif consistency == 'consistent':
        parts.append('<p>Internally consistent; not independently verified.</p>')
    else:
        parts.append('<p>Inventory reconciliation unavailable or incomplete.</p>')
    parts.append('<p>Recorded queue state: <strong>' + _state(data.get('queue_state') if available else None) + '</strong></p>')
    parts.append('<p>Snapshot freshness: ' + freshness.replace('_', ' ') + '</p><dl>')
    for key, label in [('configured_combinations', 'Configured combinations'),
                       ('selected_combinations', 'Selected combinations'),
                       ('authorization_excluded_combinations', 'Excluded pending authorization'),
                       ('assigned_combinations', 'Assigned combinations'),
                       ('unassigned_combinations', 'Unassigned combinations'),
                       ('listed_jobs', 'Listed batches'),
                       ('recorded_completed_jobs', 'Recorded completed batches')]:
        parts.append('<dt>' + label + '</dt><dd>' + _count(counts.get(key)) + '</dd>')
    parts.append('</dl><h3>Recorded batch states</h3><ol>')
    jobs = data.get('jobs') if available and isinstance(data.get('jobs'), list) else []
    for index, job in enumerate(jobs, 1):
        state = _state(job.get('state') if isinstance(job, dict) else None)
        parts.append('<li>Batch ' + str(index) + ': ' + escape(state) + '</li>')
    parts.append('</ol>')
    if available and not jobs:
        parts.append('<p>No batches listed in this snapshot. This does not establish portfolio completion.</p>')
    from queue_public_status import public_methods
    methods = public_methods(data.get('method_inventory'), {
        'configured': counts.get('configured_combinations'),
        'assigned': counts.get('assigned_combinations'),
        'unassigned': counts.get('unassigned_combinations'),
        'excluded': counts.get('authorization_excluded_combinations'),
    }, projected=True) if available and data.get('method_inventory_available') is True else None
    parts.append('<h3>Configured method coverage</h3>')
    parts.append('<p>Counts describe configured work, not readiness or completed results. Unlisted method labels hide names not approved for display.</p>')
    if methods is None:
        parts.append('<p>Method coverage unavailable: missing or inconsistent inventory.</p>')
    else:
        for dimension, rows in methods.items():
            parts.append('<h4>' + escape(dimension.replace('_', ' ').title()) + '</h4><ul>')
            for row in rows:
                parts.append('<li><strong>' + escape(row['method']) + '</strong>: Configured: '
                             + str(row['configured']) + ' · Assigned: ' + str(row['assigned'])
                             + ' · Unassigned: ' + str(row['unassigned'])
                             + ' · Excluded pending authorization: ' + str(row['excluded']) + '</li>')
            parts.append('</ul>')
    parts.append('</section>')
    return ''.join(parts)
