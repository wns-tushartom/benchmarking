"""Invoke preflight in a bounded child; validate receipt in the parent."""
import json
import os
from pathlib import Path
import sys
import uuid
from queue_child import run_child


def run_preflight(payload, *, state_dir, env, timeout=120, child_runner=None, pass_fds=()):
    state_dir = Path(state_dir).absolute()
    if state_dir.resolve() != state_dir or not state_dir.is_dir():
        raise RuntimeError('preflight state directory must be canonical')
    token = uuid.uuid4().hex
    request = state_dir / ('preflight-' + token + '.request.json')
    receipt = state_dir / ('preflight-' + token + '.receipt.json')
    log = state_dir / ('preflight-' + token + '.log')
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    fd = os.open(request, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    command = [sys.executable, str(Path(__file__).with_name('queue_preflight_cli.py')),
               '--request', str(request), '--receipt', str(receipt)]
    result = (child_runner or run_child)(command, cwd=Path(payload['root']), env=env,
                                        log_path=log, timeout=timeout, pass_fds=pass_fds)
    if result.get('returncode') != 0 or result.get('timed_out') is not False:
        raise RuntimeError('preflight child failed or timed out')
    try:
        if receipt.is_symlink() or not receipt.is_file() or receipt.stat().st_size > 8_000_000:
            raise ValueError('invalid receipt file')
        evidence = json.loads(receipt.read_text())
        if evidence.get('preflight_verified') is not True or evidence.get('batch_ready') is not False:
            raise ValueError('invalid verification flags')
        for section, flag in [('corpus', 'corpus_verified'), ('models', 'models_verified'),
                              ('vector_stores', 'vector_stores_verified')]:
            value = evidence.get(section)
            if not isinstance(value, dict) or value.get(flag) is not True:
                raise ValueError('missing verified admission section')
        ids = evidence.get('combination_ids')
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(cid, str) or not cid for cid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('invalid combination evidence')
        from queue_completion_bindings import verify_data_identity
        expected_digest = verify_data_identity(Path(payload['root']), payload['config'],
                                               payload['approved_hashes'])
        if evidence.get('dataset_hash') != expected_digest:
            raise ValueError('dataset differs from approved data')
        batches = [batch for batch in payload['approved_plan']['batches']
                   if batch['batch_id'] == payload['batch_id']]
        if len(batches) != 1 or ids != batches[0]['combination_ids']:
            raise ValueError('combinations differ from approved batch')
        if evidence.get('pending_gates') != ['producer_lock', 'launch_revalidation']:
            raise ValueError('unexpected pending gates')
        for field, expected in (
            ('batch_id', payload['batch_id']),
            ('portfolio_id', payload['approved_plan']['portfolio_id']),
            ('config_hash', payload['approved_config_hash']),
            ('promotion_status', 'not_accepted')):
            if evidence.get(field) != expected:
                raise ValueError('receipt identity mismatch')
    except Exception:
        raise RuntimeError('preflight receipt rejected') from None
    return evidence
