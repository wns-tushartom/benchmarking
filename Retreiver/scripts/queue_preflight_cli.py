"""Private, bounded-child entry point for preflight (not benchmark launch)."""
import argparse
import json
import os
from pathlib import Path
import sys


def execute(request_path, receipt_path, *, probe=None):
    request_path = Path(request_path).absolute()
    receipt_path = Path(receipt_path).absolute()
    if request_path.resolve() != request_path or receipt_path.resolve() != receipt_path:
        raise ValueError('noncanonical path')
    if receipt_path.exists() or not receipt_path.parent.is_dir():
        raise ValueError('receipt destination unavailable')
    if not request_path.is_file() or request_path.stat().st_size > 8_000_000:
        raise ValueError('invalid request file')
    payload = json.loads(request_path.read_text())
    required = {'schema_version', 'root', 'config', 'approved_plan', 'batch_id',
                'approved_hashes', 'approved_config_hash', 'allow_openai'}
    if not isinstance(payload, dict) or set(payload) != required or payload['schema_version'] != 1:
        raise ValueError('invalid request contract')
    if type(payload['allow_openai']) is not bool:
        raise ValueError('authorization must be boolean')
    if probe is None:
        from queue_batch_preflight import preflight_batch
        probe = preflight_batch
    result = probe(**{k: v for k, v in payload.items() if k != 'schema_version'})
    if result.get('preflight_verified') is not True or result.get('batch_ready') is not False:
        raise ValueError('invalid preflight result')
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    # Exclusive creation: never overwrite previous evidence. Partial files are
    # retained after write failures and are not valid JSON completion evidence.
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    try:
        execute(args.request, args.receipt)
    except Exception:
        print('preflight rejected; no batch launch authorized', file=sys.stderr)
        return 2
    print('preflight verified; remaining admission gates still required')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
