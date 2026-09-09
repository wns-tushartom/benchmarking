"""Strict CSV depth/metric validation supplement to artifact/identity verification.

This module alone does NOT verify completion: callers must first validate the
immutable plan, receipt, artifact hashes and snapshot data bindings.
"""
import csv
import math
from pathlib import Path

class DepthError(ValueError):
    pass

def verify_full_depth(summary_path, expected_ids, *, id_column, metrics):
    expected = set(expected_ids)
    if not expected or len(expected) != len(expected_ids):
        raise DepthError('Expected identities must be nonempty and unique')
    if not metrics:
        raise DepthError('Required metrics must be explicit')
    with Path(summary_path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {id_column, 'status', 'query_count', *metrics}
        if not required.issubset(reader.fieldnames or []):
            raise DepthError('Missing required summary columns')
        seen = set()
        for row in reader:
            identity = row[id_column]
            if identity not in expected or identity in seen:
                raise DepthError('Unexpected or duplicate identity')
            if row['status'] != 'completed':
                raise DepthError('Noncompleted row')
            if row['query_count'] != '500':
                raise DepthError('Full depth requires exactly 500 queries')
            for key in metrics:
                try:
                    value = float(row[key])
                except (TypeError, ValueError) as exc:
                    raise DepthError('Nonnumeric required metric: ' + key) from exc
                if not math.isfinite(value):
                    raise DepthError('Nonfinite required metric: ' + key)
            seen.add(identity)
    if seen != expected:
        raise DepthError('Missing expected combinations')
    return {'verified_full_depth_rows': len(seen), 'queries_per_row': 500}
