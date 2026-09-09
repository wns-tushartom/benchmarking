"""Read-only data binding supplement, not a standalone completion verifier.

Call alongside repository receipt/hash verification and strict full-depth CSV
verification. Approved SHA-256 identities must come from the queue selection
contract, never from the result being verified.
"""
import hashlib
import re
from pathlib import Path
from benchmarking.core.config import config_hash


def verify_data_identity(root, config, approved_hashes):
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError('repository path alias rejected')
    exp = config['experiment']
    paths = []
    for key in ('dataset', 'corpus_workbook'):
        name = exp.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError('missing data path')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('noncanonical data path')
        path = root / relative
        if path.resolve() != path or not path.is_file():
            raise ValueError('missing data file or path alias')
        expected = approved_hashes.get(name)
        if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
            raise ValueError('approved data identity missing')
        paths.append((path, expected))
    # Reproduce upstream's path-sensitive combined dataset hash, streaming
    # instead of loading a large workbook into memory.
    combined = hashlib.sha256()
    for path, expected in paths:
        digest = hashlib.sha256()
        combined.update(str(path).encode('utf-8'))
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
                combined.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('approved data identity mismatch')
    return combined.hexdigest()[:16]


def verify_bindings(root, config, snapshot, manifest, approved_hashes):
    if snapshot != config:
        raise ValueError('config snapshot mismatch')
    combined_hash = verify_data_identity(root, config, approved_hashes)
    if manifest.get('dataset_hash') != combined_hash:
        raise ValueError('manifest dataset binding mismatch')
    expected_config = config_hash(config)
    for key in ('config_hash', 'official_matrix_contract_hash', 'selected_run_config_hash'):
        if manifest.get(key) != expected_config:
            raise ValueError('manifest config binding mismatch')
    if type(manifest.get('query_count')) is not int or manifest['query_count'] != 500:
        raise ValueError('manifest must declare exactly 500 queries')
    return {'data_bindings_verified': True, 'config_binding_verified': True,
            'completion_verified': False}
