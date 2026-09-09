"""Assemble a pinned patch pair in a NEW directory; never install in place.

Failure after directory creation preserves incomplete staging for diagnosis.
A receipt is written last. This is not deployment approval or a sandbox against
concurrent hostile same-UID filesystem changes.
"""
import hashlib
import json
from pathlib import Path
import queue_writer_patch


def _canonical(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError('noncanonical staging/source path')
    return path


def stage_patch_bundle(source_root, destination):
    root = _canonical(source_root)
    target = _canonical(destination)
    if not target.parent.is_dir():
        raise ValueError('staging parent must exist')
    pins = json.loads(Path(__file__).with_name('queue_writer_source_contract.json').read_text())
    transform = Path(queue_writer_patch.__file__)
    if hashlib.sha256(transform.read_bytes()).hexdigest() != pins['transform_sha256']:
        raise ValueError('patch transformation differs from pinned identity')
    sources = {}
    for relative in pins['before_sha256']:
        path = _canonical(root / relative)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('invalid source path')
        sources[relative] = path.read_bytes().decode('utf-8')
    bundle = queue_writer_patch.prepare_patch_bundle(sources, pins['before_sha256'])
    if bundle['after_sha256'] != pins['after_sha256']:
        raise ValueError('prepared source differs from pinned output')
    for relative, source in bundle['sources'].items():
        compile(source, relative, 'exec')
    target.mkdir(mode=0o700, exist_ok=False)
    for relative, source in bundle['sources'].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(source.encode('utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != pins['after_sha256'][relative]:
            raise ValueError('staged source verification failed')
    receipt = {'before_sha256': bundle['before_sha256'],
               'after_sha256': bundle['after_sha256'],
               'release_status': 'not_deployable', 'installed': False}
    with (target / 'STAGING_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    return receipt
