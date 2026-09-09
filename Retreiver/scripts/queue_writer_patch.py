"""Review-only pure source transformation; never installs files.

Exclusive creation protects cooperating patched portfolio writers, not hostile
same-UID processes or unpatched writers. Unknown source requires manual review.
"""
ANCHOR = '    load_env_file(root)\n    cfg = official_cfg'
REPLACEMENT = ('    if portfolio_context is not None:\n'
               '        parent = output_dir.parent\n'
               '        if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent:\n'
               '            raise ValueError("canonical portfolio root must already exist")\n'
               '        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)\n' + ANCHOR)

def patch_source(source: str) -> str:
    if source.count(ANCHOR) != 1 or REPLACEMENT in source:
        raise ValueError('unsupported or already modified runner; manual review required')
    return source.replace(ANCHOR, REPLACEMENT, 1)

CLI_ANCHOR = '    selected = next(item for item in status["batches"] if item["batch_id"] == batch_id)\n'
CLI_REPLACEMENT = (CLI_ANCHOR + '    if selected.get("state") != "not_run":\n'
                   '        raise ValueError("selected batch is not unstarted; existing evidence preserved")\n')

def patch_cli_source(source: str) -> str:
    if source.count(CLI_ANCHOR) != 1 or CLI_REPLACEMENT in source:
        raise ValueError('unsupported or already modified CLI; manual review required')
    return source.replace(CLI_ANCHOR, CLI_REPLACEMENT, 1)

def prepare_patch_bundle(sources: dict, approved_before_sha256: dict) -> dict:
    """Prepare in memory only; supplied approval must come from source review.

    Hashes are over UTF-8 bytes without newline normalization. This function
    grants no installation authority and never writes repository files.
    """
    import hashlib
    transforms = {'benchmarking/core/runner.py': patch_source,
                  'scripts/benchmark_cli.py': patch_cli_source}
    if (not isinstance(sources, dict) or not isinstance(approved_before_sha256, dict)
            or set(sources) != set(transforms) or set(approved_before_sha256) != set(transforms)):
        raise ValueError('exact writer and CLI source approvals required')
    def digest(text):
        if not isinstance(text, str):
            raise ValueError('source must be UTF-8 text')
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    before = {path: digest(sources[path]) for path in transforms}
    if before != approved_before_sha256:
        raise ValueError('source differs from independently approved baseline')
    prepared = {path: transform(sources[path]) for path, transform in transforms.items()}
    return {'sources': prepared, 'before_sha256': before,
            'after_sha256': {path: digest(text) for path, text in prepared.items()}}
