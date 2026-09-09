"""Validate installed reranker identity enforcement without editing its source."""
import inspect


def verify_identity_contract(adapter_class):
    """Exercise real adapter against synthetic local responses, never network.

    This verifies guard behavior, not provider readiness. Must be executed in a
    separate preflight process because its transport binding is temporarily patched.
    """
    import os
    from types import SimpleNamespace
    from unittest.mock import patch
    module = inspect.getmodule(adapter_class)
    if module is None:
        raise RuntimeError('Cannot resolve adapter module')
    hit = SimpleNamespace(chunk=SimpleNamespace(paragraph='probe document'))
    responses = ({'scores': [0.7]}, {'model': 'wrong-model', 'scores': [0.7]})
    for response in responses:
        with patch.dict(os.environ, {'QUEUE_GUARD_URL': 'http://127.0.0.1:1'}):
            adapter = adapter_class(name='queue-contract-probe', endpoint_env='QUEUE_GUARD_URL',
                                    model='expected-model', require_response_model=True,
                                    expected_response_model='expected-model')
        with patch.object(module, '_post_json', return_value=response):
            try:
                adapter.rerank('probe', [hit], 1)
            except RuntimeError as error:
                if 'model' not in str(error) or 'mismatch' not in str(error):
                    raise RuntimeError('Failure did not establish model-identity enforcement') from error
            else:
                raise RuntimeError('Installed reranker accepts missing or mismatched model identity')
    with patch.dict(os.environ, {'QUEUE_GUARD_URL': 'http://127.0.0.1:1'}):
        adapter = adapter_class(name='queue-contract-probe', endpoint_env='QUEUE_GUARD_URL',
                                model='expected-model', require_response_model=True,
                                expected_response_model='expected-model')
    with patch.object(module, '_post_json', return_value={'model': 'expected-model', 'scores': [0.7]}):
        result = adapter.rerank('probe', [hit], 1)
    if len(result) != 1 or result[0].score != 0.7:
        raise RuntimeError('Correct-model contract probe failed')
    return {'identity_guard_verified': True, 'network_requests': 0}
