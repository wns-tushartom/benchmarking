"""Model-only preflight; vector-store/data/receipt gates remain separate.

Run this on the VM in an isolated preflight child with a bounded timeout.
No endpoints are replaced and no services are started by this module.
"""
import math
from benchmarking.core.config import technique
from benchmarking.core.schemas import Chunk, SearchHit


def _factory(kind, name, params):
    from benchmarking.core.registry import default_registry
    cls = default_registry().get(kind, params['adapter'])
    kwargs = {k: v for k, v in params.items() if k != 'adapter'}
    if kind == 'embedding':
        kwargs['model_name'] = params.get('model') or name
    else:
        kwargs['name'] = name
    return cls(**kwargs)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _identity(adapter, params):
    expected = params.get('expected_response_model') or params.get('model')
    if adapter.last_response_metadata.get('model') != expected:
        raise RuntimeError('response identity mismatch')


def probe_models(config, combinations, *, factory=None, allow_openai=False):
    factory = factory or _factory
    required = {}
    for row in combinations:
        for kind in ('embedding', 'reranker'):
            name = row[kind]
            required[kind, name] = technique(config, kind + 's', name)
    if not required:
        raise RuntimeError('empty model selection')
    # Validate the ENTIRE selected set before constructing any provider.
    for (kind, name), params in required.items():
        adapter = params.get('adapter')
        authorized_openai = kind == 'embedding' and adapter == 'openai' and allow_openai
        if adapter == 'amazon_bedrock' or (adapter == 'openai' and not authorized_openai) or (params.get('license') == 'commercial' and not authorized_openai):
            raise RuntimeError('paid provider not authorized')
        if kind == 'reranker' and name == 'none' and adapter == 'identity':
            continue
        if adapter != 'remote_http' and not authorized_openai:
            raise RuntimeError('unsupported provider remains pending')
        if (not authorized_openai and params.get('require_response_model') is not True) or not params.get('model'):
            raise RuntimeError('model identity guard required in configuration')
        if kind == 'embedding':
            dimension = params.get('dimensions')
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise RuntimeError('positive configured dimension required')
    records = []
    for (kind, name), params in sorted(required.items()):
        if kind == 'reranker' and name == 'none':
            continue
        try:
            adapter = factory(kind, name, params)
            if kind == 'embedding':
                for role in ('query', 'document'):
                    if params['adapter'] == 'openai':
                        vectors = adapter.embed_many(['queue readiness probe'])
                    else:
                        vectors = adapter.embed_many_with_role(['queue readiness probe'], role)
                    _identity(adapter, params)
                    if len(vectors) != 1 or len(vectors[0]) != params['dimensions']:
                        raise RuntimeError('embedding count or dimension mismatch')
                    if not all(_finite(v) for v in vectors[0]):
                        raise RuntimeError('embedding values must be finite numbers')
            else:
                hits = [SearchHit(Chunk(i, 'queue-probe', text), 0.0)
                        for i, text in enumerate(('A flight schedule changed.', 'A refund was requested.'))]
                result = adapter.rerank('flight schedule', hits, len(hits))
                _identity(adapter, params)
                if len(result) != len(hits) or sorted(h.chunk.id for h in result) != [0, 1]:
                    raise RuntimeError('reranker alignment mismatch')
                if any(h.chunk != hits[h.chunk.id].chunk for h in result):
                    raise RuntimeError('reranker alignment mismatch')
                scores = [h.score for h in result]
                if not all(_finite(s) for s in scores):
                    raise RuntimeError('reranker scores must be finite numbers')
                if scores != sorted(scores, reverse=True):
                    raise RuntimeError('reranker ordering mismatch')
        except Exception as error:
            # Endpoint exception bodies can contain tokens or internal addresses.
            allowed = ('response identity mismatch', 'embedding count or dimension mismatch',
                       'embedding values must be finite numbers', 'reranker alignment mismatch',
                       'reranker scores must be finite numbers', 'reranker ordering mismatch')
            reason = str(error) if str(error) in allowed else 'provider inference failed'
            raise RuntimeError(reason) from None
        records.append({'kind': kind, 'name': name, 'model': params.get('expected_response_model') or params['model'], 'verified': True})
    return {'models_verified': True, 'batch_ready': False, 'providers': records,
            'pending_gates': ['vector_store', 'corpus_identity', 'immutable_plan', 'producer_lock']}
