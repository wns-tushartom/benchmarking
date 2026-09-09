"""Synthetic vector-store round trips; never mark whole batches ready.

FAISS probes are ephemeral. Network stores remain pending until isolated,
non-destructive namespace lifecycle probes are supported.
"""
import math
from benchmarking.core.config import technique
from benchmarking.core.schemas import Chunk


def probe_vector_stores(config, combinations):
    required = {}
    for row in combinations:
        params = technique(config, 'vector_stores', row['vector_store'])
        dimension = technique(config, 'embeddings', row['embedding']).get('dimensions')
        if type(dimension) is not int or not 2 <= dimension <= 65536:
            raise RuntimeError('unsupported embedding dimension')
        if params.get('adapter') != 'faiss':
            raise RuntimeError('vector store pending isolated round-trip support')
        index = row['index_type']
        if index not in ('HNSW', 'Flat'):
            raise RuntimeError('unsupported vector index')
        if any(params.get(k) for k in ('index_dir', 'index_root', 'namespace', 'load_existing')):
            raise RuntimeError('persistent vector settings require isolated probe design')
        required[row['vector_store'], index, dimension] = params
    if not required:
        raise RuntimeError('empty vector selection')
    records = []
    for (name, index, dimension), params in sorted(required.items()):
        try:
            from benchmarking.adapters.vector_faiss import FaissVectorStoreAdapter
            # Explicitly use no persistent namespace or existing index.
            store = FaissVectorStoreAdapter(name=name, index_type=index,
                index_dir=None, load_existing=False,
                **{k: params[k] for k in ('hnsw_m', 'ef_construction', 'ef_search') if k in params})
            vectors = [[1.0, 0.0] + [0.0]*(dimension-2),
                       [0.0, 1.0] + [0.0]*(dimension-2)]
            chunks = [Chunk(1, 'queue-probe', 'first synthetic record'),
                      Chunk(2, 'queue-probe', 'second synthetic record')]
            receipt = store.upsert(chunks, vectors)
            if receipt.get('vector_count') != 2:
                raise ValueError('count mismatch')
            for expected, vector in zip(chunks, vectors):
                hits = store.search(vector, 2)
                if len(hits) != 2 or {h.chunk.id for h in hits} != {1, 2}:
                    raise ValueError('coverage mismatch')
                if hits[0].chunk.id != expected.id or hits[0].chunk.paragraph != expected.paragraph:
                    raise ValueError('identity mismatch')
                scores = [float(h.score) for h in hits]
                if not all(math.isfinite(s) for s in scores) or scores != sorted(scores, reverse=True):
                    raise ValueError('invalid scores')
        except Exception:
            raise RuntimeError('vector round-trip failed; check dependency or adapter') from None
        records.append({'name': name, 'adapter': 'faiss', 'index_type': index,
                        'dimensions': dimension, 'verified': True, 'persistent': False})
    return {'vector_stores_verified': True, 'batch_ready': False, 'stores': records}
