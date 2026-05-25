# Adapter install notes

## Order of operations on VM

1. Start VM services from `wns-vm-adapters`:

```bash
bash scripts/setup_vm.sh
bash scripts/start_vector_dbs.sh
bash scripts/start_embedding_service.sh
```

2. Install benchmark adapters from this bundle:

```bash
bash scripts/install_adapters.sh /path/to/Retreiver
```

3. Put VM URLs in the benchmark repo `.env`.

4. Run:

```bash
bash scripts/run_smoke.sh /path/to/Retreiver
```

## What the installer changes

### Embeddings

Before:

```text
jina_v3 -> local_hash
gte_multilingual_base -> local_hash
```

After:

```text
jina_v3 -> remote_http, JINA_EMBEDDING_URL
gte_multilingual_base -> remote_http, GTE_EMBEDDING_URL
```

### Vector stores

Before:

```text
Qdrant -> local_vector
PGVector -> local_vector
Weaviate -> local_vector
```

After:

```text
Qdrant -> qdrant
PGVector -> pgvector
Weaviate -> weaviate
```

## Notes

- Qdrant adapter uses cosine distance collection.
- PGVector adapter creates an HNSW cosine index.
- Weaviate adapter creates a `vectorizer: none` class with HNSW/cosine config.
- Collections/classes/tables are per-process names to avoid clashes during smoke tests.
