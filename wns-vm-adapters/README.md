# WNS VM Adapters

VM-side services for WNS benchmarking, restricted to ports `5000-5010`.

## Port map

```text
5000: embedding service, Jina/GTE
5001: Qdrant HTTP
5002: Qdrant gRPC
5003: PGVector/Postgres
5004: Weaviate
5005: optional Jupyter
```

## Quick start on VM

```bash
git clone <YOUR_GITHUB_REPO_URL> wns-vm-adapters
cd wns-vm-adapters
bash scripts/setup_vm.sh
cp .env.example .env
bash scripts/start_vector_dbs.sh
bash scripts/start_embedding_service.sh
```

In another terminal:

```bash
bash scripts/check_all.sh
```

## URLs for benchmark `.env`

If benchmark runner talks to this VM:

```env
JINA_EMBEDDING_URL=http://VM_HOST:5000/embed/jina
GTE_EMBEDDING_URL=http://VM_HOST:5000/embed/gte
QDRANT_URL=http://VM_HOST:5001
PGVECTOR_DSN=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
DATABASE_URL=postgresql://wns:wns_password@VM_HOST:5003/wns_benchmark
WEAVIATE_URL=http://VM_HOST:5004
```

If benchmark runner is on same VM, use `127.0.0.1`.

## Jupyter option

```bash
source .venv/bin/activate
jupyter lab --ip 0.0.0.0 --port 5005 --no-browser
```

Use Jupyter for smoke tests only. Use `tmux` or `systemd` for long-running services.
