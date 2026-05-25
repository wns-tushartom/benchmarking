# Deployment notes

Only ports `5000-5010` are used.

```text
5000 embedding service
5001 Qdrant HTTP
5002 Qdrant gRPC
5003 PGVector/Postgres
5004 Weaviate
5005 Jupyter optional
```

## Start service in tmux

```bash
tmux new -s wns-embed
bash scripts/start_embedding_service.sh
```

Detach: `Ctrl-b`, then `d`.

## Jupyter

```bash
source .venv/bin/activate
jupyter lab --ip 0.0.0.0 --port 5005 --no-browser
```

Open `notebooks/vm_smoke_test.py`.
