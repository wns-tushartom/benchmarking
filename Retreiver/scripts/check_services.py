#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request


def tcp_check(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_check(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def main() -> None:
    qdrant_url = env("QDRANT_URL", "http://127.0.0.1:5001")
    weaviate_url = env("WEAVIATE_URL", "http://127.0.0.1:5004")

    print("Service readiness")

    ok, msg = http_check(qdrant_url.rstrip("/") + "/")
    print(f"- Qdrant {qdrant_url}: {'OK' if ok else 'MISSING'} - {msg}")

    pg_ok = tcp_check("127.0.0.1", 5003)
    print(f"- PGVector/Postgres 127.0.0.1:5003: {'OK' if pg_ok else 'MISSING'}")

    # Weaviate may return 404 on root but should expose readiness/meta endpoints.
    candidates = [weaviate_url.rstrip("/") + "/v1/.well-known/ready", weaviate_url.rstrip("/") + "/v1/meta"]
    weaviate_results = [http_check(u) for u in candidates]
    weaviate_ok = any(ok for ok, _ in weaviate_results)
    print(f"- Weaviate {weaviate_url}: {'OK' if weaviate_ok else 'MISSING'} - {weaviate_results}")

    qwen_url = env("QWEN_RERANK_URL")
    if qwen_url:
        ok, msg = http_check(qwen_url)
        print(f"- Qwen rerank VM endpoint {qwen_url}: {'OK' if ok else 'CHECK'} - {msg}")
    else:
        print("- Qwen rerank VM endpoint: MISSING - set QWEN_RERANK_URL when VM endpoint is ready")

    bge_url = env("BGE_RERANK_URL")
    if bge_url:
        ok, msg = http_check(bge_url)
        print(f"- BGE rerank VM endpoint {bge_url}: {'OK' if ok else 'CHECK'} - {msg}")
    else:
        print("- BGE rerank VM endpoint: MISSING - set BGE_RERANK_URL when VM endpoint is ready")

    jina_url = env("JINA_EMBEDDING_URL")
    if jina_url:
        ok, msg = http_check(jina_url)
        print(f"- Jina embedding VM endpoint {jina_url}: {'OK' if ok else 'CHECK'} - {msg}")
    else:
        print("- Jina embedding VM endpoint: MISSING - set JINA_EMBEDDING_URL when VM endpoint is ready")

    gte_url = env("GTE_EMBEDDING_URL")
    if gte_url:
        ok, msg = http_check(gte_url)
        print(f"- GTE embedding VM endpoint {gte_url}: {'OK' if ok else 'CHECK'} - {msg}")
    else:
        print("- GTE embedding VM endpoint: MISSING - set GTE_EMBEDDING_URL when VM endpoint is ready")

    print("\nDocker hint")
    print("Run from CMD on Windows:")
    print("  docker compose -f docker-compose.benchmark.yml up -d qdrant postgres-pgvector weaviate")
    print("Ports: model adapters 5000, Qdrant 5001/5002, PGVector 5003, Weaviate 5004/5005")


if __name__ == "__main__":
    main()
