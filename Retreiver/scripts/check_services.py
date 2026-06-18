#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wns_env import load_env_files as load_wns_env_files, service_base_from_endpoint


def tcp_check(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "TCP open"
    except OSError as exc:
        return False, str(exc)


def http_check(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def post_json_check(url: str, payload: dict, timeout: float = 180.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        if body.get("embeddings") or body.get("data") or body.get("scores") or body.get("results"):
            return 200 <= resp.status < 300, f"HTTP {resp.status} keys={','.join(sorted(body.keys()))}"
        return False, f"HTTP {resp.status} unexpected keys={sorted(body.keys())}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}"
    except Exception as exc:
        return False, str(exc)


def pgvector_auth_check() -> tuple[bool, str]:
    dsn = os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        return False, "PGVECTOR_DSN or DATABASE_URL is not set"
    try:
        import psycopg  # type: ignore[import-not-found]
    except Exception as exc:
        return False, f"psycopg is not installed: {exc}"
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1").fetchone()
        return True, "SELECT 1 OK"
    except Exception as exc:
        return False, str(exc)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def record(label: str, ok: bool, msg: str, *, required: bool, failures: list[str]) -> None:
    status = "OK" if ok else ("MISSING" if required else "CHECK")
    print(f"- {label}: {status} - {msg}")
    if required and not ok:
        failures.append(f"{label}: {msg}")


def main() -> int:
    load_wns_env_files(ROOT)
    qdrant_url = env("QDRANT_URL", "http://127.0.0.1:5001").rstrip("/")
    weaviate_url = env("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/")
    model_adapter_url = service_base_from_endpoint(env("MODEL_ADAPTER_URL", "http://127.0.0.1:5000"))

    print("Service readiness")
    failures: list[str] = []

    ok, msg = http_check(model_adapter_url + "/health")
    record(f"Model adapter {model_adapter_url}/health", ok, msg, required=True, failures=failures)

    ok, msg = http_check(qdrant_url + "/healthz")
    record(f"Qdrant {qdrant_url}/healthz", ok, msg, required=True, failures=failures)

    ok, msg = pgvector_auth_check()
    record("PGVector DSN auth", ok, msg, required=True, failures=failures)

    candidates = [weaviate_url + "/v1/.well-known/ready", weaviate_url + "/v1/meta"]
    weaviate_results = [http_check(u) for u in candidates]
    weaviate_ok = any(item_ok for item_ok, _ in weaviate_results)
    record(f"Weaviate {weaviate_url}", weaviate_ok, str(weaviate_results), required=True, failures=failures)

    optional_posts = [
        ("Jina embedding", env("JINA_EMBEDDING_URL"), {"texts": ["refund policy", "flight change"]}),
        ("GTE embedding", env("GTE_EMBEDDING_URL"), {"texts": ["refund policy", "flight change"]}),
        ("BGE rerank", env("BGE_RERANK_URL"), {"query": "refund policy", "documents": ["refund policy details", "seat selection rules"], "top_k": 2}),
        ("Qwen rerank", env("QWEN_RERANK_URL"), {"query": "refund policy", "documents": ["refund policy details", "seat selection rules"], "top_k": 2}),
    ]
    for label, url, payload in optional_posts:
        if not url:
            print(f"- {label}: CHECK - endpoint env var not set")
            continue
        ok, msg = post_json_check(url, payload)
        record(f"{label} {url}", ok, msg, required=False, failures=failures)

    commercial_missing = []
    if env("OPENAI_API_KEY"):
        print("- OpenAI embedding credential: OK - OPENAI_API_KEY set")
    else:
        commercial_missing.append("OPENAI_API_KEY")
    aws_required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    aws_missing = [key for key in aws_required if not env(key)]
    if aws_missing:
        commercial_missing.extend(aws_missing)
    if not (env("AWS_REGION") or env("AWS_DEFAULT_REGION")):
        commercial_missing.append("AWS_REGION/AWS_DEFAULT_REGION")
    if commercial_missing:
        print("- Commercial optional adapters: CHECK - missing " + ", ".join(commercial_missing))
    else:
        print("- Commercial optional adapters: OK - OpenAI/AWS variables present")

    print("\nDocker hint")
    print("Run from CMD on Windows:")
    print("  docker compose -f docker-compose.benchmark.yml up -d qdrant postgres-pgvector weaviate")
    print("Ports: model adapters 5000, Qdrant 5001/5002, PGVector 5003, Weaviate 5004/5005, dashboard 5009")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
