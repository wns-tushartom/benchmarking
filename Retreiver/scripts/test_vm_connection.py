#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def http_get(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def http_post_json(url: str, payload: dict, timeout: float = 90.0) -> tuple[bool, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            dims = body.get("dimensions")
            count = body.get("count") or len(body.get("embeddings") or [])
            return 200 <= resp.status < 300 and bool(dims) and bool(count), f"HTTP {resp.status}, count={count}, dimensions={dims}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}"
    except Exception as exc:
        return False, str(exc)


def tcp_check_from_dsn(dsn: str, timeout: float = 5.0) -> tuple[bool, str]:
    parsed = urllib.parse.urlparse(dsn)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"TCP {host}:{port}"
    except Exception as exc:
        return False, str(exc)


def show(name: str, ok: bool, msg: str) -> bool:
    print(f"- {name}: {'OK' if ok else 'FAIL'} - {msg}")
    return ok


def main() -> int:
    load_dotenv(ROOT / ".env")
    jina = os.getenv("JINA_EMBEDDING_URL", "").strip()
    gte = os.getenv("GTE_EMBEDDING_URL", "").strip()
    qdrant = os.getenv("QDRANT_URL", "").strip().rstrip("/")
    pg = os.getenv("PGVECTOR_DSN", "").strip()
    weaviate = os.getenv("WEAVIATE_URL", "").strip().rstrip("/")

    print("WNS VM connection check")
    missing = [k for k, v in {
        "JINA_EMBEDDING_URL": jina,
        "GTE_EMBEDDING_URL": gte,
        "QDRANT_URL": qdrant,
        "PGVECTOR_DSN": pg,
        "WEAVIATE_URL": weaviate,
    }.items() if not v or "VM_HOST" in v]
    if missing:
        print("Missing or placeholder .env values:", ", ".join(missing))
        print("Run: scripts\\windows\\Configure-LocalFrontendForVM.cmd VM_HOST_OR_IP")
        return 2

    results = [
        show("Jina embedding", *http_post_json(jina, {"texts": ["smoke test"]})),
        show("GTE embedding", *http_post_json(gte, {"texts": ["smoke test"]})),
        show("Qdrant", *http_get(qdrant + "/")),
        show("PGVector", *tcp_check_from_dsn(pg)),
        show("Weaviate", *http_get(weaviate + "/v1/meta")),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
