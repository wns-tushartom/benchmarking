#!/usr/bin/env python3
"""Health check for NVIDIA RAG Blueprint services used by Project Smiley.

Checks the externally reachable Project Smiley ports:
  5006 -> NVIDIA rag-server /v1
  5007 -> NVIDIA ingestor-server /v1
  5008 -> NVIDIA reference frontend

Outputs human-readable text by default and can write JSON for the dashboard.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "nvidia_rag" / "health.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def http_json(url: str, timeout: float) -> dict[str, Any]:
    started = time.time()
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {"raw": body[:1000]}
            return {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "latency_ms": round((time.time() - started) * 1000, 2),
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "latency_ms": round((time.time() - started) * 1000, 2), "error": exc.read().decode("utf-8", errors="replace")[:1000]}
    except Exception as exc:
        return {"ok": False, "status": None, "latency_ms": round((time.time() - started) * 1000, 2), "error": str(exc)}


def service_urls() -> dict[str, str]:
    return {
        "rag_server": os.getenv("NVIDIA_RAG_SERVER_URL", "http://127.0.0.1:5006").rstrip("/"),
        "ingestor": os.getenv("NVIDIA_INGESTOR_URL", "http://127.0.0.1:5007").rstrip("/"),
        "frontend": os.getenv("NVIDIA_RAG_FRONTEND_URL", "http://127.0.0.1:5008").rstrip("/"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    load_env_file(ROOT / ".env.vm.generated")

    urls = service_urls()
    checks = {
        "rag_server": {"url": urls["rag_server"] + "/v1/health?check_dependencies=true", **http_json(urls["rag_server"] + "/v1/health?check_dependencies=true", args.timeout)},
        "ingestor": {"url": urls["ingestor"] + "/v1/health?check_dependencies=true", **http_json(urls["ingestor"] + "/v1/health?check_dependencies=true", args.timeout)},
        "frontend": {"url": urls["frontend"], **http_json(urls["frontend"], args.timeout)},
    }
    overall_ok = checks["rag_server"]["ok"] and checks["ingestor"]["ok"]
    payload = {
        "ok": overall_ok,
        "service": "nvidia_rag_blueprint",
        "ports": {"rag_server": 5006, "ingestor": 5007, "frontend": 5008},
        "urls": urls,
        "checks": checks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("NVIDIA RAG Blueprint readiness")
        for name, check in checks.items():
            print(f"- {name}: {'OK' if check['ok'] else 'MISSING'} {check['url']} {check.get('status') or ''} {check.get('error', '')}")
        print(f"Wrote {out.relative_to(ROOT)}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
