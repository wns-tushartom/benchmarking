#!/usr/bin/env python3
"""Shared WNS environment loading helpers.

Later VM-generated files should override stale local defaults, but blank generated
values must not erase real credentials already present in the process or .env.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILES = (".env", ".env.project-smiley-nvidia", ".env.vm.generated")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_env_files(root: Path, names: tuple[str, ...] = DEFAULT_ENV_FILES, *, override: bool = True) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for name in names:
        for key, value in parse_env_file(root / name).items():
            loaded[key] = value
            current = os.environ.get(key)
            if value == "" and current:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def service_base_from_endpoint(value: str, default: str = "http://127.0.0.1:5000") -> str:
    base = (value or default).strip().rstrip("/")
    for suffix in ("/embed/gte", "/embed/jina", "/rerank/bge", "/rerank/qwen", "/health"):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip("/")
    return base
