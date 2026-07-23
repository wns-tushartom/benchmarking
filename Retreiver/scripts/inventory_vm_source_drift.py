#!/usr/bin/env python3
"""Read-only inventory of VM changes that could be lost during source deployment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREFIXES = (
    "scripts/",
    "web/",
    "source/",
    "benchmarking/",
    "configs/",
    "tests/",
)
RUNTIME_PREFIXES = (
    "data/",
    "logs/",
    "runtime/",
    "artifacts/",
)


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _unique(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in paths:
        path = _normalize(raw)
        if path and path not in seen:
            seen.add(path)
            result.append(path)
    return result


def classify_paths(paths: Iterable[str]) -> dict[str, object]:
    merge_back_required: list[str] = []
    runtime_artifacts: list[str] = []
    unclassified: list[str] = []
    for path in _unique(paths):
        if path.startswith(SOURCE_PREFIXES):
            merge_back_required.append(path)
        elif path.startswith(RUNTIME_PREFIXES):
            runtime_artifacts.append(path)
        else:
            unclassified.append(path)
    return {
        "merge_back_required": merge_back_required,
        "runtime_artifacts": runtime_artifacts,
        "unclassified": unclassified,
        "safe_to_pull_source": not merge_back_required and not unclassified,
    }


def porcelain_paths(payload: bytes) -> list[str]:
    fields = payload.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        raw = fields[index]
        index += 1
        if not raw:
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        if len(text) < 4:
            continue
        status = text[:2]
        paths.append(text[3:])
        if "R" in status or "C" in status:
            if index < len(fields) and fields[index]:
                paths.append(fields[index].decode("utf-8", errors="surrogateescape"))
                index += 1
    return _unique(paths)


def classify_porcelain_status(payload: bytes) -> dict[str, object]:
    return classify_paths(porcelain_paths(payload))


def read_git_status(root: Path = ROOT) -> bytes:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or "git status failed")
    return result.stdout


def exit_code(report: dict[str, object]) -> int:
    if report.get("merge_back_required"):
        return 2
    if report.get("unclassified"):
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print("inventory_vm_source_drift.py takes no arguments", file=sys.stderr)
        return 64
    try:
        report = classify_porcelain_status(read_git_status(ROOT))
    except Exception as exc:
        report = {
            "merge_back_required": [],
            "runtime_artifacts": [],
            "unclassified": [],
            "safe_to_pull_source": False,
            "error": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
