#!/usr/bin/env python3
"""Verify the canonical dashboard release contract without mutating the tree."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "dashboard_release_contract.json"
SAFE_DIRTY_PREFIXES = ("tests/", "docs/", "artifacts/")


class _IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for name, value in attrs:
            if name == "id" and value:
                self.ids.append(value)


def duplicate_html_ids(path: Path) -> list[str]:
    parser = _IdParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return sorted(value for value, count in Counter(parser.ids).items() if count > 1)


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release contract must be a JSON object")
    return payload


def _tracked_dirty_paths(root: Path) -> list[str] | None:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None
    status = subprocess.run(
        ["git", "diff", "--name-only", "--relative", "HEAD", "--"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "git diff failed")
    return [line.strip() for line in status.stdout.splitlines() if line.strip()]


def _matrix_counts(root: Path) -> dict[str, int]:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    server = importlib.import_module("scripts.serve_benchmark_dashboard")
    options = server.benchmark_options()
    chunkers = len(options.get("chunkers") or [])
    embeddings = len(options.get("embeddings") or [])
    stores = len(options.get("vector_stores") or options.get("stores") or [])
    rerankers = len(options.get("rerankers") or [])
    return {
        "chunkers": chunkers,
        "embeddings": embeddings,
        "stores": stores,
        "rerankers": rerankers,
        "total": chunkers * embeddings * stores * rerankers,
    }


def verify_release(
    root: Path,
    contract_path: Path,
    *,
    allow_dirty_tests: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    notes: list[str] = []
    try:
        contract = _load_contract(contract_path)
    except Exception as exc:
        return {"ok": False, "errors": [f"contract: {exc}"], "notes": notes}

    required_files = ["web/index.html", "web/app.js", "scripts/serve_benchmark_dashboard.py"]
    required_files.extend(contract.get("required_files") or [])
    for relative in dict.fromkeys(required_files):
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    index_path = root / "web" / "index.html"
    server_path = root / "scripts" / "serve_benchmark_dashboard.py"
    if index_path.is_file():
        index = index_path.read_text(encoding="utf-8")
        for page in contract.get("pages") or []:
            if f'data-page-panel="{page}"' not in index:
                errors.append(f"missing page panel: {page}")
        for dom_id in contract.get("dom_ids") or []:
            if f'id="{dom_id}"' not in index:
                errors.append(f"missing DOM id: {dom_id}")
        duplicates = duplicate_html_ids(index_path)
        if duplicates:
            errors.append(f"duplicate HTML ids: {', '.join(duplicates)}")

    if server_path.is_file():
        server_text = server_path.read_text(encoding="utf-8")
        for route in contract.get("api_routes") or []:
            if route not in server_text:
                errors.append(f"missing API route: {route}")

    for relative, tokens in (contract.get("file_tokens") or {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"token file missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"missing token in {relative}: {token}")

    for relative, tokens in (contract.get("forbidden_tokens") or {}).items():
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in text:
                errors.append(f"forbidden token in {relative}: {token}")

    try:
        actual_matrix = _matrix_counts(root)
        if actual_matrix != contract.get("matrix"):
            errors.append(
                "matrix contract mismatch: "
                f"expected={contract.get('matrix')} actual={actual_matrix}"
            )
    except Exception as exc:
        errors.append(f"matrix verification failed: {exc}")

    app_path = root / "web" / "app.js"
    if app_path.is_file():
        node = subprocess.run(
            ["node", "--check", str(app_path)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if node.returncode != 0:
            errors.append(f"web/app.js syntax failed: {node.stderr.strip() or node.stdout.strip()}")

    try:
        dirty = _tracked_dirty_paths(root)
        if dirty is None:
            notes.append("git metadata unavailable; tracked-tree cleanliness not checked")
        elif dirty:
            unsafe = [path for path in dirty if not path.startswith(SAFE_DIRTY_PREFIXES)]
            if not allow_dirty_tests:
                errors.append(f"tracked tree is dirty: {', '.join(dirty)}")
            elif unsafe:
                errors.append(f"tracked source tree is dirty: {', '.join(unsafe)}")
            else:
                notes.append(f"allowed dirty non-source paths: {', '.join(dirty)}")
    except Exception as exc:
        errors.append(f"tracked-tree check failed: {exc}")

    return {"ok": not errors, "errors": errors, "notes": notes}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--allow-dirty-tests", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    contract = args.contract.resolve()
    report = verify_release(root, contract, allow_dirty_tests=args.allow_dirty_tests)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for note in report["notes"]:
            print(f"NOTE: {note}")
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        if report["ok"]:
            print("DASHBOARD_RELEASE_CONTRACT_OK")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
