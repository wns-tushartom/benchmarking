#!/usr/bin/env python3
"""Build a source-only VM overlay for the isolated retrieval/reranker candidate lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any


OVERLAY_NAME = "Retreiver-retrieval-reranker-candidates-overlay.zip"
OVERLAY_FILES = (
    "benchmarking/adapters/local.py",
    "benchmarking/adapters/remote_embeddings.py",
    "benchmarking/adapters/remote_rerankers.py",
    "benchmarking/core/registry.py",
    "benchmarking/core/runner.py",
    "benchmarking/core/schemas.py",
    "benchmarking/retrieval.py",
    "configs/benchmark.retrieval-reranker-candidates.json",
    "configs/project_matrix_candidate_catalog.json",
    "docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md",
    "requirements.txt",
    "scripts/benchmark_cli.py",
    "scripts/package_retrieval_reranker_candidate_overlay.py",
    "scripts/wns_vm_adapter_service.py",
    "tests/test_candidate_integrity_guards.py",
    "tests/test_hybrid_retriever.py",
    "tests/test_modular_candidate_matrix.py",
    "tests/test_retrieval_reranker_candidate_overlay.py",
    "tests/test_vm_candidate_rerankers.py",
)


def source_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def worktree_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def source_fingerprint(member_sha256: dict[str, str]) -> str:
    payload = json.dumps(member_sha256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def archive_member(repo_root: Path, relative_path: str) -> tuple[str, bytes]:
    path = repo_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required overlay source is missing: {path}")
    if relative_path.startswith("data/") or path.name in {".env", ".env.vm.generated"}:
        raise ValueError(f"Forbidden overlay member: {relative_path}")
    return f"{repo_root.name}/{relative_path}", path.read_bytes()


def build_overlay(repo_root: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / OVERLAY_NAME
    members = [archive_member(repo_root, relative_path) for relative_path in OVERLAY_FILES]
    member_sha256 = {
        relative_path: hashlib.sha256(content).hexdigest()
        for relative_path, (_archive_path, content) in zip(OVERLAY_FILES, members)
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, content in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as zf:
        crc_failure = zf.testzip()
        names = zf.namelist()
    receipt = {
        "archive": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "base_git_commit": source_commit(repo_root),
        "worktree_dirty": worktree_dirty(repo_root),
        "member_sha256": member_sha256,
        "source_fingerprint": source_fingerprint(member_sha256),
        "members": names,
        "member_count": len(names),
        "zip_crc_ok": crc_failure is None,
        "zip_crc_failure": crc_failure,
        "policy": "source-only candidate overlay; excludes data, artifacts, caches, secrets, and model weights",
    }
    (output_dir / f"{archive.stem}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    archive, receipt = build_overlay(repo_root, args.output_dir)
    print(json.dumps({"archive": str(archive), **receipt}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
