#!/usr/bin/env python3
"""Build the source-only all-method candidate portfolio VM overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


OVERLAY_NAME = "Retreiver-all-methods-portfolio-overlay.zip"
OVERLAY_FILES = (
    "benchmarking/adapters/vector_turbovec.py",
    "benchmarking/core/config.py",
    "benchmarking/core/portfolio.py",
    "benchmarking/core/registry.py",
    "benchmarking/core/runner.py",
    "benchmarking/core/schemas.py",
    "benchmarking/retrieval.py",
    "configs/benchmark.all-methods-portfolio.json",
    "configs/vm_adapter_ports.json",
    "docker-compose.benchmark.yml",
    ".env.example",
    "requirements-benchmark.txt",
    "scripts/benchmark_cli.py",
    "scripts/dashboard_control_api.py",
    "scripts/dashboard_portfolio_api.py",
    "scripts/package_all_methods_portfolio_overlay.py",
    "scripts/publish_portfolio_results.py",
    "scripts/serve_benchmark_dashboard.py",
    "scripts/start_vm_stack.sh",
    "scripts/vm_adapter_manager.py",
    "scripts/wns_vm_adapter_service.py",
    "web/index.html",
    "web/app.js",
    "web/styles.css",
    "web/source-state.js",
    "web/recommendations.js",
    "web/wns-logo.svg",
    "VM_ADAPTER_PORT_MAP.md",
    "docs/VM_ALL_METHODS_PORTFOLIO_RUNBOOK.md",
    "docs/superpowers/specs/2026-08-20-all-method-portfolio-and-adapter-control-design.md",
    "docs/superpowers/plans/2026-08-20-all-method-portfolio-and-adapter-control.md",
    "tests/test_all_methods_portfolio.py",
    "tests/test_all_methods_portfolio_overlay.py",
    "tests/test_dashboard_adapter_control.py",
    "tests/test_dashboard_portfolio_control.py",
    "tests/test_portfolio_batch_planner.py",
    "tests/test_portfolio_batch_execution.py",
    "tests/test_portfolio_frontend.py",
    "tests/test_portfolio_results_publication.py",
    "tests/test_start_vm_stack_script.py",
    "tests/test_turbovec_runner_persistence.py",
    "tests/test_turbovec_vector_adapter.py",
    "tests/test_vm_adapter_manager.py",
    "tests/test_vm_adapter_profiles.py",
)


def source_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def worktree_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def allowlisted_members_dirty(repo_root: Path) -> bool:
    tracked = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "ls-files", "--error-unmatch", "--", *OVERLAY_FILES],
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        return True
    changed = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "diff", "--quiet", "HEAD", "--", *OVERLAY_FILES],
        capture_output=True,
        text=True,
    )
    if changed.returncode == 0:
        return False
    if changed.returncode == 1:
        return True
    raise RuntimeError(f"could not verify overlay member state: {changed.stderr.strip()}")


def official_latest_semantically_dirty(repo_root: Path) -> bool:
    official_paths = (
        "data/modular_runs/latest/modular_details.csv",
        "data/modular_runs/latest/modular_summary.csv",
    )
    result = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repo_root),
            "diff",
            "--ignore-space-at-eol",
            "--quiet",
            "--",
            *official_paths,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise RuntimeError(f"could not verify official latest artifacts: {result.stderr.strip()}")


def source_fingerprint(member_sha256: dict[str, str]) -> str:
    payload = json.dumps(member_sha256, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_source(repo_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid overlay path: {relative_path}")
    current = repo_root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Required overlay source is missing: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Overlay source may not contain a symlink: {relative_path}")
    if not current.is_file():
        raise ValueError(f"Overlay source must be a regular file: {relative_path}")
    return current


def archive_member(repo_root: Path, relative_path: str) -> tuple[str, bytes, int]:
    path = _safe_source(repo_root, relative_path)
    if relative_path.startswith(("data/", "docs/evidence/")):
        raise ValueError(f"Forbidden runtime/artifact overlay member: {relative_path}")
    if path.name in {".env", ".env.vm.generated"}:
        raise ValueError(f"Forbidden secret overlay member: {relative_path}")
    mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
    return f"{repo_root.name}/{relative_path}", path.read_bytes(), mode


def build_overlay(repo_root: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    repo_root = repo_root.resolve()
    if official_latest_semantically_dirty(repo_root):
        raise RuntimeError(
            "official latest artifacts have semantic changes; refuse candidate portfolio packaging"
        )
    members = [archive_member(repo_root, path) for path in OVERLAY_FILES]
    member_sha256 = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, (_name, content, _mode) in zip(OVERLAY_FILES, members)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / OVERLAY_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, content, mode in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | mode) << 16
            zf.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as zf:
        crc_failure = zf.testzip()
        names = zf.namelist()
    receipt = {
        "archive": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "base_git_commit": source_commit(repo_root),
        "worktree_dirty": worktree_dirty(repo_root),
        "allowlisted_members_dirty": allowlisted_members_dirty(repo_root),
        "promotion_status": "not_accepted",
        "delivery_scope": "candidate_portfolio_source_only",
        "dashboard_delivery": "included",
        "member_sha256": member_sha256,
        "source_fingerprint": source_fingerprint(member_sha256),
        "members": names,
        "member_count": len(names),
        "zip_crc_ok": crc_failure is None,
        "zip_crc_failure": crc_failure,
        "policy": (
            "Source, tests, configs, dashboard, and runbooks only; excludes data, accepted results, "
            "runtime receipts, logs, secrets, caches, model weights, and browser evidence"
        ),
    }
    receipt_path = output_dir / f"{archive.stem}-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archive, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    archive, receipt = build_overlay(repo_root, args.output_dir)
    print(json.dumps({"archive_path": os.fspath(archive), **receipt}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
