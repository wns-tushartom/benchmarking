from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts import package_all_methods_portfolio_overlay as packager
from scripts.package_all_methods_portfolio_overlay import OVERLAY_FILES, build_overlay


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_overlay_is_dashboard_complete_and_source_only(tmp_path: Path) -> None:
    archive, receipt = build_overlay(REPO_ROOT, tmp_path)

    assert archive.exists()
    assert receipt["zip_crc_ok"] is True
    assert receipt["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert receipt["delivery_scope"] == "candidate_portfolio_source_only"
    assert receipt["promotion_status"] == "not_accepted"
    assert isinstance(receipt["allowlisted_members_dirty"], bool)
    assert receipt["member_count"] == len(OVERLAY_FILES)
    assert len(receipt["source_fingerprint"]) == 64
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert zf.testzip() is None

    assert names == [f"Retreiver/{path}" for path in OVERLAY_FILES]
    for required in (
        "benchmarking/adapters/vector_turbovec.py",
        "benchmarking/core/portfolio.py",
        "configs/benchmark.all-methods-portfolio.json",
        "configs/vm_adapter_ports.json",
        "requirements-benchmark.txt",
        "scripts/dashboard_control_api.py",
        "scripts/dashboard_portfolio_api.py",
        "scripts/serve_benchmark_dashboard.py",
        "scripts/vm_adapter_manager.py",
        "scripts/publish_portfolio_results.py",
        "web/index.html",
        "web/app.js",
        "web/styles.css",
        "docs/VM_ALL_METHODS_PORTFOLIO_RUNBOOK.md",
        "tests/test_portfolio_frontend.py",
    ):
        assert f"Retreiver/{required}" in names
        assert receipt["member_sha256"][required] == hashlib.sha256(
            (REPO_ROOT / required).read_bytes()
        ).hexdigest()
    assert not any(name.startswith("Retreiver/data/") for name in names)
    assert not any(name.startswith("Retreiver/docs/evidence/") for name in names)
    assert not any("__pycache__" in name or name.endswith((".pyc", ".log")) for name in names)
    assert not any(Path(name).name in {".env", ".env.vm.generated"} for name in names)


def test_portfolio_overlay_refuses_semantically_modified_official_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(packager, "official_latest_semantically_dirty", lambda _root: True)

    with pytest.raises(RuntimeError, match="official latest artifacts have semantic changes"):
        build_overlay(REPO_ROOT, tmp_path)


def test_portfolio_overlay_rejects_symlinked_allowlisted_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    linked = root / "linked.py"
    linked.symlink_to(outside)
    monkeypatch.setattr(packager, "OVERLAY_FILES", ("linked.py",))
    monkeypatch.setattr(packager, "official_latest_semantically_dirty", lambda _root: False)
    monkeypatch.setattr(packager, "source_commit", lambda _root: "0" * 40)
    monkeypatch.setattr(packager, "worktree_dirty", lambda _root: False)

    with pytest.raises(ValueError, match="symlink"):
        build_overlay(root, tmp_path / "out")
