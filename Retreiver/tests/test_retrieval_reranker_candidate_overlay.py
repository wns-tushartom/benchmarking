from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from scripts.package_retrieval_reranker_candidate_overlay import OVERLAY_FILES, build_overlay


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_candidate_overlay_is_receipted_and_contains_only_allowlisted_source(tmp_path: Path) -> None:
    archive, receipt = build_overlay(REPO_ROOT, tmp_path)

    assert archive.exists()
    assert receipt["zip_crc_ok"] is True
    assert receipt["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert zf.testzip() is None

    expected = [f"Retreiver/{path}" for path in OVERLAY_FILES]
    assert names == expected
    assert "Retreiver/configs/benchmark.retrieval-reranker-candidates.json" in names
    assert "Retreiver/scripts/wns_vm_adapter_service.py" in names
    assert "Retreiver/docs/VM_RETRIEVAL_RERANKER_CANDIDATE_RUNBOOK.md" in names
    assert not any(name.startswith("Retreiver/data/") for name in names)
    assert not any(Path(name).name in {".env", ".env.vm.generated"} for name in names)
