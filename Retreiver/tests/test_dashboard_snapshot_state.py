from __future__ import annotations

import json
from pathlib import Path

from scripts.dashboard_snapshot_state import publish_document_readiness, resolve_document_readiness


def _snapshot(total: int, ready: int, review: int = 0) -> dict:
    return {
        "rows": [{"pdf_name": f"doc-{index}.pdf"} for index in range(total)],
        "total": total,
        "ready_count": ready,
        "review_count": review,
        "uploaded": [],
    }


def test_failed_attempt_does_not_replace_last_successful_readiness(tmp_path: Path) -> None:
    first = publish_document_readiness(tmp_path, _snapshot(3, 3))
    assert first["display_source"] == "last_successful"

    failed = publish_document_readiness(tmp_path, _snapshot(3, 0, 3))

    assert failed["display_source"] == "last_successful"
    assert failed["total"] == 3
    assert failed["ready_count"] == 3
    assert failed["latest_attempt"]["ready_count"] == 0
    assert failed["latest_attempt"]["review_count"] == 3


def test_partial_attempt_does_not_replace_last_successful_readiness(tmp_path: Path) -> None:
    publish_document_readiness(tmp_path, _snapshot(4, 4))

    resolved = publish_document_readiness(tmp_path, _snapshot(4, 2, 2))

    assert resolved["ready_count"] == 4
    assert resolved["latest_attempt"]["ready_count"] == 2


def test_successful_attempt_promotes_atomically_and_resolves_after_restart(tmp_path: Path) -> None:
    publish_document_readiness(tmp_path, _snapshot(2, 2))

    resolved = resolve_document_readiness(tmp_path, _snapshot(2, 0, 2))

    assert resolved["ready_count"] == 2
    state_dir = tmp_path / "data" / "dashboard_state"
    assert json.loads((state_dir / "last_successful_document_readiness.json").read_text())["ready_count"] == 2
    assert not list(state_dir.glob("*.tmp"))


def test_without_prior_success_latest_attempt_remains_visible(tmp_path: Path) -> None:
    resolved = publish_document_readiness(tmp_path, _snapshot(2, 0, 2))

    assert resolved["display_source"] == "latest_attempt"
    assert resolved["ready_count"] == 0


def test_readiness_snapshots_preserve_audit_provenance(tmp_path: Path) -> None:
    snapshot = {
        **_snapshot(2, 0, 2),
        "audit_status": "text_only_fallback",
        "parser_counts": {"PyPDF2_fallback": 2},
        "review_reason_counts": {"text_only_fallback": 2},
    }

    resolved = publish_document_readiness(tmp_path, snapshot)

    assert resolved["audit_status"] == "text_only_fallback"
    assert resolved["parser_counts"] == {"PyPDF2_fallback": 2}
    assert resolved["review_reason_counts"] == {"text_only_fallback": 2}
    assert resolved["latest_attempt"]["audit_status"] == "text_only_fallback"
