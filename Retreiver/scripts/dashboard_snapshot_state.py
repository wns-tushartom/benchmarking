"""Atomic latest-attempt and last-successful dashboard readiness snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

_STATE_DIR = Path("data/dashboard_state")
_LATEST = "latest_attempt_document_readiness.json"
_SUCCESS = "last_successful_document_readiness.json"


def _normalized(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    total = snapshot.get("total", 0)
    ready = snapshot.get("ready_count", 0)
    review = snapshot.get("review_count", 0)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (total, ready, review)):
        raise ValueError("document readiness counts must be nonnegative integers")
    if ready > total or review > total:
        raise ValueError("document readiness counts exceed total")
    rows = snapshot.get("rows", [])
    uploaded = snapshot.get("uploaded", [])
    if not isinstance(rows, list) or not isinstance(uploaded, list):
        raise ValueError("document readiness rows must be arrays")
    return {
        "rows": rows,
        "total": total,
        "ready_count": ready,
        "review_count": review,
        "uploaded": uploaded,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_successful(snapshot: Mapping[str, Any]) -> bool:
    total = int(snapshot["total"])
    return total > 0 and snapshot["ready_count"] == total and snapshot["review_count"] == 0


def _state_dir(root: Path) -> Path:
    return Path(root).resolve() / _STATE_DIR


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _normalized(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def resolve_document_readiness(root: Path, latest_attempt: Mapping[str, Any]) -> dict[str, Any]:
    latest = _normalized(latest_attempt)
    previous = _read(_state_dir(root) / _SUCCESS)
    displayed = previous if previous is not None and not _is_successful(latest) else latest
    result = dict(displayed)
    result["display_source"] = "last_successful" if displayed is previous or _is_successful(latest) else "latest_attempt"
    result["latest_attempt"] = latest
    return result


def publish_document_readiness(root: Path, latest_attempt: Mapping[str, Any]) -> dict[str, Any]:
    latest = _normalized(latest_attempt)
    state_dir = _state_dir(root)
    _atomic_json(state_dir / _LATEST, latest)
    if _is_successful(latest):
        _atomic_json(state_dir / _SUCCESS, latest)
    return resolve_document_readiness(root, latest)
