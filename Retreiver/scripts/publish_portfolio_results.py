#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.core.portfolio import PORTFOLIO_STATES, PortfolioBatch, PortfolioPlan


CANDIDATE_LANE = "all-methods-portfolio"
EXPECTED_BATCH_COUNT = 12
EXPECTED_BATCH_SIZE = 180
EXPECTED_COMBINATION_COUNT = 2160
REQUIRED_BATCH_ARTIFACTS = frozenset({
    "modular_summary.csv",
    "modular_details.csv",
    "analysis.json",
    "MODULAR_REPORT.md",
    "config_snapshot.json",
    "combination_manifest.json",
    "provider_readiness_receipt.json",
})


class BatchReceiptError(ValueError):
    """A present batch directory is not a valid immutable completion."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchReceiptError(f"{label} is unreadable or malformed") from exc
    if not isinstance(payload, dict):
        raise BatchReceiptError(f"{label} must contain a JSON object")
    return payload


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise BatchReceiptError(f"{label} must not be a symlink or path alias")
    if not path.is_file():
        raise BatchReceiptError(f"{label} is missing or is not a regular file")


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink or path alias")


def load_candidate_portfolio(
    plan_or_root: str | os.PathLike[str],
) -> tuple[PortfolioPlan, Path, Path]:
    """Load a plan only from its exact candidate-only portfolio root."""
    lexical = Path(os.path.abspath(os.fspath(plan_or_root)))
    if lexical.name == "portfolio_plan.json":
        plan_path = lexical
        portfolio_root = lexical.parent
    else:
        portfolio_root = lexical
        plan_path = portfolio_root / "portfolio_plan.json"

    for path, label in (
        (portfolio_root.parent, "candidate lane"),
        (portfolio_root, "portfolio root"),
        (plan_path, "portfolio plan"),
    ):
        _reject_symlink(path, label)
        if path.resolve() != path:
            raise ValueError(f"{label} must not use a symlink or path alias")
    if portfolio_root.parent.name != CANDIDATE_LANE:
        raise ValueError("portfolio path is not an exact candidate portfolio root")
    if not portfolio_root.is_dir():
        raise ValueError("candidate portfolio root does not exist")
    _require_regular_file(plan_path, "portfolio plan")
    try:
        plan = PortfolioPlan.from_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("portfolio plan is unreadable or invalid") from exc
    if portfolio_root.name != plan.portfolio_id:
        raise ValueError("candidate portfolio root does not match immutable portfolio ID")
    return plan, portfolio_root, plan_path


def _expected_context(plan: PortfolioPlan, batch: PortfolioBatch) -> dict[str, str]:
    return {
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "batch_id": batch.batch_id,
        "promotion_status": "not_accepted",
    }


def _require_context(payload: Mapping[str, Any], expected: Mapping[str, str], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise BatchReceiptError(f"{label} {key} does not match immutable portfolio context")


def _validate_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BatchReceiptError(f"{label} is not a lowercase SHA-256 digest")
    return value


def verify_completed_batch(
    portfolio_root: Path,
    plan: PortfolioPlan,
    batch: PortfolioBatch,
    *,
    candidate_lane: str = CANDIDATE_LANE,
) -> dict[str, Any]:
    """Validate one completion receipt, every artifact hash, and exact ordered IDs."""
    if not candidate_lane or len(Path(candidate_lane).parts) != 1:
        raise BatchReceiptError("candidate lane is not a safe direct output namespace")
    batch_dir = portfolio_root / batch.batch_id
    if batch_dir.is_symlink():
        raise BatchReceiptError(f"batch {batch.batch_id} directory must not be a symlink or path alias")
    if not batch_dir.is_dir():
        raise BatchReceiptError(f"batch {batch.batch_id} directory is missing")
    expected = _expected_context(plan, batch)
    receipt_path = batch_dir / "completion_receipt.json"
    manifest_path = batch_dir / "manifest.json"
    receipt = _json_object(receipt_path, "completion receipt")
    manifest = _json_object(manifest_path, "batch manifest")

    _require_context(receipt, expected, "completion receipt")
    if receipt.get("state") != "completed":
        raise BatchReceiptError("completion receipt state is not completed")
    if receipt.get("combination_count") != len(batch.combination_ids):
        raise BatchReceiptError("completion receipt combination count does not match the batch")
    manifest_sha256 = _validate_digest(
        receipt.get("manifest_sha256"), "completion receipt manifest_sha256"
    )
    if _sha256(manifest_path) != manifest_sha256:
        raise BatchReceiptError("completion receipt manifest_sha256 does not match manifest")

    if manifest.get("status") != "completed":
        raise BatchReceiptError("batch manifest status is not completed")
    if manifest.get("promotion_status") != "not_accepted":
        raise BatchReceiptError("batch manifest promotion_status must be not_accepted")
    manifest_context = manifest.get("portfolio")
    if not isinstance(manifest_context, dict):
        raise BatchReceiptError("batch manifest portfolio context is missing")
    _require_context(manifest_context, expected, "batch manifest")
    expected_artifact_root = f"{candidate_lane}/{plan.portfolio_id}/{batch.batch_id}"
    if manifest.get("artifact_root") != expected_artifact_root:
        raise BatchReceiptError("batch manifest artifact root is not canonical")

    receipt_hashes = receipt.get("artifact_sha256")
    manifest_hashes = manifest.get("artifact_sha256")
    if not isinstance(receipt_hashes, dict) or not isinstance(manifest_hashes, dict):
        raise BatchReceiptError("batch artifact hash maps are missing")
    if receipt_hashes != manifest_hashes:
        raise BatchReceiptError("completion receipt artifact hashes differ from manifest")
    if not REQUIRED_BATCH_ARTIFACTS.issubset(receipt_hashes):
        raise BatchReceiptError("completion receipt is missing required artifact hashes")
    for name, expected_digest in receipt_hashes.items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise BatchReceiptError("artifact hash name is not a safe batch-local filename")
        digest = _validate_digest(expected_digest, f"artifact hash for {name}")
        artifact_path = batch_dir / name
        _require_regular_file(artifact_path, f"batch artifact {name}")
        if _sha256(artifact_path) != digest:
            raise BatchReceiptError(f"artifact hash mismatch for {name}")

    combination_manifest = _json_object(
        batch_dir / "combination_manifest.json", "combination manifest"
    )
    _require_context(combination_manifest, expected, "combination manifest")
    expected_ids = list(batch.combination_ids)
    if combination_manifest.get("combination_count") != len(expected_ids):
        raise BatchReceiptError("combination manifest count does not match batch")
    if combination_manifest.get("combination_ids") != expected_ids:
        raise BatchReceiptError("combination manifest ordered combination IDs do not match batch")
    combinations = combination_manifest.get("combinations")
    if not isinstance(combinations, list) or [
        item.get("combination_id") if isinstance(item, dict) else None
        for item in combinations
    ] != expected_ids:
        raise BatchReceiptError("combination manifest rows do not match ordered combination IDs")

    summary_path = batch_dir / "modular_summary.csv"
    try:
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BatchReceiptError("candidate modular summary is unreadable") from exc
    summary_ids = [row.get("combination_id") for row in summary_rows]
    if summary_ids != expected_ids:
        raise BatchReceiptError(
            "candidate modular summary must contain the exact ordered combination IDs exactly once"
        )
    for row in summary_rows:
        if row.get("status") != "completed":
            raise BatchReceiptError("candidate modular summary contains a non-completed row")
        if row.get("batch_id") != batch.batch_id:
            raise BatchReceiptError("candidate modular summary contains the wrong batch ID")
        if row.get("promotion_status") != "not_accepted":
            raise BatchReceiptError("candidate modular summary promotion_status must be not_accepted")

    return {
        "batch_id": batch.batch_id,
        "combination_count": len(expected_ids),
        "completion_receipt_sha256": _sha256(receipt_path),
        "manifest_sha256": manifest_sha256,
        "artifact_sha256": dict(sorted(receipt_hashes.items())),
        "summary_rows": summary_rows,
    }


def portfolio_status(plan_or_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Reconcile immutable plan order with fail-closed batch receipt observations."""
    plan, portfolio_root, plan_path = load_candidate_portfolio(plan_or_root)
    batches: list[dict[str, Any]] = []
    first_runnable: str | None = None
    state_counts = {"completed": 0, "failed": 0, "not_run": 0}
    for batch in plan.batches:
        batch_dir = portfolio_root / batch.batch_id
        if not batch_dir.exists() and not batch_dir.is_symlink():
            state = "not_run"
            error = None
        else:
            try:
                verify_completed_batch(portfolio_root, plan, batch)
            except BatchReceiptError as exc:
                state = "failed"
                error = str(exc)
            else:
                state = "completed"
                error = None
        if state != "completed" and first_runnable is None:
            first_runnable = batch.batch_id
        state_counts[state] += 1
        batches.append({
            "batch_id": batch.batch_id,
            "embedding": batch.embedding,
            "reranker_group": batch.reranker_group,
            "combination_count": len(batch.combination_ids),
            "state": state,
            "selected": False,
            "error": error,
        })
    if first_runnable is not None:
        next(item for item in batches if item["batch_id"] == first_runnable)["selected"] = True
    selected_count = next(
        (
            item["combination_count"]
            for item in batches
            if item["batch_id"] == first_runnable
        ),
        0,
    )
    combination_state_counts = {
        "configured": len(plan.combination_ids),
        # Excluded attempted combinations are deliberately outside the executable
        # immutable plan; no excluded ID can become runnable from this command.
        "excluded": 0,
        "selected": selected_count,
        "completed": sum(
            item["combination_count"] for item in batches if item["state"] == "completed"
        ),
        "failed": sum(
            item["combination_count"] for item in batches if item["state"] == "failed"
        ),
        "not_run": sum(
            item["combination_count"] for item in batches if item["state"] == "not_run"
        ),
    }
    return {
        "schema_version": 1,
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "promotion_status": "not_accepted",
        "plan": str(plan_path),
        "configured_combination_count": len(plan.combination_ids),
        "excluded_combination_count": 0,
        "batch_count": len(plan.batches),
        "max_combinations_per_batch": plan.max_combinations_per_batch,
        "state_values": sorted(PORTFOLIO_STATES),
        "state_counts": state_counts,
        "combination_state_counts": combination_state_counts,
        "selected_batch_id": first_runnable,
        "batches": batches,
    }


def _validate_publication_shape(plan: PortfolioPlan, portfolio_root: Path) -> None:
    if len(plan.batches) != EXPECTED_BATCH_COUNT:
        raise ValueError(f"publication requires exactly {EXPECTED_BATCH_COUNT} planned batches")
    if len(plan.combination_ids) != EXPECTED_COMBINATION_COUNT:
        raise ValueError(
            f"publication requires exactly {EXPECTED_COMBINATION_COUNT} planned combinations"
        )
    if any(
        len(batch.combination_ids) != EXPECTED_BATCH_SIZE
        or len(batch.combination_ids) > plan.max_combinations_per_batch
        or len(batch.combination_ids) > 250
        for batch in plan.batches
    ):
        raise ValueError(
            f"publication requires every expected batch to contain exactly {EXPECTED_BATCH_SIZE} combinations and at most 250"
        )
    expected_names = {batch.batch_id for batch in plan.batches}
    present_names = {
        child.name
        for child in portfolio_root.iterdir()
        if child.is_dir() or child.is_symlink()
    }
    if present_names != expected_names:
        raise ValueError("publication requires all 12 expected batch directories and no path aliases")


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    if not rows:
        raise ValueError("candidate aggregate cannot be empty")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("candidate summaries do not share an exact CSV schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_private_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def publish_portfolio_results(
    plan_or_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Publish one deterministic candidate-only aggregate after strict 12-batch verification."""
    plan, portfolio_root, plan_path = load_candidate_portfolio(plan_or_root)
    _validate_publication_shape(plan, portfolio_root)

    verified_batches: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, str]] = []
    for batch in plan.batches:
        verified = verify_completed_batch(portfolio_root, plan, batch)
        aggregate_rows.extend(verified.pop("summary_rows"))
        verified_batches.append(verified)
    aggregate_ids = [row.get("combination_id") for row in aggregate_rows]
    expected_ids = [
        combination_id
        for batch in plan.batches
        for combination_id in batch.combination_ids
    ]
    if aggregate_ids != expected_ids or len(set(aggregate_ids)) != EXPECTED_COMBINATION_COUNT:
        raise ValueError(
            "aggregate candidate summary must cover every planned combination ID exactly once"
        )

    aggregate_content = _csv_bytes(aggregate_rows)
    aggregate_digest = hashlib.sha256(aggregate_content).hexdigest()
    receipt_payload = {
        "schema_version": 1,
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "promotion_status": "not_accepted",
        "state": "completed",
        "batch_count": len(plan.batches),
        "combination_count": len(aggregate_rows),
        "plan_sha256": _sha256(plan_path),
        "aggregate_candidate_summary_sha256": aggregate_digest,
        "batches": verified_batches,
    }
    receipt_content = _json_bytes(receipt_payload)
    aggregate_path = portfolio_root / "aggregate_candidate_summary.csv"
    receipt_path = portfolio_root / "portfolio_receipt.json"

    existing = aggregate_path.exists() or receipt_path.exists()
    if existing:
        for path in (aggregate_path, receipt_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("existing publication conflicts with verified candidate output")
        if (
            aggregate_path.read_bytes() != aggregate_content
            or receipt_path.read_bytes() != receipt_content
        ):
            raise ValueError("existing publication conflicts with verified candidate output")
    else:
        _atomic_private_write(aggregate_path, aggregate_content)
        _atomic_private_write(receipt_path, receipt_content)

    return {
        "ok": True,
        "portfolio_id": plan.portfolio_id,
        "promotion_status": "not_accepted",
        "batch_count": len(plan.batches),
        "combination_count": len(aggregate_rows),
        "portfolio_receipt": str(receipt_path),
        "aggregate_candidate_summary": str(aggregate_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly verify and publish a candidate all-method portfolio"
    )
    parser.add_argument("plan_or_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish_portfolio_results(args.plan_or_root), indent=2))


if __name__ == "__main__":
    main()
