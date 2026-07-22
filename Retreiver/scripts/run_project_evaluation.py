#!/usr/bin/env python3
"""Run a single project-local FAISS ground-truth evaluation.

This runner deliberately accepts all inputs explicitly and refuses paths outside the
project root.  Retrieval and evaluation artifacts are written only under the
selected project's ``runs/<run-id>/`` directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_retrieval_groundtruth import load_groundtruth


RUN_ID_RE = re.compile(r"^(?!\.\.?$)[A-Za-z0-9_.-]+$")
RUN_ID_ERROR = "run_id must name a direct child containing only letters, digits, '.', '_' or '-'"
DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_run_dir(project_root: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(RUN_ID_ERROR)
    run_dir = project_root.resolve() / "runs" / run_id
    if not is_within(project_root, run_dir):  # defensive guard for future run-id changes
        raise ValueError("run output must be inside project root")
    return run_dir


def canonical_project_runs_dir(project_root: Path) -> Path:
    root = project_root.resolve()
    runs_dir = root / "runs"
    if runs_dir.is_symlink():
        raise ValueError("project runs directory must be a direct, non-symlink child of the project root")
    try:
        runs_dir.mkdir()
    except FileExistsError:
        pass
    try:
        runs_stat = runs_dir.lstat()
    except OSError as exc:
        raise ValueError(f"cannot access project runs directory: {runs_dir}: {exc}") from exc
    if stat.S_ISLNK(runs_stat.st_mode) or not stat.S_ISDIR(runs_stat.st_mode):
        raise ValueError("project runs directory must be a direct, non-symlink child of the project root")
    try:
        if runs_dir.resolve(strict=True) != runs_dir:
            raise ValueError("project runs directory must not be aliased")
    except OSError as exc:
        raise ValueError(f"cannot resolve project runs directory: {runs_dir}: {exc}") from exc
    return runs_dir


def open_nominal_reserved_run_dir(*, project_root: Path, run_dir: Path) -> int:
    """Open the nominal run directory without following either project child link."""
    root = project_root.resolve()
    expected_run_dir = root / "runs" / run_dir.name
    if run_dir != expected_run_dir:
        raise ValueError("project evaluation run directory must be a direct child of the project runs directory")
    try:
        root_fd = os.open(root, DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError(f"cannot securely open project root: {root}: {exc}") from exc
    try:
        try:
            runs_fd = os.open("runs", DIRECTORY_OPEN_FLAGS, dir_fd=root_fd)
        except OSError as exc:
            raise ValueError(f"cannot securely open project runs directory: {root / 'runs'}: {exc}") from exc
    finally:
        os.close(root_fd)
    try:
        try:
            return os.open(run_dir.name, DIRECTORY_OPEN_FLAGS, dir_fd=runs_fd)
        except OSError as exc:
            raise ValueError(f"cannot securely open reserved project evaluation run: {run_dir}: {exc}") from exc
    finally:
        os.close(runs_fd)


def nominal_run_dir_matches_fd(*, project_root: Path, run_dir: Path, run_dir_fd: int) -> bool:
    """Return whether the accessible nominal path still names the held run directory."""
    try:
        nominal_fd = open_nominal_reserved_run_dir(project_root=project_root, run_dir=run_dir)
    except ValueError:
        return False
    try:
        held = os.fstat(run_dir_fd)
        nominal = os.fstat(nominal_fd)
        return stat.S_ISDIR(held.st_mode) and (held.st_dev, held.st_ino) == (nominal.st_dev, nominal.st_ino)
    finally:
        os.close(nominal_fd)


def reserve_project_run_dir(project_root: Path, run_id: str) -> tuple[Path, int]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(RUN_ID_ERROR)
    runs_dir = canonical_project_runs_dir(project_root)
    root_fd: int | None = None
    runs_fd: int | None = None
    try:
        root_fd = os.open(project_root.resolve(), DIRECTORY_OPEN_FLAGS)
        runs_fd = os.open("runs", DIRECTORY_OPEN_FLAGS, dir_fd=root_fd)
    except OSError as exc:
        if root_fd is not None:
            os.close(root_fd)
        raise ValueError(f"cannot securely open project runs directory: {runs_dir}: {exc}") from exc
    try:
        try:
            os.mkdir(run_id, dir_fd=runs_fd)
        except FileExistsError as exc:
            raise ValueError(f"project evaluation run already exists: {runs_dir / run_id}") from exc
        try:
            run_fd = os.open(run_id, DIRECTORY_OPEN_FLAGS, dir_fd=runs_fd)
        except OSError as exc:
            raise ValueError(f"cannot securely open reserved project evaluation run: {runs_dir / run_id}: {exc}") from exc
    finally:
        if runs_fd is not None:
            os.close(runs_fd)
        if root_fd is not None:
            os.close(root_fd)
    run_dir = runs_dir / run_id
    return run_dir, run_fd


def preflight(
    *,
    project_root: Path,
    groundtruth: Path,
    ingestion_summary: Path,
    sheet: str,
    embedding: str,
    store: str,
    reranker: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"project root does not exist: {root}")
    if store != "FAISS":
        errors.append("project evaluation supports only FAISS")
    if reranker != "none":
        errors.append("project evaluation supports reranker=none only")
    for label, path in (("ground truth", groundtruth), ("ingestion receipt", ingestion_summary)):
        if not is_within(root, path):
            errors.append(f"{label} is outside project root: {path}")
        elif not path.is_file():
            errors.append(f"{label} does not exist: {path}")
    chunks_dir = root / "chunks"
    if root.is_dir() and not chunks_dir.is_dir():
        errors.append(f"project chunks directory does not exist: {chunks_dir}")

    rows: list[dict[str, str]] = []
    if ingestion_summary.is_file() and is_within(root, ingestion_summary):
        try:
            rows = read_csv(ingestion_summary)
        except (OSError, csv.Error) as exc:
            errors.append(f"cannot read ingestion receipt: {exc}")
    selected = [
        row for row in rows
        if row.get("status") == "ok"
        and row.get("sheet") == sheet
        and row.get("embedding") == embedding
        and row.get("store") == store
        and row.get("collection_or_table")
    ]
    if len(selected) != 1:
        errors.append(f"expected exactly one successful selected ingestion row, found {len(selected)}")
    index_dir: Path | None = None
    if len(selected) == 1:
        index_dir = Path(selected[0]["collection_or_table"]).resolve()
        if not is_within(root, index_dir):
            errors.append(f"selected FAISS index is outside project root: {index_dir}")
        elif not index_dir.is_dir():
            errors.append(f"selected FAISS index does not exist: {index_dir}")
        else:
            for filename in ("index.faiss", "chunks.json"):
                if not (index_dir / filename).is_file():
                    errors.append(f"selected FAISS index is missing {filename}: {index_dir}")

    expected_queries = 0
    gt_sha256 = ""
    if groundtruth.is_file() and is_within(root, groundtruth):
        try:
            expected_queries = len(load_groundtruth(groundtruth))
            gt_sha256 = sha256(groundtruth)
            if not expected_queries:
                errors.append("ground truth contains no query/question rows")
        except Exception as exc:
            errors.append(f"cannot read ground truth: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "project_root": str(root),
        "groundtruth": str(groundtruth.resolve()),
        "groundtruth_sha256": gt_sha256,
        "ingestion_summary": str(ingestion_summary.resolve()),
        "index_dir": str(index_dir) if index_dir else "",
        "chunks_dir": str(chunks_dir),
        "expected_queries": expected_queries,
        "components": {
            "sheet": sheet,
            "embedding": embedding,
            "store": store,
            "reranker": reranker,
        },
    }


def build_commands(
    *,
    project_root: Path,
    run_id: str,
    run_dir_fd: int,
    groundtruth: Path,
    ingestion_summary: Path,
    sheet: str,
    embedding: str,
    store: str,
    reranker: str,
    top_k: int,
) -> tuple[list[str], list[str]]:
    if reranker != "none":
        raise ValueError("project evaluation supports reranker=none only")
    del project_root, run_id
    run_dir = Path(f"/proc/self/fd/{run_dir_fd}")
    retrieval_dir = run_dir / "retrieval"
    evaluation_dir = run_dir / "evaluation"
    retrieval = [
        sys.executable,
        str(ROOT / "scripts" / "run_retrieval_smoke_from_vm_dbs.py"),
        "--queries-file", str(groundtruth.resolve()),
        "--ingestion-summary", str(ingestion_summary.resolve()),
        "--out-dir", str(retrieval_dir),
        "--sheets", sheet,
        "--embeddings", embedding,
        "--stores", store,
        "--top-k", str(top_k),
        "--max-combos", "1",
        "--query-limit", "0",
    ]
    evaluation = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_retrieval_groundtruth.py"),
        "--groundtruth", str(groundtruth.resolve()),
        "--smoke-dir", str(retrieval_dir),
        "--out-dir", str(evaluation_dir),
    ]
    return retrieval, evaluation


def completion_state(
    *,
    expected_queries: int,
    report: dict[str, Any],
    summary_rows: list[dict[str, str]],
) -> tuple[str, int, int]:
    try:
        evaluated = int((summary_rows[0] if len(summary_rows) == 1 else {}).get("evaluated_queries") or 0)
    except (TypeError, ValueError):
        evaluated = 0
    try:
        missing = int(report.get("missing_query_count") or 0)
    except (TypeError, ValueError):
        missing = expected_queries
    report_evaluated = report.get("evaluated_rows")
    complete = (
        int(report.get("summary_rows") or 0) == 1
        and len(summary_rows) == 1
        and evaluated == expected_queries
        and report_evaluated == expected_queries
        and missing == 0
    )
    return ("complete" if complete else "incomplete"), evaluated, missing


def write_manifest(
    *,
    project_root: Path,
    run_dir: Path,
    payload: dict[str, Any],
    run_dir_fd: int | None = None,
) -> Path:
    """Atomically publish a manifest through a descriptor for the reserved run directory."""
    owned_fd = run_dir_fd is None
    if owned_fd:
        run_dir_fd = open_nominal_reserved_run_dir(project_root=project_root, run_dir=run_dir)
    assert run_dir_fd is not None
    encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    temporary_name = f".evaluation_manifest.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=run_dir_fd,
        )
        with os.fdopen(temporary_fd, "wb") as f:
            temporary_fd = None
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(
            temporary_name,
            "evaluation_manifest.json",
            src_dir_fd=run_dir_fd,
            dst_dir_fd=run_dir_fd,
        )
        os.fsync(run_dir_fd)
        return run_dir / "evaluation_manifest.json"
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=run_dir_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if owned_fd:
            os.close(run_dir_fd)


def run_command(command: list[str], *, pass_fds: tuple[int, ...] = ()) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        pass_fds=pass_fds,
    )
    output = proc.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    return proc.returncode, output


def run_evaluation(
    *,
    info: dict[str, Any],
    project_root: Path,
    run_id: str,
    groundtruth: Path,
    ingestion_summary: Path,
    sheet: str,
    embedding: str,
    store: str,
    reranker: str,
    top_k: int,
) -> tuple[int, dict[str, Any]]:
    run_dir, run_dir_fd = reserve_project_run_dir(project_root, run_id)
    pinned_run_dir = Path(f"/proc/self/fd/{run_dir_fd}")
    retrieval_dir = run_dir / "retrieval"
    evaluation_dir = run_dir / "evaluation"
    pinned_retrieval_dir = pinned_run_dir / "retrieval"
    pinned_evaluation_dir = pinned_run_dir / "evaluation"
    try:
        retrieval_cmd, evaluation_cmd = build_commands(
            project_root=project_root,
            run_id=run_id,
            run_dir_fd=run_dir_fd,
            groundtruth=groundtruth,
            ingestion_summary=ingestion_summary,
            sheet=sheet,
            embedding=embedding,
            store=store,
            reranker=reranker,
            top_k=top_k,
        )

        def nominal_command(command: list[str]) -> list[str]:
            pinned_prefix = str(pinned_run_dir)
            nominal_prefix = str(run_dir)
            return [
                nominal_prefix + value[len(pinned_prefix):] if value.startswith(pinned_prefix) else value
                for value in command
            ]

        manifest: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "state": "failed",
            "run_id": run_id,
            "components": info["components"],
            "groundtruth": info["groundtruth"],
            "groundtruth_sha256": info["groundtruth_sha256"],
            "receipt": info["ingestion_summary"],
            "index_dir": info["index_dir"],
            "chunks_dir": info["chunks_dir"],
            "retrieval_dir": str(retrieval_dir),
            "evaluation_dir": str(evaluation_dir),
            "expected_queries": info["expected_queries"],
            "evaluated_queries": 0,
            "missing_query_count": info["expected_queries"],
            "commands": {
                "retrieval": nominal_command(retrieval_cmd),
                "evaluation": nominal_command(evaluation_cmd),
            },
        }
        retrieval_code, retrieval_output = run_command(retrieval_cmd, pass_fds=(run_dir_fd,))
        manifest["retrieval_exit_code"] = retrieval_code
        manifest["retrieval_output"] = retrieval_output[-20000:]
        if retrieval_code != 0:
            manifest["error"] = "retrieval command failed"
            write_manifest(
                project_root=project_root,
                run_dir=run_dir,
                run_dir_fd=run_dir_fd,
                payload=manifest,
            )
            return 1, manifest

        evaluation_code, evaluation_output = run_command(evaluation_cmd, pass_fds=(run_dir_fd,))
        manifest["evaluation_exit_code"] = evaluation_code
        manifest["evaluation_output"] = evaluation_output[-20000:]
        report_path = pinned_evaluation_dir / "groundtruth_eval_report.json"
        summary_path = pinned_evaluation_dir / "groundtruth_eval_summary.csv"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary_rows = read_csv(summary_path)
            state, evaluated, missing = completion_state(
                expected_queries=info["expected_queries"], report=report, summary_rows=summary_rows,
            )
            manifest.update({
                "state": "failed" if evaluation_code else state,
                "evaluated_queries": evaluated,
                "missing_query_count": missing,
                "evaluator_report": str(evaluation_dir / "groundtruth_eval_report.json"),
                "evaluator_summary": str(evaluation_dir / "groundtruth_eval_summary.csv"),
            })
            if evaluation_code:
                manifest["error"] = "evaluation command failed"
            elif state != "complete":
                manifest["error"] = "evaluator output did not account for every ground-truth query"
        except (OSError, json.JSONDecodeError, csv.Error) as exc:
            manifest["state"] = "failed" if evaluation_code else "incomplete"
            manifest["error"] = f"cannot validate evaluator output: {exc}"
        if manifest["state"] == "complete" and not nominal_run_dir_matches_fd(
            project_root=project_root,
            run_dir=run_dir,
            run_dir_fd=run_dir_fd,
        ):
            manifest["state"] = "failed"
            manifest["error"] = "project evaluation run path was displaced before completion"
        write_manifest(
            project_root=project_root,
            run_dir=run_dir,
            run_dir_fd=run_dir_fd,
            payload=manifest,
        )
        return (0 if manifest["state"] == "complete" else 1), manifest
    finally:
        os.close(run_dir_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--ingestion-summary", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--reranker", default="none")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    groundtruth = Path(args.groundtruth)
    ingestion_summary = Path(args.ingestion_summary)
    info = preflight(
        project_root=project_root,
        groundtruth=groundtruth,
        ingestion_summary=ingestion_summary,
        sheet=args.sheet,
        embedding=args.embedding,
        store=args.store,
        reranker=args.reranker,
    )
    if args.preflight_only or not info["ok"]:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return 0 if info["ok"] else 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        code, manifest = run_evaluation(
            info=info,
            project_root=project_root,
            run_id=run_id,
            groundtruth=groundtruth,
            ingestion_summary=ingestion_summary,
            sheet=args.sheet,
            embedding=args.embedding,
            store=args.store,
            reranker=args.reranker,
            top_k=args.top_k,
        )
    except ValueError as exc:
        print(json.dumps({"state": "failed", "error": str(exc)}, indent=2), flush=True)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
