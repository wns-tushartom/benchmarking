#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.core.config import generate_matrix, load_benchmark_config
from benchmarking.core.portfolio import PortfolioPlan, build_portfolio_plan
from benchmarking.core.runner import run_experiment, validate_output_directory
from scripts.publish_portfolio_results import portfolio_status as inspect_portfolio_status


def _output_directory(root: Path, config_path: Path, requested: str) -> Path:
    """Fail closed when a candidate experiment targets an official artifact lane."""
    output_dir = (root / requested).resolve()
    config = load_benchmark_config(config_path)
    return validate_output_directory(config, root, output_dir)


def _portfolio_root(root: Path, config: dict[str, Any], plan: PortfolioPlan) -> Path:
    lane = str(config.get("experiment", {}).get("output_lane") or "").strip()
    if not isinstance(config.get("portfolio"), dict) or not lane or len(Path(lane).parts) != 1:
        raise ValueError("configuration is not an isolated portfolio configuration")
    return root.resolve() / "data" / "modular_runs" / lane / plan.portfolio_id


def _reject_symlink_components(root: Path, path: Path) -> None:
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    if lexical_root != root.resolve():
        raise ValueError("portfolio plan path must not contain a symlink")
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("portfolio plan path is outside its canonical root") from exc
    current = lexical_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("portfolio plan path must not contain a symlink")


def _atomic_private_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        Path(temporary_name).replace(path)
        path.chmod(0o600)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_portfolio_plan(root: Path, config_path: Path) -> tuple[PortfolioPlan, Path]:
    """Create or idempotently verify the immutable plan in its candidate-only root."""
    config = load_benchmark_config(config_path)
    plan = build_portfolio_plan(config)
    portfolio_root = _portfolio_root(root, config, plan)
    lane_root = portfolio_root.parent
    plan_path = portfolio_root / "portfolio_plan.json"
    _reject_symlink_components(root, plan_path)
    for path in (lane_root, portfolio_root, plan_path):
        if path.is_symlink():
            raise ValueError("portfolio plan path must not contain a symlink")
    if plan_path.exists():
        try:
            existing = PortfolioPlan.from_json(plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("existing portfolio plan is immutable and invalid") from exc
        if existing != plan:
            raise ValueError("existing portfolio plan is immutable and differs from current config")
        return plan, plan_path
    _atomic_private_json(plan_path, plan.to_json())
    return plan, plan_path


def load_verified_portfolio_plan(
    root: Path,
    config_path: Path,
    plan_path: Path,
) -> PortfolioPlan:
    """Bind a plan file to the current config, canonical output root, and hash."""
    config = load_benchmark_config(config_path)
    expected = build_portfolio_plan(config)
    expected_path = _portfolio_root(root, config, expected) / "portfolio_plan.json"
    lexical_path = Path(os.path.abspath(plan_path))
    _reject_symlink_components(root, lexical_path)
    if lexical_path != expected_path or lexical_path.is_symlink() or lexical_path.parent.is_symlink():
        raise ValueError("portfolio plan path is outside its canonical immutable root or is a symlink")
    try:
        loaded = PortfolioPlan.from_json(lexical_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("portfolio plan is unreadable or invalid") from exc
    if loaded != expected:
        raise ValueError("portfolio plan does not match the current canonical configuration")
    return loaded


def run_portfolio_batch(
    root: Path,
    config_path: Path,
    plan_path: Path,
    batch_id: str,
    *,
    limit_queries: int = 0,
) -> dict[str, Any]:
    """Execute exactly one immutable <=250-combination batch."""
    plan = load_verified_portfolio_plan(root, config_path, plan_path)
    batch = next((item for item in plan.batches if item.batch_id == batch_id), None)
    if batch is None:
        raise ValueError(f"unknown batch ID: {batch_id}")
    config = load_benchmark_config(config_path)
    output_dir = validate_output_directory(
        config,
        root,
        _portfolio_root(root, config, plan) / batch.batch_id,
    )
    context = {
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "batch_id": batch.batch_id,
        "promotion_status": "not_accepted",
    }
    analysis = run_experiment(
        config_path,
        root,
        output_dir,
        max_runs=0,
        limit_queries=limit_queries,
        combination_ids=batch.combination_ids,
        portfolio_context=context,
    )
    return {
        "ok": True,
        **context,
        "combination_count": len(batch.combination_ids),
        "output_dir": str(output_dir),
        "best_config": analysis.get("best_config", {}),
    }


def portfolio_status(plan_or_root: Path) -> dict[str, Any]:
    """Report receipt-derived state without changing the immutable plan."""
    return inspect_portfolio_status(plan_or_root)


def run_next_portfolio_batch(
    root: Path,
    config_path: Path,
    plan_path: Path,
    *,
    limit_queries: int = 0,
) -> dict[str, Any]:
    """Run the first plan-ordered batch without a fully valid completion receipt."""
    plan = load_verified_portfolio_plan(root, config_path, plan_path)
    status = portfolio_status(plan_path)
    batch_id = status["selected_batch_id"]
    if batch_id is None:
        return {
            "ok": True,
            "portfolio_id": plan.portfolio_id,
            "state": "completed",
            "message": "all portfolio batches are already completed",
        }
    selected = next(item for item in status["batches"] if item["batch_id"] == batch_id)
    if selected.get("state") != "not_run":
        raise ValueError("selected batch is not unstarted; existing evidence preserved")
    result = run_portfolio_batch(
        root,
        config_path,
        plan_path,
        batch_id,
        limit_queries=limit_queries,
    )
    return {**result, "resumed_from_state": selected["state"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="WNS modular benchmark CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("config", nargs="?", default="configs/benchmark.local.json")

    matrix = sub.add_parser("matrix")
    matrix.add_argument("config", nargs="?", default="configs/benchmark.local.json")

    run = sub.add_parser("run")
    run.add_argument("config", nargs="?", default="configs/benchmark.local.json")
    run.add_argument("--output-dir", default="data/modular_runs/latest")
    run.add_argument("--max-runs", type=int, default=0)
    run.add_argument("--limit-queries", type=int, default=0)
    run.add_argument("--chunker", default="all")
    run.add_argument("--embedding", default="all")
    run.add_argument("--vector-store", default="all")
    run.add_argument("--index-type", default="all")
    run.add_argument("--retrieval-method", default="all")
    run.add_argument("--reranker", default="all")

    portfolio_plan = sub.add_parser("portfolio-plan")
    portfolio_plan.add_argument(
        "config", nargs="?", default="configs/benchmark.all-methods-portfolio.json"
    )

    portfolio_batch = sub.add_parser("portfolio-run-batch")
    portfolio_batch.add_argument(
        "config", nargs="?", default="configs/benchmark.all-methods-portfolio.json"
    )
    portfolio_batch.add_argument("--plan", type=Path, required=True)
    portfolio_batch.add_argument("--batch-id", required=True)
    portfolio_batch.add_argument("--limit-queries", type=int, default=0)

    portfolio_status_parser = sub.add_parser("portfolio-status")
    portfolio_status_parser.add_argument("plan_or_root", type=Path)

    portfolio_run_next = sub.add_parser("run-next")
    portfolio_run_next.add_argument(
        "config", nargs="?", default="configs/benchmark.all-methods-portfolio.json"
    )
    portfolio_run_next.add_argument("--plan", type=Path, required=True)
    portfolio_run_next.add_argument("--limit-queries", type=int, default=0)

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config if hasattr(args, "config") else None

    if args.cmd == "validate":
        assert config_path is not None
        cfg = load_benchmark_config(config_path)
        rows = generate_matrix(cfg)
        print(json.dumps({"ok": True, "experiment": cfg["experiment"]["name"], "matrix_count": len(rows)}, indent=2))
    elif args.cmd == "matrix":
        assert config_path is not None
        cfg = load_benchmark_config(config_path)
        print(json.dumps(generate_matrix(cfg), indent=2))
    elif args.cmd == "run":
        assert config_path is not None
        selections = {
            "chunker": args.chunker,
            "embedding": args.embedding,
            "vector_store": args.vector_store,
            "index_type": args.index_type,
            "retrieval_method": args.retrieval_method,
            "reranker": args.reranker,
        }
        output_dir = _output_directory(root, config_path, args.output_dir)
        analysis = run_experiment(
            config_path,
            root,
            output_dir,
            max_runs=args.max_runs,
            limit_queries=args.limit_queries,
            selections=selections,
        )
        print(json.dumps({"ok": True, "best_config": analysis.get("best_config", {}), "report": str(output_dir / "MODULAR_REPORT.md")}, indent=2))
    elif args.cmd == "portfolio-plan":
        assert config_path is not None
        plan, plan_path = create_portfolio_plan(root, config_path)
        print(json.dumps({
            "ok": True,
            "plan": str(plan_path),
            "portfolio_id": plan.portfolio_id,
            "portfolio_hash": plan.portfolio_hash,
            "combination_count": len(plan.combination_ids),
            "batch_count": len(plan.batches),
            "max_combinations_per_batch": plan.max_combinations_per_batch,
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "embedding": batch.embedding,
                    "reranker_group": batch.reranker_group,
                    "combination_count": len(batch.combination_ids),
                }
                for batch in plan.batches
            ],
        }, indent=2))
    elif args.cmd == "portfolio-run-batch":
        assert config_path is not None
        plan_path = args.plan if args.plan.is_absolute() else root / args.plan
        print(json.dumps(run_portfolio_batch(
            root,
            config_path,
            plan_path,
            args.batch_id,
            limit_queries=args.limit_queries,
        ), indent=2))
    elif args.cmd == "portfolio-status":
        plan_or_root = (
            args.plan_or_root
            if args.plan_or_root.is_absolute()
            else root / args.plan_or_root
        )
        print(json.dumps(portfolio_status(plan_or_root), indent=2))
    elif args.cmd == "run-next":
        assert config_path is not None
        plan_path = args.plan if args.plan.is_absolute() else root / args.plan
        print(json.dumps(run_next_portfolio_batch(
            root,
            config_path,
            plan_path,
            limit_queries=args.limit_queries,
        ), indent=2))


if __name__ == "__main__":
    main()
