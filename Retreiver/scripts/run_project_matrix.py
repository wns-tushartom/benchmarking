#!/usr/bin/env python3
"""Execute one validated project matrix run by project/run IDs only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from source.services.project_matrix_runner import ProjectMatrixRunner
from source.services.project_workspace import ProjectWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated project matrix")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    configured = os.environ.get("USER_PROJECTS_DIR")
    projects_root = (
        Path(configured).resolve()
        if configured
        else (_REPOSITORY_ROOT / "data" / "user_projects").resolve()
    )
    result = ProjectMatrixRunner(ProjectWorkspace(projects_root)).run(
        arguments.project_id,
        arguments.run_id,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    state = str(result.get("state") or "").strip().lower()
    if state == "completed":
        return 0
    if state == "partial":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
