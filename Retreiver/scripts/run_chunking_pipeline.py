#!/usr/bin/env python3
"""Run WNS PDF extraction plus all chunking workbook generation.

Recommended VM command:
    .venv-vm/bin/python scripts/run_chunking_pipeline.py --mode all

Modes:
- all: re-parse every PDF in data/pdfs, then regenerate all chunking sheets.
- missing: append PDFs not already present in data/benchmark_input.csv, then regenerate all chunking sheets.
- chunk-only: leave benchmark_input.csv untouched and regenerate chunking sheets.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP_TARGETS = [
    ROOT / "data" / "benchmark_input.csv",
    ROOT / "data" / "pdf_extraction_audit.csv",
    ROOT / "data" / "chunking_methods_output.xlsx",
    ROOT / "data" / "chunking_methods_output_v2.xlsx",
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def backup_outputs() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = ROOT / "data" / "chunking_backups" / stamp
    copied = 0
    for path in BACKUP_TARGETS:
        if path.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
            copied += 1
    if copied:
        print(f"Backed up {copied} existing artifacts to {backup_dir.relative_to(ROOT)}")
    else:
        print("No existing chunking artifacts to back up.")
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "missing", "chunk-only"], default="all")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--include-image-markers", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Continue to chunking even when extraction audit has needs-review PDFs.")
    parser.add_argument("--allow-text-only-fallback", action="store_true", help="Allow PyPDF2 text-only fallback during extraction. Use for recovery, not clean MinerU runs.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.no_backup:
        backup_outputs()

    if args.mode in {"all", "missing"}:
        cmd = [py, "scripts/prepare_benchmark_input_mineru.py", "--pdf-dir", args.pdf_dir]
        if args.mode == "missing":
            cmd.append("--only-missing")
        if args.include_image_markers:
            cmd.append("--include-image-markers")
        if args.allow_partial:
            cmd.append("--allow-partial")
        if args.allow_text_only_fallback:
            cmd.append("--allow-text-only-fallback")
        run(cmd)

    # The first script creates the four established chunker sheets.
    run([py, "scripts/apply_chunking_methods.py"])
    # The second script reads the first workbook and adds candidate sheets.
    run([
        py,
        "scripts/apply_candidate_chunking_methods.py",
        "--input",
        "data/benchmark_input.csv",
        "--existing",
        "data/chunking_methods_output.xlsx",
        "--output",
        "data/chunking_methods_output_v2.xlsx",
    ])
    run([py, "scripts/verify_benchmark_input.py"])
    run([py, "scripts/verify_all_chunking_methods.py"])

    print("\nDone. Final workbook: data/chunking_methods_output_v2.xlsx")
    print("Audit file: data/pdf_extraction_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
