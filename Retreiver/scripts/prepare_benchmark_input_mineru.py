#!/usr/bin/env python3
"""Prepare benchmark input using Shubham's MinerU-first document parser.

Primary path:
  data/pdfs/*.pdf -> source/services/document_parser.py -> MinerU/magic-pdf
Fallback:
  PyPDF2 when MinerU is unavailable or fails and fallback is allowed by config.

Outputs:
  data/benchmark_input.csv          id, pdf_name, paragraph, page/source metadata
  data/pdf_extraction_audit.csv     per-PDF parser/audit status, including image/table/formula counts
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.services.document_parser import MINERU_AVAILABLE, PYPDF2_AVAILABLE, document_parser  # noqa: E402

PDF_DIR = ROOT / "data" / "pdfs"
OUTPUT_CSV = ROOT / "data" / "benchmark_input.csv"
AUDIT_CSV = ROOT / "data" / "pdf_extraction_audit.csv"
MIN_PARAGRAPH_LENGTH = 40
MAX_CHARS = 1500


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def split_blocks(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            if len(block) >= MIN_PARAGRAPH_LENGTH:
                out.append(block)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", block)
        cur: list[str] = []
        cur_len = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if cur and cur_len + len(sent) > max_chars:
                chunk = " ".join(cur).strip()
                if len(chunk) >= MIN_PARAGRAPH_LENGTH:
                    out.append(chunk)
                cur = [sent]
                cur_len = len(sent)
            else:
                cur.append(sent)
                cur_len += len(sent) + 1
        if cur:
            chunk = " ".join(cur).strip()
            if len(chunk) >= MIN_PARAGRAPH_LENGTH:
                out.append(chunk)
    return out


def table_to_text(table: Any) -> str:
    if isinstance(table, dict):
        for key in ("table_body", "html", "text", "markdown", "caption"):
            value = table.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(table, ensure_ascii=False)[:4000]
    return str(table or "").strip()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


async def parse_one(pdf: Path, include_image_markers: bool, allow_text_only_fallback: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_items: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "pdf_name": pdf.name,
        "status": "ok",
        "parser_method": "",
        "total_pages": 0,
        "parsed_pages": 0,
        "text_chars": 0,
        "image_count": 0,
        "table_count": 0,
        "formula_count": 0,
        "row_count": 0,
        "needs_ocr_review": 0,
        "error": "",
    }
    try:
        await document_parser.validate_file(str(pdf))
        if not MINERU_AVAILABLE and not allow_text_only_fallback:
            raise RuntimeError("MinerU/magic-pdf is not available. Refusing PyPDF2 text-only extraction because image/table/formula/layout counts would be wrong. Install requirements-mineru.txt or pass --allow-text-only-fallback for temporary text-only rows.")
        parsed = await document_parser.parse_pdf(str(pdf), pdf.name, allow_fallback=allow_text_only_fallback)
        audit["parser_method"] = parsed.metadata.get("parsing_method", "")
        text_only_fallback = audit["parser_method"] == "PyPDF2_fallback"
        audit["total_pages"] = parsed.metadata.get("total_pdf_pages", parsed.total_pages)
        audit["parsed_pages"] = len(parsed.content)
        for page in parsed.content:
            page_num = int(page.page_number) + 1
            pieces: list[tuple[str, str]] = []
            if page.content:
                pieces.append(("text", page.content))
            for table in page.tables or []:
                pieces.append(("table", table_to_text(table)))
            for formula in page.formulas or []:
                pieces.append(("formula", str(formula)))
            if include_image_markers:
                for image_path in page.images or []:
                    pieces.append(("image", f"[image extracted from {pdf.name} page {page_num}: {image_path}]"))

            audit["text_chars"] += len(page.content or "")
            audit["image_count"] += len(page.images or [])
            audit["table_count"] += len(page.tables or [])
            audit["formula_count"] += len(page.formulas or [])

            for source_type, text in pieces:
                for block in split_blocks(text):
                    row_items.append({
                        "pdf_name": pdf.name,
                        "page_number": page_num,
                        "source_type": source_type,
                        "paragraph": block,
                        "parser_method": audit["parser_method"],
                        "image_count": len(page.images or []),
                        "table_count": len(page.tables or []),
                        "formula_count": len(page.formulas or []),
                    })
        audit["row_count"] = len(row_items)
        if text_only_fallback:
            audit["status"] = "text_only_review"
            audit["needs_ocr_review"] = 1
            audit["error"] = "PyPDF2 fallback extracts text only. Image/table/formula/layout counts are not valid until MinerU/magic-pdf extraction runs."
        elif audit["text_chars"] < 100 or audit["row_count"] == 0:
            audit["status"] = "needs_ocr"
            audit["needs_ocr_review"] = 1
        elif audit["image_count"] and audit["text_chars"] / max(1, int(audit["parsed_pages"] or 1)) < 300:
            audit["status"] = "partial_ocr_review"
            audit["needs_ocr_review"] = 1
    except Exception as exc:
        audit["status"] = "failed"
        audit["needs_ocr_review"] = 1
        audit["error"] = str(exc)[:1000]
    return row_items, audit


async def main_async(args: argparse.Namespace) -> int:
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT / pdf_dir
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    audit_path = Path(args.audit_output)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return 2

    row_fields = ["id", "pdf_name", "paragraph", "page_number", "source_type", "parser_method", "image_count", "table_count", "formula_count"]
    audit_fields = ["pdf_name", "status", "parser_method", "total_pages", "parsed_pages", "text_chars", "image_count", "table_count", "formula_count", "row_count", "needs_ocr_review", "error"]
    existing_rows: list[dict[str, Any]] = []
    existing_audits: list[dict[str, Any]] = []
    existing_pdf_names: set[str] = set()
    if args.only_missing and output.exists():
        with output.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_pdf_names = {str(r.get("pdf_name", "")).strip() for r in existing_rows if str(r.get("pdf_name", "")).strip()}
        if audit_path.exists():
            with audit_path.open("r", encoding="utf-8-sig", newline="") as f:
                existing_audits = list(csv.DictReader(f))
        reprocess_statuses = {"failed", "text_only_review", "needs_ocr", "partial_ocr_review"}
        reprocess_pdf_names = {
            str(a.get("pdf_name", "")).strip()
            for a in existing_audits
            if str(a.get("status", "")).strip() in reprocess_statuses
        }
        if reprocess_pdf_names:
            existing_rows = [r for r in existing_rows if str(r.get("pdf_name", "")).strip() not in reprocess_pdf_names]
            existing_audits = [a for a in existing_audits if str(a.get("pdf_name", "")).strip() not in reprocess_pdf_names]
            existing_pdf_names -= reprocess_pdf_names
        pdfs = [pdf for pdf in pdfs if pdf.name not in existing_pdf_names]

    print(f"MinerU available: {MINERU_AVAILABLE}; PyPDF2 available: {PYPDF2_AVAILABLE}")
    if args.only_missing:
        print(f"Existing benchmark rows: {len(existing_rows)} across {len(existing_pdf_names)} PDFs")
        print(f"Missing PDFs to process: {len(pdfs)}")
    else:
        print(f"Processing {len(pdfs)} PDFs from {pdf_dir}")
    if not pdfs and args.only_missing:
        print("No missing PDFs found. Existing benchmark_input.csv was left unchanged.")
        return 0

    parsed_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pdf in pdfs:
        rows, audit = await parse_one(pdf, include_image_markers=args.include_image_markers, allow_text_only_fallback=args.allow_text_only_fallback)
        audits.append(audit)
        parsed_rows.extend(rows)
        print(f"{audit['status']}\t{pdf.name}\tmethod={audit['parser_method']}\trows={audit['row_count']}\timages={audit['image_count']}\ttables={audit['table_count']}")

    all_rows = existing_rows + parsed_rows if args.only_missing else parsed_rows
    for idx, row in enumerate(all_rows, 1):
        row["id"] = idx
    audits_out = existing_audits + audits if args.only_missing else audits

    write_csv(output, [{field: row.get(field, "") for field in row_fields} for row in all_rows], row_fields)
    write_csv(audit_path, [{field: row.get(field, "") for field in audit_fields} for row in audits_out], audit_fields)

    needs = [a for a in audits if str(a.get("needs_ocr_review")) == "1"]
    print(f"Wrote {len(all_rows)} rows to {output.relative_to(ROOT)}")
    print(f"Wrote audit to {audit_path.relative_to(ROOT)}")
    print(f"PDFs needing OCR/review: {len(needs)}")
    for a in needs[:50]:
        print(f"needs_review\t{a['pdf_name']}\tstatus={a['status']}\terror={a['error']}")
    if needs and not args.allow_partial:
        print("Extraction is not final-product clean. Re-run after fixing MinerU/layout extraction, or pass --allow-partial only for temporary text-only/debug artifacts.")
        return 1
    if not all_rows:
        print("No benchmark rows were produced.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--output", default="data/benchmark_input.csv")
    parser.add_argument("--audit-output", default="data/pdf_extraction_audit.csv")
    parser.add_argument("--include-image-markers", action="store_true", help="Add image-reference rows when MinerU emits image paths. OCR text is still taken from MinerU text/table output.")
    parser.add_argument("--allow-text-only-fallback", action="store_true", help="Allow PyPDF2 fallback when MinerU is unavailable. Use only for temporary text-only debugging; images/tables/formulas will be marked invalid for review.")
    parser.add_argument("--allow-partial", action="store_true", help="Return success even when PDFs need OCR/layout review. Use only for temporary debugging, not final product runs.")
    parser.add_argument("--only-missing", action="store_true", help="Append only PDFs not already present in the output CSV; use this for newly added PDFs.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
