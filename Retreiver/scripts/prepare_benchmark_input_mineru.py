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


async def parse_one(pdf: Path, include_image_markers: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        parsed = await document_parser.parse_pdf(str(pdf), pdf.name)
        audit["parser_method"] = parsed.metadata.get("parsing_method", "")
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
        if audit["text_chars"] < 100 or audit["row_count"] == 0:
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
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return 2

    print(f"MinerU available: {MINERU_AVAILABLE}; PyPDF2 available: {PYPDF2_AVAILABLE}")
    print(f"Processing {len(pdfs)} PDFs from {pdf_dir}")

    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for pdf in pdfs:
        rows, audit = await parse_one(pdf, include_image_markers=args.include_image_markers)
        audits.append(audit)
        all_rows.extend(rows)
        print(f"{audit['status']}\t{pdf.name}\tmethod={audit['parser_method']}\trows={audit['row_count']}\timages={audit['image_count']}\ttables={audit['table_count']}")

    for idx, row in enumerate(all_rows, 1):
        row["id"] = idx

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    audit_path = Path(args.audit_output)
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path

    write_csv(output, all_rows, ["id", "pdf_name", "paragraph", "page_number", "source_type", "parser_method", "image_count", "table_count", "formula_count"])
    write_csv(audit_path, audits, ["pdf_name", "status", "parser_method", "total_pages", "parsed_pages", "text_chars", "image_count", "table_count", "formula_count", "row_count", "needs_ocr_review", "error"])

    needs = [a for a in audits if str(a.get("needs_ocr_review")) == "1"]
    print(f"Wrote {len(all_rows)} rows to {output.relative_to(ROOT)}")
    print(f"Wrote audit to {audit_path.relative_to(ROOT)}")
    print(f"PDFs needing OCR/review: {len(needs)}")
    for a in needs[:50]:
        print(f"needs_review\t{a['pdf_name']}\tstatus={a['status']}\terror={a['error']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--output", default="data/benchmark_input.csv")
    parser.add_argument("--audit-output", default="data/pdf_extraction_audit.csv")
    parser.add_argument("--include-image-markers", action="store_true", help="Add image-reference rows when MinerU emits image paths. OCR text is still taken from MinerU text/table output.")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
