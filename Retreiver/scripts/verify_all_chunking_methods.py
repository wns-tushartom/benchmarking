"""Verify all chunking methods in the updated Excel file."""
from __future__ import annotations

import pandas as pd

WORKBOOK = "data/chunking_methods_output_v2.xlsx"

xl = pd.ExcelFile(WORKBOOK, engine="openpyxl")

print("=" * 70)
print("ALL CHUNKING METHODS VERIFICATION")
print("=" * 70)
print(f"\nExcel file: {WORKBOOK}")
print(f"Total sheets: {len(xl.sheet_names)}")

sheets = {sheet: pd.read_excel(xl, sheet, keep_default_na=False) for sheet in xl.sheet_names}
if "original_input" not in sheets:
    raise SystemExit("Missing required sheet: original_input")

required = ["id", "pdf_name", "paragraph"]
original_count = len(sheets["original_input"])
print("\n" + "=" * 70)
print("CHUNKING METHODS SUMMARY")
print("=" * 70)

for sheet, df in sheets.items():
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"{sheet}: missing columns {missing}; available={list(df.columns)}")
    if df["pdf_name"].isna().any() or df["paragraph"].isna().any():
        raise SystemExit(f"{sheet}: null pdf_name or paragraph found")
    if df["id"].duplicated().any():
        raise SystemExit(f"{sheet}: duplicate ids found")

    lengths = df["paragraph"].astype(str).str.len()
    print(f"\n{sheet}:")
    print(f"  Total chunks: {len(df)}")
    print(f"  PDFs covered: {df['pdf_name'].nunique()}")
    print(f"  Avg length: {lengths.mean():.1f} chars")
    print(f"  Min length: {lengths.min()} chars")
    print(f"  Max length: {lengths.max()} chars")
    if sheet != "original_input":
        print(f"  Expansion ratio: {len(df) / max(original_count, 1):.2f}x")

print("\n" + "=" * 70)
print("CANDIDATE METHOD CHECK")
print("=" * 70)
for method in ["semantic_split", "fixed_tok1200_ov150"]:
    if method not in sheets:
        print(f"\n{method}: missing")
        continue
    df = sheets[method]
    lengths = df["paragraph"].astype(str).str.len()
    strategy = "Semantic similarity-based lexical fallback" if "semantic" in method else "Fixed token windows"
    print(f"\n{method}:")
    print(f"  Strategy: {strategy}")
    print(f"  Chunks created: {len(df)}")
    print(f"  Expansion: {len(df) / max(original_count, 1):.2f}x from original")
    print(f"  Avg chunk size: {lengths.mean():.0f} chars")
    print(f"  Size range: {lengths.min()}-{lengths.max()} chars")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("Workbook is structurally valid. Next step: run Recall@5/Recall@10 using ground-truth queries before selecting a chunking method.")
