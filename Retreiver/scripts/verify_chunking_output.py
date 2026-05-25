"""Verify chunking output Excel file"""
import pandas as pd

xl = pd.ExcelFile('data/chunking_methods_output.xlsx')

print("=" * 60)
print("CHUNKING OUTPUT VERIFICATION")
print("=" * 60)
print(f"\nExcel file: data/chunking_methods_output.xlsx")
print(f"Total sheets: {len(xl.sheet_names)}")
print("\nSheets and row counts:")

for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet)
    print(f"  {sheet}: {len(df)} rows")

print("\n" + "=" * 60)
print("Sample from entity_heuristic_w6 (first 3 rows):")
print("=" * 60)

df = pd.read_excel(xl, 'entity_heuristic_w6')
for idx in range(min(3, len(df))):
    row = df.iloc[idx]
    print(f"\nRow {idx + 1}:")
    print(f"  ID: {row['id']}")
    print(f"  PDF: {row['pdf_name'][:50]}...")
    print(f"  Paragraph length: {len(row['paragraph'])} chars")
    print(f"  Preview: {row['paragraph'][:100]}...")

print("\n" + "=" * 60)
print("✓ Verification complete!")
print("=" * 60)
