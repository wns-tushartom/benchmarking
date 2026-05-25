"""Quick verification of benchmark_input.csv"""
import pandas as pd

df = pd.read_csv('data/benchmark_input.csv')

print("=" * 60)
print("Benchmark Input CSV Verification")
print("=" * 60)
print(f"\nColumns: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")
print(f"\nParagraph length stats:")
print(f"  Min: {df['paragraph'].str.len().min()} chars")
print(f"  Max: {df['paragraph'].str.len().max()} chars")
print(f"  Mean: {df['paragraph'].str.len().mean():.1f} chars")
print(f"\nUnique PDFs: {df['pdf_name'].nunique()}")
print(f"PDF distribution:")
print(df['pdf_name'].value_counts())

print("\n" + "=" * 60)
print("Sample paragraphs:")
print("=" * 60)

for idx in [0, 10, 30]:
    if idx < len(df):
        row = df.iloc[idx]
        print(f"\nRow {idx} | ID={row['id']} | PDF={row['pdf_name'][:50]}")
        print(f"Length: {len(row['paragraph'])} chars")
        print(f"Preview: {row['paragraph'][:150]}...")
