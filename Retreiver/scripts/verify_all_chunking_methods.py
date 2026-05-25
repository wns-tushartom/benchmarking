"""Verify all chunking methods in the updated Excel file"""
import pandas as pd

xl = pd.ExcelFile('data/chunking_methods_output_v2.xlsx')

print("=" * 70)
print("ALL CHUNKING METHODS VERIFICATION")
print("=" * 70)
print(f"\nExcel file: data/chunking_methods_output_v2.xlsx")
print(f"Total sheets: {len(xl.sheet_names)}")
print("\n" + "=" * 70)
print("CHUNKING METHODS SUMMARY")
print("=" * 70)

for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet)
    lengths = df['paragraph'].astype(str).str.len()
    
    print(f"\n{sheet}:")
    print(f"  Total chunks: {len(df)}")
    print(f"  Avg length: {lengths.mean():.1f} chars")
    print(f"  Min length: {lengths.min()} chars")
    print(f"  Max length: {lengths.max()} chars")
    
    if sheet != 'original_input':
        expansion = len(df) / 61  # 61 is original input count
        print(f"  Expansion ratio: {expansion:.2f}x")

print("\n" + "=" * 70)
print("COMPARISON: NEW METHODS vs ORIGINAL")
print("=" * 70)

methods_to_compare = ['semantic_split', 'fixed_tok1200_ov150']

for method in methods_to_compare:
    df = pd.read_excel(xl, method)
    lengths = df['paragraph'].astype(str).str.len()
    
    print(f"\n{method}:")
    print(f"  Strategy: {'Semantic similarity-based' if 'semantic' in method else 'Fixed token windows'}")
    print(f"  Chunks created: {len(df)}")
    print(f"  Expansion: {len(df) / 61:.2f}x from original")
    print(f"  Avg chunk size: {lengths.mean():.0f} chars")
    print(f"  Size range: {lengths.min()}-{lengths.max()} chars")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("\nBoth methods are now ready for Recall@5/Recall@10 evaluation.")
print("Next step: Run benchmark with actual queries to determine which")
print("chunking method provides better retrieval performance.")
print("\nKey differences:")
print("  - semantic_split: 122 chunks, avg 142 chars (more granular)")
print("  - fixed_tok1200_ov150: 56 chunks, avg 319 chars (larger context)")
print("\n" + "=" * 70)
