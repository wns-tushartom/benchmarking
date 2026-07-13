"""
Apply different chunking methods to benchmark input data.

Creates an Excel file with separate sheets for each chunking method:
- entity_heuristic_w6
- entity_heuristic_w5
- entity_heuristic_w4
- Heading_sections_l2

Each sheet contains: id, pdf_name, paragraph, plus optional source metadata
(page_number, source_type, parser_method, image/table/formula counts) when present.

Usage:
    python scripts/apply_chunking_methods.py
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from source.services.project_chunking import (
    entity_heuristic_chunking as _shared_entity_heuristic_chunking,
    extract_entities as _shared_extract_entities,
    heading_sections_chunking as _shared_heading_sections_chunking,
)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not found. Install with: pip install pandas openpyxl")
    exit(1)

# Configuration
INPUT_CSV = Path("data/benchmark_input.csv")
OUTPUT_EXCEL = Path("data/chunking_methods_output.xlsx")
REQUIRED_COLUMNS = ['id', 'pdf_name', 'paragraph']
OPTIONAL_METADATA_COLUMNS = ['page_number', 'source_type', 'parser_method', 'image_count', 'table_count', 'formula_count']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text_for_excel(text: str) -> str:
    """
    Clean text to remove characters that are illegal in Excel.
    Excel doesn't allow certain control characters.
    """
    if not text:
        return text

    # Remove control characters except tab, newline, carriage return
    cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')

    return cleaned


def load_input_data() -> List[Dict[str, Any]]:
    """Load benchmark input CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                'id': int(row['id']),
                'pdf_name': row['pdf_name'],
                'paragraph': clean_text_for_excel(row['paragraph'])
            }
            for column in OPTIONAL_METADATA_COLUMNS:
                if column in row:
                    item[column] = row.get(column, '')
            rows.append(item)

    logger.info(f"Loaded {len(rows)} rows from {INPUT_CSV}")
    return rows



# Public legacy names are direct aliases to the project production implementations.
extract_entities = _shared_extract_entities
entity_heuristic_chunking = _shared_entity_heuristic_chunking
heading_sections_chunking = _shared_heading_sections_chunking


def apply_chunking_method(rows: List[Dict[str, Any]], method_name: str) -> pd.DataFrame:
    """
    Apply a chunking method to all input rows.

    Args:
        rows: Input data rows
        method_name: Name of chunking method

    Returns:
        DataFrame with columns: id, pdf_name, paragraph
    """
    logger.info(f"Applying chunking method: {method_name}")

    results = []
    chunk_id = 1

    for row in rows:
        original_text = row['paragraph']
        pdf_name = row['pdf_name']

        # Apply appropriate chunking method
        if method_name == 'entity_heuristic_w6':
            chunks = entity_heuristic_chunking(original_text, window_size=6)
        elif method_name == 'entity_heuristic_w5':
            chunks = entity_heuristic_chunking(original_text, window_size=5)
        elif method_name == 'entity_heuristic_w4':
            chunks = entity_heuristic_chunking(original_text, window_size=4)
        elif method_name == 'Heading_sections_l2':
            chunks = heading_sections_chunking(original_text, level=2)
        else:
            raise ValueError(f"Unknown chunking method: {method_name}")

        # Add chunks to results
        for chunk in chunks:
            item = {
                'id': chunk_id,
                'pdf_name': pdf_name,
                'paragraph': chunk
            }
            for column in OPTIONAL_METADATA_COLUMNS:
                if column in row:
                    item[column] = row.get(column, '')
            results.append(item)
            chunk_id += 1

    df = pd.DataFrame(results)
    logger.info(f"  {method_name}: {len(results)} chunks created from {len(rows)} input paragraphs")

    return df


def create_excel_output(input_rows: List[Dict[str, Any]]) -> None:
    """
    Create Excel file with separate sheets for each chunking method.
    """
    logger.info("Creating Excel output...")

    # Define chunking methods
    methods = [
        'entity_heuristic_w6',
        'entity_heuristic_w5',
        'entity_heuristic_w4',
        'Heading_sections_l2'
    ]

    # Create Excel writer
    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Add original input as first sheet
        original_df = pd.DataFrame(input_rows)
        original_df.to_excel(writer, sheet_name='original_input', index=False)
        logger.info(f"  Added sheet: original_input ({len(original_df)} rows)")

        # Apply each chunking method and add as sheet
        for method in methods:
            df = apply_chunking_method(input_rows, method)
            df.to_excel(writer, sheet_name=method, index=False)
            logger.info(f"  Added sheet: {method} ({len(df)} rows)")

    logger.info(f"✓ Excel file created: {OUTPUT_EXCEL}")


def print_summary(input_rows: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    logger.info("=" * 60)
    logger.info("CHUNKING SUMMARY")
    logger.info("=" * 60)

    methods = [
        'entity_heuristic_w6',
        'entity_heuristic_w5',
        'entity_heuristic_w4',
        'Heading_sections_l2'
    ]

    logger.info(f"Original input: {len(input_rows)} paragraphs")
    logger.info("")

    for method in methods:
        df = apply_chunking_method(input_rows, method)
        avg_length = df['paragraph'].str.len().mean()
        min_length = df['paragraph'].str.len().min()
        max_length = df['paragraph'].str.len().max()

        logger.info(f"{method}:")
        logger.info(f"  Total chunks: {len(df)}")
        logger.info(f"  Avg length: {avg_length:.0f} chars")
        logger.info(f"  Min length: {min_length} chars")
        logger.info(f"  Max length: {max_length} chars")
        logger.info(f"  Expansion ratio: {len(df) / len(input_rows):.2f}x")
        logger.info("")

    logger.info(f"Output file: {OUTPUT_EXCEL}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("CHUNKING METHODS APPLICATION")
    logger.info("=" * 60)

    try:
        # Load input data
        input_rows = load_input_data()

        # Create Excel output
        create_excel_output(input_rows)

        # Print summary
        print_summary(input_rows)

        logger.info("✓ Done!")

    except Exception as e:
        logger.error(f"✗ Failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
