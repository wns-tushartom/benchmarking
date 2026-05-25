"""
Prepare benchmark input CSV from PDFs.

Extracts paragraph-level text from all PDFs in data/pdfs/
and writes to data/benchmark_input.csv with columns:
- id: sequential integer starting from 1
- pdf_name: exact filename of source PDF
- paragraph: cleaned text content

Usage:
    python scripts/prepare_benchmark_input.py
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Tuple

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not found. Install with: pip install PyPDF2")
    exit(1)


# Configuration
PDF_DIR = Path("data/pdfs")
OUTPUT_CSV = Path("data/benchmark_input.csv")
MIN_PARAGRAPH_LENGTH = 40
MAX_PARAGRAPH_LENGTH = 1500


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyPDF2."""
    text_parts = []
    
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
            
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    print(f"  Warning: Failed to extract page {page_num + 1}: {e}")
                    continue
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        print(f"  ERROR: Failed to read PDF: {e}")
        return ""


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into meaningful paragraphs.
    
    Strategy:
    1. First split on blank lines (double newlines)
    2. If a block is too long, split on sentence boundaries
    3. Filter out very short paragraphs
    """
    paragraphs = []
    
    # First split on blank lines
    blocks = text.split('\n\n')
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # If block is short enough, use as-is
        if len(block) <= MAX_PARAGRAPH_LENGTH:
            if len(block) >= MIN_PARAGRAPH_LENGTH:
                paragraphs.append(block)
        else:
            # Split long blocks on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', block)
            current_para = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_len = len(sentence)
                
                # If adding this sentence would exceed max length, save current paragraph
                if current_length + sentence_len > MAX_PARAGRAPH_LENGTH and current_para:
                    para_text = ' '.join(current_para)
                    if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                        paragraphs.append(para_text)
                    current_para = [sentence]
                    current_length = sentence_len
                else:
                    current_para.append(sentence)
                    current_length += sentence_len + 1  # +1 for space
            
            # Don't forget the last paragraph
            if current_para:
                para_text = ' '.join(current_para)
                if len(para_text) >= MIN_PARAGRAPH_LENGTH:
                    paragraphs.append(para_text)
    
    return paragraphs


def process_pdfs() -> List[Tuple[int, str, str]]:
    """
    Process all PDFs and return list of (id, pdf_name, paragraph) tuples.
    """
    if not PDF_DIR.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        return []
    
    pdf_files = sorted([f for f in PDF_DIR.iterdir() if f.suffix.lower() == '.pdf'])
    
    if not pdf_files:
        print(f"ERROR: No PDF files found in {PDF_DIR}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)
    
    all_rows = []
    current_id = 1
    total_paragraphs = 0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract text
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text:
            print(f"  Skipped: No text extracted")
            continue
        
        # Normalize whitespace
        normalized_text = normalize_whitespace(raw_text)
        
        # Split into paragraphs
        paragraphs = split_into_paragraphs(normalized_text)
        
        if not paragraphs:
            print(f"  Skipped: No valid paragraphs found")
            continue
        
        # Add to results
        for para in paragraphs:
            all_rows.append((current_id, pdf_file.name, para))
            current_id += 1
        
        print(f"  Extracted: {len(paragraphs)} paragraphs")
        total_paragraphs += len(paragraphs)
    
    print("\n" + "=" * 60)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total paragraphs extracted: {total_paragraphs}")
    
    return all_rows


def write_csv(rows: List[Tuple[int, str, str]], output_path: Path):
    """Write rows to CSV file."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['id', 'pdf_name', 'paragraph'])
        # Write data
        writer.writerows(rows)
    
    print(f"\nOutput written to: {output_path}")
    print(f"Total rows: {len(rows)}")


def main():
    print("Benchmark Input Preparation")
    print("=" * 60)
    
    # Process PDFs
    rows = process_pdfs()
    
    if not rows:
        print("\nERROR: No data to write")
        return
    
    # Write CSV
    write_csv(rows, OUTPUT_CSV)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
