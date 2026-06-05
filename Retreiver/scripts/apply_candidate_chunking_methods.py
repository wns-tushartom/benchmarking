"""Add candidate chunking methods to WNS benchmark workbook.

Creates two extra chunking outputs:
- semantic_split
- fixed_tok1200_ov150

Input:
    data/benchmark_input.csv

Output:
    data/chunking_methods_output_v2.xlsx

Usage from Retreiver repo root:
    python scripts/apply_candidate_chunking_methods.py

Optional:
    python scripts/apply_candidate_chunking_methods.py --input data/benchmark_input.csv --existing data/chunking_methods_output.xlsx --output data/chunking_methods_output_v2.xlsx
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas/openpyxl. Install with: pip install pandas openpyxl"
    ) from exc


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "you", "your", "have", "has", "had", "not", "but", "can", "will", "all",
    "any", "our", "their", "then", "than", "into", "out", "when", "where", "what",
    "which", "who", "how", "why", "step", "page", "pdf", "created", "updated",
}


def normalize_text(text: Any) -> str:
    """Normalize whitespace and remove Excel-hostile control chars."""
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences/lines while keeping useful fragments."""
    text = normalize_text(text)
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p and p.strip()]
    return parts or [text]


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: List[str]) -> str:
    """Simple readable reconstruction after regex tokenization."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return normalize_text(text)


def content_terms(text: str) -> set[str]:
    terms = {w.lower() for w in WORD_RE.findall(text)}
    return {t for t in terms if t not in STOPWORDS and len(t) >= 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fixed_token_chunks(
    text: str,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 150,
    min_chars: int = 50,
) -> List[str]:
    """Fixed token chunking: 1200-token windows with 150-token overlap."""
    text = normalize_text(text)
    tokens = tokenize(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [text] if len(text) >= min_chars else []

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_tokens]
        chunk = detokenize(window)
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if start + chunk_tokens >= len(tokens):
            break
        start += step
    return chunks


def semantic_split_chunks(
    text: str,
    min_chars: int = 120,
    target_chars: int = 1200,
    max_chars: int = 1500,
    similarity_threshold: float = 0.12,
) -> List[str]:
    """Dependency-light semantic-ish splitting.

    This is a lexical fallback: it compares adjacent sentence term overlap and
    splits when overlap is low after a minimum chunk size is reached. It is not
    an official semantic embedding splitter; evaluate recall before selecting.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []

    for i, sent in enumerate(sentences):
        sent = normalize_text(sent)
        if not sent:
            continue

        if not current:
            current.append(sent)
            continue

        current_text = " ".join(current)
        current_len = len(current_text)
        sim = jaccard(content_terms(current[-1]), content_terms(sent))

        should_split = False
        if current_len >= min_chars and sim < similarity_threshold:
            should_split = True
        if current_len + len(sent) + 1 > max_chars:
            should_split = True
        if current_len >= target_chars and sim < (similarity_threshold * 2):
            should_split = True

        if should_split:
            chunk = normalize_text(" ".join(current))
            if len(chunk) >= 50:
                chunks.append(chunk)
            current = [sent]
        else:
            current.append(sent)

    if current:
        chunk = normalize_text(" ".join(current))
        if len(chunk) >= 50:
            chunks.append(chunk)

    # If all fragments were too short, keep original when useful.
    if not chunks and len(normalize_text(text)) >= 50:
        return [normalize_text(text)]
    return chunks


def rows_for_method(df: "pd.DataFrame", method: str) -> "pd.DataFrame":
    output: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        pdf_name = row["pdf_name"]
        paragraph = normalize_text(row["paragraph"])

        if method == "semantic_split":
            chunks = semantic_split_chunks(paragraph)
        elif method == "fixed_tok1200_ov150":
            chunks = fixed_token_chunks(paragraph, chunk_tokens=1200, overlap_tokens=150)
        else:
            raise ValueError(f"Unknown method: {method}")

        for chunk in chunks:
            output.append({
                "id": len(output) + 1,
                "pdf_name": pdf_name,
                "paragraph": chunk,
            })

    return pd.DataFrame(output, columns=["id", "pdf_name", "paragraph"])


def verify_sheet(name: str, df: "pd.DataFrame") -> None:
    expected = ["id", "pdf_name", "paragraph"]
    if list(df.columns) != expected:
        raise ValueError(f"{name}: columns must be {expected}, got {list(df.columns)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{name}: duplicate ids found")
    if df["pdf_name"].isna().any() or df["paragraph"].isna().any():
        raise ValueError(f"{name}: null pdf_name or paragraph found")
    if df.empty:
        raise ValueError(f"{name}: no rows produced")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/benchmark_input.csv")
    parser.add_argument("--existing", default="data/chunking_methods_output.xlsx")
    parser.add_argument("--output", default="data/chunking_methods_output_v2.xlsx")
    args = parser.parse_args()

    input_path = Path(args.input)
    existing_path = Path(args.existing)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    expected = ["id", "pdf_name", "paragraph"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns {missing}; available columns: {list(df.columns)}")
    # Keep the canonical benchmark columns for the Excel sheets. Source/page/parser
    # metadata may be present in benchmark_input.csv after MinerU extraction, but
    # downstream chunking workbooks intentionally remain id,pdf_name,paragraph.
    df = df[expected].copy()

    sheets: Dict[str, pd.DataFrame] = {}

    if existing_path.exists():
        existing = pd.read_excel(existing_path, sheet_name=None)
        for name, sheet_df in existing.items():
            # Keep only required columns in case Excel added formatting columns.
            sheets[name] = sheet_df[expected].copy()
    else:
        sheets["original_input"] = df[expected].copy()

    for method in ["semantic_split", "fixed_tok1200_ov150"]:
        print(f"Generating {method}...")
        method_df = rows_for_method(df, method)
        verify_sheet(method, method_df)
        sheets[method] = method_df
        lengths = method_df["paragraph"].astype(str).str.len()
        print(
            f"  {method}: rows={len(method_df)}, "
            f"min_len={lengths.min()}, max_len={lengths.max()}, "
            f"avg_len={lengths.mean():.1f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, sheet_df in sheets.items():
            verify_sheet(name, sheet_df)
            sheet_df.to_excel(writer, sheet_name=name[:31], index=False)

    print("\nDone")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Sheets:")
    for name, sheet_df in sheets.items():
        print(f"  - {name}: {len(sheet_df)} rows")


if __name__ == "__main__":
    main()
