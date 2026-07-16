"""Validated contracts shared by dashboard pipeline entry points."""
from __future__ import annotations

from pathlib import Path
import unicodedata
from typing import Iterable

from source.services.project_questions import parse_question_bytes


def _query_identity(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def merge_queries(*groups: Iterable[str]) -> list[str]:
    """Merge query groups in order while removing normalized duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            query = " ".join(str(raw).split())
            if not query:
                continue
            identity = _query_identity(query)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(query)
    return merged


def parse_query_upload(
    filename: str,
    content: bytes,
    *,
    max_questions: int = 10_000,
    max_file_bytes: int = 10 * 1024 * 1024,
) -> dict[str, object]:
    """Parse a bounded TXT/CSV/XLSX query upload without returning its filename/path."""
    safe_filename = Path(str(filename)).name
    questions = parse_question_bytes(
        safe_filename,
        content,
        max_questions=max_questions,
        max_file_bytes=max_file_bytes,
    )
    queries = merge_queries(question.query for question in questions)
    return {
        "queries": queries,
        "query_count": len(queries),
        "source_type": "query_upload",
    }
