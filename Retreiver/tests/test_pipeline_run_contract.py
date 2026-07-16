from __future__ import annotations

from io import BytesIO

import pytest

from scripts.pipeline_run_contract import merge_queries, parse_query_upload


def test_parse_query_upload_accepts_txt_one_query_per_line() -> None:
    result = parse_query_upload("batch.txt", b"First question\n\nSecond question\n")

    assert result == {
        "queries": ["First question", "Second question"],
        "query_count": 2,
        "source_type": "query_upload",
    }


def test_parse_query_upload_accepts_csv_query_alias() -> None:
    result = parse_query_upload(
        "batch.csv",
        b"query,answer\nFirst question,one\nSecond question,two\n",
    )

    assert result["queries"] == ["First question", "Second question"]
    assert result["query_count"] == 2


def test_parse_query_upload_accepts_xlsx_question_alias() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["question"])
    sheet.append(["First question"])
    sheet.append(["Second question"])
    content = BytesIO()
    workbook.save(content)
    workbook.close()

    result = parse_query_upload("batch.xlsx", content.getvalue())

    assert result["queries"] == ["First question", "Second question"]


def test_parse_query_upload_rejects_invalid_and_never_echoes_filename() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_query_upload("/private/path/questions.json", b"[]")

    result = parse_query_upload("/private/path/questions.txt", b"Safe question\n")
    serialized = repr(result)
    assert "private" not in serialized
    assert "questions.txt" not in serialized


def test_merge_queries_preserves_order_and_deduplicates_normalized_values() -> None:
    assert merge_queries(
        [" Typed question ", "SECOND question"],
        ["typed   question", "Uploaded question", "second QUESTION"],
    ) == ["Typed question", "SECOND question", "Uploaded question"]
