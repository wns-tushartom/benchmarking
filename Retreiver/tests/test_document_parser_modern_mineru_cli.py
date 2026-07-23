from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from source.services import document_parser


def test_modern_mineru_cli_uses_pipeline_with_tables_and_no_formula(
    tmp_path: Path, monkeypatch,
) -> None:
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="parsed", stderr="")
    run = Mock(return_value=result)
    legacy_api = Mock()
    monkeypatch.setattr(document_parser, "MINERU_AVAILABLE", True)
    monkeypatch.setattr(document_parser, "pdf_parse_main", legacy_api)
    monkeypatch.setattr(document_parser, "MAGIC_PDF_CLI", "/tmp/mineru-bin/mineru")
    monkeypatch.setattr(document_parser.subprocess, "run", run)
    parser = document_parser.DocumentParserService(
        output_dir=tmp_path / "parsed", force_backend="mineru"
    )

    parser._run_mineru_parsing("/tmp/input.pdf", "/tmp/output")

    assert run.call_args.args[0] == [
        "/tmp/mineru-bin/mineru",
        "-p", "/tmp/input.pdf",
        "-o", "/tmp/output",
        "-b", "pipeline",
        "-m", "auto",
        "-f", "false",
        "-t", "true",
    ]
    legacy_api.assert_not_called()


def test_modern_mineru_failure_is_visible_and_does_not_call_legacy_api(
    tmp_path: Path, monkeypatch,
) -> None:
    result = subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="DownloadModelError: unavailable"
    )
    legacy_api = Mock()
    monkeypatch.setattr(document_parser, "MINERU_AVAILABLE", True)
    monkeypatch.setattr(document_parser, "pdf_parse_main", legacy_api)
    monkeypatch.setattr(document_parser, "MAGIC_PDF_CLI", "/tmp/mineru-bin/mineru")
    monkeypatch.setattr(document_parser.subprocess, "run", lambda *_args, **_kwargs: result)
    parser = document_parser.DocumentParserService(
        output_dir=tmp_path / "parsed", force_backend="mineru"
    )

    with pytest.raises(RuntimeError, match="MinerU CLI failed"):
        parser._run_mineru_parsing("/tmp/input.pdf", "/tmp/output")

    legacy_api.assert_not_called()


def test_structured_mineru_content_retains_pages_tables_and_image_derived_text(
    tmp_path: Path,
) -> None:
    parser = document_parser.DocumentParserService(
        output_dir=tmp_path / "parsed", force_backend="mineru"
    )

    pages = parser._process_json_content(
        [
            {"type": "title", "page_idx": 2, "text": "Layout heading"},
            {
                "type": "image",
                "page_idx": 2,
                "image_path": "images/figure-1.jpg",
                "caption": "Image-derived refund diagram",
            },
            {
                "type": "table",
                "page_idx": 2,
                "table_body": "| Rule | Days |\n|---|---|\n| Refund | 30 |",
            },
        ]
    )

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert pages[0].images == ["images/figure-1.jpg"]
    assert pages[0].tables[0]["type"] == "table"
    assert "Image-derived refund diagram" in pages[0].content
    assert "| Refund | 30 |" in pages[0].content
    assert pages[0].metadata["image_count"] == 1
    assert pages[0].metadata["table_count"] == 1
