from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

from source.services import document_parser


def test_discovers_cli_next_to_unresolved_venv_interpreter(
    tmp_path: Path, monkeypatch,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.touch()
    legacy = venv_bin / "magic-pdf"
    legacy.touch()
    legacy.chmod(0o755)
    monkeypatch.setattr(document_parser.sys, "executable", str(interpreter))
    monkeypatch.setattr(document_parser.shutil, "which", lambda _name: None)

    assert document_parser.find_mineru_cli() == str(legacy)


def test_prefers_modern_mineru_when_legacy_cli_is_also_available(
    tmp_path: Path, monkeypatch,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.touch()
    modern = venv_bin / "mineru"
    legacy = venv_bin / "magic-pdf"
    modern.touch()
    legacy.touch()
    modern.chmod(0o755)
    legacy.chmod(0o755)
    monkeypatch.setattr(document_parser.sys, "executable", str(interpreter))
    monkeypatch.setattr(document_parser.shutil, "which", lambda _name: None)

    assert document_parser.find_mineru_cli() == str(modern)


def test_ignores_stale_non_executable_modern_cli_candidate(
    tmp_path: Path, monkeypatch,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.touch()
    modern = venv_bin / "mineru"
    legacy = venv_bin / "magic-pdf"
    modern.touch()
    legacy.touch()
    legacy.chmod(0o755)
    monkeypatch.setattr(document_parser.sys, "executable", str(interpreter))
    monkeypatch.setattr(document_parser.shutil, "which", lambda _name: None)

    assert document_parser.find_mineru_cli() == str(legacy)


def test_legacy_python_api_path_remains_supported(tmp_path: Path, monkeypatch) -> None:
    legacy_api = Mock()
    monkeypatch.setattr(document_parser, "MINERU_AVAILABLE", True)
    monkeypatch.setattr(document_parser, "MAGIC_PDF_CLI", None)
    monkeypatch.setattr(document_parser, "pdf_parse_main", legacy_api)
    parser = document_parser.DocumentParserService(
        output_dir=tmp_path / "parsed", force_backend="mineru"
    )

    parser._run_mineru_parsing("/tmp/input.pdf", "/tmp/output")

    legacy_api.assert_called_once_with(
        pdf_path="/tmp/input.pdf",
        parse_method=document_parser.MINERU_PARSE_METHOD,
        model_json_path=document_parser.MINERU_MODEL_JSON_PATH,
        is_json_md_dump=True,
        output_dir="/tmp/output",
    )


def test_legacy_magic_pdf_cli_path_remains_supported(tmp_path: Path, monkeypatch) -> None:
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="parsed", stderr="")
    run = Mock(return_value=result)
    monkeypatch.setattr(document_parser, "MINERU_AVAILABLE", True)
    monkeypatch.setattr(document_parser, "MAGIC_PDF_CLI", "/tmp/mineru-bin/magic-pdf")
    monkeypatch.setattr(document_parser, "pdf_parse_main", None)
    monkeypatch.setattr(document_parser.subprocess, "run", run)
    parser = document_parser.DocumentParserService(
        output_dir=tmp_path / "parsed", force_backend="mineru"
    )

    parser._run_mineru_parsing("/tmp/input.pdf", "/tmp/output")

    assert run.call_args.args[0] == [
        "/tmp/mineru-bin/magic-pdf",
        "-p", "/tmp/input.pdf",
        "-o", "/tmp/output",
        "-m", "auto",
    ]
