import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class ModernMineruCliTests(unittest.TestCase):
    def test_modern_mineru_cli_uses_pipeline_with_tables_and_no_formula(self):
        from source.services import document_parser

        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="parsed",
            stderr="",
        )
        legacy_api = Mock()
        with (
            patch.object(document_parser, "MINERU_AVAILABLE", True),
            patch.object(document_parser, "pdf_parse_main", legacy_api),
            patch.object(document_parser, "MAGIC_PDF_CLI", "/tmp/mineru-bin/mineru"),
            patch.object(document_parser.subprocess, "run", return_value=result) as run,
        ):
            parser = document_parser.DocumentParserService(
                output_dir=Path("/tmp/modern-mineru-output"),
                force_backend="mineru",
            )
            parser._run_mineru_parsing("/tmp/input.pdf", "/tmp/output")

        self.assertEqual(
            run.call_args.args[0],
            [
                "/tmp/mineru-bin/mineru",
                "-p", "/tmp/input.pdf",
                "-o", "/tmp/output",
                "-b", "pipeline",
                "-m", "auto",
                "-f", "false",
                "-t", "true",
            ],
        )
        legacy_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
