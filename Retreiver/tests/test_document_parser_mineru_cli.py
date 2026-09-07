import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class MineruCliDiscoveryTests(unittest.TestCase):
    def test_discovers_cli_next_to_unresolved_venv_interpreter(self):
        from source.services import document_parser

        with tempfile.TemporaryDirectory() as td:
            venv_bin = Path(td) / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").touch()
            cli = venv_bin / "magic-pdf"
            cli.touch()
            cli.chmod(0o755)
            with patch.object(document_parser.sys, "executable", str(venv_bin / "python")), patch.object(document_parser.shutil, "which", return_value=None):
                self.assertEqual(document_parser.find_mineru_cli(), str(cli))

    def test_prefers_modern_mineru_when_legacy_cli_is_also_available(self):
        from source.services import document_parser

        with tempfile.TemporaryDirectory() as td:
            venv_bin = Path(td) / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").touch()
            legacy = venv_bin / "magic-pdf"
            modern = venv_bin / "mineru"
            legacy.touch()
            modern.touch()
            legacy.chmod(0o755)
            modern.chmod(0o755)
            with patch.object(document_parser.sys, "executable", str(venv_bin / "python")), patch.object(document_parser.shutil, "which", return_value=None):
                self.assertEqual(document_parser.find_mineru_cli(), str(modern))

    def test_ignores_stale_non_executable_modern_cli_candidate(self):
        from source.services import document_parser

        with tempfile.TemporaryDirectory() as td:
            venv_bin = Path(td) / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").touch()
            stale_modern = venv_bin / "mineru"
            legacy = venv_bin / "magic-pdf"
            stale_modern.touch()
            legacy.touch()
            legacy.chmod(0o755)
            with patch.object(document_parser.sys, "executable", str(venv_bin / "python")), patch.object(document_parser.shutil, "which", return_value=None):
                self.assertEqual(document_parser.find_mineru_cli(), str(legacy))


if __name__ == "__main__":
    unittest.main()
