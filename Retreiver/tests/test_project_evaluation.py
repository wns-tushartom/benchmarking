import csv
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_retrieval_groundtruth as evaluator
from scripts import run_project_evaluation as project_runner
from scripts import run_retrieval_smoke_from_vm_dbs as retrieval


class ProjectRetrievalTests(unittest.TestCase):
    def test_retrieval_uses_explicit_project_receipt_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            receipt = project_root / "runs" / "setup" / "ingestion" / "summary.csv"
            out_dir = project_root / "runs" / "evaluation" / "retrieval"
            receipt.parent.mkdir(parents=True)
            with receipt.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["status", "sheet", "embedding", "store", "collection_or_table"],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "status": "ok",
                        "sheet": "Heading_sections_l2",
                        "embedding": "gte_multilingual_base",
                        "store": "FAISS",
                        "collection_or_table": str(project_root / "faiss" / "selected"),
                    },
                    {
                        "status": "failed",
                        "sheet": "Heading_sections_l2",
                        "embedding": "gte_multilingual_base",
                        "store": "FAISS",
                        "collection_or_table": str(project_root / "faiss" / "failed"),
                    },
                    {
                        "status": "ok",
                        "sheet": "other_sheet",
                        "embedding": "gte_multilingual_base",
                        "store": "FAISS",
                        "collection_or_table": str(project_root / "faiss" / "other"),
                    },
                ])

            result = retrieval.rows_from_receipt(
                receipt,
                ["Heading_sections_l2"],
                ["gte_multilingual_base"],
                ["FAISS"],
            )

            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["collection_or_table"].startswith(str(project_root)))
            self.assertEqual(retrieval.output_dir_for(out_dir), out_dir)


class ProjectEvaluationTests(unittest.TestCase):
    @staticmethod
    def evaluation_info(project_root: Path, *, expected_queries: int = 1) -> dict[str, object]:
        return {
            "components": {
                "sheet": "Heading_sections_l2",
                "embedding": "gte_multilingual_base",
                "store": "FAISS",
                "reranker": "none",
            },
            "groundtruth": str(project_root / "questions" / "groundtruth.csv"),
            "groundtruth_sha256": "fixture",
            "ingestion_summary": str(project_root / "runs" / "setup" / "ingestion" / "summary.csv"),
            "index_dir": str(project_root / "vector_indexes" / "faiss" / "selected"),
            "chunks_dir": str(project_root / "chunks"),
            "expected_queries": expected_queries,
        }

    @staticmethod
    def command_output_dir(command: list[str]) -> Path:
        return Path(command[command.index("--out-dir") + 1])

    @staticmethod
    def swap_runs_parent(project_root: Path, outside: Path) -> Path:
        runs_dir = project_root / "runs"
        original_runs_dir = project_root / "runs_original"
        runs_dir.rename(original_runs_dir)
        runs_dir.symlink_to(outside, target_is_directory=True)
        return original_runs_dir

    def run_evaluation_fixture(
        self,
        *,
        project_root: Path,
        run_id: str = "eval-1",
    ) -> tuple[int, dict[str, object]]:
        return project_runner.run_evaluation(
            info=self.evaluation_info(project_root),
            project_root=project_root,
            run_id=run_id,
            groundtruth=project_root / "questions" / "groundtruth.csv",
            ingestion_summary=project_root / "runs" / "setup" / "ingestion" / "summary.csv",
            sheet="Heading_sections_l2",
            embedding="gte_multilingual_base",
            store="FAISS",
            reranker="none",
            top_k=10,
        )

    def test_source_document_is_used_as_expected_pdf(self):
        row = {"question": "Which paper?", "source_document": "paper.pdf"}

        self.assertEqual(evaluator.load_groundtruth_rows([row])[0]["expected_pdf"], "paper.pdf")

    def test_evaluator_loads_smokes_through_pinned_run_directory_fd(self):
        with tempfile.TemporaryDirectory(dir=evaluator.ROOT) as tmpdir:
            retrieval_dir = Path(tmpdir) / "retrieval"
            retrieval_dir.mkdir()
            artifact = retrieval_dir / "result.json"
            artifact.write_text(json.dumps({"query": "Which paper?", "hits": []}), encoding="utf-8")
            run_fd = os.open(Path(tmpdir), os.O_RDONLY | os.O_DIRECTORY)
            try:
                rows = evaluator.load_smokes(Path(f"/proc/self/fd/{run_fd}/retrieval"))
            finally:
                os.close(run_fd)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "Which paper?")
        self.assertFalse(rows[0]["artifact"].startswith("/proc/self/fd/"))

    def test_project_evaluation_command_never_uses_official_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            cmd = project_runner.build_commands(
                project_root=project_root,
                run_id="eval-1",
                run_dir_fd=73,
                groundtruth=project_root / "questions" / "groundtruth.csv",
                ingestion_summary=project_root / "runs" / "setup" / "ingestion" / "summary.csv",
                sheet="Heading_sections_l2",
                embedding="gte_multilingual_base",
                store="FAISS",
                reranker="none",
                top_k=10,
            )

            rendered = " ".join(" ".join(part) for part in cmd)
            self.assertNotIn("data/db_ingestion_runs", rendered)
            self.assertNotIn("data/retrieval_smoke", rendered)
            self.assertNotIn("data/evaluation ", rendered)
            self.assertIn("/proc/self/fd/73/retrieval", rendered)
            self.assertIn("/proc/self/fd/73/evaluation", rendered)

    def test_project_run_id_rejects_dot_segments_before_creating_runs_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()

            for run_id in (".", ".."):
                with self.subTest(run_id=run_id):
                    with self.assertRaisesRegex(ValueError, "direct child"):
                        project_runner.reserve_project_run_dir(project_root, run_id)
                    self.assertFalse((project_root / "runs").exists())

            self.assertEqual(
                project_runner.project_run_dir(project_root, "eval.1_test-2"),
                project_root.resolve() / "runs" / "eval.1_test-2",
            )

    def test_existing_run_directory_is_rejected_without_overwriting_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            run_dir = project_root / "runs" / "eval-1"
            manifest_path = run_dir / "evaluation_manifest.json"
            run_dir.mkdir(parents=True)
            original_manifest = '{"state": "complete", "immutable": true}\n'
            manifest_path.write_text(original_manifest, encoding="utf-8")
            info = {
                "components": {
                    "sheet": "Heading_sections_l2",
                    "embedding": "gte_multilingual_base",
                    "store": "FAISS",
                    "reranker": "none",
                },
                "groundtruth": str(project_root / "questions" / "groundtruth.csv"),
                "groundtruth_sha256": "fixture",
                "ingestion_summary": str(project_root / "runs" / "setup" / "ingestion" / "summary.csv"),
                "index_dir": str(project_root / "vector_indexes" / "faiss" / "selected"),
                "chunks_dir": str(project_root / "chunks"),
                "expected_queries": 1,
            }

            with patch.object(project_runner, "run_command", return_value=(1, "simulated retrieval failure")) as run_command:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    project_runner.run_evaluation(
                        info=info,
                        project_root=project_root,
                        run_id="eval-1",
                        groundtruth=project_root / "questions" / "groundtruth.csv",
                        ingestion_summary=project_root / "runs" / "setup" / "ingestion" / "summary.csv",
                        sheet="Heading_sections_l2",
                        embedding="gte_multilingual_base",
                        store="FAISS",
                        reranker="none",
                        top_k=10,
                    )

            run_command.assert_not_called()
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)

    def test_concurrent_same_run_id_reserves_only_one_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            info = {
                "components": {
                    "sheet": "Heading_sections_l2",
                    "embedding": "gte_multilingual_base",
                    "store": "FAISS",
                    "reranker": "none",
                },
                "groundtruth": str(project_root / "questions" / "groundtruth.csv"),
                "groundtruth_sha256": "fixture",
                "ingestion_summary": str(project_root / "runs" / "setup" / "ingestion" / "summary.csv"),
                "index_dir": str(project_root / "vector_indexes" / "faiss" / "selected"),
                "chunks_dir": str(project_root / "chunks"),
                "expected_queries": 1,
            }
            first_command_started = threading.Event()
            release_command = threading.Event()
            command_calls: list[list[str]] = []
            command_lock = threading.Lock()
            results: list[tuple[int, dict[str, object]]] = []
            errors: list[Exception] = []

            def blocked_run_command(command: list[str], **_kwargs: object) -> tuple[int, str]:
                with command_lock:
                    command_calls.append(command)
                    first_command_started.set()
                release_command.wait(timeout=2)
                return 1, "simulated retrieval failure"

            def evaluate() -> None:
                try:
                    results.append(project_runner.run_evaluation(
                        info=info,
                        project_root=project_root,
                        run_id="eval-1",
                        groundtruth=project_root / "questions" / "groundtruth.csv",
                        ingestion_summary=project_root / "runs" / "setup" / "ingestion" / "summary.csv",
                        sheet="Heading_sections_l2",
                        embedding="gte_multilingual_base",
                        store="FAISS",
                        reranker="none",
                        top_k=10,
                    ))
                except Exception as exc:  # collected from worker threads for assertions below
                    errors.append(exc)

            with patch.object(project_runner, "run_command", side_effect=blocked_run_command):
                first = threading.Thread(target=evaluate)
                second = threading.Thread(target=evaluate)
                first.start()
                self.assertTrue(first_command_started.wait(timeout=1))
                second.start()
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    with command_lock:
                        if len(command_calls) > 1:
                            break
                    time.sleep(0.01)
                release_command.set()
                first.join(timeout=2)
                second.join(timeout=2)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(len(command_calls), 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertIn("already exists", str(errors[0]))

    def test_symlinked_runs_parent_is_rejected_before_writing_outside_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            outside = tmp_path / "outside"
            runs_dir = project_root / "runs"
            runs_dir.mkdir(parents=True)
            outside.mkdir()
            info = {
                "components": {
                    "sheet": "Heading_sections_l2",
                    "embedding": "gte_multilingual_base",
                    "store": "FAISS",
                    "reranker": "none",
                },
                "groundtruth": str(project_root / "questions" / "groundtruth.csv"),
                "groundtruth_sha256": "fixture",
                "ingestion_summary": str(project_root / "runs" / "setup" / "ingestion" / "summary.csv"),
                "index_dir": str(project_root / "vector_indexes" / "faiss" / "selected"),
                "chunks_dir": str(project_root / "chunks"),
                "expected_queries": 1,
            }

            def swap_runs_parent(_command: list[str], **_kwargs: object) -> tuple[int, str]:
                runs_dir.rename(project_root / "runs_original")
                runs_dir.symlink_to(outside, target_is_directory=True)
                return 1, "simulated retrieval failure"

            with patch.object(project_runner, "run_command", side_effect=swap_runs_parent) as run_command:
                code, manifest = project_runner.run_evaluation(
                    info=info,
                    project_root=project_root,
                    run_id="eval-1",
                    groundtruth=project_root / "questions" / "groundtruth.csv",
                    ingestion_summary=project_root / "runs" / "setup" / "ingestion" / "summary.csv",
                    sheet="Heading_sections_l2",
                    embedding="gte_multilingual_base",
                    store="FAISS",
                    reranker="none",
                    top_k=10,
                )

            run_command.assert_called_once()
            self.assertEqual(code, 1)
            self.assertEqual(manifest["state"], "failed")
            self.assertFalse((outside / "eval-1").exists())

    def test_runs_parent_swap_immediately_before_retrieval_keeps_artifacts_off_external_and_global_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            outside = tmp_path / "outside"
            outside.mkdir()
            (project_root / "runs").mkdir(parents=True)
            global_retrieval = tmp_path / "data" / "retrieval_smoke"

            def run_retrieval_after_swap(command: list[str], **_kwargs: object) -> tuple[int, str]:
                self.swap_runs_parent(project_root, outside)
                out_dir = self.command_output_dir(command)
                self.assertTrue(str(out_dir).startswith("/proc/self/fd/"))
                passed_fds = _kwargs.get("pass_fds", ())
                if not isinstance(passed_fds, tuple):
                    self.fail("run command did not receive a tuple of inherited directory fds")
                self.assertEqual(len(passed_fds), 1)
                out_dir.mkdir(parents=True)
                (out_dir / "retrieval-marker.txt").write_text("retrieval", encoding="utf-8")
                return 1, "simulated retrieval failure"

            with patch.object(project_runner, "run_command", side_effect=run_retrieval_after_swap):
                try:
                    code, manifest = self.run_evaluation_fixture(project_root=project_root)
                except ValueError:
                    # The vulnerable implementation detects the swap only after its lexical child path escaped.
                    code, manifest = 1, {}

            # This assertion is RED before the descriptor-pinned implementation: the child writes
            # to outside/eval-1/retrieval after the parent swap.
            self.assertEqual(code, 1)
            self.assertEqual(manifest.get("state", "failed"), "failed")
            self.assertFalse((outside / "eval-1").exists())
            self.assertFalse(global_retrieval.exists())
            self.assertEqual(
                (project_root / "runs_original" / "eval-1" / "retrieval" / "retrieval-marker.txt").read_text(
                    encoding="utf-8",
                ),
                "retrieval",
            )

    def test_runs_parent_swap_immediately_before_evaluation_keeps_artifacts_off_external_and_global_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            outside = tmp_path / "outside"
            outside.mkdir()
            (project_root / "runs").mkdir(parents=True)
            global_evaluation = tmp_path / "data" / "evaluation"
            calls = 0

            def run_commands_with_evaluation_swap(command: list[str], **_kwargs: object) -> tuple[int, str]:
                nonlocal calls
                calls += 1
                out_dir = self.command_output_dir(command)
                if calls == 2:
                    self.swap_runs_parent(project_root, outside)
                out_dir.mkdir(parents=True)
                if calls == 2:
                    (out_dir / "groundtruth_eval_report.json").write_text(
                        json.dumps({"summary_rows": 1, "evaluated_rows": 1, "missing_query_count": 0}),
                        encoding="utf-8",
                    )
                    (out_dir / "groundtruth_eval_summary.csv").write_text(
                        "evaluated_queries\n1\n", encoding="utf-8",
                    )
                return 0, "ok"

            with patch.object(project_runner, "run_command", side_effect=run_commands_with_evaluation_swap):
                try:
                    code, manifest = self.run_evaluation_fixture(project_root=project_root)
                except ValueError:
                    code, manifest = 1, {}

            # This assertion is RED before the descriptor-pinned implementation: evaluation
            # output is emitted through the swapped lexical runs parent.
            self.assertEqual(code, 1)
            self.assertEqual(manifest.get("state", "failed"), "failed")
            self.assertEqual(manifest.get("error"), "project evaluation run path was displaced before completion")
            self.assertFalse((outside / "eval-1").exists())
            self.assertFalse(global_evaluation.exists())
            self.assertNotIn("/proc/self/fd", json.dumps(manifest))
            self.assertTrue(
                (project_root / "runs_original" / "eval-1" / "evaluation" / "groundtruth_eval_report.json").is_file(),
            )

    def test_runs_parent_swap_immediately_before_manifest_write_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            outside = tmp_path / "outside"
            run_dir = project_root / "runs" / "eval-1"
            run_dir.mkdir(parents=True)
            (outside / "eval-1").mkdir(parents=True)
            original_dumps = json.dumps

            def dump_then_swap(*args: object, **kwargs: object) -> str:
                self.swap_runs_parent(project_root, outside)
                return original_dumps(*args, **kwargs)

            with patch.object(project_runner.json, "dumps", side_effect=dump_then_swap):
                project_runner.write_manifest(
                    project_root=project_root,
                    run_dir=run_dir,
                    payload={"state": "failed", "run_id": "eval-1"},
                )

            # This assertion is RED before the descriptor-pinned implementation: validation
            # succeeds, json encoding yields to the swap, and Path.write_text follows the symlink.
            self.assertFalse((outside / "eval-1" / "evaluation_manifest.json").exists())
            self.assertTrue((project_root / "runs_original" / "eval-1" / "evaluation_manifest.json").is_file())

    def test_project_preflight_rejects_index_outside_project_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = tmp_path / "project"
            groundtruth = project_root / "questions" / "groundtruth.csv"
            receipt = project_root / "runs" / "setup" / "ingestion" / "summary.csv"
            (project_root / "chunks").mkdir(parents=True)
            groundtruth.parent.mkdir(parents=True)
            receipt.parent.mkdir(parents=True)
            groundtruth.write_text("question\nWhich paper?\n", encoding="utf-8")
            with receipt.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["status", "sheet", "embedding", "store", "collection_or_table"],
                )
                writer.writeheader()
                writer.writerow({
                    "status": "ok",
                    "sheet": "Heading_sections_l2",
                    "embedding": "gte_multilingual_base",
                    "store": "FAISS",
                    "collection_or_table": str(tmp_path / "outside-index"),
                })

            preflight = project_runner.preflight(
                project_root=project_root,
                groundtruth=groundtruth,
                ingestion_summary=receipt,
                sheet="Heading_sections_l2",
                embedding="gte_multilingual_base",
                store="FAISS",
                reranker="none",
            )

            self.assertFalse(preflight["ok"])
            self.assertIn("outside project root", " ".join(preflight["errors"]))

    def test_project_preflight_accepts_root_contained_faiss_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            groundtruth = project_root / "questions" / "groundtruth.csv"
            receipt = project_root / "runs" / "setup" / "ingestion" / "summary.csv"
            index_dir = project_root / "vector_indexes" / "faiss" / "selected"
            (project_root / "chunks").mkdir(parents=True)
            groundtruth.parent.mkdir(parents=True)
            receipt.parent.mkdir(parents=True)
            index_dir.mkdir(parents=True)
            groundtruth.write_text("question,source_document\nWhich paper?,paper.pdf\n", encoding="utf-8")
            (index_dir / "index.faiss").write_bytes(b"fixture")
            (index_dir / "chunks.json").write_text("[]", encoding="utf-8")
            with receipt.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["status", "sheet", "embedding", "store", "collection_or_table"],
                )
                writer.writeheader()
                writer.writerow({
                    "status": "ok",
                    "sheet": "Heading_sections_l2",
                    "embedding": "gte_multilingual_base",
                    "store": "FAISS",
                    "collection_or_table": str(index_dir),
                })

            preflight = project_runner.preflight(
                project_root=project_root,
                groundtruth=groundtruth,
                ingestion_summary=receipt,
                sheet="Heading_sections_l2",
                embedding="gte_multilingual_base",
                store="FAISS",
                reranker="none",
            )

            self.assertTrue(preflight["ok"])
            self.assertEqual(preflight["expected_queries"], 1)
            self.assertEqual(preflight["index_dir"], str(index_dir))

    def test_completion_state_requires_every_groundtruth_query(self):
        state, evaluated, missing = project_runner.completion_state(
            expected_queries=2,
            report={"summary_rows": 1, "evaluated_rows": 1, "missing_query_count": 0},
            summary_rows=[{"evaluated_queries": "1"}],
        )

        self.assertEqual(state, "incomplete")
        self.assertEqual(evaluated, 1)
        self.assertEqual(missing, 0)


if __name__ == "__main__":
    unittest.main()
