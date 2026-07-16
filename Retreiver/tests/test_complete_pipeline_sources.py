from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import scripts.run_complete_pipeline as pipeline


def _ready_info(*, groundtruth: str = "") -> dict:
    return {
        "ok": True,
        "missing": [],
        "warnings": [],
        "service_checks": [],
        "groundtruth": groundtruth,
        "sheets": ["uploaded_chunks"],
        "embeddings": ["gte_multilingual_base"],
        "stores": ["FAISS"],
        "rerankers": ["bge-reranker-base"],
        "combo_count": 1,
        "query_limit": 0,
        "chunk_limit": 0,
        "top_k": 5,
        "fresh_run": False,
        "allow_partial_extraction": False,
    }


def _run_main_with_captured_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    info: dict,
    archive_calls: list[str] | None = None,
) -> list[tuple[list[str], str]]:
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda _args: info)
    monkeypatch.setattr(
        pipeline,
        "archive_fresh_dirs",
        lambda run_id: archive_calls.append(run_id) if archive_calls is not None else None,
    )
    monkeypatch.setattr(pipeline, "ensure_chunking_workbook", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_cmd",
        lambda cmd, stage: calls.append(([str(part) for part in cmd], stage)),
    )
    monkeypatch.setattr(sys, "argv", ["run_complete_pipeline.py", *argv])

    previous_cwd = Path.cwd()
    try:
        assert pipeline.main() == 0
    finally:
        os.chdir(previous_cwd)
    return calls


def test_evidence_only_uses_selected_workbook_and_never_runs_scored_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "project" / "chunks" / "chunking_methods_output_v2.xlsx"
    workbook.parent.mkdir(parents=True)
    workbook.touch()

    calls = _run_main_with_captured_stages(
        monkeypatch,
        tmp_path,
        [
            "--workbook",
            str(workbook),
            "--evidence-only",
            "--query",
            "How are involuntary changes handled?",
            "--query",
            "Which document contains the refund rule?",
            "--sheets",
            "uploaded_chunks",
            "--embeddings",
            "gte_multilingual_base",
            "--stores",
            "FAISS",
            "--top-k",
            "5",
        ],
        _ready_info(),
    )

    stages = [stage for _cmd, stage in calls]
    ingestion_cmd = next(cmd for cmd, stage in calls if stage == "ingestion")
    retrieval_cmd = next(cmd for cmd, stage in calls if stage == "evidence-only retrieval")

    assert ingestion_cmd[ingestion_cmd.index("--workbook") + 1] == str(workbook)
    assert "--queries" in retrieval_cmd
    assert "How are involuntary changes handled?" in retrieval_cmd
    assert "Which document contains the refund rule?" in retrieval_cmd
    assert "--queries-file" not in retrieval_cmd
    assert not any("evaluation" in stage for stage in stages)
    assert not any("reranker" in stage for stage in stages)
    assert not any("lift" in stage.lower() or "analysis" in stage.lower() for stage in stages)


def test_evaluated_mode_uses_selected_workbook_and_groundtruth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "project" / "chunks" / "chunking_methods_output_v2.xlsx"
    groundtruth = tmp_path / "groundtruth" / "questions.csv"
    workbook.parent.mkdir(parents=True)
    groundtruth.parent.mkdir(parents=True)
    workbook.touch()
    groundtruth.write_text("query,answer\nq,a\n", encoding="utf-8")

    calls = _run_main_with_captured_stages(
        monkeypatch,
        tmp_path,
        [
            "--workbook",
            str(workbook),
            "--groundtruth",
            str(groundtruth),
            "--sheets",
            "uploaded_chunks",
            "--embeddings",
            "gte_multilingual_base",
            "--stores",
            "FAISS",
            "--rerankers",
            "bge-reranker-base",
        ],
        _ready_info(groundtruth=str(groundtruth)),
    )

    stages = [stage for _cmd, stage in calls]
    ingestion_cmd = next(cmd for cmd, stage in calls if stage == "ingestion")
    retrieval_cmd = next(cmd for cmd, stage in calls if stage == "ground-truth retrieval")

    assert ingestion_cmd[ingestion_cmd.index("--workbook") + 1] == str(workbook)
    assert retrieval_cmd[retrieval_cmd.index("--queries-file") + 1] == str(groundtruth)
    assert "no-reranker evaluation" in stages
    assert "reranker pass" in stages
    assert "reranked evaluation" in stages


def test_evaluated_mode_separates_retrieval_and_reranked_output_depth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "project.xlsx"
    groundtruth = tmp_path / "questions.csv"
    workbook.touch()
    groundtruth.write_text("query,answer\nq,a\n", encoding="utf-8")

    calls = _run_main_with_captured_stages(
        monkeypatch,
        tmp_path,
        [
            "--workbook",
            str(workbook),
            "--groundtruth",
            str(groundtruth),
            "--top-k",
            "20",
            "--reranked-output-k",
            "5",
            "--rerankers",
            "bge-reranker-base",
        ],
        _ready_info(groundtruth=str(groundtruth)),
    )

    retrieval_cmd = next(cmd for cmd, stage in calls if stage == "ground-truth retrieval")
    reranker_cmd = next(cmd for cmd, stage in calls if stage == "reranker pass")
    assert retrieval_cmd[retrieval_cmd.index("--top-k") + 1] == "20"
    assert reranker_cmd[reranker_cmd.index("--candidate-k") + 1] == "20"
    assert reranker_cmd[reranker_cmd.index("--top-k") + 1] == "5"
    assert reranker_cmd[reranker_cmd.index("--limit-artifacts") + 1] == "0"


def test_run_manifest_persists_depth_contract_and_unlimited_chunks() -> None:
    payload = pipeline.run_manifest_payload(
        "20260716_120000",
        {
            **_ready_info(),
            "retrieval_top_k": 20,
            "reranked_output_k": 5,
        },
        status="running",
    )

    assert payload["run_id"] == "20260716_120000"
    assert payload["status"] == "running"
    assert payload["retrieval_top_k"] == 20
    assert payload["reranked_output_k"] == 5
    assert payload["chunk_limit"] == 0


def test_normalized_queries_rejects_blank_values() -> None:
    assert pipeline.normalized_queries(["  ", "real question", "\n"]) == ["real question"]


def test_alternate_workbook_uses_its_own_runnable_sheets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.xlsx"
    alternate = tmp_path / "project.xlsx"
    monkeypatch.setattr(pipeline, "WORKBOOK_PATH", canonical)
    monkeypatch.setattr(pipeline, "workbook_sheet_names", lambda path: ["uploaded_chunks"] if path == alternate else ["official"])

    assert pipeline.dataset_sheet_options(canonical, ["official"]) == ["official"]
    assert pipeline.dataset_sheet_options(alternate, ["official"]) == ["uploaded_chunks"]


def test_alternate_benchmark_input_is_not_claimed_as_rebuildable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "canonical.xlsx"
    alternate_input = tmp_path / "alternate.csv"
    alternate_input.write_text("query\ncustom\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "WORKBOOK_PATH", workbook)
    monkeypatch.setattr(pipeline, "BENCHMARK_INPUT_PATH", tmp_path / "default.csv")

    available, missing, can_rebuild = pipeline.chunking_gap(
        ["uploaded_chunks"],
        workbook_path=workbook,
        benchmark_input_path=alternate_input,
    )

    assert available == []
    assert missing == ["uploaded_chunks"]
    assert can_rebuild is False


def test_evidence_only_never_archives_scored_artifacts_even_when_fresh_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "project.xlsx"
    workbook.touch()
    archives: list[str] = []
    _run_main_with_captured_stages(
        monkeypatch,
        tmp_path,
        ["--workbook", str(workbook), "--evidence-only", "--query", "q", "--fresh-run"],
        {**_ready_info(), "mode": "evidence_only", "queries": ["q"]},
        archive_calls=archives,
    )
    assert archives == []
