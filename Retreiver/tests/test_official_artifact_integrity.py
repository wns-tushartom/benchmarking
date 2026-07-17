from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

from scripts import serve_benchmark_dashboard as dashboard


SOURCE_STATE = Path(__file__).resolve().parents[1] / "web" / "source-state.js"


def _complete_metric_row() -> dict[str, object]:
    return {
        "status": "completed",
        "evaluated_queries": 10,
        "recall_at_1": 0.5,
        "recall_at_3": 0.5,
        "recall_at_5": 0.5,
        "recall_at_10": 0.5,
        "mrr": 0.5,
        "precision_at_5": 0.5,
        "ndcg_at_5": 0.5,
        "avg_first_relevant_rank": 1.0,
        "no_hit_queries": 0,
        "avg_latency_seconds": 0.01,
    }


def test_official_metric_validation_rejects_impossible_finite_values() -> None:
    row = _complete_metric_row()
    row.update(
        {
            "recall_at_5": 999,
            "no_hit_queries": 11,
            "avg_latency_seconds": -1,
        }
    )

    invalid = dashboard.benchmark_missing_metrics(row)

    assert "recall_at_5" in invalid
    assert "no_hit_queries" in invalid
    assert "avg_latency_seconds" in invalid


def test_frontend_official_metric_validation_rejects_impossible_finite_values() -> None:
    row = _complete_metric_row()
    row.update(
        {
            "recall_at_5": 999,
            "no_hit_queries": 11,
            "avg_latency_seconds": -1,
        }
    )
    script = f"""
const sourceState = require({json.dumps(str(SOURCE_STATE))});
console.log(JSON.stringify(sourceState.officialMetricCompleteness({json.dumps(row)})));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["complete"] is False
    assert "recall_at_5" in result["missing_metrics"]
    assert "no_hit_queries" in result["missing_metrics"]
    assert "avg_latency_seconds" in result["missing_metrics"]


def test_unmanifested_modular_details_never_enter_official_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    copied_run = tmp_path / "runs" / "archive" / "copied"
    copied_run.mkdir(parents=True)
    detail_path = copied_run / "modular_details.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "chunker",
                "embedding",
                "vector_store",
                "reranker",
                "query_id",
                "query",
                "category",
                "top_ids",
                "top_scores",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "chunker": "chunk",
                "embedding": "embed",
                "vector_store": "faiss",
                "reranker": "reranker",
                "query_id": "q1",
                "query": "question",
                "category": "demo",
                "top_ids": "1",
                "top_scores": "0.9",
            }
        )

    monkeypatch.setattr(dashboard, "MODULAR_DIR", tmp_path / "runs" / "latest")
    monkeypatch.setattr(
        dashboard,
        "official_matrix_keys",
        lambda: {("chunk", "embed", "faiss", "reranker")},
    )
    monkeypatch.setattr(
        dashboard,
        "chunk_lookup_for_sheet",
        lambda _sheet: {
            1: SimpleNamespace(id=1, pdf_name="demo.pdf", paragraph="trusted-looking text")
        },
    )
    monkeypatch.setattr(dashboard, "benchmark_detail_sources", lambda: [("modular_details:orphan", detail_path)])
    assert dashboard.benchmark_detail_evidence() == []


def test_official_manifest_binds_artifact_location_and_content(tmp_path: Path, monkeypatch) -> None:
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import config_hash, load_benchmark_config

    monkeypatch.setattr(dashboard, "MODULAR_DIR", tmp_path / "data" / "modular_runs" / "latest")
    run_dir = tmp_path / "data" / "modular_runs" / "archive" / "bound-run"
    run_dir.mkdir(parents=True)
    summary = run_dir / "modular_summary.csv"
    summary.write_text("status,evaluated_queries\ncompleted,5\n", encoding="utf-8")
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    cfg = load_benchmark_config(dashboard.CONFIG_PATH)
    manifest = {
        "status": "completed",
        "run_id": run_dir.name,
        "config_hash": config_hash(cfg),
        "dataset_id": dashboard.DEFAULT_DATASET_ID,
        "groundtruth_id": dashboard.OFFICIAL_GROUNDTRUTH_ID,
        "query_count": 5,
        "artifact_root": "archive/bound-run",
        "artifact_sha256": {"modular_summary.csv": digest},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    trusted, reason, _ = dashboard.official_artifact_provenance(summary)
    assert trusted is True
    assert reason == "trusted"

    summary.write_text("status,evaluated_queries\ncompleted,500\n", encoding="utf-8")
    trusted, reason, _ = dashboard.official_artifact_provenance(summary)
    assert trusted is False
    assert reason == "artifact_digest_mismatch"

    summary.write_text("status,evaluated_queries\ncompleted,5\n", encoding="utf-8")
    copied_parent = tmp_path / "data" / "modular_runs" / "copied" / "bound-run"
    shutil.copytree(run_dir, copied_parent)
    copied_summary = copied_parent / "modular_summary.csv"
    trusted, reason, _ = dashboard.official_artifact_provenance(copied_summary)
    assert trusted is False
    assert reason == "artifact_root_mismatch"

    moved_summary = tmp_path / "moved-summary.csv"
    shutil.move(summary, moved_summary)
    summary.symlink_to(moved_summary)
    trusted, reason, _ = dashboard.official_artifact_provenance(summary)
    assert trusted is False
    assert reason == "artifact_symlink"


def test_official_details_require_an_admitted_summary_from_same_run(
    tmp_path: Path, monkeypatch
) -> None:
    from benchmarking.core.config import config_hash, load_benchmark_config

    monkeypatch.setattr(dashboard, "MODULAR_DIR", tmp_path / "runs" / "latest")
    run_dir = tmp_path / "runs" / "archive" / "broken-summary"
    run_dir.mkdir(parents=True)
    key = ("chunk", "embed", "faiss", "reranker")
    summary = run_dir / "modular_summary.csv"
    summary.write_text(
        "chunker,embedding,vector_store,reranker,status,evaluated_queries\n"
        "chunk,embed,faiss,reranker,completed,5\n",
        encoding="utf-8",
    )
    details = run_dir / "modular_details.csv"
    details.write_text(
        "chunker,embedding,vector_store,reranker,query_id,query,top_ids,top_scores\n"
        "chunk,embed,faiss,reranker,q1,question,1,0.9\n",
        encoding="utf-8",
    )
    cfg = load_benchmark_config(dashboard.CONFIG_PATH)
    manifest = {
        "status": "completed",
        "run_id": run_dir.name,
        "config_hash": config_hash(cfg),
        "dataset_id": dashboard.DEFAULT_DATASET_ID,
        "groundtruth_id": dashboard.OFFICIAL_GROUNDTRUTH_ID,
        "query_count": 5,
        "artifact_root": "archive/broken-summary",
        "artifact_sha256": {
            "modular_summary.csv": "0" * 64,
            "modular_details.csv": hashlib.sha256(details.read_bytes()).hexdigest(),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {key})
    monkeypatch.setattr(
        dashboard,
        "chunk_lookup_for_sheet",
        lambda _sheet: {1: SimpleNamespace(id=1, pdf_name="demo.pdf", paragraph="text")},
    )
    monkeypatch.setattr(
        dashboard, "benchmark_detail_sources", lambda: [("modular_details:broken", details)]
    )

    assert dashboard.official_artifact_provenance(details)[:2] == (True, "trusted")
    assert dashboard.benchmark_detail_evidence() == []


def test_duplicate_official_run_ids_are_diagnostic_only(tmp_path: Path, monkeypatch) -> None:
    from benchmarking.core.config import config_hash, load_benchmark_config

    monkeypatch.setattr(dashboard, "MODULAR_DIR", tmp_path / "runs" / "latest")
    key = ("chunk", "embed", "faiss", "reranker")
    cfg = load_benchmark_config(dashboard.CONFIG_PATH)
    sources = []
    for parent_name, score in (("a", "0.2"), ("b", "0.9")):
        run_dir = tmp_path / "runs" / parent_name / "dupe"
        run_dir.mkdir(parents=True)
        summary = run_dir / "modular_summary.csv"
        summary.write_text(
            "chunker,embedding,vector_store,reranker,status,evaluated_queries,"
            "recall_at_1,recall_at_3,recall_at_5,recall_at_10,mrr,precision_at_5,"
            "ndcg_at_5,avg_first_relevant_rank,no_hit_queries,avg_latency_seconds\n"
            f"chunk,embed,faiss,reranker,completed,5,{score},{score},{score},{score},"
            f"{score},{score},{score},1,0,0.1\n",
            encoding="utf-8",
        )
        manifest = {
            "status": "completed",
            "run_id": "dupe",
            "config_hash": config_hash(cfg),
            "dataset_id": dashboard.DEFAULT_DATASET_ID,
            "groundtruth_id": dashboard.OFFICIAL_GROUNDTRUTH_ID,
            "query_count": 5,
            "artifact_root": summary.parent.resolve().as_posix(),
            "artifact_sha256": {
                "modular_summary.csv": hashlib.sha256(summary.read_bytes()).hexdigest()
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        sources.append((f"modular_run:{parent_name}/dupe", summary))
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {key})
    monkeypatch.setattr(dashboard, "benchmark_reference_sources", lambda: sources)

    result = dashboard.read_benchmark_reference()
    assert result["summary"] == []
    assert len(result["diagnostics"]) == 2
    assert {row["admission_reason"] for row in result["diagnostics"]} == {"duplicate_run_id"}
