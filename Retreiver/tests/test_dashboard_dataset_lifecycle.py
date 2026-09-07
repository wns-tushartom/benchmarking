import json
import tempfile
from pathlib import Path

import pytest


def test_groundtruth_validation_requires_question_and_reference_column():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        valid = Path(td) / "valid.csv"
        valid.write_text("question,ground truth,context\nHow do I rebook?,Rebook from Manage My Booking,policy text\n", encoding="utf-8")
        invalid = Path(td) / "invalid.csv"
        invalid.write_text("question,category\nHow do I rebook?,travel\n", encoding="utf-8")
        assert dashboard.validate_groundtruth_file(valid)["valid"] is True
        result = dashboard.validate_groundtruth_file(invalid)

    assert result["valid"] is False
    assert "reference" in result["reason"].lower()




def test_project_pdf_rebuild_requires_layout_aware_extraction(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    class Page:
        page_number = 0
        content = "Layout-aware paragraph."
        images = ["images/figure-1.jpg"]
        tables = [{"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}]

    class Parsed:
        content = [Page()]
        metadata = {"parsing_method": "MinerU"}

    class Parser:
        def __init__(self, **kwargs):
            assert kwargs["force_backend"] == "mineru"

        async def parse_pdf(self, path, filename):
            return Parsed()

    monkeypatch.setattr(dashboard, "DocumentParserService", Parser)
    with tempfile.TemporaryDirectory() as td:
        project_dir = Path(td)
        raw = project_dir / "raw_uploads"
        raw.mkdir()
        (raw / "visual.pdf").write_bytes(b"%PDF-test")
        index = dashboard.rebuild_project_search_index(project_dir)

    assert index["chunk_count"] >= 1
    assert {chunk["parser_method"] for chunk in index["chunks"]} == {"MinerU"}
    assert {chunk["page_number"] for chunk in index["chunks"]} == {"1"}
    assert "| A | B |" in "\n".join(chunk["paragraph"] for chunk in index["chunks"])


def test_project_manifest_binds_its_uploaded_groundtruth_and_never_accepts_other_project_path():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            bound = dashboard.attach_project_groundtruth(project["project_id"], "gt.csv", b"question,ground truth,context\nrefund?,refund,policy\n")
            manifest = json.loads((dashboard.project_root(project["project_id"]) / "manifest.json").read_text(encoding="utf-8"))
        finally:
            dashboard.USER_PROJECTS_DIR = old

    assert bound["mode"] == "evaluated"
    assert manifest["groundtruth"]["valid"] is True
    assert manifest["groundtruth"]["path"].endswith("/questions/gt.csv")


def test_project_query_stays_evidence_only_without_valid_bound_groundtruth():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Customers can request a refund.", "Refund")
            payload = dashboard.query_user_project(project["project_id"], "refund")
        finally:
            dashboard.USER_PROJECTS_DIR = old

    assert payload["mode"] == "evidence_only"
    assert payload["recommendation_available"] is False


def test_setup_job_is_persisted_with_project_matrix_and_stage_provenance():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old_projects, old_jobs = dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR
        dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = Path(td) / "projects", Path(td) / "jobs"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            job = dashboard.create_project_setup_job(
                project["project_id"],
                {"chunkers": ["fixed_tok1200_ov150"], "embeddings": ["gte_multilingual_base"], "vector_stores": ["FAISS"]},
            )
            stored = json.loads((dashboard.JOB_DIR / f"{job['job_id']}.json").read_text(encoding="utf-8"))
            manifest = json.loads((dashboard.project_root(project["project_id"]) / "manifest.json").read_text(encoding="utf-8"))
        finally:
            dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = old_projects, old_jobs

    assert stored["kind"] == "dataset_setup"
    assert stored["project_id"] == project["project_id"]
    assert stored["matrix"]["vector_stores"] == ["FAISS"]
    assert [stage["name"] for stage in stored["stages"]] == ["extract", "chunk", "embed", "index"]
    assert manifest["setup"]["job_id"] == job["job_id"]


def test_project_setup_command_uses_only_project_paths_and_collection_prefix():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory(dir=dashboard.ROOT) as td:
        old_projects, old_jobs = dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR
        dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = Path(td) / "projects", Path(td) / "jobs"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            record = dashboard.create_project_setup_job(project["project_id"], {
                "chunkers": ["fixed_tok1200_ov150"],
                "embeddings": ["gte_multilingual_base"],
                "vector_stores": ["FAISS"],
            })
            cmd = dashboard.project_setup_cmd(record)
        finally:
            dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = old_projects, old_jobs

    project_rel = f"{Path(td).name}/projects/{project['project_id']}"
    assert cmd[0] == dashboard.sys.executable
    assert cmd[1] == "scripts/run_long_db_ingestion.py"
    assert cmd[cmd.index("--workbook") + 1].startswith(f"{project_rel}/chunks/")
    assert cmd[cmd.index("--output-dir") + 1].startswith(f"{project_rel}/runs/{record['job_id']}/")
    assert cmd[cmd.index("--cache-dir") + 1].startswith(f"{project_rel}/vector_indexes/")
    assert cmd[cmd.index("--faiss-index-root") + 1].startswith(f"{project_rel}/vector_indexes/")
    assert cmd[cmd.index("--collection-prefix") + 1] == f"project_{project['project_id']}"
    assert "data/db_ingestion_runs" not in cmd
    assert "data/embedding_cache" not in cmd
    assert "data/faiss_indexes" not in cmd


def test_ingestion_runner_resolves_project_paths_and_namespaces_stores():
    import scripts.run_long_db_ingestion as ingestion

    paths = ingestion.resolve_runtime_paths(
        "data/user_projects/demo/runs/job/ingestion",
        "data/user_projects/demo/vector_indexes/embedding_cache",
        "data/user_projects/demo/vector_indexes/faiss",
    )
    store = ingestion.make_store(
        "FAISS", "fixed_tok1200_ov150", "gte_multilingual_base", "project_demo", paths["faiss_index_root"],
    )

    assert paths["output_dir"] == ingestion.ROOT / "data/user_projects/demo/runs/job/ingestion"
    assert paths["cache_dir"] == ingestion.ROOT / "data/user_projects/demo/vector_indexes/embedding_cache"
    assert paths["faiss_index_root"] == ingestion.ROOT / "data/user_projects/demo/vector_indexes/faiss"
    assert "project_demo_fixed_tok1200_ov150_gte_multilingual" == ingestion.store_prefix("project_demo", "fixed_tok1200_ov150", "gte_multilingual_base")
    assert str(paths["faiss_index_root"]) in str(store.index_dir)
    assert "data/faiss_indexes" not in str(store.index_dir)


def test_ingestion_runner_override_paths_are_contained_under_repository_root():
    import scripts.run_long_db_ingestion as ingestion

    paths = ingestion.resolve_runtime_paths(
        "data/user_projects/demo/runs/job/ingestion",
        "data/user_projects/demo/vector_indexes/embedding_cache",
        "data/user_projects/demo/vector_indexes/faiss",
    )

    assert paths["output_dir"] == ingestion.ROOT / "data/user_projects/demo/runs/job/ingestion"
    with pytest.raises(ValueError, match="rooted under repository"):
        ingestion.resolve_runtime_paths("../../outside", "", "")
    with pytest.raises(ValueError, match="rooted under repository"):
        ingestion.resolve_runtime_paths("", "/tmp/outside", "")


@pytest.mark.parametrize(
    ("current_stage", "runner_line"),
    [
        ("embed", "ERROR embedding sheet=fixed_tok1200_ov150 embedding=gte_multilingual_base: RuntimeError('offline')"),
        ("index", "ERROR store sheet=fixed_tok1200_ov150 embedding=gte_multilingual_base store=FAISS: RuntimeError('offline')"),
    ],
)
def test_running_setup_job_persists_runner_error_prefix_as_immediate_failure(current_stage, runner_line):
    import scripts.serve_benchmark_dashboard as dashboard

    class RunningProcess:
        def poll(self):
            return None

    with tempfile.TemporaryDirectory() as td:
        old_job_dir, old_jobs = dashboard.JOB_DIR, dict(dashboard.JOBS)
        dashboard.JOB_DIR = Path(td) / "jobs"
        dashboard.JOBS.clear()
        try:
            states = {"extract": "complete", "chunk": "complete", "embed": "complete", "index": "queued"}
            states[current_stage] = "running"
            if current_stage == "embed":
                states["index"] = "queued"
            record = {
                "job_id": f"runner-error-{current_stage}",
                "kind": "dataset_setup",
                "state": "running",
                "created_at": "2026-07-21T10:00:00",
                "started_at": "2026-07-21T10:00:00",
                "finished_at": "",
                "project_id": "",
                "stages": [
                    {"name": name, "state": states[name], "started_at": "", "finished_at": "", "output": "", "error": ""}
                    for name in ("extract", "chunk", "embed", "index")
                ],
            }
            log_path = dashboard.JOB_DIR / f"{record['job_id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(runner_line + "\n", encoding="utf-8")
            record["log_path"] = str(log_path)
            dashboard.persist_job(record)
            dashboard.JOBS[record["job_id"]] = {"process": RunningProcess(), "log_path": str(log_path)}

            status = dashboard.job_status(record["job_id"])
            persisted = dashboard.read_dashboard_job(record["job_id"])
        finally:
            dashboard.JOBS.clear()
            dashboard.JOBS.update(old_jobs)
            dashboard.JOB_DIR = old_job_dir

    assert status["running"] is True
    assert status["exit_code"] is None
    assert status["state"] == "failed"
    assert status["stages"][2 if current_stage == "embed" else 3]["state"] == "failed"
    assert runner_line in status["stages"][2 if current_stage == "embed" else 3]["error"]
    assert persisted is not None
    assert persisted["state"] == "failed"
    assert persisted["finished_at"]


def test_cumulative_setup_log_attributes_latest_embedding_failure_to_embed_stage():
    import scripts.serve_benchmark_dashboard as dashboard

    record = {
        "kind": "dataset_setup",
        "state": "running",
        "finished_at": "",
        "stages": [
            {"name": name, "state": "complete", "started_at": "", "finished_at": "", "output": "", "error": ""}
            for name in ("extract", "chunk", "embed", "index")
        ],
    }
    output = (
        "loaded_cached_vectors path=data/cache-a.npy count=42 dim=768\n"
        "OK sheet=first embedding=gte store=FAISS\n"
        "ERROR embedding sheet=second embedding=gte: RuntimeError('endpoint offline')\n"
    )

    dashboard.refresh_setup_job(record, output, exit_code=None)

    assert record["state"] == "failed"
    assert dashboard.setup_stage(record, "embed")["state"] == "failed"
    assert dashboard.setup_stage(record, "index")["state"] == "complete"


def test_setup_completion_marks_cached_embedding_stage_complete():
    import scripts.serve_benchmark_dashboard as dashboard

    record = {
        "kind": "dataset_setup",
        "state": "running",
        "stages": [
            {"name": name, "state": "complete" if name in {"extract", "chunk"} else ("running" if name == "embed" else "queued"), "started_at": "", "finished_at": "", "output": "", "error": ""}
            for name in ("extract", "chunk", "embed", "index")
        ],
    }
    output = "loaded_cached_vectors path=data/cache.npy count=42 dim=768\nOK sheet=fixed embedding=gte store=FAISS\n"

    dashboard.refresh_setup_job(record, output, exit_code=0)

    assert record["state"] == "complete"
    assert dashboard.setup_stage(record, "embed")["state"] == "complete"
    assert dashboard.setup_stage(record, "index")["state"] == "complete"


def test_setup_restart_recovers_completion_from_runner_done_marker_without_process():
    import scripts.serve_benchmark_dashboard as dashboard

    record = {
        "kind": "dataset_setup",
        "state": "running",
        "finished_at": "",
        "stages": [
            {"name": name, "state": "complete" if name in {"extract", "chunk"} else ("running" if name == "embed" else "queued"), "started_at": "", "finished_at": "", "output": "", "error": ""}
            for name in ("extract", "chunk", "embed", "index")
        ],
    }
    output = (
        "loaded_cached_vectors path=data/cache.npy count=42 dim=768\n"
        "OK sheet=fixed embedding=gte store=FAISS\n"
        "DONE summary=data/user_projects/demo/runs/setup/ingestion/summary.csv failures=0\n"
    )

    dashboard.refresh_setup_job(record, output, exit_code=None)

    assert record["state"] == "complete"
    assert record["finished_at"]
    assert dashboard.setup_stage(record, "embed")["state"] == "complete"
    assert dashboard.setup_stage(record, "index")["state"] == "complete"


def test_setup_restart_does_not_complete_with_missing_required_stages():
    import scripts.serve_benchmark_dashboard as dashboard

    record = {
        "kind": "dataset_setup",
        "state": "running",
        "finished_at": "",
        "stages": [
            {"name": name, "state": "complete", "started_at": "", "finished_at": "", "output": "", "error": ""}
            for name in ("extract", "chunk")
        ],
    }
    output = "DONE summary=data/user_projects/demo/runs/setup/ingestion/summary.csv failures=0\n"

    dashboard.refresh_setup_job(record, output, exit_code=None)

    assert record["state"] == "running"
    assert record["finished_at"] == ""


def test_results_project_listing_reconciles_persisted_setup_failure_after_browser_closes():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old_projects, old_job_dir, old_jobs = dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR, dict(dashboard.JOBS)
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        dashboard.JOB_DIR = Path(td) / "jobs"
        dashboard.JOBS.clear()
        try:
            project_id = "reconcile-project"
            root = dashboard.project_root(project_id)
            root.mkdir(parents=True)
            job_id = "persisted-failure"
            stages = [
                {"name": name, "state": "complete" if name in {"extract", "chunk"} else ("running" if name == "embed" else "queued"), "started_at": "", "finished_at": "", "output": "", "error": ""}
                for name in ("extract", "chunk", "embed", "index")
            ]
            dashboard.write_project_manifest(root, {
                "project_id": project_id,
                "groundtruth": {"valid": True, "reason": ""},
                "setup": {"state": "running", "job_id": job_id, "stages": stages, "artifact_root": dashboard.display_path(root)},
            })
            log_path = dashboard.JOB_DIR / f"{job_id}.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("ERROR embedding sheet=s embedding=gte: RuntimeError('endpoint missing')\n", encoding="utf-8")
            dashboard.persist_job({
                "job_id": job_id,
                "kind": "dataset_setup",
                "state": "running",
                "created_at": "2026-07-22T02:45:00",
                "started_at": "2026-07-22T02:45:00",
                "finished_at": "",
                "project_id": project_id,
                "artifact_root": dashboard.display_path(root),
                "stages": stages,
                "log_path": dashboard.display_path(log_path),
            })

            projects = dashboard.list_user_projects()
            manifest = dashboard.read_project_manifest(root)
        finally:
            dashboard.JOBS.clear()
            dashboard.JOBS.update(old_jobs)
            dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = old_projects, old_job_dir

    assert projects[0]["setup"]["state"] == "failed"
    assert projects[0]["setup"]["stages"][2]["state"] == "failed"
    assert "endpoint missing" in projects[0]["setup"]["stages"][2]["error"]
    assert manifest["setup"]["state"] == "failed"


def test_results_project_listing_reconciles_incomplete_stage_on_completed_setup():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old_projects, old_job_dir, old_jobs = dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR, dict(dashboard.JOBS)
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        dashboard.JOB_DIR = Path(td) / "jobs"
        dashboard.JOBS.clear()
        try:
            project_id = "cached-setup-project"
            root = dashboard.project_root(project_id)
            root.mkdir(parents=True)
            job_id = "completed-with-running-stage"
            stages = [
                {"name": name, "state": "complete" if name in {"extract", "chunk", "index"} else "running", "started_at": "", "finished_at": "", "output": "", "error": ""}
                for name in ("extract", "chunk", "embed", "index")
            ]
            dashboard.write_project_manifest(root, {
                "project_id": project_id,
                "groundtruth": {"valid": True, "reason": ""},
                "setup": {"state": "complete", "job_id": job_id, "stages": stages, "artifact_root": dashboard.display_path(root)},
            })
            log_path = dashboard.JOB_DIR / f"{job_id}.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("loaded_cached_vectors path=data/cache.npy count=42 dim=768\nOK sheet=fixed embedding=gte store=FAISS\n", encoding="utf-8")
            dashboard.persist_job({
                "job_id": job_id,
                "kind": "dataset_setup",
                "state": "complete",
                "created_at": "2026-07-22T02:45:00",
                "started_at": "2026-07-22T02:45:00",
                "finished_at": "2026-07-22T02:45:01",
                "project_id": project_id,
                "artifact_root": dashboard.display_path(root),
                "stages": stages,
                "log_path": dashboard.display_path(log_path),
            })

            projects = dashboard.list_user_projects()
        finally:
            dashboard.JOBS.clear()
            dashboard.JOBS.update(old_jobs)
            dashboard.USER_PROJECTS_DIR, dashboard.JOB_DIR = old_projects, old_job_dir

    assert projects[0]["setup"]["state"] == "complete"
    assert projects[0]["setup"]["stages"][2]["state"] == "complete"


def test_project_run_gate_blocks_evidence_only_and_incomplete_setup():
    import scripts.serve_benchmark_dashboard as dashboard

    with tempfile.TemporaryDirectory() as td:
        old = dashboard.USER_PROJECTS_DIR
        dashboard.USER_PROJECTS_DIR = Path(td) / "projects"
        try:
            project = dashboard.create_user_project_upload("policy.txt", b"Refund policy text.", "Refund")
            evidence_only = dashboard.project_run_gate(project["project_id"])
            dashboard.attach_project_groundtruth(project["project_id"], "gt.csv", b"question,ground truth\nrefund?,refund\n")
            incomplete = dashboard.project_run_gate(project["project_id"])
        finally:
            dashboard.USER_PROJECTS_DIR = old

    assert evidence_only == {"error": "project_run_blocked", "mode": "evidence_only", "reason": "No ground truth attached"}
    assert incomplete["error"] == "project_run_blocked"
    assert incomplete["mode"] == "evaluated"
    assert "setup" in incomplete["reason"].lower()


def test_project_evaluation_command_uses_only_server_resolved_project_inputs(tmp_path):
    import scripts.serve_benchmark_dashboard as dashboard

    old = dashboard.USER_PROJECTS_DIR
    dashboard.USER_PROJECTS_DIR = tmp_path / "projects"
    try:
        project_id = "safe-project"
        root = dashboard.project_root(project_id)
        groundtruth = root / "questions" / "groundtruth.csv"
        receipt = root / "runs" / "setup-1" / "ingestion" / "summary.csv"
        index_dir = root / "vector_indexes" / "faiss" / "lane"
        groundtruth.parent.mkdir(parents=True)
        receipt.parent.mkdir(parents=True)
        index_dir.mkdir(parents=True)
        groundtruth.write_text("question,ground truth\nrefund?,refund policy\n", encoding="utf-8")
        receipt.write_text(
            "status,sheet,embedding,store,collection_or_table\n"
            f"ok,Heading_sections_l2,gte_multilingual_base,FAISS,{index_dir}\n",
            encoding="utf-8",
        )
        dashboard.write_project_manifest(root, {
            "project_id": project_id,
            "mode": "evaluated",
            "groundtruth": {"path": dashboard.display_path(groundtruth), "valid": True, "reason": ""},
            "setup": {"state": "complete", "job_id": "setup-1", "stages": []},
        })
        qs = {
            "sheet": ["Heading_sections_l2"],
            "embedding": ["gte_multilingual_base"],
            "store": ["FAISS"],
            "reranker": ["none"],
            "groundtruth": ["/tmp/browser-supplied.csv"],
            "ingestion_summary": ["/tmp/browser-supplied-summary.csv"],
        }
        cmd = dashboard.project_evaluation_cmd(project_id, qs)
    finally:
        dashboard.USER_PROJECTS_DIR = old

    assert cmd[1] == "scripts/run_project_evaluation.py"
    assert "scripts/run_complete_pipeline.py" not in cmd
    assert cmd[cmd.index("--project-root") + 1] == str(root)
    assert cmd[cmd.index("--groundtruth") + 1] == str(groundtruth)
    assert cmd[cmd.index("--ingestion-summary") + 1] == str(receipt)
    assert "/tmp/browser-supplied.csv" not in cmd
    assert "/tmp/browser-supplied-summary.csv" not in cmd
    assert all(
        not value.startswith("data/db_ingestion_runs")
        and not value.startswith("data/retrieval_smoke")
        and not value.startswith("data/evaluation")
        for value in cmd
    )


def test_project_evaluation_command_rejects_evidence_only_incomplete_and_tuple_mismatch(tmp_path):
    import scripts.serve_benchmark_dashboard as dashboard

    old = dashboard.USER_PROJECTS_DIR
    dashboard.USER_PROJECTS_DIR = tmp_path / "projects"
    try:
        project_id = "rejected-project"
        root = dashboard.project_root(project_id)
        root.mkdir(parents=True)
        dashboard.write_project_manifest(root, {
            "project_id": project_id,
            "mode": "evidence_only",
            "groundtruth": {"path": "", "valid": False, "reason": "No ground truth attached"},
            "setup": {"state": "complete", "job_id": "setup-1", "stages": []},
        })
        qs = {"sheet": ["Heading_sections_l2"], "embedding": ["gte_multilingual_base"], "store": ["FAISS"], "reranker": ["none"]}
        with pytest.raises(ValueError, match="ground truth"):
            dashboard.project_evaluation_cmd(project_id, qs)

        groundtruth = root / "questions" / "groundtruth.csv"
        groundtruth.parent.mkdir(parents=True)
        groundtruth.write_text("question,ground truth\nrefund?,refund policy\n", encoding="utf-8")
        dashboard.write_project_manifest(root, {
            "project_id": project_id,
            "mode": "evaluated",
            "groundtruth": {"path": dashboard.display_path(groundtruth), "valid": True, "reason": ""},
            "setup": {"state": "running", "job_id": "setup-1", "stages": []},
        })
        with pytest.raises(ValueError, match="setup"):
            dashboard.project_evaluation_cmd(project_id, qs)

        receipt = root / "runs" / "setup-1" / "ingestion" / "summary.csv"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            "status,sheet,embedding,store,collection_or_table\n"
            "ok,Heading_sections_l2,gte_multilingual_base,FAISS,vector_indexes/faiss/lane\n",
            encoding="utf-8",
        )
        dashboard.write_project_manifest(root, {
            "project_id": project_id,
            "mode": "evaluated",
            "groundtruth": {"path": dashboard.display_path(groundtruth), "valid": True, "reason": ""},
            "setup": {"state": "complete", "job_id": "setup-1", "stages": []},
        })
        with pytest.raises(ValueError, match="matching successful setup receipt"):
            dashboard.project_evaluation_cmd(project_id, {**qs, "sheet": ["fixed_tok1200_ov150"]})
    finally:
        dashboard.USER_PROJECTS_DIR = old


def test_official_complete_pipeline_command_remains_on_global_runner():
    import scripts.serve_benchmark_dashboard as dashboard

    cmd = dashboard.complete_pipeline_cmd({"sheet": ["Heading_sections_l2"]})

    assert cmd[:2] == [dashboard.sys.executable, "scripts/run_complete_pipeline.py"]
    assert "scripts/run_project_evaluation.py" not in cmd


def test_dashboard_ui_routes_only_selected_projects_to_project_evaluation_endpoints():
    app = (Path(__file__).parents[1] / "web" / "app.js").read_text(encoding="utf-8")

    assert "projectScoped ? '/api/run/project-evaluation/preflight' : '/api/run/preflight-complete-pipeline'" in app
    assert "projectScoped ? '/api/run/project-evaluation' : '/api/run/complete-pipeline'" in app


if __name__ == "__main__":
    test_groundtruth_validation_requires_question_and_reference_column()
    test_project_manifest_binds_its_uploaded_groundtruth_and_never_accepts_other_project_path()
    test_project_query_stays_evidence_only_without_valid_bound_groundtruth()
    test_setup_job_is_persisted_with_project_matrix_and_stage_provenance()
    test_project_setup_command_uses_only_project_paths_and_collection_prefix()
    test_ingestion_runner_resolves_project_paths_and_namespaces_stores()
    test_project_run_gate_blocks_evidence_only_and_incomplete_setup()
    print("dashboard dataset lifecycle tests passed")
