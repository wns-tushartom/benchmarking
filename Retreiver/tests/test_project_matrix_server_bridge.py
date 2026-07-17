from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

from scripts import serve_benchmark_dashboard as dashboard
from source.services.project_workspace import ProjectWorkspace


def _public_request(project_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": f"project:{project_id}",
        "top_k": 3,
        "questions_source": {"type": "typed", "query": "What is the refund rule?"},
        "selections": {
            "chunkers": ["fixed_tok1200_ov150", "entity_heuristic_w6"],
            "embeddings": ["jina_v3"],
            "vector_stores": ["Qdrant", "FAISS"],
            "rerankers": ["Qwen3:4B Rerank"],
        },
        "large_matrix_confirmation": None,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, ProjectWorkspace]:
    root = tmp_path / "repo"
    projects = root / "data" / "user_projects"
    projects.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "ROOT", root)
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", projects)
    monkeypatch.setattr(dashboard, "JOB_DIR", root / "data" / "dashboard_jobs")
    result = dashboard.create_user_project_upload(
        original_name="policy.txt",
        content=b"Refunds are available for thirty days.",
        label="Customer policy",
    )
    project_id = str(result["project_id"])
    return root, project_id, ProjectWorkspace(projects)


def test_preflight_resolves_only_project_source_and_does_not_create_run(project):
    root, project_id, workspace = project

    response = dashboard.preflight_project_matrix(
        _public_request(project_id), root=root, workspace=workspace
    )

    assert response["ok"] is True
    assert response["dataset_id"] == f"project:{project_id}"
    assert response["project_id"] == project_id
    assert response["combination_count"] == 4
    assert response["request_fingerprint"]
    assert response["confirmation_required"] is False
    assert response["confirmation_token"] is None
    assert response["stages"] == {
        "reusable": ["extraction"],
        "stale": [],
        "required": ["chunking", "embedding", "vector_store", "retrieval", "reranking", "publication"],
    }
    assert list(workspace.layout(project_id)["runs"].iterdir()) == []
    public_json = json.dumps(response)
    assert str(root) not in public_json
    assert "request.json" not in public_json


def test_preflight_reports_stale_stage_without_artifact_names(project):
    root, project_id, workspace = project
    private_index = workspace.layout(project_id)["indexes"] / "customer-secret.index"
    private_index.write_text("stale", encoding="utf-8")

    response = dashboard.preflight_project_matrix(
        _public_request(project_id), root=root, workspace=workspace
    )

    assert response["stages"]["stale"] == ["indexes"]
    assert "customer-secret" not in json.dumps(response)


@pytest.mark.parametrize("dataset_id", ["dataset:wns-default", "project:missing", "../project:escape"])
def test_preflight_rejects_every_non_resolved_project_source(project, dataset_id: str):
    root, project_id, workspace = project
    payload = _public_request(project_id, dataset_id=dataset_id)

    with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
        dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)

    assert caught.value.status in {400, 404}
    assert str(root) not in caught.value.public_message


def test_preflight_rejects_unknown_request_keys_and_duplicate_ids(project):
    root, project_id, workspace = project
    unknown = _public_request(project_id, filesystem_path="/tmp/private")
    duplicate = _public_request(project_id)
    duplicate["selections"] = {
        **duplicate["selections"],  # type: ignore[arg-type]
        "chunkers": ["fixed_tok1200_ov150", "fixed_tok1200_ov150"],
    }

    for payload in (unknown, duplicate):
        with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
            dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)
        assert caught.value.status == 400
        assert str(root) not in caught.value.public_message


def test_question_set_source_is_loaded_by_project_and_content_hash(project):
    root, project_id, workspace = project
    question_set = workspace.create_question_set(
        project_id, "queries.txt", b"What is the refund rule?\n"
    )
    source = {
        "type": "question_set",
        "question_set_id": question_set["question_set_id"],
        "content_sha256": question_set["content_sha256"],
    }

    response = dashboard.preflight_project_matrix(
        _public_request(project_id, questions_source=source), root=root, workspace=workspace
    )
    assert response["ok"] is True

    source["content_sha256"] = "0" * 64
    with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
        dashboard.preflight_project_matrix(
            _public_request(project_id, questions_source=source), root=root, workspace=workspace
        )
    assert caught.value.status == 400
    assert str(root) not in caught.value.public_message


def test_large_matrix_preflight_issues_request_bound_confirmation_without_a_run(
    project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, project_id, workspace = project
    key = tmp_path / "matrix-confirmation.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setenv("PROJECT_MATRIX_CONFIRMATION_KEY_FILE", str(key))
    payload = _public_request(project_id)
    payload["selections"] = {
        "chunkers": [
            "entity_heuristic_w6",
            "entity_heuristic_w5",
            "entity_heuristic_w4",
            "Heading_sections_l2",
            "fixed_tok1200_ov150",
        ],
        "embeddings": [
            "jina_v3",
            "gte_multilingual_base",
            "openai_text-embedding-3-large",
        ],
        "vector_stores": ["Qdrant", "PGVector", "Weaviate", "FAISS"],
        "rerankers": ["Amazon Rerank v1", "Qwen3:4B Rerank"],
    }

    response = dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)

    assert response["combination_count"] == 120
    assert response["confirmation_required"] is True
    assert response["confirmation_verified"] is False
    assert isinstance(response["confirmation_token"], str)
    assert list(workspace.layout(project_id)["runs"].iterdir()) == []

    payload["large_matrix_confirmation"] = response["confirmation_token"]
    confirmed = dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)
    assert confirmed["confirmation_verified"] is True
    assert confirmed["confirmation_token"] is None


def test_launch_revalidates_publishes_runner_envelope_and_uses_only_id_arguments(
    project, monkeypatch: pytest.MonkeyPatch
):
    root, project_id, workspace = project
    launched: list[list[str]] = []

    def fake_launch(command: list[str]) -> dict[str, object]:
        launched.append(command)
        return {"job_id": "job-safe", "running": True, "exit_code": None}

    monkeypatch.setattr(dashboard, "launch_job", fake_launch)

    response = dashboard.launch_project_matrix(
        _public_request(project_id), root=root, workspace=workspace
    )

    run_id = response["run_id"]
    assert response == {
        "ok": True,
        "dataset_id": f"project:{project_id}",
        "project_id": project_id,
        "run_id": run_id,
        "combination_count": 4,
        "request_fingerprint": response["request_fingerprint"],
        "job": {"job_id": "job-safe", "running": True, "exit_code": None},
    }
    assert launched == [[
        sys.executable,
        "scripts/run_project_matrix.py",
        "--project-id",
        project_id,
        "--run-id",
        run_id,
    ]]
    assert all("run_complete_pipeline.py" not in value for value in launched[0])

    request_path = workspace.run_layout(project_id, str(run_id))["root"] / "request.json"
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    assert set(envelope) == {
        "schema_version", "request", "request_fingerprint", "combination_count"
    }
    assert envelope["request"]["project_id"] == project_id
    assert "dataset_id" not in envelope["request"]
    assert envelope["request_fingerprint"] == response["request_fingerprint"]
    assert envelope["combination_count"] == 4
    assert str(root) not in json.dumps(response)


def test_launch_failure_removes_only_the_new_run_root(project, monkeypatch: pytest.MonkeyPatch):
    root, project_id, workspace = project
    preserved = workspace.layout(project_id)["runs"] / "preserved.txt"
    preserved.write_text("keep", encoding="utf-8")

    def fail_launch(_command: list[str]) -> dict[str, object]:
        raise RuntimeError(f"provider failed at {root / 'secret'}")

    monkeypatch.setattr(dashboard, "launch_job", fail_launch)

    with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
        dashboard.launch_project_matrix(
            _public_request(project_id), root=root, workspace=workspace
        )

    assert caught.value.status == 500
    assert caught.value.public_message == "Project matrix could not be launched"
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert list(workspace.layout(project_id)["runs"].iterdir()) == [preserved]


def test_request_publication_failure_removes_new_run_before_launch(
    project, monkeypatch: pytest.MonkeyPatch
):
    root, project_id, workspace = project
    monkeypatch.setattr(
        dashboard,
        "_publish_project_matrix_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )
    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda _command: pytest.fail("launch must not follow failed publication"),
    )

    with pytest.raises(dashboard.ProjectMatrixBridgeError):
        dashboard.launch_project_matrix(
            _public_request(project_id), root=root, workspace=workspace
        )

    assert list(workspace.layout(project_id)["runs"].iterdir()) == []


def test_job_status_redacts_internal_executable_log_and_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FinishedProcess:
        @staticmethod
        def poll() -> int:
            return 0

    log = tmp_path / "job.log"
    log.write_text(f"failed under {dashboard.ROOT / 'data' / 'user_projects'}\n", encoding="utf-8")
    monkeypatch.setattr(
        dashboard,
        "JOBS",
        {
            "job-safe": {
                "cmd": [
                    "/private/venv/bin/python",
                    "scripts/run_project_matrix.py",
                    "--project-id",
                    "project-safe",
                    "--run-id",
                    "run-safe",
                ],
                "log_path": str(log),
                "process": FinishedProcess(),
                "started_at": "2026-07-17T00:00:00",
            }
        },
    )

    status = dashboard.job_status("job-safe")

    assert status["cmd"] == [
        "python",
        "scripts/run_project_matrix.py",
        "--project-id",
        "project-safe",
        "--run-id",
        "run-safe",
    ]
    assert "log_path" not in status
    serialized = json.dumps(status)
    assert str(dashboard.ROOT) not in serialized
    assert "/private/" not in serialized


class _JsonRequest:
    def __init__(self, path: str, payload: object):
        body = json.dumps(payload).encode("utf-8")
        self.path = path
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        self.rfile = BytesIO(body)
        self.responses: list[tuple[int, dict[str, object]]] = []
        self.close_connection = False

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        self.responses.append((status, payload))


def test_post_routes_project_preflight_and_returns_safe_validation_error(
    project, monkeypatch: pytest.MonkeyPatch
):
    _root, project_id, _workspace = project
    request = _JsonRequest(
        "/api/run/preflight-project-matrix",
        _public_request(project_id, unknown="private"),
    )

    dashboard.Handler.do_POST(request)  # type: ignore[arg-type]

    assert len(request.responses) == 1
    status, payload = request.responses[0]
    assert status == 400
    assert payload["error"]["code"] == "invalid_project_matrix_request"  # type: ignore[index]
    assert payload["error"]["request_id"]  # type: ignore[index]
    assert "private" not in json.dumps(payload)
