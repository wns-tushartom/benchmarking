from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts import serve_benchmark_dashboard as dashboard
from source.services.project_workspace import ProjectWorkspace


def _browser_request(project_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": f"project:{project_id}",
        "groundtruth_id": "groundtruth:none",
        "typed_queries": ["What is the refund rule?", "When does coverage end?"],
        "top_k": 3,
        "selections": {
            "chunkers": ["fixed_tok1200_ov150"],
            "embeddings": ["jina_v3"],
            "vector_stores": ["FAISS"],
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
    created = dashboard.create_user_project_upload(
        original_name="policy.txt",
        content=b"Refunds are available for thirty days.",
        label="Customer policy",
    )
    project_id = str(created["project_id"])
    return root, project_id, ProjectWorkspace(projects)


def _attach_project_groundtruth(
    root: Path,
    workspace: ProjectWorkspace,
    project_id: str,
    *,
    filename: str = "scored.csv",
    content: bytes = (
        b"query,reference_context,answer\n"
        b"What is the refund rule?,Refunds are available for thirty days.,Thirty days\n"
        b"When does coverage end?,Coverage ends after thirty days.,After thirty days\n"
    ),
) -> str:
    layout = workspace.layout(project_id)
    source = layout["questions"] / filename
    source.write_bytes(content)
    manifest_path = layout["root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["question_file"] = str(source.relative_to(root))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return f"groundtruth:project:{project_id}"


def test_typed_query_preflight_reports_evidence_mode_without_persisting(project):
    root, project_id, workspace = project
    before = list(workspace.layout(project_id)["questions"].iterdir())

    response = dashboard.preflight_project_matrix(
        _browser_request(project_id), root=root, workspace=workspace
    )

    assert response["ok"] is True
    assert response["groundtruth_id"] == "groundtruth:none"
    assert response["question_mode"] == "evidence_only"
    assert response["question_count"] == 2
    assert list(workspace.layout(project_id)["questions"].iterdir()) == before
    public = json.dumps(response)
    assert str(root) not in public
    assert "What is the refund rule?" not in public


def test_retrieval_only_preflight_does_not_claim_reranking_is_required(project):
    root, project_id, workspace = project
    payload = _browser_request(project_id)
    payload["selections"]["rerankers"] = []  # type: ignore[index]

    response = dashboard.preflight_project_matrix(
        payload, root=root, workspace=workspace
    )

    assert response["combination_count"] == 1
    assert "retrieval" in response["stages"]["required"]
    assert "reranking" not in response["stages"]["required"]


def test_typed_query_launch_persists_question_set_before_id_only_runner_launch(
    project, monkeypatch: pytest.MonkeyPatch
):
    root, project_id, workspace = project
    launched: list[list[str]] = []
    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda command: launched.append(command)
        or {"job_id": "job-safe", "running": True, "exit_code": None},
    )

    response = dashboard.launch_project_matrix(
        _browser_request(project_id), root=root, workspace=workspace
    )

    run_id = str(response["run_id"])
    envelope = json.loads(
        (workspace.run_layout(project_id, run_id)["root"] / "request.json").read_text(
            encoding="utf-8"
        )
    )
    source = envelope["request"]["questions_source"]
    assert source["type"] == "question_set"
    questions = workspace.load_question_set(
        project_id,
        source["question_set_id"],
        expected_content_sha256=source["content_sha256"],
    )
    assert [question.query for question in questions] == [
        "What is the refund rule?",
        "When does coverage end?",
    ]
    assert all(question.labels.is_empty() for question in questions)
    assert launched == [[
        sys.executable,
        "scripts/run_project_matrix.py",
        "--project-id",
        project_id,
        "--run-id",
        run_id,
    ]]
    assert str(root) not in json.dumps(response)
    assert "What is the refund rule?" not in json.dumps(response)


def test_project_groundtruth_is_catalog_resolved_parsed_and_persisted_immutably(
    project, monkeypatch: pytest.MonkeyPatch
):
    root, project_id, workspace = project
    groundtruth_id = _attach_project_groundtruth(root, workspace, project_id)
    payload = _browser_request(
        project_id,
        groundtruth_id=groundtruth_id,
        typed_queries=[],
    )
    before = {path.name for path in workspace.layout(project_id)["questions"].iterdir()}

    preflight = dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)

    assert preflight["groundtruth_id"] == groundtruth_id
    assert preflight["question_mode"] == "retrieval_labels"
    assert preflight["question_count"] == 2
    assert {path.name for path in workspace.layout(project_id)["questions"].iterdir()} == before

    monkeypatch.setattr(
        dashboard,
        "launch_job",
        lambda _command: {"job_id": "job-safe", "running": True, "exit_code": None},
    )
    launched = dashboard.launch_project_matrix(payload, root=root, workspace=workspace)
    envelope = json.loads(
        (
            workspace.run_layout(project_id, str(launched["run_id"]))["root"]
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    source = envelope["request"]["questions_source"]
    questions = workspace.load_question_set(
        project_id,
        source["question_set_id"],
        expected_content_sha256=source["content_sha256"],
    )
    assert len(questions) == 2
    assert all(not question.labels.is_empty() for question in questions)
    assert str(root) not in json.dumps(preflight)
    assert str(root) not in json.dumps(launched)


@pytest.mark.parametrize(
    "typed_queries",
    [
        [],
        ["   "],
        ["valid question", "   "],
        "not-an-array",
        [42],
        ["first\nsecond"],
        ["x" * 4001],
    ],
)
def test_none_groundtruth_rejects_missing_or_malformed_typed_queries(
    project, typed_queries: object
):
    root, project_id, workspace = project

    with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
        dashboard.preflight_project_matrix(
            _browser_request(project_id, typed_queries=typed_queries),
            root=root,
            workspace=workspace,
        )

    assert caught.value.status == 400
    assert str(root) not in caught.value.public_message
    assert list(workspace.layout(project_id)["questions"].iterdir()) == []


def test_selected_groundtruth_rejects_typed_queries_and_incompatible_project_link(project):
    root, project_id, workspace = project
    other = dashboard.create_user_project_upload(
        original_name="other.txt",
        content=b"Other project corpus.",
        label="Other",
    )
    other_id = str(other["project_id"])
    other_groundtruth = _attach_project_groundtruth(root, workspace, other_id)

    matching_groundtruth = _attach_project_groundtruth(root, workspace, project_id)
    for payload in (
        _browser_request(
            project_id,
            groundtruth_id=other_groundtruth,
            typed_queries=[],
        ),
        _browser_request(
            project_id,
            groundtruth_id=matching_groundtruth,
            typed_queries=["ambiguous"],
        ),
    ):
        with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
            dashboard.preflight_project_matrix(payload, root=root, workspace=workspace)
        assert caught.value.status == 400
        assert str(root) not in caught.value.public_message


@pytest.mark.parametrize(
    "groundtruth_id",
    [
        "groundtruth:project:../escape",
        "groundtruth:repository:../secret.csv",
        "/tmp/private.csv",
        "groundtruth:repository:unsupported.json",
        "groundtruth:missing",
    ],
)
def test_groundtruth_ids_are_resolved_fail_closed_without_path_disclosure(
    project, groundtruth_id: str
):
    root, project_id, workspace = project

    with pytest.raises(dashboard.ProjectMatrixBridgeError) as caught:
        dashboard.preflight_project_matrix(
            _browser_request(
                project_id,
                groundtruth_id=groundtruth_id,
                typed_queries=[],
            ),
            root=root,
            workspace=workspace,
        )

    assert caught.value.status == 400
    assert caught.value.public_message == "Invalid project matrix request"
    assert str(root) not in caught.value.public_message
