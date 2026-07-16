import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.dashboard_source_catalog import (
    DatasetSource,
    GroundtruthSource,
    build_source_catalog,
    register_groundtruth_upload,
    resolve_dataset,
    resolve_groundtruth,
)
from scripts.serve_benchmark_dashboard import (
    complete_pipeline_cmd,
    nvidia_benchmark_cmd,
    nvidia_ingest_cmd,
)
import scripts.serve_benchmark_dashboard as dashboard


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_workbook(path: Path, rows: int = 2) -> None:
    import openpyxl

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.title = "uploaded_chunks"
    worksheet.append(["id", "pdf_name", "paragraph"])
    for index in range(1, rows + 1):
        worksheet.append([index, "source.pdf", f"chunk {index}"])
    workbook.save(path)


def _fixture_root(tmp_path: Path) -> Path:
    pdf_dir = tmp_path / "data" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "one.pdf").write_bytes(b"%PDF fixture one")
    (pdf_dir / "two.PDF").write_bytes(b"%PDF fixture two")
    _write_csv(
        tmp_path / "data" / "benchmark_input.csv",
        ["id", "pdf_name", "paragraph"],
        [
            {"id": "1", "pdf_name": "one.pdf", "paragraph": "a"},
            {"id": "2", "pdf_name": "one.pdf", "paragraph": "b"},
            {"id": "3", "pdf_name": "two.PDF", "paragraph": "c"},
        ],
    )
    _write_workbook(tmp_path / "data" / "chunking_methods_output_v2.xlsx", rows=3)

    project = tmp_path / "data" / "user_projects" / "customer-demo"
    project.mkdir(parents=True)
    raw_uploads = project / "raw_uploads"
    raw_uploads.mkdir()
    (raw_uploads / "source.pdf").write_bytes(b"%PDF uploaded source")
    workbook = project / "chunks" / "chunking_methods_output_v2.xlsx"
    _write_workbook(workbook, rows=2)
    project_questions = project / "questions" / "project_questions.csv"
    _write_csv(
        project_questions,
        ["user_query", "expected_text", "expected_pdf"],
        [{"user_query": "project q", "expected_text": "project a", "expected_pdf": "source.pdf"}],
    )
    (project / "manifest.json").write_text(
        json.dumps(
            {
                "project_id": "customer-demo",
                "label": "Customer demo",
                "question_file": "data/user_projects/customer-demo/questions/project_questions.csv",
                "workbook": "data/user_projects/customer-demo/chunks/chunking_methods_output_v2.xlsx",
            }
        ),
        encoding="utf-8",
    )
    (project / "search_index.json").write_text(
        json.dumps(
            {
                "source_files": ["raw_uploads/source.pdf"],
                "chunk_count": 2,
                "chunks": [{"id": 1}, {"id": 2}],
                "workbook": "data/user_projects/customer-demo/chunks/chunking_methods_output_v2.xlsx",
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        tmp_path / "data" / "groundtruth" / "repository.csv",
        ["query", "answer", "source"],
        [
            {"query": "q1", "answer": "a1", "source": "one.pdf"},
            {"query": "q2", "answer": "a2", "source": "two.PDF"},
        ],
    )

    uploaded = tmp_path / "data" / "user_groundtruth" / "upload-123"
    uploaded.mkdir(parents=True)
    _write_csv(
        uploaded / "questions.csv",
        ["question", "ground_truth", "pdf_name"],
        [{"question": "uploaded q", "ground_truth": "uploaded a", "pdf_name": "source.pdf"}],
    )
    (uploaded / "manifest.json").write_text(
        json.dumps(
            {
                "id": "upload-123",
                "label": "Uploaded questions",
                "filename": "questions.csv",
                "validation": "valid",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_build_source_catalog_discovers_allowlisted_sources_and_truthful_counts(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    catalog = build_source_catalog(root)

    assert all(isinstance(source, dict) for source in catalog["datasets"])
    assert all(isinstance(source, dict) for source in catalog["groundtruth"])

    datasets = {source["id"]: source for source in catalog["datasets"]}
    assert list(datasets) == ["dataset:wns-default", "project:customer-demo"]
    assert datasets["dataset:wns-default"]["document_count"] == 2
    assert datasets["dataset:wns-default"]["chunk_count"] == 3
    assert datasets["dataset:wns-default"]["ready"] is True
    assert datasets["dataset:wns-default"]["sheets"] == ["uploaded_chunks"]
    assert datasets["project:customer-demo"]["label"] == "Customer demo"
    assert datasets["project:customer-demo"]["document_count"] == 1
    assert datasets["project:customer-demo"]["chunk_count"] == 2
    assert datasets["project:customer-demo"]["ready"] is True
    assert datasets["project:customer-demo"]["sheets"] == ["uploaded_chunks"]

    groundtruth = {source["id"]: source for source in catalog["groundtruth"]}
    assert catalog["groundtruth"][0]["id"] == "groundtruth:none"
    assert set(groundtruth) == {
        "groundtruth:none",
        "groundtruth:project:customer-demo",
        "groundtruth:repository:repository.csv",
        "groundtruth:upload:upload-123",
    }
    repository = groundtruth["groundtruth:repository:repository.csv"]
    assert repository["row_count"] == 2
    assert repository["valid"] is True
    assert (repository["query_column"], repository["answer_column"], repository["source_column"]) == (
        "query",
        "answer",
        "source",
    )
    upload = groundtruth["groundtruth:upload:upload-123"]
    assert upload["row_count"] == 1
    assert upload["valid"] is True
    assert upload["query_column"] == "question"
    project_gt = groundtruth["groundtruth:project:customer-demo"]
    assert project_gt["valid"] is True
    assert project_gt["project_id"] == "customer-demo"
    assert (project_gt["query_column"], project_gt["answer_column"], project_gt["source_column"]) == (
        "user_query",
        "expected_text",
        "expected_pdf",
    )

    public_json = json.dumps(catalog)
    assert str(root) not in public_json
    assert "manifest.json" not in public_json
    assert "search_index.json" not in public_json
    assert "questions.csv" not in public_json
    assert "project_questions.csv" not in public_json
    assert "chunking_methods_output_v2.xlsx" not in public_json


def test_resolvers_return_only_private_paths_for_known_catalog_ids(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    default = resolve_dataset(root, "dataset:wns-default")
    assert isinstance(default, DatasetSource)
    assert default.document_path == (root / "data" / "pdfs").resolve()
    assert default.workbook_path == (root / "data" / "chunking_methods_output_v2.xlsx").resolve()

    project = resolve_dataset(root, "project:customer-demo")
    assert isinstance(project, DatasetSource)
    assert project.document_count == 1
    project_root = (root / "data" / "user_projects" / "customer-demo").resolve()
    assert project.document_path == project_root / "raw_uploads"
    assert project.workbook_path == project_root / "chunks" / "chunking_methods_output_v2.xlsx"
    assert project.manifest_path == project_root / "manifest.json"
    assert project.search_index_path == project_root / "search_index.json"

    assert resolve_groundtruth(root, "groundtruth:none") is None
    repository = resolve_groundtruth(root, "groundtruth:repository:repository.csv")
    assert isinstance(repository, GroundtruthSource)
    assert repository.path == (root / "data" / "groundtruth" / "repository.csv").resolve()
    upload = resolve_groundtruth(root, "groundtruth:upload:upload-123")
    assert isinstance(upload, GroundtruthSource)
    assert upload.path == (root / "data" / "user_groundtruth" / "upload-123" / "questions.csv").resolve()
    project_gt = resolve_groundtruth(root, "groundtruth:project:customer-demo")
    assert isinstance(project_gt, GroundtruthSource)
    assert project_gt.path == project_root / "questions" / "project_questions.csv"


@pytest.mark.parametrize(
    ("resolver", "source_id"),
    [
        (resolve_dataset, "project:../customer-demo"),
        (resolve_dataset, "../../data/pdfs"),
        (resolve_dataset, "project:not-catalogued"),
        (resolve_groundtruth, "groundtruth:repository:../repository.csv"),
        (resolve_groundtruth, "groundtruth:upload:../upload-123"),
        (resolve_groundtruth, "groundtruth:project:../customer-demo"),
        (resolve_groundtruth, "/tmp/questions.csv"),
    ],
)
def test_resolvers_reject_unknown_and_traversal_ids(tmp_path: Path, resolver, source_id: str) -> None:
    root = _fixture_root(tmp_path)

    with pytest.raises(ValueError, match="Unknown source ID"):
        resolver(root, source_id)


def test_catalog_rejects_symlink_escapes_from_approved_roots(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_csv(outside / "secret.csv", ["query", "answer"], [{"query": "q", "answer": "a"}])
    (root / "data" / "groundtruth" / "leak.csv").symlink_to(outside / "secret.csv")

    external_project = outside / "evil"
    external_project.mkdir()
    (external_project / "manifest.json").write_text('{"label":"evil"}', encoding="utf-8")
    (external_project / "search_index.json").write_text('{"source_files":[],"chunks":[]}', encoding="utf-8")
    (root / "data" / "user_projects" / "evil").symlink_to(external_project, target_is_directory=True)

    catalog = build_source_catalog(root)
    assert "groundtruth:repository:leak.csv" not in {row["id"] for row in catalog["groundtruth"]}
    assert "project:evil" not in {row["id"] for row in catalog["datasets"]}
    with pytest.raises(ValueError, match="Unknown source ID"):
        resolve_groundtruth(root, "groundtruth:repository:leak.csv")
    with pytest.raises(ValueError, match="Unknown source ID"):
        resolve_dataset(root, "project:evil")


def test_allowlisted_root_symlink_to_in_repo_directory_is_rejected(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    approved = root / "data" / "groundtruth"
    decoy = root / "other_tables"
    decoy.mkdir()
    _write_csv(decoy / "leak.csv", ["query", "answer"], [{"query": "q", "answer": "a"}])
    for child in approved.iterdir():
        child.unlink()
    approved.rmdir()
    approved.symlink_to(decoy, target_is_directory=True)

    ids = {row["id"] for row in build_source_catalog(root)["groundtruth"]}
    assert "groundtruth:repository:leak.csv" not in ids
    with pytest.raises(ValueError, match="Unknown source ID"):
        resolve_groundtruth(root, "groundtruth:repository:leak.csv")


def test_readiness_requires_real_documents_chunks_and_readable_workbook(tmp_path: Path) -> None:
    root = tmp_path
    (root / "data" / "pdfs").mkdir(parents=True)
    (root / "data" / "pdfs" / "one.pdf").write_bytes(b"%PDF")
    _write_csv(root / "data" / "benchmark_input.csv", ["query"], [{"query": "fallback chunk"}])

    project = root / "data" / "user_projects" / "broken"
    project.mkdir(parents=True)
    (project / "manifest.json").write_text('{"label":"Broken"}', encoding="utf-8")
    (project / "search_index.json").write_text(
        json.dumps({"source_files": ["raw_uploads/missing.pdf"], "chunk_count": 9, "chunks": []}),
        encoding="utf-8",
    )
    fake_workbook = project / "chunks" / "chunking_methods_output_v2.xlsx"
    fake_workbook.parent.mkdir()
    fake_workbook.write_text("not an xlsx", encoding="utf-8")

    datasets = {row["id"]: row for row in build_source_catalog(root)["datasets"]}
    assert datasets["dataset:wns-default"]["chunk_count"] == 1
    assert datasets["dataset:wns-default"]["ready"] is False
    assert datasets["project:broken"]["document_count"] == 0
    assert datasets["project:broken"]["chunk_count"] == 0
    assert datasets["project:broken"]["ready"] is False
    assert resolve_dataset(root, "project:broken").workbook_path is None


def test_root_relative_project_question_file_wins_with_multiple_candidates(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    questions = root / "data" / "user_projects" / "customer-demo" / "questions"
    _write_csv(questions / "other.csv", ["question", "answer"], [{"question": "wrong", "answer": "wrong"}])

    source = resolve_groundtruth(root, "groundtruth:project:customer-demo")
    assert source is not None
    assert source.path == (questions / "project_questions.csv").resolve()


def test_root_relative_declared_paths_win_over_nested_decoys(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    project = root / "data" / "user_projects" / "customer-demo"
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((project / "search_index.json").read_text(encoding="utf-8"))

    workbook_declared = str(index["workbook"])
    question_declared = str(manifest["question_file"])
    decoy_workbook = project / workbook_declared
    decoy_question = project / question_declared
    _write_workbook(decoy_workbook)
    _write_csv(decoy_question, ["question", "answer"], [{"question": "decoy", "answer": "wrong"}])

    dataset = resolve_dataset(root, "project:customer-demo")
    groundtruth = resolve_groundtruth(root, "groundtruth:project:customer-demo")
    assert dataset.workbook_path == (root / workbook_declared).resolve()
    assert groundtruth is not None
    assert groundtruth.path == (root / question_declared).resolve()


def test_invalid_or_zero_query_groundtruth_is_not_resolvable(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    invalid = root / "data" / "groundtruth" / "invalid.csv"
    _write_csv(invalid, ["query", "answer"], [{"query": "", "answer": "not usable"}])

    row = next(item for item in build_source_catalog(root)["groundtruth"] if item["id"] == "groundtruth:repository:invalid.csv")
    assert row["row_count"] == 0
    assert row["valid"] is False
    with pytest.raises(ValueError, match="Unknown or invalid source ID"):
        resolve_groundtruth(root, "groundtruth:repository:invalid.csv")


def test_groundtruth_with_blank_relevance_values_is_not_resolvable(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    invalid = root / "data" / "groundtruth" / "blank_relevance.csv"
    _write_csv(invalid, ["query", "answer"], [{"query": "real question", "answer": ""}])

    row = next(item for item in build_source_catalog(root)["groundtruth"] if item["id"].endswith("blank_relevance.csv"))
    assert row["validation"] == "missing_relevance_values"
    assert row["valid"] is False
    with pytest.raises(ValueError, match="Unknown or invalid source ID"):
        resolve_groundtruth(root, row["id"])


def test_project_groundtruth_is_discovered_without_dataset_index(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    project = root / "data" / "user_projects" / "questions-only"
    project.mkdir()
    questions = project / "questions" / "gt.csv"
    _write_csv(questions, ["query", "answer"], [{"query": "q", "answer": "a"}])
    (project / "manifest.json").write_text(
        json.dumps({"label": "Questions only", "question_file": str(questions.relative_to(root))}),
        encoding="utf-8",
    )

    ids = {item["id"] for item in build_source_catalog(root)["groundtruth"]}
    assert "groundtruth:project:questions-only" in ids
    source = resolve_groundtruth(root, "groundtruth:project:questions-only")
    assert source is not None
    assert source.path == questions.resolve()


def test_project_workbook_requires_ingestion_columns(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    project = root / "data" / "user_projects" / "customer-demo"
    workbook_path = project / "chunks" / "chunking_methods_output_v2.xlsx"
    import openpyxl

    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    assert worksheet is not None
    worksheet.append(["wrong", "columns"])
    worksheet.append(["x", "y"])
    workbook.save(workbook_path)

    dataset = resolve_dataset(root, "project:customer-demo")
    assert dataset.ready is False
    assert dataset.workbook_path is None


def test_missing_query_column_is_reported_before_empty_rows(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    invalid = root / "data" / "groundtruth" / "missing_query.csv"
    _write_csv(invalid, ["answer"], [{"answer": "not usable"}])

    row = next(item for item in build_source_catalog(root)["groundtruth"] if item["id"] == "groundtruth:repository:missing_query.csv")
    assert row["validation"] == "missing_query_column"
    assert row["valid"] is False


def test_register_groundtruth_upload_creates_reusable_valid_source(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    project_manifests = set((root / "data" / "user_projects").glob("*/manifest.json"))

    entry = register_groundtruth_upload(
        root,
        "claims.csv",
        b"query,ground_truth\nq1,a1\nq2,a2\n",
        "Claims GT",
    )

    assert entry["kind"] == "uploaded"
    assert entry["row_count"] == 2
    assert entry["source_id"].startswith("groundtruth:upload:")
    assert resolve_groundtruth(root, entry["source_id"]).path.is_file()
    assert set((root / "data" / "user_projects").glob("*/manifest.json")) == project_manifests


def test_register_groundtruth_upload_rejects_invalid_schema_without_artifact(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    upload_root = root / "data" / "user_groundtruth"
    existing_manifests = set(upload_root.glob("*/manifest.json"))

    with pytest.raises(ValueError, match="query column"):
        register_groundtruth_upload(root, "bad.csv", b"answer\na1\n", "Bad")

    assert set(upload_root.glob("*/manifest.json")) == existing_manifests


def test_complete_pipeline_command_resolves_dataset_and_groundtruth_ids(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    cmd = complete_pipeline_cmd(
        {
            "dataset_id": ["project:customer-demo"],
            "groundtruth_id": ["groundtruth:project:customer-demo"],
            "sheet": ["uploaded_chunks"],
        },
        root=root,
    )

    dataset = resolve_dataset(root, "project:customer-demo")
    groundtruth = resolve_groundtruth(root, "groundtruth:project:customer-demo")
    assert cmd[cmd.index("--workbook") + 1] == str(dataset.workbook_path)
    assert cmd[cmd.index("--groundtruth") + 1] == str(groundtruth.path)
    assert "--skip-extraction-audit" in cmd
    assert "project:customer-demo" not in cmd

    default_sheet_cmd = complete_pipeline_cmd(
        {
            "dataset_id": ["project:customer-demo"],
            "groundtruth_id": ["groundtruth:project:customer-demo"],
        },
        root=root,
    )
    assert default_sheet_cmd[default_sheet_cmd.index("--sheets") + 1] == "uploaded_chunks"


def test_complete_pipeline_command_separates_retrieval_and_reranked_depth(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    cmd = complete_pipeline_cmd(
        {
            "dataset_id": ["dataset:wns-default"],
            "groundtruth_id": ["groundtruth:repository:repository.csv"],
            "retrieval_top_k": ["20"],
            "reranked_output_k": ["5"],
        },
        root=root,
    )

    assert cmd[cmd.index("--top-k") + 1] == "20"
    assert cmd[cmd.index("--reranked-output-k") + 1] == "5"


@pytest.mark.parametrize(
    ("retrieval_top_k", "reranked_output_k"),
    [("0", "5"), ("ten", "5"), ("10", "0"), ("10", "11")],
)
def test_complete_pipeline_command_rejects_invalid_depths(
    tmp_path: Path,
    retrieval_top_k: str,
    reranked_output_k: str,
) -> None:
    root = _fixture_root(tmp_path)

    with pytest.raises(ValueError, match="depth|positive|cannot exceed"):
        complete_pipeline_cmd(
            {
                "dataset_id": ["dataset:wns-default"],
                "groundtruth_id": ["groundtruth:repository:repository.csv"],
                "retrieval_top_k": [retrieval_top_k],
                "reranked_output_k": [reranked_output_k],
            },
            root=root,
        )


def test_complete_pipeline_command_enforces_evidence_only_queries(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)

    with pytest.raises(ValueError, match="query"):
        complete_pipeline_cmd(
            {"dataset_id": ["dataset:wns-default"], "groundtruth_id": ["groundtruth:none"]},
            root=root,
        )

    cmd = complete_pipeline_cmd(
        {
            "dataset_id": ["dataset:wns-default"],
            "groundtruth_id": ["groundtruth:none"],
            "query": ["first question\nsecond question"],
        },
        root=root,
    )
    assert "--evidence-only" in cmd
    assert cmd.count("--query") == 2
    assert "--groundtruth" not in cmd


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sheet", "--skip-extraction-audit"),
        ("embedding", "--fresh-run"),
        ("store", "--allow-partial-extraction"),
        ("reranker", "--evidence-only"),
    ],
)
def test_complete_pipeline_command_rejects_unallowlisted_matrix_options(
    tmp_path: Path, field: str, value: str
) -> None:
    root = _fixture_root(tmp_path)
    params = {
        "dataset_id": ["dataset:wns-default"],
        "groundtruth_id": ["groundtruth:repository:repository.csv"],
        field: ["uploaded_chunks", value] if field == "sheet" else [value],
    }

    with pytest.raises(ValueError, match="Unsupported"):
        complete_pipeline_cmd(params, root=root)


def test_nvidia_commands_resolve_ids_and_reject_none_groundtruth(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    dataset = resolve_dataset(root, "project:customer-demo")
    groundtruth = resolve_groundtruth(root, "groundtruth:project:customer-demo")

    ingest = nvidia_ingest_cmd(
        {"dataset_id": ["project:customer-demo"], "collection": ["wns_text"]},
        root=root,
    )
    assert ingest[ingest.index("--path") + 1] == str(dataset.document_path)

    benchmark = nvidia_benchmark_cmd(
        {"groundtruth_id": ["groundtruth:project:customer-demo"], "collection": ["wns_text"]},
        root=root,
    )
    assert benchmark[benchmark.index("--groundtruth") + 1] == str(groundtruth.path)
    with pytest.raises(ValueError, match="ground truth"):
        nvidia_benchmark_cmd({"groundtruth_id": ["groundtruth:none"]}, root=root)


def test_legacy_stage_actions_reject_non_default_datasets_and_evidence_only_scoring() -> None:
    assert hasattr(dashboard, "validate_legacy_action_request")

    with pytest.raises(ValueError, match="default dataset"):
        dashboard.validate_legacy_action_request(
            "/api/run/retrieval-smoke",
            {
                "dataset_id": ["project:customer-demo"],
                "groundtruth_id": ["groundtruth:repository:repository.csv"],
            },
        )

    with pytest.raises(ValueError, match="ground truth"):
        dashboard.validate_legacy_action_request(
            "/api/run/evaluate-groundtruth",
            {
                "dataset_id": ["dataset:wns-default"],
                "groundtruth_id": ["groundtruth:none"],
            },
        )

    dashboard.validate_legacy_action_request(
        "/api/run/retrieval-smoke",
        {
            "dataset_id": ["dataset:wns-default"],
            "groundtruth_id": ["groundtruth:none"],
        },
    )


def test_frontend_run_mode_disables_incompatible_legacy_actions() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for dashboard JavaScript regression test")
    app_path = Path(__file__).resolve().parents[1] / "web" / "app.js"
    node_script = f"""
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function makeElement(id) {{
  return {{
    id,
    value: '',
    textContent: '',
    innerHTML: '',
    disabled: false,
    checked: true,
    hidden: false,
    options: [],
    selectedOptions: [],
    classList: {{toggle(){{}}, add(){{}}, remove(){{}}}},
    addEventListener(){{}},
    querySelectorAll(){{return [];}},
    setAttribute(){{}},
  }};
}}
const document = {{
  getElementById(id) {{ return elements[id] || (elements[id] = makeElement(id)); }},
  querySelectorAll() {{ return []; }},
  addEventListener() {{}},
  body: {{insertAdjacentHTML(){{}}}},
}};
const context = {{
  console,
  document,
  window: {{confirm(){{return true;}}}},
  location: {{hash:'#overview'}},
  history: {{replaceState(){{}}}},
  fetch: async()=>({{ok:true,json:async()=>({{}}),text:async()=>''}}),
  setTimeout(){{}},
}};
vm.createContext(context);
let code = fs.readFileSync({str(app_path)!r}, 'utf8');
code = code.split('loadOptions().then(refresh)')[0];
vm.runInContext(code + `
state = {{sourceCatalog: {{datasets: [
  {{id:'dataset:wns-default', kind:'default', ready:true}},
  {{id:'project:customer-demo', kind:'project', ready:true}},
]}}}};
document.getElementById('runDataset').value = 'dataset:wns-default';
document.getElementById('runGroundtruth').value = 'groundtruth:none';
syncRunMode();
globalThis.__evidence = {{
  freshDisabled: document.getElementById('runFresh').disabled,
  freshChecked: document.getElementById('runFresh').checked,
  evalDisabled: document.getElementById('runEvalBtn').disabled,
  fullDisabled: document.getElementById('runFullGtBtn').disabled,
  smokeDisabled: document.getElementById('runSmokeBtn').disabled,
}};
document.getElementById('runDataset').value = 'project:customer-demo';
document.getElementById('runGroundtruth').value = 'groundtruth:repository:repository.csv';
syncRunMode();
globalThis.__project = {{
  parseDisabled: document.getElementById('runParseMineruBtn').disabled,
  ingestDisabled: document.getElementById('runIngestBtn').disabled,
  smokeDisabled: document.getElementById('runSmokeBtn').disabled,
  rerankerDisabled: document.getElementById('runRerankerBtn').disabled,
  mainDisabled: document.getElementById('runCompletePipelineBtn').disabled,
}};
`, context);
if (!context.__evidence.freshDisabled || context.__evidence.freshChecked) process.exit(1);
if (!context.__evidence.evalDisabled || !context.__evidence.fullDisabled) process.exit(2);
if (context.__evidence.smokeDisabled) process.exit(3);
if (!context.__project.parseDisabled || !context.__project.ingestDisabled || !context.__project.smokeDisabled || !context.__project.rerankerDisabled) process.exit(4);
if (context.__project.mainDisabled) process.exit(5);
"""
    proc = subprocess.run([node, "-e", node_script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
