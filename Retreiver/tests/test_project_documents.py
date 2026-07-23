from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from source.services import project_documents
from source.services.project_documents import (
    ProjectDocument,
    ProjectDocumentStorageError,
    ProjectDocumentValidationError,
    load_project_documents,
    write_project_documents,
)


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return payload.getvalue()


def _text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    payload = BytesIO()
    writer.write(payload)
    return payload.getvalue()


def _stub_mineru(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    class Page:
        page_number = 0
        content = text
        images: list[str] = []
        tables: list[dict[str, str]] = []

    class Parsed:
        content = [Page()]
        metadata = {"parsing_method": "MinerU"}

    monkeypatch.setattr(
        project_documents,
        "_parse_pdf_with_mineru",
        lambda *_args, **_kwargs: Parsed(),
    )


def test_project_documents_round_trip_with_canonical_hashes(tmp_path: Path):
    document = ProjectDocument(
        source_name="policy.txt",
        text="PROJECT_ALPHA_SENTINEL refunds are available within thirty days.",
        metadata={"page_number": "1", "parser_method": "utf8_text"},
    )
    target = tmp_path / "extracted_text" / "documents.jsonl"

    manifest = write_project_documents(target, [document])
    loaded = load_project_documents(target, expected_sha256=manifest["corpus_sha256"])

    assert loaded == [document]
    row = json.loads(target.read_text(encoding="utf-8").strip())
    assert row == {
        "schema_version": 1,
        "source_id": document.source_id,
        "source_name": "policy.txt",
        "page_number": "1",
        "parser_method": "utf8_text",
        "text": document.text,
        "content_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
    }
    assert manifest["document_count"] == 1
    assert manifest["source_count"] == 1


def test_project_document_is_deeply_immutable_and_validates_identity_fields():
    caller_metadata = {"page_number": "1", "parser_method": "utf8_text_v1"}
    document = ProjectDocument("policy.txt", "Immutable policy text", caller_metadata)
    original_source_id = document.source_id

    caller_metadata["page_number"] = "2"
    assert document.source_id == original_source_id
    with pytest.raises(TypeError):
        document.metadata["page_number"] = "2"  # type: ignore[index]
    with pytest.raises(ProjectDocumentValidationError):
        ProjectDocument(
            "policy.txt",
            "Policy text",
            {
                "page_number": "1",
                "parser_method": "utf8_text_v1",
                "nested": {"labels": ["mutable"]},
            },
        )
    with pytest.raises(ProjectDocumentValidationError):
        ProjectDocument(
            "policy.txt",
            "Policy text",
            {"page_number": "1", "parser_method": "utf8_text_v1", "extra": "value"},
        )
    assert dict(document.metadata) == {
        "page_number": "1",
        "parser_method": "utf8_text_v1",
    }
    for invalid_page in (True, 0, -1, "01", "one"):
        with pytest.raises(ProjectDocumentValidationError):
            ProjectDocument(
                "policy.txt",
                "Policy text",
                {"page_number": invalid_page, "parser_method": "utf8_text_v1"},
            )
    for invalid_name in (" policy.txt", "policy.txt ", "docs//policy.txt", "docs\\policy.txt"):
        with pytest.raises(ProjectDocumentValidationError):
            ProjectDocument(
                invalid_name,
                "Policy text",
                {"page_number": "1", "parser_method": "utf8_text_v1"},
            )


def test_project_document_corpus_is_canonical_across_input_order(tmp_path: Path):
    first = ProjectDocument(
        "b.txt",
        "Second source policy text",
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )
    second = ProjectDocument(
        "a.txt",
        "First source policy text",
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )
    left = tmp_path / "left" / "documents.jsonl"
    right = tmp_path / "right" / "documents.jsonl"

    left_manifest = write_project_documents(left, [first, second])
    right_manifest = write_project_documents(right, [second, first])

    assert left.read_bytes() == right.read_bytes()
    assert left_manifest["corpus_sha256"] == right_manifest["corpus_sha256"]
    assert [document.source_name for document in load_project_documents(left)] == ["a.txt", "b.txt"]


def test_project_document_io_rejects_symlinked_ancestor(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    document = ProjectDocument(
        "policy.txt",
        "Project policy text",
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )

    with pytest.raises((ProjectDocumentStorageError, ProjectDocumentValidationError)):
        write_project_documents(alias / "documents.jsonl", [document])
    assert not (outside / "documents.jsonl").exists()

    write_project_documents(outside / "documents.jsonl", [document])
    with pytest.raises(ProjectDocumentValidationError):
        load_project_documents(alias / "documents.jsonl")


def test_project_document_publication_rolls_back_after_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent = tmp_path / "extracted_text"
    parent.mkdir()
    target = parent / "documents.jsonl"
    identity = (parent.stat().st_dev, parent.stat().st_ino)
    real_fsync = project_documents.os.fsync
    failed = False

    def fail_once_on_parent(descriptor: int) -> None:
        nonlocal failed
        opened = project_documents.os.fstat(descriptor)
        if not failed and (opened.st_dev, opened.st_ino) == identity:
            failed = True
            raise OSError("forced parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(project_documents.os, "fsync", fail_once_on_parent)
    document = ProjectDocument(
        "policy.txt",
        "Project policy text",
        {"page_number": "1", "parser_method": "utf8_text_v1"},
    )
    with pytest.raises(ProjectDocumentStorageError):
        write_project_documents(target, [document])

    assert not target.exists()
    assert not list(parent.glob(".*.tmp"))


def test_project_document_loader_rejects_hash_tampering_and_duplicates(tmp_path: Path):
    target = tmp_path / "documents.jsonl"
    document = ProjectDocument(
        source_name="policy.txt",
        text="PROJECT_ALPHA_SENTINEL original text",
        metadata={"page_number": "1", "parser_method": "utf8_text"},
    )
    write_project_documents(target, [document])
    row = json.loads(target.read_text(encoding="utf-8").strip())
    row["text"] = "PROJECT_BETA_SENTINEL substituted text"
    target.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ProjectDocumentValidationError):
        load_project_documents(target)


def test_project_document_loader_refuses_noncanonical_artifact_names(tmp_path: Path):
    substitute = tmp_path / "search_index.json"
    substitute.write_text("{}", encoding="utf-8")
    with pytest.raises(ProjectDocumentValidationError):
        load_project_documents(substitute)

    workbook = tmp_path / "chunking_methods_output.xlsx"
    workbook.write_bytes(b"PK\x03\x04not-a-canonical-corpus")
    with pytest.raises(ProjectDocumentValidationError):
        load_project_documents(workbook)


def test_invalid_utf8_document_upload_leaves_no_visible_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    with pytest.raises(dashboard.UploadValidationError):
        dashboard.create_user_project_upload("policy.txt", b"\xff\xfe\xfa", "Bad text")

    assert not list(base.glob("*"))
    assert not list(base.glob(".staging-*"))


@pytest.mark.parametrize("name", ["data.csv", "data.xlsx", "notes.md", "data.json"])
def test_dashboard_document_upload_rejects_unsupported_direct_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    with pytest.raises(dashboard.UnsupportedUploadError):
        dashboard.create_user_project_upload(name, b"not a document corpus", "Invalid")
    assert not list(base.glob("*"))


def test_dashboard_document_zip_rejects_non_txt_pdf_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    archive = _zip_bytes([("questions.csv", b"question\nrefund?\n")])
    with pytest.raises(dashboard.UploadValidationError):
        dashboard.create_user_project_upload("bundle.zip", archive, "Invalid archive")
    assert not list(base.glob("*"))


def test_dashboard_upload_publishes_canonical_txt_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)

    payload = dashboard.create_user_project_upload(
        "policy.txt",
        b"PROJECT_ALPHA_SENTINEL refunds are available within thirty days.",
        "Alpha",
    )

    root = base / payload["project_id"]
    corpus = root / "extracted_text" / "documents.jsonl"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents = load_project_documents(corpus, expected_sha256=manifest["corpus_sha256"])
    assert manifest["extraction_status"] == "complete"
    assert manifest["document_count"] == 1
    assert documents[0].source_name == "policy.txt"
    assert "PROJECT_ALPHA_SENTINEL" in documents[0].text


def test_missing_pdf_extraction_dependency_is_sanitized_as_storage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)

    def unavailable(*_args, **_kwargs):
        raise dashboard.ProjectDocumentStorageError("private dependency detail")

    monkeypatch.setattr(dashboard, "extract_project_documents", unavailable)
    with pytest.raises(dashboard.UploadStorageError) as raised:
        dashboard.create_user_project_upload(
            "policy.pdf",
            _text_pdf("PROJECT_ALPHA_SENTINEL refund rules."),
            "No parser",
        )

    assert str(raised.value) == dashboard.UploadStorageError.public_message
    assert "dependency" not in str(raised.value).lower()
    assert not list(base.glob("*"))


def test_dashboard_upload_publishes_canonical_pdf_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    _stub_mineru(
        monkeypatch,
        "PROJECT_ALPHA_SENTINEL refunds require a booking reference.",
    )
    payload = dashboard.create_user_project_upload(
        "policy.pdf",
        _text_pdf("PROJECT_ALPHA_SENTINEL refunds require a booking reference."),
        "Alpha PDF",
    )

    root = base / payload["project_id"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents = load_project_documents(
        root / "extracted_text" / "documents.jsonl",
        expected_sha256=manifest["corpus_sha256"],
    )
    assert manifest["extraction_status"] == "complete"
    assert manifest["parser_versions"] == ["MinerU"]
    assert documents[0].source_name == "policy.pdf"
    assert "PROJECT_ALPHA_SENTINEL" in documents[0].text


def test_dashboard_zip_corpus_uses_only_that_projects_validated_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    _stub_mineru(
        monkeypatch,
        "PROJECT_ALPHA_SENTINEL baggage rules require a receipt.",
    )
    archive = _zip_bytes(
        [
            ("docs/alpha.txt", b"PROJECT_ALPHA_SENTINEL refund rules."),
            (
                "docs/baggage.pdf",
                _text_pdf("PROJECT_ALPHA_SENTINEL baggage rules require a receipt."),
            ),
        ]
    )

    payload = dashboard.create_user_project_upload("bundle.zip", archive, "Alpha bundle")

    root = base / payload["project_id"]
    documents = load_project_documents(root / "extracted_text" / "documents.jsonl")
    assert {document.source_name for document in documents} == {
        "docs/alpha.txt",
        "docs/baggage.pdf",
    }
    assert all("PROJECT_ALPHA_SENTINEL" in document.text for document in documents)
