from io import BytesIO
from email.message import Message
import json
from pathlib import Path
import stat
import zipfile

import pytest

import source.services.project_workspace as project_workspace
from source.services.project_workspace import (
    ProjectWorkspace,
    safe_slug,
    validate_resource_id,
)


LAYOUT_CHILDREN = (
    "raw_uploads",
    "extracted_text",
    "chunks",
    "questions",
    "indexes",
    "runs",
)


def test_resource_ids_are_uuid_backed_and_reject_path_syntax(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = workspace.new_project_id("Refund Data")
    assert project_id.startswith("refund-data_")
    assert validate_resource_id(project_id) == project_id
    for value in ("../other", "/tmp/project", "a/b", "", ".", "project id"):
        with pytest.raises(ValueError):
            validate_resource_id(value)


@pytest.mark.parametrize(
    "value",
    (
        "project_00000000000000000000000000000000",
        "project_00000000000010008000000000000000",
    ),
)
def test_resource_ids_reject_nil_and_non_v4_uuids(value: str):
    with pytest.raises(ValueError):
        validate_resource_id(value)


def test_project_root_cannot_escape_base(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    with pytest.raises(ValueError):
        workspace.project_root("../official")


def test_project_root_rejects_symlink_to_another_project(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    project_a = workspace.new_project_id("Project A")
    project_b = workspace.new_project_id("Project B")
    project_b_root = base / project_b
    project_b_root.mkdir(parents=True)
    (base / project_a).symlink_to(project_b_root, target_is_directory=True)

    with pytest.raises(ValueError):
        workspace.project_root(project_a)


def test_project_root_rejects_symlink_outside_workspace(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    project_id = workspace.new_project_id("Project A")
    outside = tmp_path / "outside"
    outside.mkdir()
    base.mkdir()
    (base / project_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        workspace.project_root(project_id)


@pytest.mark.parametrize("child", LAYOUT_CHILDREN)
@pytest.mark.parametrize("destination_kind", ("outside", "another_project"))
def test_layout_rejects_symlinked_children(
    tmp_path: Path, child: str, destination_kind: str
):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    project_id = workspace.new_project_id("Project A")
    root = base / project_id
    root.mkdir(parents=True)

    if destination_kind == "outside":
        destination = tmp_path / "outside" / child
    else:
        other_project_id = workspace.new_project_id("Project B")
        destination = base / other_project_id / child
    destination.mkdir(parents=True)
    (root / child).symlink_to(destination, target_is_directory=True)

    with pytest.raises(ValueError):
        workspace.layout(project_id)


def test_relative_base_is_canonicalized_once(tmp_path: Path, monkeypatch):
    original_cwd = tmp_path / "original"
    original_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    workspace = ProjectWorkspace(Path("projects"))
    canonical_base = (original_cwd / "projects").resolve()
    project_id = workspace.new_project_id("Project A")

    new_cwd = tmp_path / "new"
    new_cwd.mkdir()
    monkeypatch.chdir(new_cwd)

    assert workspace.base == canonical_base
    assert workspace.project_root(project_id) == canonical_base / project_id


def test_safe_slug_normalizes_and_bounds_project_labels():
    assert safe_slug(" Refund Data / 2026 ") == "refund-data-2026"
    assert safe_slug(".-") == "project"
    assert safe_slug("A" * 41) == "a" * 40
    assert safe_slug(".-" * 30 + "Policy") == "policy"
    assert safe_slug("a" * 39 + ".b") == "a" * 39
    assert safe_slug("a" * 39 + "-b") == "a" * 39


def test_run_id_is_uuid_backed_and_valid(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    run_id = workspace.new_run_id()

    assert run_id.startswith("run_")
    assert validate_resource_id(run_id) == run_id


def test_layout_returns_only_paths_beneath_the_project_root(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = workspace.new_project_id("Refund Data")
    root = workspace.project_root(project_id)

    assert workspace.layout(project_id) == {
        "root": root,
        "raw_uploads": root / "raw_uploads",
        "extracted_text": root / "extracted_text",
        "documents": root / "extracted_text" / "documents.jsonl",
        "chunks": root / "chunks",
        "questions": root / "questions",
        "indexes": root / "indexes",
        "runs": root / "runs",
    }


def test_layout_rejects_symlinked_canonical_documents_file(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = workspace.new_project_id("Refund Data")
    root = workspace.project_root(project_id)
    root.mkdir(parents=True)
    for child in LAYOUT_CHILDREN:
        (root / child).mkdir()
    outside = tmp_path / "outside-documents.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    (root / "extracted_text" / "documents.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="Canonical project corpus"):
        workspace.layout(project_id)


def _zip_bytes(
    entries: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=compression) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return payload.getvalue()


def _xlsx_bytes(
    *,
    content_types: bytes = b"types",
    workbook: bytes = b"workbook",
    extra_entries: list[tuple[str | zipfile.ZipInfo, bytes]] | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    return _zip_bytes(
        [
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", b"relationships"),
            ("xl/workbook.xml", workbook),
            *(extra_entries or []),
        ],
        compression=compression,
    )


def _real_xlsx_bytes() -> bytes:
    from openpyxl import Workbook

    payload = BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet["A1"] = "refund policy"
    workbook.save(payload)
    return payload.getvalue()


def _corrupt_deflated_member(payload: bytes, member_name: str) -> bytes:
    corrupted = bytearray(payload)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)
    offset = info.header_offset
    filename_length = int.from_bytes(payload[offset + 26 : offset + 28], "little")
    extra_length = int.from_bytes(payload[offset + 28 : offset + 30], "little")
    data_start = offset + 30 + filename_length + extra_length
    corrupted[data_start] = 0x06  # raw DEFLATE block with reserved/invalid BTYPE
    return bytes(corrupted)


def _assert_no_upload_residue(base: Path) -> None:
    assert not base.exists() or list(base.iterdir()) == []


def test_upload_limits_have_safe_defaults():
    limits = project_workspace.UploadLimits()

    assert limits.max_upload_bytes == 100 * 1024 * 1024
    assert limits.max_json_bytes == 1024 * 1024
    assert limits.max_zip_entries == 500
    assert limits.max_zip_entry_bytes == 100 * 1024 * 1024
    assert limits.max_zip_expanded_bytes == 500 * 1024 * 1024
    assert limits.max_zip_ratio == 100
    assert limits.max_zip_depth == 8


@pytest.mark.parametrize("nested", (False, True))
def test_real_openpyxl_workbook_is_accepted_directly_and_inside_zip(
    tmp_path: Path,
    nested: bool,
):
    base = tmp_path / "projects"
    xlsx = _real_xlsx_bytes()
    name = "bundle.zip" if nested else "dataset.xlsx"
    payload = _zip_bytes([("dataset.xlsx", xlsx)], compression=zipfile.ZIP_STORED) if nested else xlsx

    manifest = ProjectWorkspace(base).create_upload(name, payload, "Real workbook")

    root = base / manifest["project_id"]
    assert root.is_dir()
    assert (root / "manifest.json").is_file()


@pytest.mark.parametrize("nested", (False, True))
def test_xlsx_requires_content_types_root_relationships_and_workbook(
    tmp_path: Path,
    nested: bool,
):
    base = tmp_path / "projects"
    impostor = _zip_bytes(
        [
            ("[Content_Types].xml", b"types"),
            ("_rels/.rels", b"relationships"),
            ("xl/not-workbook.xml", b"not a workbook"),
        ]
    )
    name = "bundle.zip" if nested else "dataset.xlsx"
    payload = _zip_bytes([("dataset.xlsx", impostor)], compression=zipfile.ZIP_STORED) if nested else impostor

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload(name, payload, "Impostor workbook")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize("nested", (False, True))
def test_malformed_deflate_in_xlsx_is_typed_validation_failure(
    tmp_path: Path,
    nested: bool,
):
    base = tmp_path / "projects"
    xlsx = _corrupt_deflated_member(_real_xlsx_bytes(), "xl/workbook.xml")
    name = "bundle.zip" if nested else "dataset.xlsx"
    payload = _zip_bytes([("dataset.xlsx", xlsx)], compression=zipfile.ZIP_STORED) if nested else xlsx

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload(name, payload, "Corrupt workbook")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize(
    ("case", "limits"),
    (
        ("entries", {"max_zip_entries": 1}),
        ("depth", {"max_zip_depth": 1}),
        ("entry-bytes", {"max_zip_entry_bytes": 4}),
        ("expanded-bytes", {"max_zip_expanded_bytes": 12}),
        ("ratio", {"max_zip_ratio": 2}),
    ),
)
def test_top_level_xlsx_enforces_each_container_resource_limit(
    tmp_path: Path, case: str, limits: dict[str, int]
):
    base = tmp_path / "projects"
    workbook = b"0" * 10_000 if case == "ratio" else b"workbook"

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base, project_workspace.UploadLimits(**limits)).create_upload(
            "dataset.xlsx", _xlsx_bytes(workbook=workbook), case
        )

    _assert_no_upload_residue(base)


def test_outer_zip_and_nested_xlsx_share_expanded_byte_budget(tmp_path: Path):
    base = tmp_path / "projects"
    workbook = bytes(range(256)) * 4
    xlsx = _xlsx_bytes(workbook=workbook)
    outer_expanded = 2 * len(xlsx)
    inner_expanded = len(b"types") + len(workbook)
    shared_limit = max(outer_expanded, inner_expanded) + 1
    assert outer_expanded <= shared_limit
    assert inner_expanded <= shared_limit
    assert outer_expanded + (2 * inner_expanded) > shared_limit
    payload = _zip_bytes(
        [("a.xlsx", xlsx), ("b.xlsx", xlsx)],
        compression=zipfile.ZIP_STORED,
    )

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(
            base,
            project_workspace.UploadLimits(max_zip_expanded_bytes=shared_limit),
        ).create_upload("bundle.zip", payload, "Shared expanded budget")

    _assert_no_upload_residue(base)


def test_outer_zip_and_nested_xlsx_share_member_count_budget(tmp_path: Path):
    base = tmp_path / "projects"
    xlsx = _xlsx_bytes()
    payload = _zip_bytes(
        [("a.xlsx", xlsx), ("b.xlsx", xlsx)],
        compression=zipfile.ZIP_STORED,
    )

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(
            base,
            project_workspace.UploadLimits(max_zip_entries=5),
        ).create_upload("bundle.zip", payload, "Shared entry budget")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize("unsafe_name", ("../escape.xml", "/absolute.xml", "C:\\absolute.xml"))
def test_xlsx_rejects_traversal_and_absolute_member_paths_without_residue(
    tmp_path: Path, unsafe_name: str
):
    base = tmp_path / "projects"
    payload = _xlsx_bytes(extra_entries=[(unsafe_name, b"x")])

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.xlsx", payload, "Unsafe XLSX")

    _assert_no_upload_residue(base)


def test_xlsx_rejects_normalized_duplicate_member_paths_without_residue(tmp_path: Path):
    base = tmp_path / "projects"
    payload = _xlsx_bytes(extra_entries=[("xl//workbook.xml", b"duplicate")])

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.xlsx", payload, "Duplicate XLSX")

    _assert_no_upload_residue(base)


def test_xlsx_rejects_symlink_member_mode_without_residue(tmp_path: Path):
    symlink = zipfile.ZipInfo("xl/linked.xml")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    base = tmp_path / "projects"
    payload = _xlsx_bytes(extra_entries=[(symlink, b"target")])

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.xlsx", payload, "Symlink XLSX")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize("file_type", (stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK))
def test_outer_zip_rejects_non_regular_member_modes_without_residue(
    tmp_path: Path,
    file_type: int,
):
    special = zipfile.ZipInfo("special.txt")
    special.create_system = 3
    special.external_attr = (file_type | 0o600) << 16
    base = tmp_path / "projects"
    payload = _zip_bytes([(special, b"unsafe")], compression=zipfile.ZIP_STORED)

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("bundle.zip", payload, "Special file")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize("unsafe_name", ("xl/nested.zip", "xl/program.exe"))
def test_xlsx_strict_entry_policy_rejects_archives_and_executables(
    tmp_path: Path, unsafe_name: str
):
    base = tmp_path / "projects"
    payload = _xlsx_bytes(extra_entries=[(unsafe_name, b"unsafe")])

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.xlsx", payload, "Unsafe XLSX")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize(
    ("case", "limits", "workbook", "extra_entries"),
    (
        ("entries", {"max_zip_entries": 1}, b"workbook", []),
        ("depth", {"max_zip_depth": 1}, b"workbook", []),
        ("entry-bytes", {"max_zip_entry_bytes": 1_000}, b"0" * 10_000, []),
        ("expanded-bytes", {"max_zip_expanded_bytes": 1_000}, b"0" * 10_000, []),
        ("ratio", {"max_zip_ratio": 2}, b"0" * 10_000, []),
    ),
)
def test_xlsx_inside_outer_zip_enforces_each_container_resource_limit(
    tmp_path: Path,
    case: str,
    limits: dict[str, int],
    workbook: bytes,
    extra_entries: list[tuple[str | zipfile.ZipInfo, bytes]],
):
    base = tmp_path / "projects"
    xlsx = _xlsx_bytes(workbook=workbook, extra_entries=extra_entries)
    payload = _zip_bytes([("dataset.xlsx", xlsx)], compression=zipfile.ZIP_STORED)

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base, project_workspace.UploadLimits(**limits)).create_upload(
            "bundle.zip", payload, case
        )

    _assert_no_upload_residue(base)


def test_normal_bounded_xlsx_is_accepted_top_level_and_inside_outer_zip(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    xlsx = _xlsx_bytes()

    direct = workspace.create_upload("direct.xlsx", xlsx, "Direct XLSX")
    bundled = workspace.create_upload(
        "bundle.zip",
        _zip_bytes([("nested/book.xlsx", xlsx)], compression=zipfile.ZIP_STORED),
        "Bundled XLSX",
    )

    direct_root = workspace.project_root(direct["project_id"])
    bundled_root = workspace.project_root(bundled["project_id"])
    assert (direct_root / "raw_uploads" / "direct.xlsx").read_bytes() == xlsx
    assert (
        bundled_root / "raw_uploads" / "extracted" / "nested" / "book.xlsx"
    ).read_bytes() == xlsx
    assert bundled["extracted_count"] == 1
    assert list(base.glob(".staging-*")) == []


def test_low_disk_headroom_fails_before_any_upload_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    base = tmp_path / "projects"
    payload = _zip_bytes([("doc.txt", b"content")], compression=zipfile.ZIP_STORED)
    limits = project_workspace.UploadLimits(
        max_upload_bytes=1024,
        max_json_bytes=1024,
        max_zip_entries=10,
        max_zip_entry_bytes=1024,
        max_zip_expanded_bytes=4096,
        max_zip_ratio=100,
        max_zip_depth=8,
    )
    monkeypatch.setattr(
        project_workspace.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": 8192, "used": 8191, "free": 1})(),
    )

    writes: list[Path] = []

    def record_write(path: Path, *_args, **_kwargs):
        writes.append(path)
        raise OSError("write must not start")

    monkeypatch.setattr(project_workspace, "_write_durable", record_write)

    with pytest.raises(project_workspace.UploadStorageError):
        ProjectWorkspace(base, limits).create_upload("bundle.zip", payload, "No space")

    assert writes == []
    assert not base.exists()


def test_txt_upload_atomically_publishes_manifest_and_raw_file(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)

    manifest = workspace.create_upload(
        original_name="policy.txt",
        content=b"Refund policy",
        label="Customer Policy",
    )

    root = workspace.project_root(manifest["project_id"])
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert (root / "raw_uploads" / "policy.txt").read_bytes() == b"Refund policy"
    assert list(base.glob(".staging-*")) == []
    assert [path.name for path in base.iterdir()] == [manifest["project_id"]]


def test_upload_finalizer_artifacts_publish_only_with_complete_project(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    observed: dict[str, Path] = {}

    def finalize(staging: Path, final_root: Path, manifest: dict) -> dict:
        assert staging.name.startswith(".staging-")
        assert not final_root.exists()
        assert list(base.iterdir()) == [staging]
        derived = staging / "extracted_text" / "derived.txt"
        derived.write_text("complete", encoding="utf-8")
        observed.update(staging=staging, final_root=final_root)
        return {**manifest, "derived_file": "extracted_text/derived.txt"}

    manifest = workspace.create_upload(
        "policy.txt",
        b"refunds",
        "Policy",
        finalizer=finalize,
    )

    assert not observed["staging"].exists()
    assert observed["final_root"].is_dir()
    assert (observed["final_root"] / manifest["derived_file"]).read_text() == "complete"
    assert json.loads((observed["final_root"] / "manifest.json").read_text()) == manifest


def test_upload_finalizer_failure_leaves_no_project_or_staging_residue(tmp_path: Path):
    base = tmp_path / "projects"

    def fail_finalize(staging: Path, final_root: Path, manifest: dict) -> dict:
        (staging / "extracted_text" / "partial.txt").write_text("partial")
        raise RuntimeError("simulated enrichment failure")

    with pytest.raises(project_workspace.UploadStorageError):
        ProjectWorkspace(base).create_upload(
            "policy.txt",
            b"refunds",
            "Policy",
            finalizer=fail_finalize,
        )

    _assert_no_upload_residue(base)


@pytest.mark.parametrize(
    ("entries", "limits"),
    (
        ([("../escape.txt", b"x")], {}),
        ([("/absolute.txt", b"x")], {}),
        ([("C:\\absolute.txt", b"x")], {}),
        ([("docs/a.txt", b"a"), ("docs//a.txt", b"b")], {}),
        ([("a.txt", b"a"), ("b.txt", b"b")], {"max_zip_entries": 1}),
        ([("large.txt", b"12345")], {"max_zip_entry_bytes": 4}),
        ([("a.txt", b"1234"), ("b.txt", b"5678")], {"max_zip_expanded_bytes": 7}),
        ([("a/b/c.txt", b"x")], {"max_zip_depth": 2}),
        ([("bomb.txt", b"0" * 10_000)], {"max_zip_ratio": 2}),
        ([("nested.zip", _zip_bytes([("inner.txt", b"x")]))], {}),
        ([("program.exe", b"MZ")], {}),
    ),
    ids=(
        "traversal",
        "absolute",
        "windows-absolute",
        "duplicate-normalized-destination",
        "too-many-entries",
        "oversized-entry",
        "oversized-cumulative-expansion",
        "depth-limit",
        "compression-ratio",
        "nested-archive",
        "unsupported-entry-extension",
    ),
)
def test_unsafe_zip_is_rejected_without_residue(
    tmp_path: Path,
    entries: list[tuple[str | zipfile.ZipInfo, bytes]],
    limits: dict[str, int],
):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base, project_workspace.UploadLimits(**limits))

    with pytest.raises(project_workspace.UploadValidationError):
        workspace.create_upload("dataset.zip", _zip_bytes(entries), "Unsafe")

    _assert_no_upload_residue(base)


def test_backslash_and_slash_zip_destinations_are_duplicates(tmp_path: Path):
    base = tmp_path / "projects"
    payload = _zip_bytes([("docs/policy.txt", b"a"), ("docs\\policy.txt", b"b")])

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.zip", payload, "Duplicates")

    _assert_no_upload_residue(base)


def test_zip_symlink_mode_is_rejected_without_residue(tmp_path: Path):
    info = zipfile.ZipInfo("linked.txt")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)

    with pytest.raises(project_workspace.UploadValidationError):
        workspace.create_upload("dataset.zip", _zip_bytes([(info, b"target")]), "Unsafe")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize(
    ("name", "content"),
    (
        ("dataset.pdf", b"plain text"),
        ("dataset.zip", b"not a zip"),
        ("dataset.txt", b"%PDF-1.7\n"),
        ("dataset.csv", b"PK\x03\x04binary"),
        ("dataset.xlsx", b"plain text"),
    ),
)
def test_extension_signature_mismatch_is_rejected_without_residue(
    tmp_path: Path, name: str, content: bytes
):
    base = tmp_path / "projects"

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload(name, content, "Mismatch")

    _assert_no_upload_residue(base)


def test_malformed_zip_with_zip_signature_is_rejected_without_residue(tmp_path: Path):
    base = tmp_path / "projects"

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.zip", b"PK\x03\x04broken", "Broken")

    _assert_no_upload_residue(base)


def test_xlsx_with_corrupt_member_is_rejected_without_residue(tmp_path: Path):
    base = tmp_path / "projects"
    payload = bytearray(
        _zip_bytes(
            [
                ("[Content_Types].xml", b"content-types"),
                ("xl/workbook.xml", b"workbook-body"),
            ],
            compression=zipfile.ZIP_STORED,
        )
    )
    member_offset = payload.index(b"workbook-body")
    payload[member_offset] ^= 0x01

    with pytest.raises(project_workspace.UploadValidationError):
        ProjectWorkspace(base).create_upload("dataset.xlsx", bytes(payload), "Corrupt")

    _assert_no_upload_residue(base)


def test_unsupported_upload_extension_is_typed_and_leaves_no_residue(tmp_path: Path):
    base = tmp_path / "projects"

    with pytest.raises(project_workspace.UnsupportedUploadError):
        ProjectWorkspace(base).create_upload("payload.exe", b"MZ", "Unsupported")

    _assert_no_upload_residue(base)


def test_total_upload_cap_and_json_cap_leave_no_residue(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(
        base,
        project_workspace.UploadLimits(max_upload_bytes=4, max_json_bytes=3),
    )

    with pytest.raises(project_workspace.UploadTooLargeError):
        workspace.create_upload("large.txt", b"12345", "Large")
    with pytest.raises(project_workspace.UploadValidationError):
        workspace.create_upload("data.json", b"[10]", "JSON")

    _assert_no_upload_residue(base)


def test_upload_filename_and_zip_entry_cannot_write_to_another_project(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    other_id = workspace.new_project_id("Other")
    other_root = workspace.project_root(other_id)
    other_root.mkdir(parents=True)
    marker = other_root / "marker.txt"
    marker.write_text("untouched", encoding="utf-8")

    with pytest.raises(project_workspace.UploadValidationError):
        workspace.create_upload(f"../{other_id}/marker.txt", b"changed", "Attack")
    with pytest.raises(project_workspace.UploadValidationError):
        workspace.create_upload(
            "attack.zip",
            _zip_bytes([(f"../{other_id}/marker.txt", b"changed")]),
            "Attack",
        )

    assert marker.read_text(encoding="utf-8") == "untouched"
    assert list(base.glob(".staging-*")) == []
    assert [path.name for path in base.iterdir()] == [other_id]


def test_normal_zip_extracts_supported_files_inside_final_project(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    content = _zip_bytes([("docs/policy.txt", b"refunds"), ("notes.md", b"# Notes")])

    manifest = workspace.create_upload("dataset.zip", content, "Normal ZIP")

    root = workspace.project_root(manifest["project_id"])
    assert (root / "raw_uploads" / "dataset.zip").read_bytes() == content
    assert (root / "raw_uploads" / "extracted" / "docs" / "policy.txt").read_bytes() == b"refunds"
    assert (root / "raw_uploads" / "extracted" / "notes.md").read_bytes() == b"# Notes"
    assert manifest["extracted_count"] == 2
    assert list(base.glob(".staging-*")) == []


def test_final_project_publish_uses_os_replace(tmp_path: Path, monkeypatch):
    base = tmp_path / "projects"
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = project_workspace.os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(project_workspace.os, "replace", recording_replace)
    manifest = ProjectWorkspace(base).create_upload("policy.txt", b"refunds", "Policy")

    final_root = base / manifest["project_id"]
    assert any(
        source.name.startswith(".staging-") and destination == final_root
        for source, destination in replace_calls
    )


def test_post_publish_fsync_failure_removes_new_project(tmp_path: Path, monkeypatch):
    base = (tmp_path / "projects").resolve()
    real_fsync_directory = project_workspace._fsync_directory

    def fail_final_base_fsync(path: Path):
        if path == base and any(
            not child.name.startswith(".staging-") for child in base.iterdir()
        ):
            raise OSError("simulated final directory fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(project_workspace, "_fsync_directory", fail_final_base_fsync)

    with pytest.raises(project_workspace.UploadStorageError):
        ProjectWorkspace(base).create_upload("policy.txt", b"refunds", "Policy")

    _assert_no_upload_residue(base)


def test_two_projects_never_cross_return_lexical_evidence(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    alpha = workspace.create_upload("same.txt", b"alpha-only cancellation remedy", "Alpha")
    beta = workspace.create_upload("same.txt", b"beta-only baggage remedy", "Beta")
    alpha_root = workspace.project_root(alpha["project_id"])
    beta_root = workspace.project_root(beta["project_id"])
    (alpha_root / "search_index.json").write_text(json.dumps({"chunks": [{"chunk_id": 1, "pdf_name": "same.txt", "paragraph": "alpha-only cancellation remedy", "page_number": 1}]}))
    (beta_root / "search_index.json").write_text(json.dumps({"chunks": [{"chunk_id": 1, "pdf_name": "same.txt", "paragraph": "beta-only baggage remedy", "page_number": 1}]}))

    alpha_result = workspace.lexical_preview(alpha["project_id"], "alpha cancellation", top_k=5)
    beta_result = workspace.lexical_preview(beta["project_id"], "beta baggage", top_k=5)

    assert alpha_result["mode"] == beta_result["mode"] == "lexical_preview"
    assert all("beta-only" not in row["paragraph"] for row in alpha_result["hits"])
    assert all("alpha-only" not in row["paragraph"] for row in beta_result["hits"])
    assert "rows" not in alpha_result
    assert "combo_count" not in alpha_result
    assert "selections" not in alpha_result
    assert alpha_result["run_id"] != beta_result["run_id"]
    for root, result in ((alpha_root, alpha_result), (beta_root, beta_result)):
        run_root = root / "runs" / result["run_id"]
        assert json.loads((run_root / "evidence.json").read_text()) == result
        manifest = json.loads((run_root / "manifest.json").read_text())
        assert manifest["mode"] == "lexical_preview"
        assert manifest["project_id"] == result["project_id"]
        assert "hits" not in manifest
        assert "selections" not in manifest
        assert not list((root / "runs").glob(".staging-*"))


def test_legacy_direct_child_project_can_still_run_lexical_preview(tmp_path: Path):
    base = tmp_path / "projects"
    workspace = ProjectWorkspace(base)
    legacy_id = "policy_20260711_114725_abcdef"
    root = base / legacy_id
    for child in LAYOUT_CHILDREN:
        (root / child).mkdir(parents=True, exist_ok=True)
    (root / "search_index.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": 1,
                        "pdf_name": "policy.txt",
                        "paragraph": "historical refund policy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = workspace.lexical_preview(legacy_id, "refund", top_k=1)

    assert result["project_id"] == legacy_id
    assert result["hits"][0]["paragraph"] == "historical refund policy"
    assert (root / "runs" / result["run_id"] / "evidence.json").is_file()


def test_legacy_project_id_symlink_is_rejected(tmp_path: Path):
    base = tmp_path / "projects"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    legacy_id = "policy_20260711_114725_abcdef"
    (base / legacy_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        ProjectWorkspace(base).lexical_preview(legacy_id, "refund")


def test_lexical_preview_requires_explicit_existing_project_and_bounded_input(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    with pytest.raises(ValueError):
        workspace.lexical_preview("", "refund")
    missing = workspace.new_project_id("missing")
    with pytest.raises(FileNotFoundError):
        workspace.lexical_preview(missing, "refund")

    project = workspace.create_upload("policy.txt", b"refund", "Policy")
    root = workspace.project_root(project["project_id"])
    (root / "search_index.json").write_text(json.dumps({"chunks": []}))
    with pytest.raises(ValueError):
        workspace.lexical_preview(project["project_id"], "x" * 4001)
    with pytest.raises(ValueError):
        workspace.lexical_preview(project["project_id"], "refund", top_k=0)
    with pytest.raises(ValueError):
        workspace.lexical_preview(project["project_id"], "refund", top_k=51)
    assert not list((root / "runs").iterdir())


def test_lexical_preview_fsync_failure_leaves_no_visible_run(tmp_path: Path, monkeypatch):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project = workspace.create_upload("policy.txt", b"refund", "Policy")
    root = workspace.project_root(project["project_id"])
    (root / "search_index.json").write_text(
        json.dumps({"chunks": [{"paragraph": "refund policy", "pdf_name": "policy.txt"}]})
    )
    runs = root / "runs"
    real_fsync_directory = project_workspace._fsync_directory

    def fail_final_runs_fsync(path: Path):
        if path == runs and any(not child.name.startswith(".staging-") for child in runs.iterdir()):
            raise OSError("simulated runs fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(project_workspace, "_fsync_directory", fail_final_runs_fsync)
    with pytest.raises(project_workspace.UploadStorageError):
        workspace.lexical_preview(project["project_id"], "refund")
    assert list(runs.iterdir()) == []


def _multipart_body(filename: str | None, content: bytes = b"x") -> tuple[str, bytes]:
    boundary = "----bounded-upload-test"
    parts = []
    if filename is not None:
        parts.append(
            b"--" + boundary.encode() + b"\r\n"
            + f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
            + b"Content-Type: application/octet-stream\r\n\r\n"
            + content + b"\r\n"
        )
    parts.append(b"--" + boundary.encode() + b"--\r\n")
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


class _UploadRequest:
    path = "/api/upload-dataset"

    def __init__(self, headers, body: bytes, rfile=None):
        self.headers = headers
        self.rfile = rfile if rfile is not None else BytesIO(body)
        self.responses: list[tuple[int, dict]] = []
        self.close_connection = False

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.responses.append((status, payload))


def _call_upload_endpoint(headers, body: bytes) -> tuple[int, dict]:
    import scripts.serve_benchmark_dashboard as dashboard

    request = _UploadRequest(headers, body)
    dashboard.Handler.do_POST(request)
    assert len(request.responses) == 1
    return request.responses[0]


def _call_project_query_endpoint(headers, body: bytes) -> tuple[int, dict]:
    import scripts.serve_benchmark_dashboard as dashboard

    request = _UploadRequest(headers, body)
    request.path = "/api/project-query"
    dashboard.Handler.do_POST(request)
    assert len(request.responses) == 1
    return request.responses[0]


class _CountingStream(BytesIO):
    def __init__(self, body: bytes):
        super().__init__(body)
        self.consumed = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.consumed += len(chunk)
        return chunk

    def readline(self, size: int = -1) -> bytes:
        chunk = super().readline(size)
        self.consumed += len(chunk)
        return chunk


def _assert_malformed_upload(response: tuple[int, dict]) -> None:
    status, payload = response
    assert status == 400
    request_id = payload["error"]["request_id"]
    assert isinstance(request_id, str) and request_id
    assert payload == {
        "error": {
            "code": "malformed_request",
            "message": "Malformed upload request",
            "request_id": request_id,
        }
    }


def test_project_query_endpoint_accepts_only_explicit_lexical_request(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    body = json.dumps({"project_id": "policy_123", "query": "refund", "top_k": 7}).encode()
    captured = {}

    def fake_query(project_id, query, top_k=5):
        captured.update(project_id=project_id, query=query, top_k=top_k)
        return {"ok": True, "mode": "lexical_preview", "hits": []}

    monkeypatch.setattr(dashboard, "query_user_project", fake_query)
    status, payload = _call_project_query_endpoint(
        {"Content-Type": "application/json", "Content-Length": str(len(body))}, body
    )

    assert status == 200
    assert payload["mode"] == "lexical_preview"
    assert captured == {"project_id": "policy_123", "query": "refund", "top_k": 7}


@pytest.mark.parametrize(
    "forbidden_key,forbidden_value",
    (
        ("selections", {"rerankers": ["bge"]}),
        ("rerankers", ["bge-reranker-base"]),
        ("vector_stores", ["Qdrant"]),
        ("embeddings", ["gte_multilingual_base"]),
        ("chunkers", ["fixed_tok1200_ov150"]),
        ("unexpected", True),
    ),
)
def test_project_query_endpoint_rejects_non_lexical_fields(
    monkeypatch, forbidden_key, forbidden_value
):
    import scripts.serve_benchmark_dashboard as dashboard

    request_payload = {"project_id": "policy_123", "query": "refund"}
    request_payload[forbidden_key] = forbidden_value
    body = json.dumps(request_payload).encode()
    monkeypatch.setattr(
        dashboard,
        "query_user_project",
        lambda *args, **kwargs: pytest.fail("rejected request must not reach project query"),
    )

    status, payload = _call_project_query_endpoint(
        {"Content-Type": "application/json", "Content-Length": str(len(body))}, body
    )

    assert status == 400
    assert payload["error"]["code"] == "invalid_request"
    assert forbidden_key not in json.dumps(payload)


def test_project_query_endpoint_rejects_duplicate_and_oversized_lengths(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    body = json.dumps({"project_id": "policy_123", "query": "refund"}).encode()
    monkeypatch.setattr(
        dashboard,
        "query_user_project",
        lambda *args, **kwargs: pytest.fail("invalid framing must not reach project query"),
    )

    duplicate_headers = Message()
    duplicate_headers.add_header("Content-Type", "application/json")
    duplicate_headers.add_header("Content-Length", str(len(body)))
    duplicate_headers.add_header("Content-Length", str(len(body)))
    duplicate_status, duplicate_payload = _call_project_query_endpoint(
        duplicate_headers, body
    )
    assert duplicate_status == 400
    assert duplicate_payload["error"]["code"] == "malformed_request"

    monkeypatch.setattr(
        dashboard,
        "dashboard_upload_limits",
        lambda: project_workspace.UploadLimits(max_json_bytes=len(body) - 1),
    )
    oversized_status, oversized_payload = _call_project_query_endpoint(
        {"Content-Type": "application/json", "Content-Length": str(len(body))}, body
    )
    assert oversized_status == 413
    assert oversized_payload["error"]["code"] == "request_too_large"


def test_project_query_endpoint_maps_missing_project_without_leaking_path(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    body = json.dumps({"project_id": "missing_123", "query": "refund"}).encode()
    monkeypatch.setattr(
        dashboard,
        "query_user_project",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("/private/project/path")),
    )

    status, payload = _call_project_query_endpoint(
        {"Content-Type": "application/json", "Content-Length": str(len(body))}, body
    )
    assert status == 404
    assert payload["error"]["code"] == "project_not_found"
    assert "/private/project/path" not in json.dumps(payload)


def test_upload_endpoint_underdeclared_body_never_reads_past_content_length(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    content_type, body = _multipart_body("dataset.txt", b"x" * (2 * 1024 * 1024))
    limits = project_workspace.UploadLimits(max_upload_bytes=1024 * 1024)
    stream = _CountingStream(body)
    request = _UploadRequest(
        {"Content-Type": content_type, "Content-Length": "1"}, body, rfile=stream
    )
    monkeypatch.setattr(dashboard, "dashboard_upload_limits", lambda: limits)
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: pytest.fail("upload service must not be called"),
    )

    dashboard.Handler.do_POST(request)

    assert stream.consumed <= 1
    assert request.close_connection is True
    assert len(request.responses) == 1
    _assert_malformed_upload(request.responses[0])


def test_upload_endpoint_rejects_truncated_declared_multipart_body(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    content_type, complete_body = _multipart_body("dataset.txt", b"safe")
    truncated_body = complete_body[:-12]
    stream = _CountingStream(truncated_body)
    request = _UploadRequest(
        {
            "Content-Type": content_type,
            "Content-Length": str(len(complete_body)),
        },
        truncated_body,
        rfile=stream,
    )
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: pytest.fail("truncated upload must not reach storage"),
    )

    dashboard.Handler.do_POST(request)

    assert stream.consumed == len(truncated_body)
    assert request.close_connection is True
    assert len(request.responses) == 1
    _assert_malformed_upload(request.responses[0])


def test_upload_endpoint_rejects_multipart_without_terminal_boundary(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    content_type, complete_body = _multipart_body("dataset.txt", b"safe")
    closing_boundary = b"------bounded-upload-test--\r\n"
    assert complete_body.endswith(closing_boundary)
    malformed_body = complete_body[: -len(closing_boundary)]
    stream = _CountingStream(malformed_body)
    request = _UploadRequest(
        {
            "Content-Type": content_type,
            "Content-Length": str(len(malformed_body)),
        },
        malformed_body,
        rfile=stream,
    )
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: pytest.fail("malformed upload must not reach storage"),
    )

    dashboard.Handler.do_POST(request)

    assert stream.consumed == len(malformed_body)
    assert request.close_connection is True
    assert len(request.responses) == 1
    _assert_malformed_upload(request.responses[0])


def test_upload_endpoint_rejects_terminal_boundary_spoofed_by_file_content(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    boundary = "----bounded-upload-test"
    terminal = b"--" + boundary.encode() + b"--"
    body = (
        b"--" + boundary.encode() + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="dataset.txt"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
        b"payload-with-no-mime-delimiter" + terminal
    )
    request = _UploadRequest(
        {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        body,
    )
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: pytest.fail("spoofed boundary must not reach storage"),
    )

    dashboard.Handler.do_POST(request)

    assert request.close_connection is True
    assert len(request.responses) == 1
    _assert_malformed_upload(request.responses[0])


def test_upload_endpoint_rejects_pathologically_long_content_length():
    content_type, body = _multipart_body("dataset.txt")

    response = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": "9" * 5000}, body
    )

    _assert_malformed_upload(response)


def test_upload_endpoint_rejects_duplicate_content_length_headers(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    content_type, body = _multipart_body("dataset.txt")
    headers = Message()
    headers.add_header("Content-Type", content_type)
    headers.add_header("Content-Length", str(len(body)))
    headers.add_header("Content-Length", str(len(body)))
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: pytest.fail("duplicate length must be rejected before storage"),
    )

    response = _call_upload_endpoint(headers, body)

    _assert_malformed_upload(response)


def test_upload_endpoint_bounds_defensive_file_read(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    read_sizes: list[int] = []
    service_called = False

    class OversizedFile:
        def read(self, size=-1):
            read_sizes.append(size)
            return b"x" * size

    class Form:
        def __contains__(self, key):
            return key == "file"

        def __getitem__(self, key):
            return type("Item", (), {"filename": "dataset.txt", "file": OversizedFile()})()

        def getfirst(self, key, default=None):
            return default

    body = b"\r\n--x--\r\n"

    def fake_field_storage(*, fp, environ):
        assert fp.read() == body
        return Form()

    def fake_service(**kwargs):
        nonlocal service_called
        service_called = True

    limits = project_workspace.UploadLimits(max_upload_bytes=16)
    monkeypatch.setattr(dashboard, "dashboard_upload_limits", lambda: limits)
    monkeypatch.setattr(dashboard.cgi, "FieldStorage", fake_field_storage)
    monkeypatch.setattr(dashboard, "create_user_project_upload", fake_service)
    request = _UploadRequest(
        {
            "Content-Type": "multipart/form-data; boundary=x",
            "Content-Length": str(len(body)),
        },
        body,
    )

    dashboard.Handler.do_POST(request)

    assert read_sizes == [limits.max_upload_bytes + 1]
    assert service_called is False
    assert request.responses[0][0] == 413
    assert request.responses[0][1]["error"]["code"] == "upload_too_large"


@pytest.mark.parametrize("content_length", (None, "abc", "-1"))
def test_upload_endpoint_rejects_missing_invalid_or_negative_content_length(content_length):
    content_type, body = _multipart_body("dataset.txt")
    headers = {"Content-Type": content_type}
    if content_length is not None:
        headers["Content-Length"] = content_length

    status, payload = _call_upload_endpoint(headers, body)

    assert status == 400
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "request_id"}
    assert payload["error"]["code"] == "malformed_request"


def test_upload_endpoint_rejects_over_limit_before_multipart_parsing(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    monkeypatch.setenv("DASHBOARD_MAX_UPLOAD_MB", "0")
    monkeypatch.setattr(
        dashboard.cgi,
        "FieldStorage",
        lambda *args, **kwargs: pytest.fail("multipart parser must not be called"),
    )

    status, payload = _call_upload_endpoint(
        {"Content-Type": "multipart/form-data; boundary=x", "Content-Length": "1"},
        b"x",
    )

    assert status == 413
    assert payload["error"]["code"] == "upload_too_large"


def test_upload_endpoint_maps_media_validation_and_malformed_errors():
    unsupported_type = _call_upload_endpoint(
        {"Content-Type": "application/json", "Content-Length": "2"}, b"{}"
    )
    content_type, unsupported_body = _multipart_body("payload.exe", b"MZ")
    unsupported_file = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(unsupported_body))},
        unsupported_body,
    )
    content_type, invalid_body = _multipart_body("broken.zip", b"PK\x03\x04broken")
    invalid_upload = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(invalid_body))},
        invalid_body,
    )
    content_type, malformed_body = _multipart_body(None)
    malformed = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(malformed_body))},
        malformed_body,
    )

    assert unsupported_type[0] == 415
    assert unsupported_file[0] == 415
    assert invalid_upload[0] == 422
    assert malformed[0] == 400
    for status, payload in (unsupported_type, unsupported_file, invalid_upload, malformed):
        assert status in {400, 415, 422}
        assert set(payload) == {"error"}
        assert set(payload["error"]) == {"code", "message", "request_id"}


def test_upload_endpoint_never_leaks_raw_exception_or_path(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    secret = "/private/customer/project.txt: database exploded"
    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    content_type, body = _multipart_body("dataset.txt", b"safe")

    status, payload = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(body))}, body
    )

    assert status == 500
    assert payload["error"]["code"] == "internal_error"
    assert secret not in json.dumps(payload)
    assert "/private" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    (
        (project_workspace.UploadTooLargeError("secret"), 413, "upload_too_large"),
        (project_workspace.UnsupportedUploadError("secret"), 415, "unsupported_media_type"),
        (project_workspace.UploadValidationError("secret"), 422, "upload_validation_failed"),
        (project_workspace.UploadStorageError("secret"), 500, "internal_error"),
    ),
)
def test_upload_endpoint_maps_typed_service_errors_without_details(
    monkeypatch, error, expected_status: int, expected_code: str
):
    import scripts.serve_benchmark_dashboard as dashboard

    monkeypatch.setattr(
        dashboard,
        "create_user_project_upload",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    content_type, body = _multipart_body("dataset.txt", b"safe")

    status, payload = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(body))}, body
    )

    assert status == expected_status
    assert payload["error"]["code"] == expected_code
    assert "secret" not in json.dumps(payload)


def test_upload_endpoint_maps_malformed_multipart_parser_error(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    monkeypatch.setattr(
        dashboard.cgi,
        "FieldStorage",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("secret malformed body")),
    )
    content_type, body = _multipart_body("dataset.txt", b"safe")

    status, payload = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(body))}, body
    )

    assert status == 400
    assert payload["error"]["code"] == "malformed_request"
    assert "secret" not in json.dumps(payload)


def test_upload_endpoint_calls_project_upload_with_expected_arguments(monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    captured: dict = {}

    def create_upload(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(dashboard, "create_user_project_upload", create_upload)
    content_type, body = _multipart_body("dataset.txt", b"safe")

    status, payload = _call_upload_endpoint(
        {"Content-Type": content_type, "Content-Length": str(len(body))}, body
    )

    assert status == 200
    assert payload == {"ok": True}
    assert captured == {
        "original_name": "dataset.txt",
        "content": b"safe",
        "label": "dataset",
    }


def test_dashboard_enrichment_failure_leaves_no_visible_project(tmp_path: Path, monkeypatch):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    monkeypatch.setattr(
        dashboard,
        "write_project_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            dashboard.ProjectDocumentStorageError("simulated corpus write failure")
        ),
    )

    with pytest.raises(project_workspace.UploadStorageError):
        dashboard.create_user_project_upload("policy.txt", b"refund policy", "Policy")

    _assert_no_upload_residue(base)


@pytest.mark.parametrize("nested", (False, True))
def test_dashboard_rejects_malformed_pdf_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    malformed_pdf = b"%PDF-1.7\nnot a valid PDF package"
    name = "bundle.zip" if nested else "broken.pdf"
    content = (
        _zip_bytes([("docs/broken.pdf", malformed_pdf)], compression=zipfile.ZIP_STORED)
        if nested
        else malformed_pdf
    )

    payload = dashboard.create_user_project_upload(name, content, "Broken PDF")

    root = base / payload["project_id"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extraction_status"] == "failed"
    assert manifest["document_count"] == 0
    assert manifest["page_count"] == 0
    assert manifest["extraction_failures"] == [
        {
            "source_name": "docs/broken.pdf" if nested else "broken.pdf",
            "code": "extraction_failed",
        }
    ]
    assert not (root / "extracted_text" / "documents.jsonl").exists()
    assert not list(base.glob(".staging-*"))


def test_dashboard_preserves_duplicate_basenames_as_distinct_derived_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)
    archive = _zip_bytes(
        [
            ("a/policy.txt", b"alpha refund policy"),
            ("b/policy.txt", b"beta baggage policy"),
        ],
        compression=zipfile.ZIP_STORED,
    )

    payload = dashboard.create_user_project_upload("bundle.zip", archive, "Two policies")

    root = base / payload["project_id"]
    corpus_rows = [
        json.loads(line)
        for line in (root / "extracted_text" / "documents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_name"] for row in corpus_rows} == {
        "a/policy.txt",
        "b/policy.txt",
    }
    assert {row["text"] for row in corpus_rows} == {
        "alpha refund policy",
        "beta baggage policy",
    }
    assert not (root / "search_index.json").exists()


def test_dashboard_atomic_upload_publishes_complete_queryable_project(
    tmp_path: Path, monkeypatch
):
    import scripts.serve_benchmark_dashboard as dashboard

    base = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", base)

    payload = dashboard.create_user_project_upload(
        "policy.txt",
        b"Refunds are available within thirty days.",
        "Policy",
    )
    root = base / payload["project_id"]
    serialized = json.dumps(payload)

    assert root.is_dir()
    assert not list(base.glob(".staging-*"))
    assert ".staging-" not in serialized
    assert (root / "manifest.json").is_file()
    assert (root / "extracted_text" / "documents.jsonl").is_file()
    assert not (root / "search_index.json").exists()
    assert not (root / "questions" / "policy.txt").exists()
    assert (root / "indexes").is_dir()
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8"))["project_id"] == payload["project_id"]

    query = dashboard.query_user_project(payload["project_id"], "refunds", top_k=1)
    assert query["project_id"] == payload["project_id"]
    assert query["mode"] == "lexical_preview"
    assert query["hits"][0]["paragraph"].startswith("Refunds are available")


def test_dashboard_upload_limit_env_defaults_are_documented():
    env = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")

    assert "DASHBOARD_MAX_UPLOAD_MB=100" in env
    assert "DASHBOARD_MAX_JSON_KB=1024" in env
    assert "DASHBOARD_MAX_ZIP_FILES=500" in env
    assert "DASHBOARD_MAX_ZIP_EXPANDED_MB=500" in env
    assert "DASHBOARD_MAX_ZIP_RATIO=100" in env
