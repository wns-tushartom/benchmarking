from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import source.services.project_run_artifacts as project_run_artifacts
from source.services.project_run_artifacts import (
    ProjectRunArtifactError,
    read_project_artifact,
    read_project_json_rows_page,
)


EVIDENCE_METADATA = {
    "schema_version": 1,
    "project_id": "project_123",
    "run_id": "run_123",
    "mode": "evidence_only",
}


def _assert_error_code(caught: pytest.ExceptionInfo[ProjectRunArtifactError], code: str) -> None:
    assert caught.value.code == code
    assert str(caught.value) in {
        "Artifact path is invalid",
        "Artifact is unavailable",
        "Artifact exceeds the configured size limit",
        "Artifact content is invalid",
    }


def _write_evidence(root: Path, rows: list[object]) -> Path:
    root.mkdir()
    path = root / "evidence.json"
    path.write_text(
        json.dumps({**EVIDENCE_METADATA, "rows": rows}),
        encoding="utf-8",
    )
    return path


def test_project_run_artifact_public_api_is_available():
    error = ProjectRunArtifactError("invalid_path")

    assert isinstance(error, RuntimeError)
    assert error.code == "invalid_path"
    assert callable(read_project_artifact)
    assert callable(read_project_json_rows_page)


def test_read_project_artifact_reads_safe_nested_regular_file(tmp_path: Path):
    root = tmp_path / "run"
    nested = root / "logs"
    nested.mkdir(parents=True)
    content = b"bounded artifact\n"
    (nested / "worker.log").write_bytes(content)

    assert read_project_artifact(
        root,
        "logs/worker.log",
        max_bytes=len(content),
    ) == content


@pytest.mark.parametrize(
    "relative_path",
    (
        "",
        ".",
        "..",
        "../evidence.json",
        "logs/../evidence.json",
        "/evidence.json",
        "//evidence.json",
        "logs//evidence.json",
        "./evidence.json",
        "logs/./evidence.json",
        "logs/",
        "logs\\evidence.json",
        "logs/ev\x00idence.json",
    ),
)
def test_read_project_artifact_rejects_invalid_relative_paths(
    tmp_path: Path,
    relative_path: str,
):
    root = tmp_path / "run"
    root.mkdir()

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, relative_path, max_bytes=1024)

    _assert_error_code(caught, "invalid_path")


def test_read_project_artifact_rejects_symlinked_root(tmp_path: Path):
    real_root = tmp_path / "real-run"
    real_root.mkdir()
    (real_root / "evidence.json").write_bytes(b"outside")
    linked_root = tmp_path / "linked-run"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(linked_root, "evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")
    assert str(real_root) not in str(caught.value)


def test_read_project_artifact_rejects_non_directory_root(tmp_path: Path):
    root = tmp_path / "run"
    root.write_bytes(b"not a directory")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")


def test_read_project_artifact_rejects_intermediate_symlink(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence.json").write_bytes(b"outside")
    (root / "logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "logs/evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")
    assert str(outside) not in str(caught.value)


def test_read_project_artifact_rejects_symlink_leaf(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    (root / "evidence.json").symlink_to(outside)

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")
    assert str(outside) not in str(caught.value)


def test_read_project_artifact_rejects_non_directory_component(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "logs").write_bytes(b"not a directory")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "logs/evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")


def test_read_project_artifact_rejects_directory_leaf(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").mkdir()

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")


def test_read_project_artifact_rejects_oversized_file(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_bytes(b"12345")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "evidence.json", max_bytes=4)

    _assert_error_code(caught, "artifact_too_large")


def test_read_project_artifact_rejects_leaf_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_bytes(b"content")
    real_fstat = os.fstat
    regular_fstat_calls = 0

    def changing_fstat(descriptor: int):
        nonlocal regular_fstat_calls
        metadata = real_fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode):
            regular_fstat_calls += 1
            if regular_fstat_calls == 2:
                return SimpleNamespace(
                    st_mode=metadata.st_mode,
                    st_dev=metadata.st_dev,
                    st_ino=metadata.st_ino + 1,
                )
        return metadata

    monkeypatch.setattr(project_run_artifacts.os, "fstat", changing_fstat)

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_artifact(root, "evidence.json", max_bytes=1024)

    _assert_error_code(caught, "unsafe_artifact")
    assert regular_fstat_calls == 2


def test_read_project_json_rows_page_rejects_malformed_utf8(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_bytes(b"\xff")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


def test_read_project_json_rows_page_rejects_malformed_json(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


def test_read_project_json_rows_page_rejects_duplicate_json_keys(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text(
        '{"schema_version":1,"project_id":"project_123",'
        '"project_id":"project_other","run_id":"run_123",'
        '"mode":"evidence_only","rows":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


def test_read_project_json_rows_page_rejects_nonfinite_json_constants(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text(
        '{"schema_version":1,"project_id":"project_123",'
        '"run_id":"run_123","mode":"evidence_only",'
        '"rows":[{"combo_id":"combo-a","score":NaN}]}',
        encoding="utf-8",
    )

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {**EVIDENCE_METADATA},
        {**EVIDENCE_METADATA, "rows": [], "extra": True},
        {**EVIDENCE_METADATA, "schema_version": 2, "rows": []},
        {**EVIDENCE_METADATA, "schema_version": 1.0, "rows": []},
        {**EVIDENCE_METADATA, "project_id": 123, "rows": []},
        {**EVIDENCE_METADATA, "run_id": None, "rows": []},
        {**EVIDENCE_METADATA, "mode": ["evidence_only"], "rows": []},
        {**EVIDENCE_METADATA, "rows": {}},
    ),
)
def test_read_project_json_rows_page_rejects_invalid_evidence_envelope(
    tmp_path: Path,
    payload: object,
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


def test_read_project_json_rows_page_rejects_non_dict_rows(tmp_path: Path):
    root = tmp_path / "run"
    _write_evidence(
        root,
        [
            {"combo_id": "combo-a", "value": 1},
            "not an object",
        ],
    )

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


@pytest.mark.parametrize(
    ("offset", "limit"),
    ((-1, 1), (0, 0), (0, 101), (True, 1), (0, True)),
)
def test_read_project_json_rows_page_rejects_invalid_pagination(
    tmp_path: Path,
    offset: int,
    limit: int,
):
    root = tmp_path / "run"
    _write_evidence(root, [])

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=offset,
            limit=limit,
            max_bytes=1024,
        )

    _assert_error_code(caught, "invalid_artifact")


def test_read_project_json_rows_page_uses_bounded_reader(tmp_path: Path):
    root = tmp_path / "run"
    path = _write_evidence(root, [{"combo_id": "combo-a", "value": "large"}])

    with pytest.raises(ProjectRunArtifactError) as caught:
        read_project_json_rows_page(
            root,
            "evidence.json",
            combo_id="combo-a",
            offset=0,
            limit=10,
            max_bytes=path.stat().st_size - 1,
        )

    _assert_error_code(caught, "artifact_too_large")


def test_read_project_json_rows_page_filters_exact_combo_and_pages_matched_rows(
    tmp_path: Path,
):
    root = tmp_path / "run"
    rows = [
        {"combo_id": "combo-a", "value": 0},
        {"combo_id": "combo-a-extra", "value": "not exact"},
        {"combo_id": "combo-b", "value": "other combo"},
        {"combo_id": "combo-a", "value": 1},
        {"combo_id": "combo-a", "value": 2},
        {"combo_id": "combo-a", "value": 3},
    ]
    path = _write_evidence(root, rows)
    max_bytes = path.stat().st_size

    first_page = read_project_json_rows_page(
        root,
        "evidence.json",
        combo_id="combo-a",
        offset=1,
        limit=2,
        max_bytes=max_bytes,
    )
    final_page = read_project_json_rows_page(
        root,
        "evidence.json",
        combo_id="combo-a",
        offset=3,
        limit=2,
        max_bytes=max_bytes,
    )
    past_end = read_project_json_rows_page(
        root,
        "evidence.json",
        combo_id="combo-a",
        offset=10,
        limit=2,
        max_bytes=max_bytes,
    )

    assert first_page == {
        **EVIDENCE_METADATA,
        "rows": [rows[3], rows[4]],
        "next_offset": 3,
    }
    assert final_page == {
        **EVIDENCE_METADATA,
        "rows": [rows[5]],
        "next_offset": None,
    }
    assert past_end == {
        **EVIDENCE_METADATA,
        "rows": [],
        "next_offset": None,
    }
