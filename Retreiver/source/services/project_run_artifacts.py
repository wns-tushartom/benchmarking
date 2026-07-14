"""Bounded, no-follow readers for artifacts within one project run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, NoReturn


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_CHUNK_BYTES = 64 * 1024
_EVIDENCE_ENVELOPE_KEYS = {
    "schema_version",
    "project_id",
    "run_id",
    "mode",
    "rows",
}
_ERROR_MESSAGES = {
    "invalid_path": "Artifact path is invalid",
    "unsafe_artifact": "Artifact is unavailable",
    "artifact_too_large": "Artifact exceeds the configured size limit",
    "invalid_artifact": "Artifact content is invalid",
}


class ProjectRunArtifactError(RuntimeError):
    """A project run artifact could not be read safely."""

    code: str

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_ERROR_MESSAGES.get(code, "Artifact could not be read"))


def _raise_artifact_error(code: str) -> NoReturn:
    raise ProjectRunArtifactError(code)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite JSON constant")


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _relative_components(relative_path: str) -> tuple[str, ...]:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path.startswith("/")
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        _raise_artifact_error("invalid_path")
    components = tuple(relative_path.split("/"))
    if any(
        not component or component in {".", ".."} or "/" in component
        for component in components
    ):
        _raise_artifact_error("invalid_path")
    return components


def _open_root(root: Path) -> int:
    try:
        lexical_root = Path(root)
        lexical_metadata = lexical_root.lstat()
    except (OSError, TypeError, ValueError):
        _raise_artifact_error("unsafe_artifact")
    if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(
        lexical_metadata.st_mode
    ):
        _raise_artifact_error("unsafe_artifact")
    try:
        resolved_root = lexical_root.resolve(strict=True)
    except (OSError, RuntimeError):
        _raise_artifact_error("unsafe_artifact")
    if resolved_root != lexical_root:
        _raise_artifact_error("unsafe_artifact")

    descriptor = -1
    try:
        descriptor = os.open(lexical_root, _DIRECTORY_FLAGS)
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_metadata.st_mode) or not _same_identity(
            opened_metadata, lexical_metadata
        ):
            _raise_artifact_error("unsafe_artifact")
        return descriptor
    except ProjectRunArtifactError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        _raise_artifact_error("unsafe_artifact")


def _open_intermediate(parent_fd: int, component: str) -> int:
    descriptor = -1
    try:
        lexical_metadata = os.stat(
            component,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(
            lexical_metadata.st_mode
        ):
            _raise_artifact_error("unsafe_artifact")
        descriptor = os.open(
            component,
            _DIRECTORY_FLAGS,
            dir_fd=parent_fd,
        )
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_metadata.st_mode) or not _same_identity(
            opened_metadata, lexical_metadata
        ):
            _raise_artifact_error("unsafe_artifact")
        return descriptor
    except ProjectRunArtifactError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        _raise_artifact_error("unsafe_artifact")


def _read_leaf(parent_fd: int, leaf_name: str, max_bytes: int) -> bytes:
    descriptor = -1
    try:
        lexical_metadata = os.stat(
            leaf_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISREG(
            lexical_metadata.st_mode
        ):
            _raise_artifact_error("unsafe_artifact")
        descriptor = os.open(leaf_name, _FILE_FLAGS, dir_fd=parent_fd)
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode) or not _same_identity(
            opened_metadata, lexical_metadata
        ):
            _raise_artifact_error("unsafe_artifact")

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

        final_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(final_metadata.st_mode) or not _same_identity(
            final_metadata, opened_metadata
        ):
            _raise_artifact_error("unsafe_artifact")
        content = b"".join(chunks)
        if len(content) > max_bytes:
            _raise_artifact_error("artifact_too_large")
        return content
    except ProjectRunArtifactError:
        raise
    except OSError:
        _raise_artifact_error("unsafe_artifact")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_project_artifact(
    root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    """Read one bounded regular file beneath a canonical run root."""
    components = _relative_components(relative_path)
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 0
    ):
        _raise_artifact_error("invalid_artifact")

    current_fd = _open_root(root)
    try:
        for component in components[:-1]:
            next_fd = _open_intermediate(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        return _read_leaf(current_fd, components[-1], max_bytes)
    finally:
        os.close(current_fd)


def read_project_json_rows_page(
    root: Path,
    relative_path: str,
    *,
    combo_id: str,
    offset: int,
    limit: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Read and page one bounded evidence envelope by exact combination ID."""
    if (
        not isinstance(combo_id, str)
        or not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 100
    ):
        _raise_artifact_error("invalid_artifact")

    content = read_project_artifact(
        root,
        relative_path,
        max_bytes=max_bytes,
    )
    try:
        envelope = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        _raise_artifact_error("invalid_artifact")

    if (
        not isinstance(envelope, dict)
        or set(envelope) != _EVIDENCE_ENVELOPE_KEYS
        or type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or not isinstance(envelope["project_id"], str)
        or not envelope["project_id"]
        or not isinstance(envelope["run_id"], str)
        or not envelope["run_id"]
        or not isinstance(envelope["mode"], str)
        or not envelope["mode"]
        or not isinstance(envelope["rows"], list)
    ):
        _raise_artifact_error("invalid_artifact")

    rows = envelope["rows"]
    if any(not isinstance(row, dict) for row in rows):
        _raise_artifact_error("invalid_artifact")

    matched_index = 0
    page: list[dict[str, Any]] = []
    has_more = False
    for row in rows:
        if row.get("combo_id") != combo_id:
            continue
        if matched_index < offset:
            matched_index += 1
            continue
        if len(page) == limit:
            has_more = True
            break
        page.append(row)
        matched_index += 1

    return {
        "schema_version": envelope["schema_version"],
        "project_id": envelope["project_id"],
        "run_id": envelope["run_id"],
        "mode": envelope["mode"],
        "rows": page,
        "next_offset": offset + len(page) if has_more else None,
    }
