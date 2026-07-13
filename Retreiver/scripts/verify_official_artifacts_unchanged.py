#!/usr/bin/env python3
"""Create and verify immutable baselines for official benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_HASH_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_MISSING_PREFIX = "!MISSING "


class ArtifactVerificationError(RuntimeError):
    """Raised when a baseline or protected artifact fails closed validation."""


def _repository_root(repo_root: Path | None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ArtifactVerificationError(f"repository root is unavailable: {root}: {exc}") from exc
    if not resolved.is_dir():
        raise ArtifactVerificationError(f"repository root is not a directory: {resolved}")
    return resolved


def _normalized_relative_path(value: str | os.PathLike[str]) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise ArtifactVerificationError("path must be text")
    if not raw or raw != raw.strip() or "\\" in raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        raise ArtifactVerificationError(
            f"path is not a normalized repository-relative path: {raw!r}"
        )
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactVerificationError(
            f"path is not a normalized repository-relative path: {raw!r}"
        )
    return raw


def _assert_no_symlink_components(path: Path, repo_root: Path) -> None:
    try:
        relative = path.relative_to(repo_root)
    except ValueError as exc:
        raise ArtifactVerificationError(f"path is outside repository: {path}") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ArtifactVerificationError(f"cannot inspect path component {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVerificationError(f"path component is a symlink: {current}")


def _entry_path(repo_root: Path, relative: str) -> Path:
    normalized = _normalized_relative_path(relative)
    candidate = repo_root.joinpath(*PurePosixPath(normalized).parts)
    _assert_no_symlink_components(candidate, repo_root)
    try:
        candidate.resolve(strict=False).relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise ArtifactVerificationError(
            f"path resolves outside repository: {normalized}"
        ) from exc
    return candidate


def _control_path(
    value: str | os.PathLike[str],
    repo_root: Path,
    *,
    require_exists: bool = False,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ArtifactVerificationError(f"control path is outside repository: {path}") from exc
        path = _entry_path(repo_root, _normalized_relative_path(relative))
    else:
        path = _entry_path(repo_root, _normalized_relative_path(os.fspath(value)))
    if require_exists:
        try:
            path.lstat()
        except FileNotFoundError as exc:
            raise ArtifactVerificationError(f"required control file is missing: {path}") from exc
        except OSError as exc:
            raise ArtifactVerificationError(f"cannot inspect required control file {path}: {exc}") from exc
    return path


def _relative_for_manifest(path: Path, repo_root: Path) -> str:
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ArtifactVerificationError(f"path is outside repository: {path}") from exc
    return _normalized_relative_path(relative)


def _hash_regular_file(path: Path, relative: str) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ArtifactVerificationError(f"protected artifact is missing: {relative}") from exc
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot inspect protected artifact {relative}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactVerificationError(f"protected artifact is a symlink: {relative}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactVerificationError(f"protected artifact is not a regular file: {relative}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot open protected artifact {relative}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ArtifactVerificationError(f"protected artifact is not a regular file: {relative}")
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _read_regular_text(path: Path, description: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(f"cannot open {description} {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVerificationError(f"{description} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactVerificationError(f"cannot decode {description} {path}: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_publish(path: Path, payload: bytes, *, replace: bool, repo_root: Path) -> None:
    destination = _control_path(path, repo_root)
    parent = destination.parent
    _assert_no_symlink_components(parent, repo_root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, flags)
    except OSError as exc:
        raise ArtifactVerificationError(f"baseline parent directory is unavailable: {parent}: {exc}") from exc

    temporary_name = f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    temporary_created = False
    published = False
    try:
        parent_metadata = os.fstat(parent_fd)
        lexical_metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or not stat.S_ISDIR(lexical_metadata.st_mode)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != (lexical_metadata.st_dev, lexical_metadata.st_ino)
            or parent.resolve(strict=True) != parent
        ):
            raise ArtifactVerificationError(f"baseline parent directory identity changed: {parent}")

        try:
            destination_metadata = os.stat(
                destination.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            destination_metadata = None
        if destination_metadata is not None and stat.S_ISLNK(destination_metadata.st_mode):
            raise ArtifactVerificationError(f"baseline destination is a symlink: {destination}")
        if destination_metadata is not None and not replace:
            raise ArtifactVerificationError(
                f"baseline already exists: {destination}; pass --replace-baseline to replace it"
            )

        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, create_flags, 0o644, dir_fd=parent_fd)
        temporary_created = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        _assert_no_symlink_components(parent, repo_root)
        lexical_metadata = os.stat(parent, follow_symlinks=False)
        if (
            (parent_metadata.st_dev, parent_metadata.st_ino)
            != (lexical_metadata.st_dev, lexical_metadata.st_ino)
            or parent.resolve(strict=True) != parent
        ):
            raise ArtifactVerificationError(f"baseline parent directory identity changed: {parent}")

        if replace:
            os.replace(
                temporary_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_created = False
        else:
            try:
                os.link(
                    temporary_name,
                    destination.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ArtifactVerificationError(
                    f"baseline already exists: {destination}; pass --replace-baseline to replace it"
                ) from exc
        published = True
        os.fsync(parent_fd)
        if temporary_created:
            os.unlink(temporary_name, dir_fd=parent_fd)
            temporary_created = False
            os.fsync(parent_fd)
    except ArtifactVerificationError:
        raise
    except OSError as exc:
        raise ArtifactVerificationError(f"failed to publish baseline {destination}: {exc}") from exc
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                if not published:
                    raise
        os.close(parent_fd)


def _render_manifest(entries: Mapping[str, str]) -> bytes:
    lines = []
    for relative in sorted(entries):
        value = entries[relative]
        if value == _MISSING_PREFIX.rstrip():
            lines.append(f"{_MISSING_PREFIX}{relative}")
        else:
            lines.append(f"{value}  {relative}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _read_manifest(
    manifest: Path,
    repo_root: Path,
    *,
    allow_missing: bool,
) -> dict[str, str]:
    path = _control_path(manifest, repo_root, require_exists=True)
    text = _read_regular_text(path, "manifest")
    lines = text.splitlines()
    if not lines:
        raise ArtifactVerificationError(f"manifest is empty: {path}")

    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if allow_missing and line.startswith(_MISSING_PREFIX):
            relative = line[len(_MISSING_PREFIX) :]
            value = _MISSING_PREFIX.rstrip()
        else:
            match = _HASH_LINE.fullmatch(line)
            if match is None:
                raise ArtifactVerificationError(
                    f"malformed manifest line {line_number} in {path}: {line!r}"
                )
            value, relative = match.groups()
        normalized = _normalized_relative_path(relative)
        _entry_path(repo_root, normalized)
        if normalized in entries:
            raise ArtifactVerificationError(
                f"duplicate manifest entry at line {line_number}: {normalized}"
            )
        entries[normalized] = value
    if list(entries) != sorted(entries):
        raise ArtifactVerificationError(
            f"manifest entries are not sorted by repository-relative path: {path}"
        )
    return entries


def write_tracked_baseline(
    manifest: str | os.PathLike[str],
    tracked_paths: Iterable[str | os.PathLike[str]],
    *,
    repo_root: Path | None = None,
    replace: bool = False,
) -> None:
    root = _repository_root(repo_root)
    entries: dict[str, str] = {}
    for value in tracked_paths:
        relative = _normalized_relative_path(value)
        if relative in entries:
            raise ArtifactVerificationError(f"duplicate tracked path: {relative}")
        path = _entry_path(root, relative)
        entries[relative] = _hash_regular_file(path, relative)
    if not entries:
        raise ArtifactVerificationError("at least one --tracked-path is required")
    destination = _control_path(manifest, root)
    _atomic_publish(destination, _render_manifest(entries), replace=replace, repo_root=root)


def check_tracked_baseline(
    manifest: str | os.PathLike[str],
    *,
    repo_root: Path | None = None,
) -> None:
    root = _repository_root(repo_root)
    entries = _read_manifest(Path(manifest), root, allow_missing=False)
    errors: list[str] = []
    for relative, expected_hash in entries.items():
        path = _entry_path(root, relative)
        try:
            actual_hash = _hash_regular_file(path, relative)
        except ArtifactVerificationError as exc:
            errors.append(str(exc))
            continue
        if actual_hash != expected_hash:
            errors.append(
                f"protected artifact changed: {relative} "
                f"(expected {expected_hash}, got {actual_hash})"
            )
    if errors:
        raise ArtifactVerificationError("; ".join(errors))


def _read_roots(roots_file: Path, repo_root: Path) -> list[str]:
    path = _control_path(roots_file, repo_root, require_exists=True)
    lines = _read_regular_text(path, "runtime roots file").splitlines()
    if not lines:
        raise ArtifactVerificationError(f"runtime roots file is empty: {path}")

    roots: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        try:
            relative = _normalized_relative_path(line)
        except ArtifactVerificationError as exc:
            raise ArtifactVerificationError(
                f"invalid runtime root at line {line_number}: {exc}"
            ) from exc
        if relative in seen:
            raise ArtifactVerificationError(f"duplicate runtime root at line {line_number}: {relative}")
        _entry_path(repo_root, relative)
        seen.add(relative)
        roots.append(relative)
    return roots


def _snapshot_runtime(roots: Sequence[str], repo_root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for root_relative in roots:
        root = _entry_path(repo_root, root_relative)
        if not root.exists():
            entries[root_relative] = _MISSING_PREFIX.rstrip()
            continue
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise ArtifactVerificationError(f"cannot inspect runtime root {root_relative}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVerificationError(f"runtime root is a symlink: {root_relative}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVerificationError(f"runtime root is not a directory: {root_relative}")

        for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_path = Path(directory)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = directory_path / name
                if child.is_symlink():
                    relative = _relative_for_manifest(child, repo_root)
                    raise ArtifactVerificationError(f"runtime artifact is a symlink: {relative}")
            for name in file_names:
                child = directory_path / name
                relative = _relative_for_manifest(child, repo_root)
                if relative in entries:
                    raise ArtifactVerificationError(f"duplicate runtime artifact entry: {relative}")
                entries[relative] = _hash_regular_file(child, relative)
    return entries


def write_runtime_baseline(
    manifest: str | os.PathLike[str],
    roots_file: str | os.PathLike[str],
    *,
    repo_root: Path | None = None,
    replace: bool = False,
) -> None:
    root = _repository_root(repo_root)
    roots = _read_roots(Path(roots_file), root)
    entries = _snapshot_runtime(roots, root)
    destination = _control_path(manifest, root)
    _atomic_publish(destination, _render_manifest(entries), replace=replace, repo_root=root)


def _runtime_difference(expected: Mapping[str, str], actual: Mapping[str, str]) -> str:
    expected_paths = set(expected)
    actual_paths = set(actual)
    added = sorted(actual_paths - expected_paths)
    removed = sorted(expected_paths - actual_paths)
    changed = sorted(
        path for path in expected_paths & actual_paths if expected[path] != actual[path]
    )
    details = []
    if added:
        details.append(f"added={added}")
    if removed:
        details.append(f"missing={removed}")
    if changed:
        details.append(f"changed={changed}")
    return ", ".join(details) or "unknown difference"


def check_runtime_baseline(
    manifest: str | os.PathLike[str],
    roots_file: str | os.PathLike[str],
    *,
    repo_root: Path | None = None,
) -> None:
    root = _repository_root(repo_root)
    expected = _read_manifest(Path(manifest), root, allow_missing=True)
    roots = _read_roots(Path(roots_file), root)
    actual = _snapshot_runtime(roots, root)
    if actual != expected:
        raise ArtifactVerificationError(
            "runtime artifact baseline differs: " + _runtime_difference(expected, actual)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--write-tracked-baseline", metavar="MANIFEST")
    operations.add_argument("--check", metavar="MANIFEST")
    operations.add_argument("--write-runtime-baseline", metavar="MANIFEST")
    operations.add_argument("--check-runtime-baseline", metavar="MANIFEST")
    parser.add_argument("--tracked-path", action="append", default=[])
    parser.add_argument("--roots", metavar="ROOTS_FILE")
    parser.add_argument("--replace-baseline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.write_tracked_baseline:
            if args.roots:
                parser.error("--roots is only valid for runtime baseline operations")
            write_tracked_baseline(
                args.write_tracked_baseline,
                args.tracked_path,
                replace=args.replace_baseline,
            )
            print(f"wrote tracked baseline: {args.write_tracked_baseline}")
        elif args.check:
            if args.tracked_path or args.roots or args.replace_baseline:
                parser.error("--check accepts only its manifest path")
            check_tracked_baseline(args.check)
            print("tracked official artifacts unchanged")
        elif args.write_runtime_baseline:
            if args.tracked_path or not args.roots:
                parser.error("--write-runtime-baseline requires --roots and no --tracked-path")
            write_runtime_baseline(
                args.write_runtime_baseline,
                args.roots,
                replace=args.replace_baseline,
            )
            print(f"wrote runtime baseline: {args.write_runtime_baseline}")
        else:
            if args.tracked_path or not args.roots or args.replace_baseline:
                parser.error("--check-runtime-baseline requires --roots and is read-only")
            check_runtime_baseline(args.check_runtime_baseline, args.roots)
            print("runtime official artifacts unchanged")
    except ArtifactVerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
