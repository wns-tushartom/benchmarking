from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from uuid import uuid4


_SCHEMA_VERSION = 1
_ALLOWED_SOURCE_SUFFIXES = {".txt", ".pdf"}
_REQUIRED_METADATA_KEYS = {"page_number", "parser_method"}
_REQUIRED_ROW_KEYS = {
    "schema_version",
    "source_id",
    "source_name",
    "page_number",
    "parser_method",
    "text",
    "content_sha256",
}


class ProjectDocumentValidationError(ValueError):
    """Raised when a canonical project corpus is malformed or incomplete."""


class ProjectDocumentStorageError(RuntimeError):
    """Raised when canonical extraction infrastructure is unavailable."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalized_source_name(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProjectDocumentValidationError("source_name is required")
    if "\\" in value or "//" in value:
        raise ProjectDocumentValidationError("source_name must be a normalized relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ProjectDocumentValidationError("source_name must be a normalized relative path")
    return str(path)


def _source_id(source_name: str, page_number: int, content_sha256: str) -> str:
    identity = f"{source_name}\0{page_number}\0{content_sha256}".encode("utf-8")
    return f"source_{_sha256(identity)}"


@dataclass(frozen=True)
class ProjectDocument:
    source_name: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_name = _normalized_source_name(self.source_name)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ProjectDocumentValidationError("document text is required")
        if not isinstance(self.metadata, Mapping):
            raise ProjectDocumentValidationError("document metadata is invalid")
        metadata = dict(self.metadata)
        if set(metadata) != _REQUIRED_METADATA_KEYS:
            raise ProjectDocumentValidationError(
                "document metadata must contain only page_number and parser_method"
            )
        raw_page_number = metadata["page_number"]
        if isinstance(raw_page_number, bool):
            raise ProjectDocumentValidationError("page_number must be a positive integer")
        try:
            page_number = int(raw_page_number)
        except (TypeError, ValueError):
            raise ProjectDocumentValidationError("page_number must be a positive integer") from None
        if page_number < 1 or str(raw_page_number).strip() != str(page_number):
            raise ProjectDocumentValidationError("page_number must be a positive integer")
        parser_method = metadata.get("parser_method", "")
        if (
            not isinstance(parser_method, str)
            or not parser_method
            or parser_method != parser_method.strip()
            or not parser_method.replace("_", "").isalnum()
            or not parser_method[0].isalpha()
        ):
            raise ProjectDocumentValidationError("page_number and parser_method are required")
        metadata["page_number"] = str(page_number)
        metadata["parser_method"] = parser_method
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @property
    def content_sha256(self) -> str:
        return _sha256(self.text.encode("utf-8"))

    @property
    def source_id(self) -> str:
        return _source_id(
            self.source_name,
            int(self.metadata["page_number"]),
            self.content_sha256,
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "page_number": str(self.metadata["page_number"]),
            "parser_method": str(self.metadata["parser_method"]),
            "text": self.text,
            "content_sha256": self.content_sha256,
        }


def _open_directory_fd(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(child)
                raise ProjectDocumentValidationError(
                    "canonical corpus directory is invalid"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_atomic(path: Path, content: bytes) -> None:
    path = Path(path)
    if path.name != "documents.jsonl":
        raise ProjectDocumentValidationError("canonical documents.jsonl is required")
    try:
        parent_fd = _open_directory_fd(path.parent, create=True)
    except ProjectDocumentValidationError:
        raise
    except OSError:
        raise ProjectDocumentStorageError("canonical corpus directory is unavailable") from None
    temporary_name = f".{path.name}.{uuid4().hex}.tmp"
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise ProjectDocumentValidationError("canonical corpus already exists") from None
        published = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        try:
            os.fsync(parent_fd)
        except OSError:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
                published = False
                os.fsync(parent_fd)
            except OSError:
                raise ProjectDocumentStorageError(
                    "canonical corpus publication state is indeterminate"
                ) from None
            raise ProjectDocumentStorageError("canonical corpus could not be stored") from None
    except (ProjectDocumentValidationError, ProjectDocumentStorageError):
        raise
    except OSError:
        if published:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise ProjectDocumentStorageError("canonical corpus could not be stored") from None
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(parent_fd)


def write_project_documents(path: Path, documents: Iterable[ProjectDocument]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_pages: set[tuple[str, int]] = set()
    source_names: set[str] = set()
    for document in documents:
        if not isinstance(document, ProjectDocument):
            raise ProjectDocumentValidationError("invalid project document")
        row = document.to_row()
        page_identity = (row["source_name"], int(row["page_number"]))
        if row["source_id"] in seen_source_ids or page_identity in seen_pages:
            raise ProjectDocumentValidationError("duplicate canonical document source page")
        seen_source_ids.add(row["source_id"])
        seen_pages.add(page_identity)
        source_names.add(row["source_name"])
        rows.append(row)
    if not rows:
        raise ProjectDocumentValidationError("project corpus is empty")
    rows.sort(
        key=lambda row: (
            row["source_name"],
            int(row["page_number"]),
            row["content_sha256"],
        )
    )
    content = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    _write_atomic(Path(path), content)
    return {
        "schema_version": _SCHEMA_VERSION,
        "document_count": len(rows),
        "source_count": len(source_names),
        "corpus_sha256": _sha256(content),
    }


def _document_from_row(row: Any) -> ProjectDocument:
    if not isinstance(row, dict) or set(row) != _REQUIRED_ROW_KEYS:
        raise ProjectDocumentValidationError("invalid canonical document row")
    if (
        isinstance(row.get("schema_version"), bool)
        or row.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ProjectDocumentValidationError("unsupported canonical document schema")
    for key in ("source_id", "source_name", "page_number", "parser_method", "text", "content_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ProjectDocumentValidationError(f"invalid {key}")
    if (
        len(row["content_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["content_sha256"])
        or not row["source_id"].startswith("source_")
        or len(row["source_id"]) != 71
        or any(character not in "0123456789abcdef" for character in row["source_id"][7:])
    ):
        raise ProjectDocumentValidationError("invalid canonical document digest")
    document = ProjectDocument(
        source_name=row["source_name"],
        text=row["text"],
        metadata={
            "page_number": row["page_number"],
            "parser_method": row["parser_method"],
        },
    )
    if row["content_sha256"] != document.content_sha256 or row["source_id"] != document.source_id:
        raise ProjectDocumentValidationError("canonical document hash mismatch")
    return document


def _read_regular_at(parent_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ProjectDocumentValidationError("canonical documents.jsonl is required")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_project_documents(path: Path, expected_sha256: str | None = None) -> list[ProjectDocument]:
    path = Path(path)
    if path.name != "documents.jsonl":
        raise ProjectDocumentValidationError("canonical documents.jsonl is required")
    try:
        parent_fd = _open_directory_fd(path.parent, create=False)
        try:
            content = _read_regular_at(parent_fd, path.name)
        finally:
            os.close(parent_fd)
    except ProjectDocumentValidationError:
        raise
    except OSError:
        raise ProjectDocumentValidationError("canonical corpus could not be read") from None
    content_sha256 = _sha256(content)
    if expected_sha256 is not None and content_sha256 != expected_sha256:
        raise ProjectDocumentValidationError("canonical corpus hash mismatch")
    documents: list[ProjectDocument] = []
    seen_source_ids: set[str] = set()
    seen_pages: set[tuple[str, int]] = set()
    previous_key: tuple[str, int, str] | None = None
    try:
        lines = content.decode("utf-8").splitlines()
        for line in lines:
            if not line.strip():
                raise ProjectDocumentValidationError("blank canonical document row")
            document = _document_from_row(json.loads(line))
            page_identity = (document.source_name, int(document.metadata["page_number"]))
            canonical_key = (
                document.source_name,
                page_identity[1],
                document.content_sha256,
            )
            if document.source_id in seen_source_ids or page_identity in seen_pages:
                raise ProjectDocumentValidationError("duplicate canonical document source page")
            if previous_key is not None and canonical_key <= previous_key:
                raise ProjectDocumentValidationError("canonical document order is invalid")
            seen_source_ids.add(document.source_id)
            seen_pages.add(page_identity)
            previous_key = canonical_key
            documents.append(document)
    except (UnicodeError, json.JSONDecodeError):
        raise ProjectDocumentValidationError("canonical corpus is malformed") from None
    if not documents:
        raise ProjectDocumentValidationError("project corpus is empty")
    return documents


def _display_source_name(project_root: Path, source_path: Path) -> str:
    raw_root = project_root / "raw_uploads"
    relative = source_path.relative_to(raw_root)
    if relative.parts and relative.parts[0] == "extracted":
        relative = Path(*relative.parts[1:])
    return _normalized_source_name(relative.as_posix())


def _parse_pdf_with_mineru(source_path: Path, output_dir: Path):
    """Parse one project PDF with layout-aware MinerU and no text-only fallback."""
    from source.services.document_parser import DocumentParserService

    try:
        parser = DocumentParserService(output_dir=output_dir, force_backend="mineru")
        return asyncio.run(
            parser.parse_pdf(
                str(source_path),
                source_path.name,
                allow_fallback=False,
            )
        )
    except ProjectDocumentStorageError:
        raise
    except Exception:
        raise ProjectDocumentStorageError("MinerU PDF extraction is unavailable") from None


def _mineru_page_text(page: Any) -> str:
    parts = [str(getattr(page, "content", "") or "").strip()]
    for table in getattr(page, "tables", []) or []:
        if not isinstance(table, Mapping):
            continue
        table_text = str(
            table.get("table_body")
            or table.get("table_caption")
            or table.get("text")
            or table.get("content")
            or ""
        ).strip()
        if table_text and table_text not in parts[0]:
            parts.append(table_text)
    image_count = len(getattr(page, "images", []) or [])
    if image_count:
        parts.append(f"MinerU image regions: {image_count}")
    return "\n\n".join(part for part in parts if part).strip()


def extract_project_documents(project_root: Path, source_paths: Iterable[Path]) -> list[ProjectDocument]:
    project_root = Path(project_root).resolve()
    raw_root = project_root / "raw_uploads"
    if raw_root.resolve() != raw_root or not raw_root.is_dir():
        raise ProjectDocumentValidationError("raw upload directory is invalid")
    documents: list[ProjectDocument] = []
    seen_source_names: set[str] = set()
    ordered_source_paths = sorted(
        (Path(source_path) for source_path in source_paths),
        key=lambda source_path: _display_source_name(project_root, source_path),
    )
    for source_path in ordered_source_paths:
        resolved = source_path.resolve()
        if resolved != source_path or raw_root not in resolved.parents or source_path.is_symlink():
            raise ProjectDocumentValidationError("source path escapes the project")
        suffix = source_path.suffix.lower()
        if suffix not in _ALLOWED_SOURCE_SUFFIXES or not source_path.is_file():
            raise ProjectDocumentValidationError("unsupported canonical document source")
        source_name = _display_source_name(project_root, source_path)
        if source_name in seen_source_names:
            raise ProjectDocumentValidationError("duplicate canonical source name")
        seen_source_names.add(source_name)
        if suffix == ".txt":
            try:
                text = source_path.read_bytes().decode("utf-8-sig")
            except (OSError, UnicodeError):
                raise ProjectDocumentValidationError("text source is unreadable") from None
            documents.append(
                ProjectDocument(
                    source_name=source_name,
                    text=text,
                    metadata={"page_number": "1", "parser_method": "utf8_text_v1"},
                )
            )
            continue
        parsed = _parse_pdf_with_mineru(
            source_path,
            project_root / "extracted_text" / "mineru",
        )
        pages = getattr(parsed, "content", None)
        metadata = getattr(parsed, "metadata", None)
        parser_method = metadata.get("parsing_method") if isinstance(metadata, Mapping) else None
        if not isinstance(parser_method, str) or not parser_method.strip():
            parser_method = "MinerU"
        if not isinstance(pages, list) or not pages:
            raise ProjectDocumentValidationError("MinerU PDF extraction returned no pages")
        for index, page in enumerate(pages, 1):
            raw_page_number = getattr(page, "page_number", index - 1)
            try:
                page_number = int(raw_page_number) + 1
            except (TypeError, ValueError):
                page_number = index
            text = _mineru_page_text(page)
            if not text:
                raise ProjectDocumentValidationError("MinerU PDF page has no searchable content")
            documents.append(
                ProjectDocument(
                    source_name=source_name,
                    text=text,
                    metadata={
                        "page_number": str(page_number),
                        "parser_method": parser_method,
                    },
                )
            )
    if not documents:
        raise ProjectDocumentValidationError("project has no canonical document sources")
    return documents
