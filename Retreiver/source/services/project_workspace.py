"""Strict, project-scoped filesystem paths and bounded atomic uploads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Callable, IO
from uuid import RFC_4122, UUID, uuid4
import zipfile
import zlib


_RESOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,79}_[0-9a-f]{32}$")
_LEGACY_PROJECT_ID_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9.-]{0,79}_[0-9]{8}_[0-9]{6}_[0-9a-f]{6,32}$"
)
_ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".zip", ".csv", ".xlsx", ".txt", ".md", ".json"}
_ALLOWED_ARCHIVE_ENTRY_EXTENSIONS = _ALLOWED_UPLOAD_EXTENSIONS - {".zip"}
_XLSX_ALLOWED_ROOTS = {"_rels", "customXml", "docProps", "xl"}
_XLSX_ALLOWED_ENTRY_EXTENSIONS = {
    ".bmp",
    ".emf",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".rels",
    ".svg",
    ".tif",
    ".tiff",
    ".vml",
    ".wmf",
    ".xml",
}
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_COPY_CHUNK_BYTES = 64 * 1024
_RUN_CHILDREN = ("logs", "chunks", "indexes", "retrieval", "reranking")
UploadFinalizer = Callable[[Path, Path, dict], dict]


class UploadError(Exception):
    """Base class for upload failures with a safe public message."""

    public_message = "Upload could not be processed"


class UnsupportedUploadError(UploadError):
    """The request names an unsupported upload media type."""

    public_message = "Unsupported upload type"


class UploadValidationError(UploadError):
    """The upload does not satisfy the bounded upload contract."""

    public_message = "Upload validation failed"


class UploadTooLargeError(UploadError):
    """The request body exceeds the configured upload byte cap."""

    public_message = "Upload exceeds the configured size limit"


class UploadStorageError(UploadError):
    """The upload could not be durably stored."""

    public_message = "Upload could not be stored"


@dataclass(frozen=True)
class UploadLimits:
    """Resource limits applied before and during upload publication."""

    max_upload_bytes: int = 100 * 1024 * 1024
    max_json_bytes: int = 1024 * 1024
    max_zip_entries: int = 500
    max_zip_entry_bytes: int = 100 * 1024 * 1024
    max_zip_expanded_bytes: int = 500 * 1024 * 1024
    max_zip_ratio: int = 100
    max_zip_depth: int = 8

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("Upload limits must be nonnegative")


@dataclass
class _ArchiveBudget:
    """Upload-wide resource use shared by outer ZIPs and nested XLSX files."""

    entries: int = 0
    expanded_bytes: int = 0


def safe_slug(value: str) -> str:
    """Return a bounded, filesystem-safe slug for a project label."""
    normalized = re.sub(r"[^a-z0-9.-]+", "-", value.lower()).strip(".-")
    slug = normalized[:40].strip(".-")
    return slug or "project"


def validate_resource_id(value: str) -> str:
    """Return a valid resource ID, rejecting all other values."""
    if not isinstance(value, str) or _RESOURCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid resource ID: {value!r}")
    resource_uuid = UUID(hex=value.rsplit("_", 1)[1])
    if resource_uuid.variant != RFC_4122 or resource_uuid.version != 4:
        raise ValueError(f"Invalid resource ID: {value!r}")
    return value


def _is_zip_signature(prefix: bytes) -> bool:
    return any(prefix.startswith(signature) for signature in _ZIP_SIGNATURES)


def _safe_original_name(original_name: str) -> str:
    if not isinstance(original_name, str) or not original_name or "\x00" in original_name:
        raise UploadValidationError(UploadValidationError.public_message)
    if (
        original_name.startswith(("/", "\\"))
        or "/" in original_name
        or "\\" in original_name
        or re.match(r"^[A-Za-z]:", original_name)
        or original_name in {".", ".."}
    ):
        raise UploadValidationError(UploadValidationError.public_message)
    return original_name


def _validate_xlsx(
    source: bytes | Path,
    limits: UploadLimits,
    budget: _ArchiveBudget | None = None,
) -> None:
    _validate_zip_container(
        source,
        limits,
        _validate_xlsx_entry,
        require_xlsx_signature=True,
        budget=budget,
    )


def _validate_json_bytes(content: bytes, max_json_bytes: int) -> None:
    if len(content) > max_json_bytes:
        raise UploadValidationError(UploadValidationError.public_message)
    try:
        json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UploadValidationError(UploadValidationError.public_message) from None


def _validate_payload_signature(name: str, content: bytes, limits: UploadLimits) -> None:
    suffix = Path(name).suffix.lower()
    prefix = content[:8]
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError(UploadValidationError.public_message)
        return
    if suffix == ".zip":
        if not _is_zip_signature(prefix):
            raise UploadValidationError(UploadValidationError.public_message)
        return
    if suffix == ".xlsx":
        if not _is_zip_signature(prefix):
            raise UploadValidationError(UploadValidationError.public_message)
        _validate_xlsx(content, limits)
        return
    if content.startswith(b"%PDF-") or _is_zip_signature(prefix) or b"\x00" in content:
        raise UploadValidationError(UploadValidationError.public_message)
    if suffix == ".json":
        _validate_json_bytes(content, limits.max_json_bytes)
        return
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadValidationError(UploadValidationError.public_message) from None
    if any(ord(character) < 32 and character not in "\t\r\n" for character in decoded):
        raise UploadValidationError(UploadValidationError.public_message)


def _validate_extracted_file(
    path: Path,
    suffix: str,
    limits: UploadLimits,
    budget: _ArchiveBudget | None = None,
) -> None:
    if suffix == ".xlsx":
        _validate_xlsx(path, limits, budget)
        return
    try:
        content = path.read_bytes()
    except OSError:
        raise UploadStorageError(UploadStorageError.public_message) from None
    _validate_payload_signature(path.name, content, limits)


def _normalized_zip_parts(filename: str, max_depth: int) -> tuple[str, ...]:
    if not filename or "\x00" in filename:
        raise UploadValidationError(UploadValidationError.public_message)
    normalized_slashes = filename.replace("\\", "/")
    if normalized_slashes.startswith("/") or re.match(r"^[A-Za-z]:", normalized_slashes):
        raise UploadValidationError(UploadValidationError.public_message)
    parts: list[str] = []
    for part in normalized_slashes.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise UploadValidationError(UploadValidationError.public_message)
        parts.append(part)
    if not parts or len(parts) > max_depth:
        raise UploadValidationError(UploadValidationError.public_message)
    return tuple(parts)


def _zip_info_has_unsafe_type(info: zipfile.ZipInfo, is_directory: bool) -> bool:
    if info.create_system != 3:
        return False
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type == 0:
        return False
    expected_type = stat.S_IFDIR if is_directory else stat.S_IFREG
    return file_type != expected_type


_ZipEntryPolicy = Callable[[tuple[str, ...], bool], None]


def _validate_xlsx_entry(parts: tuple[str, ...], is_directory: bool) -> None:
    normalized = PurePosixPath(*parts)
    if normalized in {
        PurePosixPath("[Content_Types].xml"),
        PurePosixPath("_rels/.rels"),
    }:
        if is_directory:
            raise UploadValidationError(UploadValidationError.public_message)
        return
    if parts[0] not in _XLSX_ALLOWED_ROOTS:
        raise UploadValidationError(UploadValidationError.public_message)
    if not is_directory and Path(parts[-1]).suffix.lower() not in _XLSX_ALLOWED_ENTRY_EXTENSIONS:
        raise UploadValidationError(UploadValidationError.public_message)


def _preflight_zip_container(
    archive: zipfile.ZipFile,
    limits: UploadLimits,
    entry_policy: _ZipEntryPolicy,
    budget: _ArchiveBudget | None = None,
) -> tuple[list[tuple[zipfile.ZipInfo, tuple[str, ...]]], set[PurePosixPath]]:
    budget = budget or _ArchiveBudget()
    infos = archive.infolist()
    if (
        len(infos) > limits.max_zip_entries
        or budget.entries + len(infos) > limits.max_zip_entries
    ):
        raise UploadValidationError(UploadValidationError.public_message)
    budget.entries += len(infos)

    destinations: set[PurePosixPath] = set()
    files: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    expanded_metadata = 0
    for info in infos:
        is_directory = info.is_dir() or info.filename.endswith(("/", "\\"))
        if _zip_info_has_unsafe_type(info, is_directory) or info.flag_bits & 0x1:
            raise UploadValidationError(UploadValidationError.public_message)
        parts = _normalized_zip_parts(info.filename, limits.max_zip_depth)
        normalized = PurePosixPath(*parts)
        if normalized in destinations:
            raise UploadValidationError(UploadValidationError.public_message)
        destinations.add(normalized)

        entry_policy(parts, is_directory)
        if is_directory:
            continue
        if info.file_size < 0 or info.compress_size < 0:
            raise UploadValidationError(UploadValidationError.public_message)
        if info.file_size > limits.max_zip_entry_bytes:
            raise UploadValidationError(UploadValidationError.public_message)
        expanded_metadata += info.file_size
        if expanded_metadata > limits.max_zip_expanded_bytes:
            raise UploadValidationError(UploadValidationError.public_message)
        if info.file_size and (
            info.compress_size == 0
            or info.file_size > limits.max_zip_ratio * info.compress_size
        ):
            raise UploadValidationError(UploadValidationError.public_message)
        if info.file_size == 0 and info.CRC != 0:
            raise UploadValidationError(UploadValidationError.public_message)
        files.append((info, parts))
    return files, destinations


def _validate_zip_container(
    source: bytes | Path,
    limits: UploadLimits,
    entry_policy: _ZipEntryPolicy,
    *,
    require_xlsx_signature: bool = False,
    budget: _ArchiveBudget | None = None,
) -> None:
    budget = budget or _ArchiveBudget()
    try:
        with zipfile.ZipFile(BytesIO(source) if isinstance(source, bytes) else source) as archive:
            files, _ = _preflight_zip_container(
                archive,
                limits,
                entry_policy,
                budget,
            )
            if require_xlsx_signature:
                file_destinations = {
                    PurePosixPath(*parts) for _, parts in files
                }
                required_members = {
                    PurePosixPath("[Content_Types].xml"),
                    PurePosixPath("_rels/.rels"),
                    PurePosixPath("xl/workbook.xml"),
                }
                if not required_members.issubset(file_destinations):
                    raise UploadValidationError(UploadValidationError.public_message)

            for info, _ in files:
                with archive.open(info, "r") as member:
                    budget.expanded_bytes += _copy_bounded(
                        member,
                        None,
                        expected_size=info.file_size,
                        entry_limit=limits.max_zip_entry_bytes,
                        cumulative_limit=limits.max_zip_expanded_bytes,
                        cumulative_start=budget.expanded_bytes,
                    )
    except UploadValidationError:
        raise
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
    ):
        raise UploadValidationError(UploadValidationError.public_message) from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _write_atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    try:
        _write_durable(temporary, encoded)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_bounded(
    source: IO[bytes],
    destination: Path | None,
    *,
    expected_size: int,
    entry_limit: int,
    cumulative_limit: int,
    cumulative_start: int,
) -> int:
    entry_bytes = 0
    output = destination.open("xb") if destination is not None else None
    try:
        while True:
            remaining_expected = expected_size - entry_bytes
            if remaining_expected == 0:
                break
            remaining_entry = entry_limit - entry_bytes
            remaining_cumulative = cumulative_limit - cumulative_start - entry_bytes
            read_size = min(
                _COPY_CHUNK_BYTES,
                remaining_expected,
                remaining_entry,
                remaining_cumulative,
            )
            if read_size <= 0:
                raise UploadValidationError(UploadValidationError.public_message)
            chunk = source.read(read_size)
            if not chunk:
                raise UploadValidationError(UploadValidationError.public_message)
            if len(chunk) > read_size:
                raise UploadValidationError(UploadValidationError.public_message)
            entry_bytes += len(chunk)
            if output is not None:
                output.write(chunk)
        if entry_bytes != expected_size:
            raise UploadValidationError(UploadValidationError.public_message)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()
    return entry_bytes


@dataclass(frozen=True)
class ProjectWorkspace:
    """Construct paths and publish uploads within one project's workspace."""

    base: Path
    limits: UploadLimits = field(default_factory=UploadLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", Path(self.base).resolve())

    def new_project_id(self, label: str) -> str:
        return f"{safe_slug(label)}_{uuid4().hex}"

    def new_run_id(self) -> str:
        return f"run_{uuid4().hex}"

    def _validate_run_id(self, run_id: str) -> str:
        validated = validate_resource_id(run_id)
        if not validated.startswith("run_"):
            raise ValueError("run_id is invalid")
        return validated

    def run_layout(self, project_id: str, run_id: str) -> dict[str, Path]:
        """Resolve one existing run without following project/run child symlinks."""
        run_id = self._validate_run_id(run_id)
        runs_root = self.layout(project_id)["runs"]
        root = runs_root / run_id
        try:
            root_metadata = root.lstat()
        except FileNotFoundError:
            raise FileNotFoundError("run does not exist") from None
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root.resolve(strict=True) != root
            or root.parent != runs_root
        ):
            raise ValueError("Run directory escapes the project workspace")
        result = {"root": root}
        for child_name in _RUN_CHILDREN:
            child = root / child_name
            try:
                child_metadata = child.lstat()
            except FileNotFoundError:
                raise ValueError("Run layout is incomplete") from None
            if (
                stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISDIR(child_metadata.st_mode)
                or child.resolve(strict=True) != child
                or child.parent != root
            ):
                raise ValueError("Run child escapes the run workspace")
            result[child_name] = child
        return result

    def create_run_layout(self, project_id: str, run_id: str) -> dict[str, Path]:
        """Exclusively create one project-owned run directory and canonical children."""
        run_id = self._validate_run_id(run_id)
        runs_root = self.layout(project_id)["runs"]
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(runs_root, flags)
        created = False
        try:
            opened_parent = os.fstat(parent_fd)
            lexical_parent = runs_root.lstat()
            if (
                not stat.S_ISDIR(opened_parent.st_mode)
                or stat.S_ISLNK(lexical_parent.st_mode)
                or (opened_parent.st_dev, opened_parent.st_ino)
                != (lexical_parent.st_dev, lexical_parent.st_ino)
                or runs_root.resolve(strict=True) != runs_root
            ):
                raise ValueError("Runs directory escapes the project workspace")
            os.mkdir(run_id, mode=0o700, dir_fd=parent_fd)
            created = True
            run_fd = os.open(run_id, flags, dir_fd=parent_fd)
            try:
                for child_name in _RUN_CHILDREN:
                    os.mkdir(child_name, mode=0o700, dir_fd=run_fd)
                os.fsync(run_fd)
            finally:
                os.close(run_fd)
            os.fsync(parent_fd)
        except Exception:
            if created:
                run_root = runs_root / run_id
                if run_root.exists() and not run_root.is_symlink():
                    shutil.rmtree(run_root)
                    try:
                        os.fsync(parent_fd)
                    except OSError:
                        pass
            raise
        finally:
            os.close(parent_fd)
        return self.run_layout(project_id, run_id)

    def new_question_set_id(self) -> str:
        return f"questions_{uuid4().hex}"

    def create_question_set(
        self,
        project_id: str,
        filename: str,
        content: bytes,
        *,
        max_questions: int | None = None,
    ) -> dict:
        """Parse and atomically publish an immutable project-owned question set."""
        from source.services.project_questions import create_question_set

        project_id = validate_resource_id(project_id)
        question_set_id = self.new_question_set_id()
        result = create_question_set(
            self.layout(project_id)["questions"],
            question_set_id,
            filename,
            content,
            max_questions=max_questions,
            upload_limits=self.limits,
        )
        return {"project_id": project_id, **result}

    def load_question_set(
        self,
        project_id: str,
        question_set_id: str,
        *,
        expected_content_sha256: str,
    ):
        """Load one immutable question set, failing closed across project boundaries."""
        from source.services.project_questions import load_question_set

        project_id = validate_resource_id(project_id)
        question_set_id = validate_resource_id(question_set_id)
        return load_question_set(
            self.layout(project_id)["questions"],
            question_set_id,
            expected_content_sha256=expected_content_sha256,
        )

    def project_root(self, project_id: str) -> Path:
        project_id = validate_resource_id(project_id)
        lexical_root = self.base / project_id
        root = lexical_root.resolve()
        if root != lexical_root or root.parent != self.base:
            raise ValueError("Project root must remain directly beneath workspace base")
        return root

    def _layout_for_root(self, root: Path) -> dict[str, Path]:
        layout = {
            "raw_uploads": root / "raw_uploads",
            "extracted_text": root / "extracted_text",
            "chunks": root / "chunks",
            "questions": root / "questions",
            "indexes": root / "indexes",
            "runs": root / "runs",
        }
        for child in layout.values():
            resolved_child = child.resolve()
            if resolved_child != child or resolved_child.parent != root:
                raise ValueError("Project layout path must remain directly beneath project root")
        documents = layout["extracted_text"] / "documents.jsonl"
        resolved_documents = documents.resolve()
        if (
            resolved_documents != documents
            or resolved_documents.parent != layout["extracted_text"]
        ):
            raise ValueError("Canonical project corpus path is invalid")
        return {"root": root, **layout, "documents": documents}

    def layout(self, project_id: str) -> dict[str, Path]:
        return self._layout_for_root(self.project_root(project_id))

    def _existing_project_root(self, project_id: str) -> Path:
        try:
            return self.project_root(project_id)
        except ValueError:
            if (
                not isinstance(project_id, str)
                or _LEGACY_PROJECT_ID_PATTERN.fullmatch(project_id) is None
            ):
                raise
        lexical_root = self.base / project_id
        root = lexical_root.resolve()
        if root != lexical_root or root.parent != self.base or lexical_root.is_symlink():
            raise ValueError("Project root must remain directly beneath workspace base")
        return root

    def _existing_layout(self, project_id: str) -> dict[str, Path]:
        return self._layout_for_root(self._existing_project_root(project_id))

    def lexical_preview(self, project_id: str, query: str, top_k: int = 5) -> dict:
        """Run project-local token-overlap search and atomically persist its evidence."""
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id is required")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if len(query) > 4000:
            raise ValueError("query exceeds 4000 characters")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")

        layout = self._existing_layout(project_id)
        root = layout["root"]
        if not root.is_dir():
            raise FileNotFoundError("project does not exist")
        manifest_path = root / "manifest.json"
        chunks: list[dict[str, Any]] | None = None
        if manifest_path.resolve() == manifest_path and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise UploadValidationError(UploadValidationError.public_message) from None
            corpus_sha256 = manifest.get("corpus_sha256") if isinstance(manifest, dict) else None
            if (
                isinstance(corpus_sha256, str)
                and manifest.get("extraction_status") == "complete"
            ):
                from source.services.project_documents import load_project_documents

                documents = load_project_documents(
                    layout["documents"],
                    expected_sha256=corpus_sha256,
                )
                chunks = [
                    {
                        "id": document.source_id,
                        "chunk_id": document.source_id,
                        "pdf_name": document.source_name,
                        "paragraph": document.text,
                        "page_number": document.metadata["page_number"],
                        "source_type": "canonical_project_document",
                        "parser_method": document.metadata["parser_method"],
                    }
                    for document in documents
                ]
        if chunks is None:
            index_path = root / "search_index.json"
            if index_path.resolve() != index_path or not index_path.is_file():
                raise ValueError("project has no canonical corpus")
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise UploadValidationError(UploadValidationError.public_message) from None
            legacy_chunks = index.get("chunks") if isinstance(index, dict) else None
            if not isinstance(legacy_chunks, list) or not legacy_chunks:
                raise ValueError("project has no indexed text")
            chunks = legacy_chunks

        query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]{3,}", query)}

        def score(paragraph: str) -> float:
            text_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]{3,}", paragraph)}
            if not query_terms or not text_terms:
                return 0.0
            overlap = len(query_terms & text_terms)
            return overlap / len(query_terms) + min(0.25, overlap / len(text_terms))

        hits: list[dict] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            paragraph = chunk.get("paragraph", "")
            if not isinstance(paragraph, str):
                continue
            hits.append(
                {
                    "rank": 0,
                    "score": round(score(paragraph), 6),
                    "pdf_name": str(chunk.get("pdf_name", "")),
                    "chunk_id": chunk.get("chunk_id", chunk.get("id", "")),
                    "paragraph": paragraph,
                    "page_number": chunk.get("page_number", ""),
                    "source_type": str(chunk.get("source_type", "uploaded_document")),
                    "parser_method": str(chunk.get("parser_method", "dashboard_upload_text_index")),
                }
            )
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        hits = hits[:top_k]
        for rank, hit in enumerate(hits, 1):
            hit["rank"] = rank

        run_id = self.new_run_id()
        created_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "ok": True,
            "mode": "lexical_preview",
            "project_id": project_id,
            "run_id": run_id,
            "created_at": created_at,
            "query": query,
            "top_k": top_k,
            "hits": hits,
            "message": (
                "Lexical preview only. Embedding, vector database, and reranker "
                "selections were not executed."
            ),
        }
        run_manifest = {
            "project_id": project_id,
            "run_id": run_id,
            "created_at": created_at,
            "query": query,
            "top_k": top_k,
            "mode": "lexical_preview",
            "status": "completed",
            "outputs": ["manifest.json", "evidence.json"],
        }

        runs = layout["runs"]
        if not runs.is_dir() or runs.is_symlink() or runs.resolve() != runs:
            raise UploadStorageError(UploadStorageError.public_message)
        final_run = runs / run_id
        staging = runs / f".staging-{uuid4().hex}"
        staging_created = False
        final_published = False
        try:
            if final_run.exists() or final_run.is_symlink():
                raise UploadStorageError(UploadStorageError.public_message)
            staging.mkdir(mode=0o700)
            staging_created = True
            _write_atomic_json(staging / "evidence.json", payload)
            _write_atomic_json(staging / "manifest.json", run_manifest)
            _fsync_directory(staging)
            os.replace(staging, final_run)
            staging_created = False
            final_published = True
            _fsync_directory(runs)
            final_published = False
        except UploadError:
            raise
        except Exception:
            raise UploadStorageError(UploadStorageError.public_message) from None
        finally:
            if staging_created and staging.parent == runs:
                if staging.is_symlink():
                    staging.unlink(missing_ok=True)
                elif staging.exists():
                    shutil.rmtree(staging)
            if final_published and final_run.parent == runs:
                if final_run.is_symlink():
                    final_run.unlink(missing_ok=True)
                elif final_run.exists():
                    shutil.rmtree(final_run)
        return payload

    def _preflight_archive(
        self,
        archive: zipfile.ZipFile,
        budget: _ArchiveBudget,
    ) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
        def validate_outer_entry(parts: tuple[str, ...], is_directory: bool) -> None:
            if is_directory:
                return
            suffix = Path(parts[-1]).suffix.lower()
            if suffix == ".zip" or suffix not in _ALLOWED_ARCHIVE_ENTRY_EXTENSIONS:
                raise UploadValidationError(UploadValidationError.public_message)
        files, _ = _preflight_zip_container(
            archive,
            self.limits,
            validate_outer_entry,
            budget,
        )
        for info, parts in files:
            if Path(parts[-1]).suffix.lower() == ".json" and info.file_size > self.limits.max_json_bytes:
                raise UploadValidationError(UploadValidationError.public_message)
        return files

    def _extract_zip(self, archive_path: Path, raw_uploads: Path) -> list[str]:
        extract_root = raw_uploads / "extracted"
        extract_root.mkdir()
        if extract_root.resolve() != extract_root or extract_root.parent != raw_uploads:
            raise UploadValidationError(UploadValidationError.public_message)

        try:
            with zipfile.ZipFile(archive_path) as archive:
                budget = _ArchiveBudget()
                files = self._preflight_archive(archive, budget)
                extracted: list[str] = []
                for info, parts in files:
                    destination = extract_root.joinpath(*parts)
                    resolved_destination = destination.resolve()
                    if extract_root not in resolved_destination.parents:
                        raise UploadValidationError(UploadValidationError.public_message)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.parent.resolve() != destination.parent:
                        raise UploadValidationError(UploadValidationError.public_message)
                    with archive.open(info, "r") as source:
                        written = _copy_bounded(
                            source,
                            destination,
                            expected_size=info.file_size,
                            entry_limit=self.limits.max_zip_entry_bytes,
                            cumulative_limit=self.limits.max_zip_expanded_bytes,
                            cumulative_start=budget.expanded_bytes,
                        )
                    budget.expanded_bytes += written
                    _validate_extracted_file(
                        destination,
                        destination.suffix.lower(),
                        self.limits,
                        budget,
                    )
                    extracted.append(str(PurePosixPath("raw_uploads", "extracted", *parts)))
                return extracted
        except UploadError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error):
            raise UploadValidationError(UploadValidationError.public_message) from None

    def create_upload(
        self,
        original_name: str,
        content: bytes,
        label: str = "",
        finalizer: UploadFinalizer | None = None,
    ) -> dict:
        """Validate, finalize in staging, and atomically publish one upload."""
        original = _safe_original_name(original_name)
        suffix = Path(original).suffix.lower()
        if suffix not in _ALLOWED_UPLOAD_EXTENSIONS:
            raise UnsupportedUploadError(UnsupportedUploadError.public_message)
        if not isinstance(content, bytes):
            raise UploadValidationError(UploadValidationError.public_message)
        if len(content) > self.limits.max_upload_bytes:
            raise UploadTooLargeError(UploadTooLargeError.public_message)
        _validate_payload_signature(original, content, self.limits)
        required_disk_bytes = len(content)
        if suffix == ".zip":
            required_disk_bytes += self.limits.max_zip_expanded_bytes
        try:
            free_disk_bytes = shutil.disk_usage(self.base.parent).free
        except OSError:
            raise UploadStorageError(UploadStorageError.public_message) from None
        if free_disk_bytes < required_disk_bytes:
            raise UploadStorageError(UploadStorageError.public_message)

        project_label = label or Path(original).stem
        project_id = self.new_project_id(project_label)
        final_root = self.project_root(project_id)
        staging = self.base / f".staging-{uuid4().hex}"
        staging_created = False
        final_published = False
        try:
            self.base.mkdir(parents=True, exist_ok=True)
            if self.base.resolve() != self.base or not self.base.is_dir():
                raise UploadStorageError(UploadStorageError.public_message)
            if staging.resolve() != staging or staging.parent != self.base:
                raise UploadStorageError(UploadStorageError.public_message)
            staging.mkdir(mode=0o700)
            staging_created = True

            child_names = ("raw_uploads", "extracted_text", "chunks", "questions", "indexes", "runs")
            for child_name in child_names:
                child = staging / child_name
                child.mkdir()
                if child.resolve() != child or child.parent != staging:
                    raise UploadStorageError(UploadStorageError.public_message)

            saved_path = staging / "raw_uploads" / original
            _write_durable(saved_path, content)
            extracted = self._extract_zip(saved_path, staging / "raw_uploads") if suffix == ".zip" else []
            manifest = {
                "ok": True,
                "project_id": project_id,
                "label": project_label,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "saved_path": str(PurePosixPath("raw_uploads", original)),
                "bytes": len(content),
                "extracted_files": extracted,
                "extracted_count": len(extracted),
                "storage_layout": {
                    "root": ".",
                    **{name: name for name in child_names},
                },
            }
            if finalizer is not None:
                finalized = finalizer(staging, final_root, dict(manifest))
                if (
                    not isinstance(finalized, dict)
                    or finalized.get("project_id") != project_id
                ):
                    raise UploadStorageError(UploadStorageError.public_message)
                manifest = finalized
            _write_atomic_json(staging / "manifest.json", manifest)
            _fsync_directory(staging)
            if final_root.exists() or final_root.is_symlink():
                raise UploadStorageError(UploadStorageError.public_message)
            os.replace(staging, final_root)
            staging_created = False
            final_published = True
            _fsync_directory(self.base)
            final_published = False
            return manifest
        except UploadError:
            raise
        except Exception:
            raise UploadStorageError(UploadStorageError.public_message) from None
        finally:
            if staging_created and staging.parent == self.base:
                if staging.is_symlink():
                    staging.unlink(missing_ok=True)
                elif staging.exists():
                    shutil.rmtree(staging)
            if final_published and final_root.parent == self.base:
                if final_root.is_symlink():
                    final_root.unlink(missing_ok=True)
                elif final_root.exists():
                    shutil.rmtree(final_root)
