"""Real, project-scoped selected-matrix execution with atomic run artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import StringIO
import itertools
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from benchmarking.core.registry import default_registry
from benchmarking.core.schemas import Chunk, SearchHit
from source.services.project_chunking import chunk_project_documents
from source.services.project_documents import load_project_documents
from source.services.project_matrix_contract import (
    ProjectMatrixRequest,
    ValidatedMatrix,
    validate_project_matrix_request,
)
from source.services.project_questions import (
    ProjectQuestion,
    parse_typed_question,
)
from source.services.project_relevance import (
    hit_is_relevant,
    label_applies,
    relevant_corpus_count,
)
from source.services.project_run_metrics import QueryMetricInput, summarize_query_metrics
from source.services.project_workspace import ProjectWorkspace


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG = _REPO_ROOT / "configs" / "project_matrix_catalog.json"
_RETRIEVAL_ONLY_RERANKER_ID = "none"
_REQUEST_ENVELOPE_KEYS = {
    "schema_version",
    "request",
    "request_fingerprint",
    "combination_count",
}
_SUMMARY_FIELDS = (
    "project_id",
    "run_id",
    "combo_id",
    "status",
    "summary_schema_version",
    "chunker_id",
    "embedding_id",
    "vector_store_id",
    "reranker_id",
    "physical_namespace",
    "query_count",
    "labelled_queries",
    "unlabelled_queries",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "ndcg_at_5",
    "avg_first_relevant_rank",
    "no_hit_queries",
    "retrieval_latency_s",
    "rerank_latency_s",
    "avg_query_latency_s",
    "evidence_count",
    "embedding_input_tokens",
    "embedding_usage_scope",
    "embedding_usage_key",
    "rerank_search_units",
    "rerank_usage_scope",
    "error_code",
    "error_detail",
)
_SUMMARY_SCHEMA_VERSION = 3
_EMPTY_QUALITY = {
    "recall_at_k": None,
    "mrr_at_k": None,
    "ndcg_at_k": None,
    "recall_at_1": None,
    "recall_at_3": None,
    "recall_at_5": None,
    "recall_at_10": None,
    "precision_at_5": None,
    "ndcg_at_5": None,
    "avg_first_relevant_rank": None,
    "no_hit_queries": None,
}


class ProjectMatrixRunnerError(RuntimeError):
    """A run cannot be safely loaded or executed."""


class ProjectMatrixRequestError(ValueError):
    """The immutable run request is invalid or no longer matches its fingerprint."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _config_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_regular_bytes(path: Path) -> bytes:
    """Read one regular file without following its leaf symlink."""
    path = Path(path)
    parent = path.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_metadata = parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise ProjectMatrixRunnerError("run storage is unavailable")
        parent_fd = os.open(parent, flags)
        try:
            opened_parent = os.fstat(parent_fd)
            if (
                (opened_parent.st_dev, opened_parent.st_ino)
                != (parent_metadata.st_dev, parent_metadata.st_ino)
                or parent.resolve(strict=True) != parent
            ):
                raise ProjectMatrixRunnerError("run storage changed during loading")
            leaf = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(leaf.st_mode) or not stat.S_ISREG(leaf.st_mode):
                raise ProjectMatrixRunnerError("run input must be a regular file")
            descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                opened_leaf = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened_leaf.st_mode)
                    or (opened_leaf.st_dev, opened_leaf.st_ino)
                    != (leaf.st_dev, leaf.st_ino)
                ):
                    raise ProjectMatrixRunnerError("run input changed during loading")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
    except (ProjectMatrixRunnerError, FileNotFoundError):
        raise
    except OSError:
        raise ProjectMatrixRunnerError("run input is unavailable") from None


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one runner-owned artifact through a held no-follow parent FD."""
    path = Path(path)
    parent = path.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    temporary_name = f".{path.name}.{uuid4().hex}.tmp"
    parent_fd = -1
    try:
        parent_metadata = parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise ProjectMatrixRunnerError("run artifact directory is unavailable")
        parent_fd = os.open(parent, flags)
        opened_parent = os.fstat(parent_fd)
        if (
            (opened_parent.st_dev, opened_parent.st_ino)
            != (parent_metadata.st_dev, parent_metadata.st_ino)
            or parent.resolve(strict=True) != parent
        ):
            raise ProjectMatrixRunnerError("run artifact directory changed")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(content)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short artifact write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    except ProjectMatrixRunnerError:
        raise
    except OSError:
        raise ProjectMatrixRunnerError("run artifact could not be published") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(parent_fd)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(payload) + b"\n")


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


_EVIDENCE_PART_MAX_BYTES = 4 * 1024 * 1024


def _write_evidence_index(
    run_layout: Mapping[str, Path],
    *,
    project_id: str,
    run_id: str,
    mode: str,
    summary_rows: Iterable[Mapping[str, Any]],
    evidence_rows: Iterable[Mapping[str, Any]],
) -> str:
    rows_by_combo: dict[str, list[Mapping[str, Any]]] = {}
    for evidence_row in evidence_rows:
        rows_by_combo.setdefault(str(evidence_row.get("combo_id") or ""), []).append(evidence_row)
    entries: list[dict[str, Any]] = []
    for summary in summary_rows:
        combo_id = str(summary["combo_id"])
        serialized = [_canonical_bytes(row) + b"\n" for row in rows_by_combo.get(combo_id, [])]
        parts: list[dict[str, Any]] = []
        current: list[bytes] = []
        current_size = 0

        def flush_part() -> None:
            nonlocal current, current_size
            if not current:
                return
            part_number = len(parts)
            relative_path = f"reranking/{combo_id}.evidence.{part_number:05d}.jsonl"
            content = b"".join(current)
            _atomic_write(run_layout["root"] / relative_path, content)
            parts.append(
                {
                    "path": relative_path,
                    "row_count": len(current),
                    "size_bytes": len(content),
                    "sha256": _sha256_bytes(content),
                }
            )
            current = []
            current_size = 0

        for line in serialized:
            if len(line) > _EVIDENCE_PART_MAX_BYTES:
                raise ValueError("one evidence row exceeds the indexed part limit")
            if current and current_size + len(line) > _EVIDENCE_PART_MAX_BYTES:
                flush_part()
            current.append(line)
            current_size += len(line)
        flush_part()
        entries.append(
            {
                "combo_id": combo_id,
                "row_count": len(serialized),
                "parts": parts,
            }
        )
    payload = {
        "schema_version": 1,
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "entries": entries,
    }
    content = _canonical_bytes(payload) + b"\n"
    _atomic_write(run_layout["root"] / "evidence_index.json", content)
    return _sha256_bytes(content)


def _load_catalog(path: Path) -> tuple[dict[str, Any], str]:
    content = _read_regular_bytes(path)
    try:
        catalog = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ProjectMatrixRunnerError("project matrix catalog is invalid") from None
    if not isinstance(catalog, dict) or not isinstance(catalog.get("techniques"), dict):
        raise ProjectMatrixRunnerError("project matrix catalog is invalid")
    return catalog, _sha256_bytes(content)


def production_adapter_classes(
    catalog_path: str | Path = _DEFAULT_CATALOG,
) -> dict[str, dict[str, type[Any]]]:
    """Resolve every canonical provider ID to the production registry class."""
    catalog, _ = _load_catalog(Path(catalog_path))
    registry = default_registry()
    output: dict[str, dict[str, type[Any]]] = {}
    kind_map = {
        "embeddings": "embedding",
        "vector_stores": "vector_store",
        "rerankers": "reranker",
    }
    for dimension, kind in kind_map.items():
        configurations = catalog["techniques"].get(dimension)
        if not isinstance(configurations, dict):
            raise ProjectMatrixRunnerError("project matrix catalog is invalid")
        output[dimension] = {}
        for canonical_id, configuration in configurations.items():
            if not isinstance(configuration, dict) or not isinstance(
                configuration.get("adapter"), str
            ):
                raise ProjectMatrixRunnerError("project matrix catalog is invalid")
            adapter_class = registry.get(kind, configuration["adapter"])
            if not isinstance(adapter_class, type):
                raise ProjectMatrixRunnerError("production adapter registry is invalid")
            output[dimension][canonical_id] = adapter_class
    return output


def _adapter_kwargs(configuration: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"adapter", "license", "execution_location", "provider_ready_env"}
    return {key: value for key, value in configuration.items() if key not in ignored}


def _chunk_row(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "pdf_name": chunk.pdf_name,
        "paragraph": chunk.paragraph,
        "parent_id": chunk.parent_id,
        "metadata": dict(chunk.metadata or {}),
    }


def _combo_identity(
    project_id: str,
    run_id: str,
    chunker_id: str,
    embedding_id: str,
    vector_store_id: str,
    reranker_id: str,
) -> str:
    payload = {
        "project_id": project_id,
        "run_id": run_id,
        "chunker_id": chunker_id,
        "embedding_id": embedding_id,
        "vector_store_id": vector_store_id,
        "reranker_id": reranker_id,
    }
    return f"combo_{_sha256_bytes(_canonical_bytes(payload))[:24]}"


def _retrieval_identity(
    project_id: str,
    corpus_sha256: str,
    chunk_sha256: str,
    chunker_id: str,
    embedding_id: str,
    vector_store_id: str,
    embedding_config_sha256: str,
    vector_config_sha256: str,
) -> str:
    payload = {
        "project_id": project_id,
        "corpus_sha256": corpus_sha256,
        "chunk_sha256": chunk_sha256,
        "chunker_id": chunker_id,
        "embedding_id": embedding_id,
        "vector_store_id": vector_store_id,
        "embedding_config_sha256": embedding_config_sha256,
        "vector_config_sha256": vector_config_sha256,
    }
    return f"project_{_sha256_bytes(_canonical_bytes(payload))[:40]}"


def _index_receipt_path(index_root: Path, namespace: str) -> Path:
    return index_root / f"index_{_sha256_bytes(namespace.encode('utf-8'))}.json"


def _load_index_receipt(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError, ProjectMatrixRunnerError):
        return None
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        return None
    dimensions = payload.get("dimensions")
    vector_count = payload.get("vector_count")
    if (
        type(dimensions) is not int
        or dimensions < 1
        or type(vector_count) is not int
        or vector_count < 1
        or not isinstance(payload.get("physical_namespace"), str)
        or not payload["physical_namespace"]
        or not isinstance(payload.get("embedding"), dict)
        or payload["embedding"].get("canonical_id") != expected.get("embedding_id")
        or payload["embedding"].get("dimensions") != dimensions
        or payload["embedding"].get("vector_count") != vector_count
    ):
        return None
    return payload


def _safe_error_code(stage: str) -> str:
    return {
        "chunking": "chunker_failed",
        "embedding": "embedding_failed",
        "vector_store": "vector_store_failed",
        "retrieval": "retrieval_failed",
        "reranker": "reranker_failed",
    }.get(stage, "combination_failed")


def _safe_error_detail(exc: BaseException) -> str:
    """Return a short operator-safe failure detail without free-form provider text."""
    name = type(exc).__name__
    if isinstance(exc, FileExistsError):
        return f"{name}: target already exists"
    if isinstance(exc, FileNotFoundError):
        return f"{name}: required path missing"
    if isinstance(exc, PermissionError):
        return f"{name}: permission denied"
    if isinstance(exc, TimeoutError):
        return f"{name}: timed out"
    if isinstance(exc, ConnectionError):
        return f"{name}: connection failed"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) is not None:
        return f"{name}: errno={exc.errno}"
    if isinstance(exc, ValueError):
        message = " ".join(str(exc).split())
        if (
            message
            and len(message) <= 80
            and "://" not in message
            and "/home/" not in message
            and "token" not in message.lower()
            and "key" not in message.lower()
        ):
            return f"{name}: {message}"
    return name


class ProjectMatrixRunner:
    """Execute one immutable selected matrix entirely inside one project run."""

    def __init__(
        self,
        workspace: ProjectWorkspace,
        *,
        catalog_path: str | Path = _DEFAULT_CATALOG,
    ) -> None:
        if not isinstance(workspace, ProjectWorkspace):
            raise TypeError("workspace must be a ProjectWorkspace")
        self.workspace = workspace
        self.catalog_path = Path(catalog_path).resolve()

    def _load_request(
        self, project_id: str, run_id: str, run_root: Path
    ) -> ValidatedMatrix:
        try:
            envelope = json.loads(
                _read_regular_bytes(run_root / "request.json").decode("utf-8")
            )
        except (UnicodeError, json.JSONDecodeError):
            raise ProjectMatrixRequestError("request.json is invalid") from None
        if not isinstance(envelope, dict) or set(envelope) != _REQUEST_ENVELOPE_KEYS:
            raise ProjectMatrixRequestError("request.json schema is invalid")
        if envelope["schema_version"] != 1 or isinstance(
            envelope["schema_version"], bool
        ):
            raise ProjectMatrixRequestError("request.json schema is invalid")
        validated = validate_project_matrix_request(
            envelope["request"], catalog_path=self.catalog_path
        )
        if validated.request.project_id != project_id:
            raise ProjectMatrixRequestError("request project_id does not match the run")
        if envelope["request_fingerprint"] != validated.request_fingerprint:
            raise ProjectMatrixRequestError("request fingerprint does not match request.json")
        if envelope["combination_count"] != validated.combination_count:
            raise ProjectMatrixRequestError("request combination count does not match")
        return validated

    def _load_documents(self, project_id: str) -> tuple[list[Any], dict[str, Any]]:
        layout = self.workspace.layout(project_id)
        manifest_path = layout["root"] / "manifest.json"
        try:
            manifest = json.loads(_read_regular_bytes(manifest_path).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ProjectMatrixRunnerError("project manifest is invalid") from None
        if (
            not isinstance(manifest, dict)
            or manifest.get("project_id") != project_id
            or manifest.get("extraction_status") != "complete"
            or not isinstance(manifest.get("corpus_sha256"), str)
            or len(manifest["corpus_sha256"]) != 64
        ):
            raise ProjectMatrixRunnerError("project extraction is incomplete")
        documents = load_project_documents(
            layout["documents"], expected_sha256=manifest["corpus_sha256"]
        )
        return documents, manifest

    def _load_questions(
        self, project_id: str, request: ProjectMatrixRequest
    ) -> list[ProjectQuestion]:
        if request.questions_source_type == "typed":
            assert request.typed_query is not None
            return parse_typed_question(request.typed_query)
        assert request.question_set_id is not None
        assert request.question_set_content_sha256 is not None
        return self.workspace.load_question_set(
            project_id,
            request.question_set_id,
            expected_content_sha256=request.question_set_content_sha256,
        )

    def _new_embedding(
        self,
        canonical_id: str,
        configuration: Mapping[str, Any],
        classes: Mapping[str, Mapping[str, type[Any]]],
    ) -> Any:
        return classes["embeddings"][canonical_id](
            model_name=canonical_id,
            **_adapter_kwargs(configuration),
        )

    def _new_vector_store(
        self,
        canonical_id: str,
        configuration: Mapping[str, Any],
        classes: Mapping[str, Mapping[str, type[Any]]],
        *,
        namespace: str,
        index_root: Path,
        load_existing: bool = False,
    ) -> Any:
        kwargs = _adapter_kwargs(configuration)
        if canonical_id == "FAISS":
            kwargs["index_root"] = str(index_root)
            kwargs["load_existing"] = load_existing
        return classes["vector_stores"][canonical_id](
            name=canonical_id,
            namespace=namespace,
            **kwargs,
        )

    def _new_reranker(
        self,
        canonical_id: str,
        configuration: Mapping[str, Any],
        classes: Mapping[str, Mapping[str, type[Any]]],
    ) -> Any:
        return classes["rerankers"][canonical_id](
            name=canonical_id,
            **_adapter_kwargs(configuration),
        )

    def run(self, project_id: str, run_id: str) -> dict[str, Any]:
        created_at = _utc_now()
        run_layout = self.workspace.run_layout(project_id, run_id)
        validated = self._load_request(project_id, run_id, run_layout["root"])
        request = validated.request
        documents, project_manifest = self._load_documents(project_id)
        project_index_root = self.workspace.layout(project_id)["indexes"]
        questions = self._load_questions(project_id, request)
        catalog, catalog_sha256 = _load_catalog(self.catalog_path)
        classes = production_adapter_classes(self.catalog_path)
        techniques = catalog["techniques"]

        chunks_by_id: dict[str, list[Chunk]] = {}
        chunk_artifacts: dict[str, dict[str, Any]] = {}
        chunk_failures: dict[str, str] = {}
        for chunker_id in request.chunkers:
            try:
                chunks = chunk_project_documents(documents, chunker_id)
            except Exception:
                chunk_failures[chunker_id] = "chunker_failed"
                continue
            content = _jsonl(_chunk_row(chunk) for chunk in chunks)
            relative = Path("chunks") / f"{chunker_id}.jsonl"
            _atomic_write(run_layout["root"] / relative, content)
            chunks_by_id[chunker_id] = chunks
            chunk_artifacts[chunker_id] = {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(content),
                "chunk_count": len(chunks),
            }
        _atomic_json(
            run_layout["chunks"] / "manifest.json",
            {
                "schema_version": 1,
                "project_id": project_id,
                "run_id": run_id,
                "corpus_sha256": project_manifest["corpus_sha256"],
                "artifacts": chunk_artifacts,
                "failures": chunk_failures,
            },
        )
        relevant_chunk_counts = {
            (chunker_id, question.question_id): relevant_corpus_count(
                question,
                chunks_by_id.get(chunker_id, ()),
                chunker_id,
            )
            for chunker_id in request.chunkers
            for question in questions
            if label_applies(question, chunker_id)
        }

        summary_rows: list[dict[str, Any]] = []
        details_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        progress_rows: list[dict[str, Any]] = []
        embedding_cache: dict[
            tuple[str, str],
            tuple[Any, list[list[float]], list[list[float]], dict[str, Any]],
        ] = {}
        reused_query_embedding_cache: dict[str, tuple[Any, list[list[float]]]] = {}
        retrieval_cache: dict[
            tuple[str, str, str],
            tuple[
                Any,
                dict[str, list[SearchHit]],
                dict[str, float],
                dict[str, Any],
            ],
        ] = {}

        selections = itertools.product(
            request.chunkers,
            request.embeddings,
            request.vector_stores,
            request.rerankers or (_RETRIEVAL_ONLY_RERANKER_ID,),
        )
        for chunker_id, embedding_id, vector_store_id, reranker_id in selections:
            combo_id = _combo_identity(
                project_id,
                run_id,
                chunker_id,
                embedding_id,
                vector_store_id,
                reranker_id,
            )
            chunks = chunks_by_id.get(chunker_id, [])
            chunk_artifact = chunk_artifacts.get(chunker_id)
            labelled_queries = sum(
                1 for question in questions if label_applies(question, chunker_id)
            )
            unlabelled_queries = len(questions) - labelled_queries
            retrieval_key = (chunker_id, embedding_id, vector_store_id)
            embedding_config = techniques["embeddings"][embedding_id]
            vector_config = techniques["vector_stores"][vector_store_id]
            chunk_sha256 = str((chunk_artifact or {}).get("sha256") or "")
            embedding_config_sha256 = _config_sha256(embedding_config)
            vector_config_sha256 = _config_sha256(vector_config)
            namespace = _retrieval_identity(
                project_id,
                project_manifest["corpus_sha256"],
                chunk_sha256,
                chunker_id,
                embedding_id,
                vector_store_id,
                embedding_config_sha256,
                vector_config_sha256,
            )
            index_receipt_expected = {
                "schema_version": 1,
                "project_id": project_id,
                "corpus_sha256": project_manifest["corpus_sha256"],
                "chunk_sha256": chunk_sha256,
                "chunker_id": chunker_id,
                "embedding_id": embedding_id,
                "vector_store_id": vector_store_id,
                "embedding_config_sha256": embedding_config_sha256,
                "vector_config_sha256": vector_config_sha256,
                "namespace": namespace,
            }
            index_receipt_path = _index_receipt_path(project_index_root, namespace)
            reranker_config = None
            combination_config = {
                "chunker": {"id": chunker_id, **techniques["chunkers"][chunker_id]},
                "embedding": {"id": embedding_id, **embedding_config},
                "vector_store": {"id": vector_store_id, **vector_config},
            }
            if reranker_id != _RETRIEVAL_ONLY_RERANKER_ID:
                reranker_config = techniques["rerankers"][reranker_id]
                combination_config["reranker"] = {
                    "id": reranker_id,
                    **reranker_config,
                }
            config_sha256 = _config_sha256(combination_config)
            physical_namespace = ""
            embedding_adapter: Any | None = None
            vector_adapter: Any | None = None
            reranker: Any = None
            retrieval_by_question: dict[str, list[SearchHit]] = {}
            retrieval_latency_by_question: dict[str, float] = {}
            retrieval_receipt: dict[str, Any] = {}
            embedding_input_tokens: int | None = None
            embedding_usage_scope = (
                "shared_embedding"
                if embedding_id == "openai_text-embedding-3-large"
                else None
            )
            embedding_usage_key = (
                f"{chunker_id}|{embedding_id}"
                if embedding_usage_scope is not None
                else None
            )
            rerank_search_units: int | None = None
            rerank_usage_scope = (
                "combination" if reranker_id == "Amazon Rerank v1" else None
            )
            failure_stage: str | None = None
            started = time.perf_counter()
            try:
                if chunker_id in chunk_failures:
                    failure_stage = "chunking"
                    raise RuntimeError("project chunker failed")
                if retrieval_key not in retrieval_cache:
                    persisted_index = _load_index_receipt(
                        index_receipt_path, index_receipt_expected
                    )
                    index_reused = persisted_index is not None
                    if index_reused:
                        assert persisted_index is not None
                        if embedding_id not in reused_query_embedding_cache:
                            failure_stage = "embedding"
                            embedding_adapter = self._new_embedding(
                                embedding_id, embedding_config, classes
                            )
                            query_vectors = embedding_adapter.embed_many(
                                question.query for question in questions
                            )
                            if len(query_vectors) != len(questions):
                                raise RuntimeError("query embedding count mismatch")
                            reused_query_embedding_cache[embedding_id] = (
                                embedding_adapter,
                                query_vectors,
                            )
                        else:
                            embedding_adapter, query_vectors = reused_query_embedding_cache[
                                embedding_id
                            ]
                        embedding_receipt = dict(persisted_index["embedding"])
                        dimensions = int(persisted_index["dimensions"])
                        vectors: list[list[float]] = []
                        failure_stage = "vector_store"
                        vector_adapter = self._new_vector_store(
                            vector_store_id,
                            vector_config,
                            classes,
                            namespace=namespace,
                            index_root=project_index_root,
                            load_existing=True,
                        )
                        if vector_store_id == "PGVector":
                            vector_adapter.dimensions = dimensions
                            vector_adapter.pg_vector_type = (
                                "halfvec" if dimensions > 2000 else "vector"
                            )
                        upsert_metrics = {
                            "vector_count": int(persisted_index["vector_count"])
                        }
                    else:
                        embedding_key = (chunker_id, embedding_id)
                        if embedding_key not in embedding_cache:
                            failure_stage = "embedding"
                            embedding_adapter = self._new_embedding(
                                embedding_id, embedding_config, classes
                            )
                            vectors = embedding_adapter.embed_many(
                                chunk.paragraph for chunk in chunks
                            )
                            if len(vectors) != len(chunks) or not vectors:
                                raise RuntimeError("embedding vector count mismatch")
                            dimensions = len(vectors[0])
                            if dimensions < 1 or any(
                                len(vector) != dimensions
                                or any(not math.isfinite(float(value)) for value in vector)
                                for vector in vectors
                            ):
                                raise RuntimeError("embedding vectors are invalid")
                            query_vectors = embedding_adapter.embed_many(
                                question.query for question in questions
                            )
                            if len(query_vectors) != len(questions):
                                raise RuntimeError("query embedding count mismatch")
                            embedding_receipt = {
                                "canonical_id": embedding_id,
                                "adapter_class": type(embedding_adapter).__name__,
                                "dimensions": dimensions,
                                "vector_count": len(vectors),
                                "provider_metadata": dict(
                                    getattr(
                                        embedding_adapter,
                                        "last_response_metadata",
                                        {},
                                    )
                                    or {}
                                ),
                            }
                            embedding_cache[embedding_key] = (
                                embedding_adapter,
                                vectors,
                                query_vectors,
                                embedding_receipt,
                            )
                        else:
                            (
                                embedding_adapter,
                                vectors,
                                query_vectors,
                                embedding_receipt,
                            ) = embedding_cache[embedding_key]
                        failure_stage = "vector_store"
                        vector_adapter = self._new_vector_store(
                            vector_store_id,
                            vector_config,
                            classes,
                            namespace=namespace,
                            index_root=project_index_root,
                        )
                        upsert_metrics = vector_adapter.upsert(chunks, vectors)
                        physical_namespace = str(vector_adapter.physical_namespace)
                        _atomic_json(
                            index_receipt_path,
                            {
                                **index_receipt_expected,
                                "physical_namespace": physical_namespace,
                                "dimensions": dimensions,
                                "vector_count": int(
                                    upsert_metrics.get("vector_count", len(vectors))
                                ),
                                "embedding": dict(embedding_receipt),
                            },
                        )
                    physical_namespace = str(vector_adapter.physical_namespace)
                    failure_stage = "retrieval"
                    retrieval_rows: list[dict[str, Any]] = []
                    for question, query_vector in zip(questions, query_vectors):
                        retrieval_start = time.perf_counter()
                        try:
                            hits = vector_adapter.search(query_vector, request.top_k)
                        except Exception:
                            if not index_reused or retrieval_rows:
                                raise
                            failure_stage = "embedding"
                            vectors = embedding_adapter.embed_many(
                                chunk.paragraph for chunk in chunks
                            )
                            if len(vectors) != len(chunks) or not vectors:
                                raise RuntimeError("embedding vector count mismatch")
                            dimensions = len(vectors[0])
                            if dimensions < 1 or any(
                                len(vector) != dimensions
                                or any(not math.isfinite(float(value)) for value in vector)
                                for vector in vectors
                            ):
                                raise RuntimeError("embedding vectors are invalid")
                            embedding_receipt = {
                                "canonical_id": embedding_id,
                                "adapter_class": type(embedding_adapter).__name__,
                                "dimensions": dimensions,
                                "vector_count": len(vectors),
                                "provider_metadata": dict(
                                    getattr(embedding_adapter, "last_response_metadata", {})
                                    or {}
                                ),
                            }
                            failure_stage = "vector_store"
                            upsert_metrics = vector_adapter.upsert(chunks, vectors)
                            physical_namespace = str(vector_adapter.physical_namespace)
                            index_reused = False
                            _atomic_json(
                                index_receipt_path,
                                {
                                    **index_receipt_expected,
                                    "physical_namespace": physical_namespace,
                                    "dimensions": dimensions,
                                    "vector_count": int(
                                        upsert_metrics.get("vector_count", len(vectors))
                                    ),
                                    "embedding": dict(embedding_receipt),
                                },
                            )
                            failure_stage = "retrieval"
                            hits = vector_adapter.search(query_vector, request.top_k)
                        retrieval_latency = time.perf_counter() - retrieval_start
                        retrieval_by_question[question.question_id] = hits
                        retrieval_latency_by_question[
                            question.question_id
                        ] = retrieval_latency
                        for rank, hit in enumerate(hits, 1):
                            retrieval_rows.append(
                                {
                                    "project_id": project_id,
                                    "run_id": run_id,
                                    "chunker_id": chunker_id,
                                    "embedding_id": embedding_id,
                                    "vector_store_id": vector_store_id,
                                    "query_id": question.question_id,
                                    "rank": rank,
                                    "chunk_id": str(hit.chunk.id),
                                    "source_name": hit.chunk.pdf_name,
                                    "page_number": str(
                                        (hit.chunk.metadata or {}).get("page_number", "")
                                    ),
                                    "paragraph": hit.chunk.paragraph,
                                    "base_score": float(hit.score),
                                    "latency_s": retrieval_latency,
                                }
                            )
                    retrieval_artifact = (
                        run_layout["retrieval"]
                        / f"retrieval_{_sha256_bytes(_canonical_bytes(retrieval_key))[:24]}.jsonl"
                    )
                    retrieval_content = _jsonl(retrieval_rows)
                    _atomic_write(retrieval_artifact, retrieval_content)
                    retrieval_receipt = {
                        "embedding": dict(embedding_receipt),
                        "vector_store": {
                            "canonical_id": vector_store_id,
                            "adapter_class": type(vector_adapter).__name__,
                            "physical_namespace": physical_namespace,
                            "reused": index_reused,
                            "vector_count": int(
                                upsert_metrics.get("vector_count", len(vectors))
                            ),
                            "provider_metadata": dict(
                                getattr(vector_adapter, "last_response_metadata", {}) or {}
                            ),
                        },
                        "retrieval_artifact": {
                            "path": retrieval_artifact.relative_to(
                                run_layout["root"]
                            ).as_posix(),
                            "sha256": _sha256_bytes(retrieval_content),
                        },
                    }
                    retrieval_cache[retrieval_key] = (
                        vector_adapter,
                        retrieval_by_question,
                        retrieval_latency_by_question,
                        retrieval_receipt,
                    )
                else:
                    (
                        vector_adapter,
                        retrieval_by_question,
                        retrieval_latency_by_question,
                        retrieval_receipt,
                    ) = retrieval_cache[retrieval_key]
                    embedding_adapter = embedding_cache[
                        (chunker_id, embedding_id)
                    ][0]
                    physical_namespace = retrieval_receipt["vector_store"][
                        "physical_namespace"
                    ]

                measured_embedding_tokens = retrieval_receipt["embedding"][
                    "provider_metadata"
                ].get("embedding_input_tokens")
                if (
                    embedding_usage_scope is not None
                    and isinstance(measured_embedding_tokens, int)
                    and not isinstance(measured_embedding_tokens, bool)
                    and measured_embedding_tokens >= 0
                ):
                    embedding_input_tokens = measured_embedding_tokens

                if reranker_config is not None:
                    failure_stage = "reranker"
                    reranker = self._new_reranker(
                        reranker_id, reranker_config, classes
                    )
                combo_detail_rows: list[dict[str, Any]] = []
                combo_evidence_rows: list[dict[str, Any]] = []
                query_metric_inputs: list[QueryMetricInput] = []
                for question in questions:
                    base_hits = retrieval_by_question[question.question_id]
                    base_scores = {
                        (str(hit.chunk.id), hit.chunk.parent_id): float(hit.score)
                        for hit in base_hits
                    }
                    if reranker is None:
                        reranked = base_hits
                        rerank_latency = 0.0
                    else:
                        rerank_start = time.perf_counter()
                        failure_stage = "reranker"
                        reranked = reranker.rerank(
                            question.query, base_hits, request.top_k
                        )
                        rerank_latency = time.perf_counter() - rerank_start
                    if reranker_id == "Amazon Rerank v1":
                        rerank_search_units = int(rerank_search_units or 0)
                        if base_hits:
                            rerank_search_units += math.ceil(len(base_hits) / 100)
                    failure_stage = "combination"
                    applies = label_applies(question, chunker_id)
                    relevant_ranks: list[int] = []
                    for rank, hit in enumerate(reranked, 1):
                        relevant = (
                            hit_is_relevant(question, hit, chunker_id)
                            if applies
                            else None
                        )
                        if relevant:
                            relevant_ranks.append(rank)
                        metadata = hit.chunk.metadata or {}
                        detail = {
                            "project_id": project_id,
                            "run_id": run_id,
                            "combo_id": combo_id,
                            "chunker_id": chunker_id,
                            "embedding_id": embedding_id,
                            "vector_store_id": vector_store_id,
                            "reranker_id": reranker_id,
                            "query_id": question.question_id,
                            "query": question.query,
                            "rank": rank,
                            "chunk_id": str(hit.chunk.id),
                            "source_id": str(metadata.get("source_id", hit.chunk.parent_id)),
                            "source_name": hit.chunk.pdf_name,
                            "page_number": str(metadata.get("page_number", "")),
                            "paragraph": hit.chunk.paragraph,
                            "base_score": base_scores.get(
                                (str(hit.chunk.id), hit.chunk.parent_id), 0.0
                            ),
                            "rerank_score": (
                                float(hit.score) if reranker is not None else None
                            ),
                            "rerank_latency_s": rerank_latency,
                            "relevant": relevant,
                        }
                        combo_detail_rows.append(detail)
                        combo_evidence_rows.append(
                            {
                                "combo_id": combo_id,
                                "query_id": question.question_id,
                                "query": question.query,
                                "source_name": hit.chunk.pdf_name,
                                "page_number": str(metadata.get("page_number", "")),
                                "excerpt": hit.chunk.paragraph,
                                "base_score": detail["base_score"],
                                "rerank_score": detail["rerank_score"],
                                "latency_s": rerank_latency,
                                "rank": rank,
                            }
                        )
                    query_metric_inputs.append(
                        QueryMetricInput(
                            query_id=question.question_id,
                            label_applies=applies,
                            relevant_ranks=tuple(relevant_ranks),
                            relevant_corpus_count=relevant_chunk_counts.get(
                                (chunker_id, question.question_id), 0
                            ),
                            retrieval_latency_s=retrieval_latency_by_question[
                                question.question_id
                            ],
                            rerank_latency_s=rerank_latency,
                        )
                    )
                reranking_artifact = run_layout["reranking"] / f"{combo_id}.jsonl"
                reranking_content = _jsonl(combo_detail_rows)
                failure_stage = "combination"
                query_metrics = summarize_query_metrics(
                    query_metric_inputs,
                    k=request.top_k,
                )
                retrieval_latency_s = (
                    math.fsum(
                        metric.retrieval_latency_s for metric in query_metric_inputs
                    )
                    / len(query_metric_inputs)
                    if query_metric_inputs
                    else None
                )
                rerank_latency_s = (
                    math.fsum(metric.rerank_latency_s for metric in query_metric_inputs)
                    / len(query_metric_inputs)
                    if query_metric_inputs
                    else None
                )
                summary = {
                    "project_id": project_id,
                    "run_id": run_id,
                    "combo_id": combo_id,
                    "status": "completed",
                    "summary_schema_version": _SUMMARY_SCHEMA_VERSION,
                    "chunker_id": chunker_id,
                    "embedding_id": embedding_id,
                    "vector_store_id": vector_store_id,
                    "reranker_id": reranker_id,
                    "physical_namespace": physical_namespace,
                    "query_count": len(query_metric_inputs),
                    "labelled_queries": query_metrics["labelled_queries"],
                    "unlabelled_queries": query_metrics["unlabelled_queries"],
                    "recall_at_k": query_metrics["recall_at_k"],
                    "mrr_at_k": query_metrics["mrr_at_k"],
                    "ndcg_at_k": query_metrics["ndcg_at_k"],
                    "recall_at_1": query_metrics["recall_at_1"],
                    "recall_at_3": query_metrics["recall_at_3"],
                    "recall_at_5": query_metrics["recall_at_5"],
                    "recall_at_10": query_metrics["recall_at_10"],
                    "precision_at_5": query_metrics["precision_at_5"],
                    "ndcg_at_5": query_metrics["ndcg_at_5"],
                    "avg_first_relevant_rank": query_metrics["avg_first_relevant_rank"],
                    "no_hit_queries": query_metrics["no_hit_queries"],
                    "retrieval_latency_s": retrieval_latency_s,
                    "rerank_latency_s": rerank_latency_s,
                    "avg_query_latency_s": query_metrics["avg_query_latency_s"],
                    "evidence_count": len(combo_evidence_rows),
                    "error_code": "",
                    "error_detail": "",
                }
                receipt = {
                    "combo_id": combo_id,
                    "status": "completed",
                    "config_sha256": config_sha256,
                    "adapters": {
                        "chunker": {
                            "canonical_id": chunker_id,
                            "adapter_class": "chunk_project_documents",
                            "chunk_count": len(chunks),
                            "artifact_sha256": (
                                chunk_artifact["sha256"] if chunk_artifact else None
                            ),
                        },
                        "embedding": dict(retrieval_receipt["embedding"]),
                        "vector_store": dict(retrieval_receipt["vector_store"]),
                    },
                    "artifacts": {
                        "retrieval": retrieval_receipt["retrieval_artifact"],
                    },
                    "latency_s": time.perf_counter() - started,
                }
                if reranker is not None:
                    receipt["adapters"]["reranker"] = {
                        "canonical_id": reranker_id,
                        "adapter_class": type(reranker).__name__,
                        "provider_metadata": dict(
                            getattr(reranker, "last_response_metadata", {}) or {}
                        ),
                    }
                    receipt["artifacts"]["reranking"] = {
                        "path": reranking_artifact.relative_to(
                            run_layout["root"]
                        ).as_posix(),
                        "sha256": _sha256_bytes(reranking_content),
                    }
                    _atomic_write(reranking_artifact, reranking_content)
                details_rows.extend(combo_detail_rows)
                evidence_rows.extend(combo_evidence_rows)
            except Exception as exc:
                if failure_stage == "reranker":
                    rerank_search_units = None
                error_code = _safe_error_code(failure_stage or "combination")
                error_detail = _safe_error_detail(exc)
                summary = {
                    "project_id": project_id,
                    "run_id": run_id,
                    "combo_id": combo_id,
                    "status": "failed",
                    "summary_schema_version": _SUMMARY_SCHEMA_VERSION,
                    "chunker_id": chunker_id,
                    "embedding_id": embedding_id,
                    "vector_store_id": vector_store_id,
                    "reranker_id": reranker_id,
                    "physical_namespace": physical_namespace,
                    "query_count": len(questions),
                    "labelled_queries": labelled_queries,
                    "unlabelled_queries": unlabelled_queries,
                    **_EMPTY_QUALITY,
                    "retrieval_latency_s": None,
                    "rerank_latency_s": None,
                    "avg_query_latency_s": None,
                    "evidence_count": 0,
                    "error_code": error_code,
                    "error_detail": error_detail,
                }
                receipt = {
                    "combo_id": combo_id,
                    "status": "failed",
                    "error_code": error_code,
                    "error_detail": error_detail,
                    "config_sha256": config_sha256,
                    "adapters": {
                        "chunker": {
                            "canonical_id": chunker_id,
                            "adapter_class": "chunk_project_documents",
                            "chunk_count": len(chunks),
                            "artifact_sha256": (
                                chunk_artifact["sha256"] if chunk_artifact else None
                            ),
                        },
                        "embedding": {
                            "canonical_id": embedding_id,
                            "adapter_class": classes["embeddings"][embedding_id].__name__,
                            "dimensions": int(
                                getattr(embedding_adapter, "dimensions", 0) or 0
                            ),
                            "vector_count": len(chunks) if embedding_adapter else 0,
                            "provider_metadata": {},
                        },
                        "vector_store": {
                            "canonical_id": vector_store_id,
                            "adapter_class": classes["vector_stores"][
                                vector_store_id
                            ].__name__,
                            "physical_namespace": physical_namespace,
                            "provider_metadata": {},
                        },
                    },
                    "artifacts": {},
                    "latency_s": time.perf_counter() - started,
                }
                if reranker_id != _RETRIEVAL_ONLY_RERANKER_ID:
                    receipt["adapters"]["reranker"] = {
                        "canonical_id": reranker_id,
                        "adapter_class": classes["rerankers"][reranker_id].__name__,
                        "provider_metadata": {},
                    }
            usage = {
                "embedding_input_tokens": embedding_input_tokens,
                "embedding_usage_scope": embedding_usage_scope,
                "embedding_usage_key": embedding_usage_key,
                "rerank_search_units": rerank_search_units,
                "rerank_usage_scope": rerank_usage_scope,
            }
            summary.update(usage)
            receipt["usage"] = usage
            receipt_path = run_layout["reranking"] / f"{combo_id}.receipt.json"
            receipt["receipt_path"] = receipt_path.relative_to(
                run_layout["root"]
            ).as_posix()
            _atomic_json(receipt_path, receipt)
            summary_rows.append(summary)
            receipts.append(receipt)
            progress_rows.append(
                {
                    "combo_id": combo_id,
                    "status": summary["status"],
                    "error_code": summary["error_code"],
                }
            )
            _atomic_json(
                run_layout["root"] / "progress.json",
                {
                    "schema_version": 1,
                    "project_id": project_id,
                    "run_id": run_id,
                    "completed_combinations": len(progress_rows),
                    "combination_count": validated.combination_count,
                    "combinations": progress_rows,
                },
            )

        succeeded = sum(row["status"] == "completed" for row in summary_rows)
        failed = len(summary_rows) - succeeded
        state = "completed" if failed == 0 else "failed" if succeeded == 0 else "partial"
        scoring_mode = (
            "retrieval_labels"
            if any(row["labelled_queries"] > 0 for row in summary_rows)
            else "evidence_only"
        )
        evidence_index_sha256 = _write_evidence_index(
            run_layout,
            project_id=project_id,
            run_id=run_id,
            mode=scoring_mode,
            summary_rows=summary_rows,
            evidence_rows=evidence_rows,
        )
        _atomic_write(run_layout["root"] / "details.jsonl", _jsonl(details_rows))
        _atomic_json(
            run_layout["root"] / "evidence.json",
            {
                "schema_version": 1,
                "project_id": project_id,
                "run_id": run_id,
                "mode": scoring_mode,
                "rows": evidence_rows,
            },
        )
        csv_buffer = StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key, "")
                    for key in _SUMMARY_FIELDS
                }
            )
        _atomic_write(
            run_layout["root"] / "summary.csv",
            csv_buffer.getvalue().encode("utf-8"),
        )
        completed_at = _utc_now()
        scored_rows = [
            {
                "combo_id": row["combo_id"],
                "chunker_id": row["chunker_id"],
                "embedding_id": row["embedding_id"],
                "vector_store_id": row["vector_store_id"],
                "reranker_id": row["reranker_id"],
                "labelled_queries": row["labelled_queries"],
                "unlabelled_queries": row["unlabelled_queries"],
                "recall_at_k": row["recall_at_k"],
                "mrr_at_k": row["mrr_at_k"],
                "ndcg_at_k": row["ndcg_at_k"],
                "recall_at_1": row.get("recall_at_1"),
                "recall_at_3": row.get("recall_at_3"),
                "recall_at_5": row.get("recall_at_5"),
                "recall_at_10": row.get("recall_at_10"),
                "precision_at_5": row.get("precision_at_5"),
                "ndcg_at_5": row.get("ndcg_at_5"),
                "avg_first_relevant_rank": row.get("avg_first_relevant_rank"),
                "no_hit_queries": row.get("no_hit_queries"),
                "retrieval_latency_s": row["retrieval_latency_s"],
                "rerank_latency_s": row["rerank_latency_s"],
                "avg_query_latency_s": row["avg_query_latency_s"],
                "evidence_count": row["evidence_count"],
            }
            for row in summary_rows
            if row["status"] == "completed" and row["labelled_queries"] > 0
        ]
        if scored_rows:
            _atomic_json(
                run_layout["root"] / "analysis.json",
                {
                    "schema_version": _SUMMARY_SCHEMA_VERSION,
                    "summary_schema_version": _SUMMARY_SCHEMA_VERSION,
                    "project_id": project_id,
                    "run_id": run_id,
                    "metric_k": request.top_k,
                    "scoring_mode": scoring_mode,
                    "created_at": created_at,
                    "completed_at": completed_at,
                    "rows": scored_rows,
                },
            )
        _atomic_json(
            run_layout["root"] / "manifest.json",
            {
                "schema_version": 2,
                "summary_schema_version": _SUMMARY_SCHEMA_VERSION,
                "project_id": project_id,
                "run_id": run_id,
                "request_fingerprint": validated.request_fingerprint,
                "catalog_sha256": catalog_sha256,
                "evidence_index_sha256": evidence_index_sha256,
                "state": state,
                "scoring_mode": scoring_mode,
                "metric_k": request.top_k,
                "created_at": created_at,
                "completed_at": completed_at,
                "combination_count": validated.combination_count,
                "succeeded": succeeded,
                "failed": failed,
                "receipts": receipts,
                "artifacts": {
                    "chunks": "chunks/manifest.json",
                    "details": "details.jsonl",
                    "evidence": "evidence.json",
                    "evidence_index": "evidence_index.json",
                    "summary": "summary.csv",
                    "analysis": "analysis.json" if scored_rows else None,
                },
            },
        )
        return {
            "project_id": project_id,
            "run_id": run_id,
            "state": state,
            "combination_count": validated.combination_count,
            "succeeded": succeeded,
            "failed": failed,
        }
