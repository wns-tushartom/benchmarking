"""Real, project-scoped selected-matrix execution with atomic run artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
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
from source.services.project_relevance import hit_is_relevant, label_applies
from source.services.project_workspace import ProjectWorkspace


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG = _REPO_ROOT / "configs" / "project_matrix_catalog.json"
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
    "chunker_id",
    "embedding_id",
    "vector_store_id",
    "reranker_id",
    "physical_namespace",
    "query_count",
    "labelled_queries",
    "unlabelled_queries",
    "recall_at_k",
    "error_code",
)


class ProjectMatrixRunnerError(RuntimeError):
    """A run cannot be safely loaded or executed."""


class ProjectMatrixRequestError(ValueError):
    """The immutable run request is invalid or no longer matches its fingerprint."""


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
    run_id: str,
    chunker_id: str,
    embedding_id: str,
    vector_store_id: str,
) -> str:
    return "|".join(
        (project_id, run_id, chunker_id, embedding_id, vector_store_id)
    )


def _safe_error_code(stage: str) -> str:
    return {
        "chunking": "chunker_failed",
        "embedding": "embedding_failed",
        "vector_store": "vector_store_failed",
        "retrieval": "retrieval_failed",
        "reranker": "reranker_failed",
    }.get(stage, "combination_failed")


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
    ) -> Any:
        kwargs = _adapter_kwargs(configuration)
        if canonical_id == "FAISS":
            kwargs["index_root"] = str(index_root)
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
        run_layout = self.workspace.run_layout(project_id, run_id)
        validated = self._load_request(project_id, run_id, run_layout["root"])
        request = validated.request
        documents, project_manifest = self._load_documents(project_id)
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

        summary_rows: list[dict[str, Any]] = []
        details_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        progress_rows: list[dict[str, Any]] = []
        embedding_cache: dict[
            tuple[str, str],
            tuple[Any, list[list[float]], list[list[float]], dict[str, Any]],
        ] = {}
        retrieval_cache: dict[
            tuple[str, str, str],
            tuple[Any, dict[str, list[SearchHit]], dict[str, Any]],
        ] = {}

        selections = itertools.product(
            request.chunkers,
            request.embeddings,
            request.vector_stores,
            request.rerankers,
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
            reranker_config = techniques["rerankers"][reranker_id]
            combination_config = {
                "chunker": {"id": chunker_id, **techniques["chunkers"][chunker_id]},
                "embedding": {"id": embedding_id, **embedding_config},
                "vector_store": {"id": vector_store_id, **vector_config},
                "reranker": {"id": reranker_id, **reranker_config},
            }
            config_sha256 = _config_sha256(combination_config)
            physical_namespace = ""
            embedding_adapter: Any | None = None
            vector_adapter: Any | None = None
            retrieval_by_question: dict[str, list[SearchHit]] = {}
            retrieval_receipt: dict[str, Any] = {}
            failure_stage: str | None = None
            started = time.perf_counter()
            try:
                if chunker_id in chunk_failures:
                    failure_stage = "chunking"
                    raise RuntimeError("project chunker failed")
                if retrieval_key not in retrieval_cache:
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
                    namespace = _retrieval_identity(
                        project_id,
                        run_id,
                        chunker_id,
                        embedding_id,
                        vector_store_id,
                    )
                    vector_adapter = self._new_vector_store(
                        vector_store_id,
                        vector_config,
                        classes,
                        namespace=namespace,
                        index_root=run_layout["indexes"],
                    )
                    physical_namespace = str(vector_adapter.physical_namespace)
                    upsert_metrics = vector_adapter.upsert(chunks, vectors)
                    failure_stage = "retrieval"
                    retrieval_rows: list[dict[str, Any]] = []
                    for question, query_vector in zip(questions, query_vectors):
                        retrieval_start = time.perf_counter()
                        hits = vector_adapter.search(query_vector, request.top_k)
                        retrieval_latency = time.perf_counter() - retrieval_start
                        retrieval_by_question[question.question_id] = hits
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
                        retrieval_receipt,
                    )
                else:
                    (
                        vector_adapter,
                        retrieval_by_question,
                        retrieval_receipt,
                    ) = retrieval_cache[retrieval_key]
                    embedding_adapter = embedding_cache[
                        (chunker_id, embedding_id)
                    ][0]
                    physical_namespace = retrieval_receipt["vector_store"][
                        "physical_namespace"
                    ]

                failure_stage = "reranker"
                reranker = self._new_reranker(
                    reranker_id, reranker_config, classes
                )
                combo_detail_rows: list[dict[str, Any]] = []
                combo_evidence_rows: list[dict[str, Any]] = []
                relevant_queries = 0
                for question in questions:
                    base_hits = retrieval_by_question[question.question_id]
                    base_scores = {
                        (str(hit.chunk.id), hit.chunk.parent_id): float(hit.score)
                        for hit in base_hits
                    }
                    rerank_start = time.perf_counter()
                    reranked = reranker.rerank(
                        question.query, base_hits, request.top_k
                    )
                    rerank_latency = time.perf_counter() - rerank_start
                    question_relevant = False
                    for rank, hit in enumerate(reranked, 1):
                        relevant = (
                            hit_is_relevant(question, hit, chunker_id)
                            if label_applies(question, chunker_id)
                            else None
                        )
                        if relevant:
                            question_relevant = True
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
                            "rerank_score": float(hit.score),
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
                            }
                        )
                    if question_relevant:
                        relevant_queries += 1
                reranking_artifact = run_layout["reranking"] / f"{combo_id}.jsonl"
                reranking_content = _jsonl(combo_detail_rows)
                _atomic_write(reranking_artifact, reranking_content)
                details_rows.extend(combo_detail_rows)
                evidence_rows.extend(combo_evidence_rows)
                recall = (
                    relevant_queries / labelled_queries
                    if labelled_queries > 0
                    else None
                )
                summary = {
                    "project_id": project_id,
                    "run_id": run_id,
                    "combo_id": combo_id,
                    "status": "completed",
                    "chunker_id": chunker_id,
                    "embedding_id": embedding_id,
                    "vector_store_id": vector_store_id,
                    "reranker_id": reranker_id,
                    "physical_namespace": physical_namespace,
                    "query_count": len(questions),
                    "labelled_queries": labelled_queries,
                    "unlabelled_queries": unlabelled_queries,
                    "recall_at_k": recall,
                    "error_code": "",
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
                        "reranker": {
                            "canonical_id": reranker_id,
                            "adapter_class": type(reranker).__name__,
                            "provider_metadata": dict(
                                getattr(reranker, "last_response_metadata", {}) or {}
                            ),
                        },
                    },
                    "artifacts": {
                        "retrieval": retrieval_receipt["retrieval_artifact"],
                        "reranking": {
                            "path": reranking_artifact.relative_to(
                                run_layout["root"]
                            ).as_posix(),
                            "sha256": _sha256_bytes(reranking_content),
                        },
                    },
                    "latency_s": time.perf_counter() - started,
                }
            except Exception:
                error_code = _safe_error_code(failure_stage or "combination")
                summary = {
                    "project_id": project_id,
                    "run_id": run_id,
                    "combo_id": combo_id,
                    "status": "failed",
                    "chunker_id": chunker_id,
                    "embedding_id": embedding_id,
                    "vector_store_id": vector_store_id,
                    "reranker_id": reranker_id,
                    "physical_namespace": physical_namespace,
                    "query_count": len(questions),
                    "labelled_queries": labelled_queries,
                    "unlabelled_queries": unlabelled_queries,
                    "recall_at_k": None,
                    "error_code": error_code,
                }
                receipt = {
                    "combo_id": combo_id,
                    "status": "failed",
                    "error_code": error_code,
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
                        "reranker": {
                            "canonical_id": reranker_id,
                            "adapter_class": classes["rerankers"][reranker_id].__name__,
                            "provider_metadata": {},
                        },
                    },
                    "artifacts": {},
                    "latency_s": time.perf_counter() - started,
                }
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
        _atomic_write(run_layout["root"] / "details.jsonl", _jsonl(details_rows))
        _atomic_json(
            run_layout["root"] / "evidence.json",
            {
                "schema_version": 1,
                "project_id": project_id,
                "run_id": run_id,
                "mode": (
                    "retrieval_labels"
                    if any(row["labelled_queries"] > 0 for row in summary_rows)
                    else "evidence_only"
                ),
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
            }
            for row in summary_rows
            if row["status"] == "completed" and row["labelled_queries"] > 0
        ]
        if scored_rows:
            _atomic_json(
                run_layout["root"] / "analysis.json",
                {
                    "schema_version": 1,
                    "project_id": project_id,
                    "run_id": run_id,
                    "scoring_mode": "retrieval_labels",
                    "rows": scored_rows,
                },
            )
        _atomic_json(
            run_layout["root"] / "manifest.json",
            {
                "schema_version": 1,
                "project_id": project_id,
                "run_id": run_id,
                "request_fingerprint": validated.request_fingerprint,
                "catalog_sha256": catalog_sha256,
                "state": state,
                "combination_count": validated.combination_count,
                "succeeded": succeeded,
                "failed": failed,
                "receipts": receipts,
                "artifacts": {
                    "chunks": "chunks/manifest.json",
                    "details": "details.jsonl",
                    "evidence": "evidence.json",
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
