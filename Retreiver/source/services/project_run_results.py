"""Fail-closed, project-scoped readers for completed matrix-run results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, NoReturn

from source.services.project_matrix_contract import (
    ProjectMatrixValidationError,
    validate_project_matrix_request,
)
from source.services.project_run_artifacts import (
    ProjectRunArtifactError,
    read_project_artifact,
    read_project_json_rows_page,
)
from source.services.project_workspace import ProjectWorkspace, validate_resource_id


_DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "configs" / "project_matrix_catalog.json"
_LEGACY_FIELDS = (
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
_V2_FIELDS = (
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
)
_V2_FIELDS_WITH_DETAIL = _V2_FIELDS + ("error_detail",)
_EVIDENCE_FIELDS = (
    "combo_id",
    "query_id",
    "query",
    "source_name",
    "page_number",
    "excerpt",
    "base_score",
    "rerank_score",
    "latency_s",
    "rank",
)
_ERROR_CODES = {
    "chunker_failed",
    "embedding_failed",
    "vector_store_failed",
    "retrieval_failed",
    "reranker_failed",
    "combination_failed",
}


class ProjectRunResultsError(RuntimeError):
    """A safe public project-result error."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status = status


def _fail(code: str, message: str, status: int) -> NoReturn:
    raise ProjectRunResultsError(code, message, status)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError("non-finite constant")


def _decode_json(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        _fail("run_unavailable", "Run results are unavailable", 409)
    if not isinstance(value, dict):
        _fail("run_unavailable", "Run results are unavailable", 409)
    return value


def _artifact_error(exc: ProjectRunArtifactError) -> NoReturn:
    if exc.code == "artifact_too_large":
        _fail("artifact_too_large", "Run artifact exceeds the size limit", 413)
    _fail("run_unavailable", "Run results are unavailable", 409)


def _safe_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _nonnegative_int(value: Any, *, nullable: bool = False) -> int | None:
    if nullable and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ValueError
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError from None
    if str(parsed) != str(value).strip() or parsed < 0:
        raise ValueError
    return parsed


def _nonnegative_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError from None
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError
    return parsed


def _quality_float(value: Any) -> float | None:
    parsed = _nonnegative_float(value)
    if parsed is not None and parsed > 1:
        raise ValueError
    return parsed


def _required_nonnegative_int(value: Any) -> int:
    parsed = _nonnegative_int(value)
    if parsed is None:
        raise ValueError
    return parsed


def _finite_evidence_number(value: Any, *, nonnegative: bool = False) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError from None
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise ValueError
    return parsed


def _public_evidence_row(raw: Any, combo_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        _fail("run_unavailable", "Run results are unavailable", 409)
    required = {"combo_id", "query_id", "query", "source_name", "excerpt", "rank"}
    if not required.issubset(raw) or raw.get("combo_id") != combo_id:
        _fail("run_unavailable", "Run results are unavailable", 409)
    query_id = raw.get("query_id")
    query = raw.get("query")
    source_name = raw.get("source_name")
    excerpt = raw.get("excerpt")
    rank = raw.get("rank")
    if (
        not isinstance(query_id, str)
        or not query_id
        or len(query_id) > 256
        or not isinstance(query, str)
        or not query
        or len(query) > 20_000
        or not isinstance(source_name, str)
        or not source_name
        or len(source_name) > 512
        or "\x00" in source_name
        or "\\" in source_name
        or source_name.startswith("/")
        or (len(source_name) >= 2 and source_name[1] == ":")
        or any(part in {"", ".", ".."} for part in source_name.split("/"))
        or not isinstance(excerpt, str)
        or len(excerpt) > 200_000
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
    ):
        _fail("run_unavailable", "Run results are unavailable", 409)
    page_number = raw.get("page_number")
    if page_number is not None and page_number != "" and (
        isinstance(page_number, bool)
        or not isinstance(page_number, (str, int))
        or len(str(page_number)) > 32
    ):
        _fail("run_unavailable", "Run results are unavailable", 409)
    try:
        _finite_evidence_number(raw.get("base_score"))
        _finite_evidence_number(raw.get("rerank_score"))
        _finite_evidence_number(raw.get("latency_s"), nonnegative=True)
    except ValueError:
        _fail("run_unavailable", "Run results are unavailable", 409)
    return {key: raw[key] for key in _EVIDENCE_FIELDS if key in raw}


class ProjectRunResultService:
    MAX_PROJECT_MANIFEST_BYTES = 2 * 1024 * 1024
    MAX_RUN_MANIFEST_BYTES = 8 * 1024 * 1024
    MAX_REQUEST_BYTES = 2 * 1024 * 1024
    MAX_SUMMARY_BYTES = 16 * 1024 * 1024
    MAX_SUMMARY_ROWS = 10_000
    MAX_LEGACY_EVIDENCE_BYTES = 16 * 1024 * 1024
    MAX_EVIDENCE_INDEX_BYTES = 2 * 1024 * 1024
    MAX_EVIDENCE_PART_BYTES = 4 * 1024 * 1024
    MAX_EVIDENCE_PAGE = 100

    _TABLE_SUFFIXES = {".csv", ".xlsx"}

    def _project_owned_groundtruth_identity(
        self, project_id: str, project_manifest: dict[str, Any] | None = None
    ) -> dict[str, str] | None:
        """Map project-owned GT files to the catalog ID the UI locks onto."""
        try:
            questions_dir = self.workspace.layout(project_id)["questions"]
        except (KeyError, TypeError, ValueError, OSError):
            return None
        if not questions_dir.is_dir() or questions_dir.is_symlink():
            return None
        owned_files = sorted(
            candidate
            for candidate in questions_dir.iterdir()
            if candidate.is_file()
            and not candidate.is_symlink()
            and candidate.suffix.lower() in self._TABLE_SUFFIXES
        )
        if len(owned_files) != 1:
            return None
        label = str((project_manifest or {}).get("label") or project_id).strip() or project_id
        return {
            "groundtruth_id": f"groundtruth:project:{project_id}",
            "groundtruth_label": f"{label} — uploaded ground truth",
        }

    def _result_source_identity(
        self,
        request: Any,
        *,
        project_id: str,
        project_manifest: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        # Browser launches convert catalog ground-truth IDs into immutable question sets.
        # The UI still locks on groundtruth:project:<id> / groundtruth:none, so result
        # identity must match those catalog IDs or Metrics/Recommendations stay empty.
        source = request.questions_source
        if source.get("type") == "question_set":
            owned = self._project_owned_groundtruth_identity(project_id, project_manifest)
            if owned is not None:
                return owned
            return {
                "groundtruth_id": "groundtruth:none",
                "groundtruth_label": "None (evidence-only)",
            }
        return {
            "groundtruth_id": "groundtruth:none",
            "groundtruth_label": "None (evidence-only)",
        }

    def __init__(self, workspace: ProjectWorkspace, *, catalog_path: Path = _DEFAULT_CATALOG):
        if not isinstance(workspace, ProjectWorkspace):
            raise TypeError("workspace must be a ProjectWorkspace")
        self.workspace = workspace
        self.catalog_path = Path(catalog_path).resolve()
        self._catalog = self._load_catalog()
        self._canonical = {
            dimension: set(self._catalog["matrix"][dimension])
            for dimension in ("chunkers", "embeddings", "vector_stores", "rerankers")
        }
        self._commercial = {
            technique_id
            for dimension in ("chunkers", "embeddings", "vector_stores", "rerankers")
            for technique_id, config in self._catalog["techniques"][dimension].items()
            if config.get("license") == "commercial"
        }

    def _load_catalog(self) -> dict[str, Any]:
        try:
            metadata = self.catalog_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise OSError
            value = json.loads(
                self.catalog_path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("matrix catalog is unavailable") from None
        if not isinstance(value, dict) or not isinstance(value.get("matrix"), dict) or not isinstance(value.get("techniques"), dict):
            raise ValueError("matrix catalog is unavailable")
        return value

    def _read(self, root: Path, name: str, maximum: int) -> bytes:
        try:
            return read_project_artifact(root, name, max_bytes=maximum)
        except ProjectRunArtifactError as exc:
            _artifact_error(exc)

    def _project(self, project_id: str) -> tuple[dict[str, Path], dict[str, Any]]:
        try:
            validate_resource_id(project_id)
            layout = self.workspace.layout(project_id)
            root = layout["root"]
            metadata = root.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError
        except (TypeError, ValueError, FileNotFoundError, OSError):
            _fail("not_found", "Project result was not found", 404)
        try:
            manifest = _decode_json(self._read(root, "manifest.json", self.MAX_PROJECT_MANIFEST_BYTES))
        except ProjectRunResultsError as exc:
            if exc.code == "artifact_too_large":
                raise
            _fail("not_found", "Project result was not found", 404)
        if (
            manifest.get("ok") is not True
            or manifest.get("project_id") != project_id
            or not isinstance(manifest.get("label"), str)
            or not manifest["label"].strip()
        ):
            _fail("not_found", "Project result was not found", 404)
        return layout, manifest

    def _run_context(self, project_id: str, run_id: str) -> dict[str, Any]:
        project_layout, project_manifest = self._project(project_id)
        try:
            run_layout = self.workspace.run_layout(project_id, run_id)
        except (TypeError, ValueError, FileNotFoundError, OSError):
            _fail("not_found", "Run result was not found", 404)
        root = run_layout["root"]
        manifest = _decode_json(self._read(root, "manifest.json", self.MAX_RUN_MANIFEST_BYTES))
        required = {"project_id", "run_id", "request_fingerprint", "combination_count", "state", "artifacts"}
        if (
            not required.issubset(manifest)
            or manifest.get("project_id") != project_id
            or manifest.get("run_id") != run_id
            or manifest.get("mode") == "lexical_preview"
            or manifest.get("state") not in {"completed", "partial", "failed"}
            or not isinstance(manifest.get("request_fingerprint"), str)
            or len(manifest["request_fingerprint"]) != 64
            or isinstance(manifest.get("combination_count"), bool)
            or not isinstance(manifest.get("combination_count"), int)
            or manifest["combination_count"] < 1
            or not isinstance(manifest.get("artifacts"), dict)
            or manifest["artifacts"].get("summary") != "summary.csv"
            or manifest["artifacts"].get("evidence") != "evidence.json"
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)

        request_envelope = _decode_json(self._read(root, "request.json", self.MAX_REQUEST_BYTES))
        if set(request_envelope) != {"schema_version", "request", "request_fingerprint", "combination_count"}:
            _fail("run_unavailable", "Run results are unavailable", 409)
        try:
            validated = validate_project_matrix_request(
                request_envelope["request"], catalog_path=self.catalog_path
            )
        except (ProjectMatrixValidationError, TypeError, ValueError):
            _fail("run_unavailable", "Run results are unavailable", 409)
        if (
            request_envelope.get("schema_version") != 1
            or request_envelope.get("request_fingerprint") != validated.request_fingerprint
            or manifest["request_fingerprint"] != validated.request_fingerprint
            or request_envelope.get("combination_count") != validated.combination_count
            or manifest["combination_count"] != validated.combination_count
            or validated.request.project_id != project_id
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)
        return {
            "project_layout": project_layout,
            "project_manifest": project_manifest,
            "run_layout": run_layout,
            "manifest": manifest,
            "validated": validated,
        }

    def result_sources(self, *, official_configured: int, official_evaluated: int) -> dict[str, Any]:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (official_configured, official_evaluated)
        ):
            _fail("invalid_request", "Invalid result request", 400)
        projects: list[dict[str, Any]] = []
        try:
            entries = list(self.workspace.base.iterdir()) if self.workspace.base.is_dir() else []
        except OSError:
            entries = []
        for entry in entries:
            try:
                _layout, manifest = self._project(entry.name)
                created = _safe_time(manifest.get("created_at"))
                artifact_time = entry.joinpath("manifest.json").lstat().st_mtime
                sort_time = created.timestamp() if created else artifact_time
                projects.append(
                    {
                        "project_id": entry.name,
                        "dataset_id": f"project:{entry.name}",
                        "label": manifest["label"].strip(),
                        "created_at": manifest.get("created_at") if created else None,
                        "artifact_time": _iso_timestamp(artifact_time),
                        "_sort": sort_time,
                    }
                )
            except (ProjectRunResultsError, OSError, ValueError):
                continue
        projects.sort(key=lambda row: (-row["_sort"], row["project_id"]))
        for row in projects:
            row.pop("_sort", None)
        return {
            "official": {
                "source_type": "official",
                "configured": official_configured,
                "evaluated": official_evaluated,
            },
            "projects": projects,
        }

    def project_runs(self, project_id: str) -> list[dict[str, Any]]:
        project_layout, _manifest = self._project(project_id)
        runs: list[dict[str, Any]] = []
        try:
            entries = list(project_layout["runs"].iterdir())
        except OSError:
            return []
        for entry in entries:
            try:
                context = self._run_context(project_id, entry.name)
                summary_rows, legacy = self._parse_summary(context)
                self._validate_manifest_summary(context, summary_rows, legacy=legacy)
                result = self.project_run_results(project_id, entry.name)
                manifest = context["manifest"]
                completed = _safe_time(manifest.get("completed_at"))
                created = _safe_time(manifest.get("created_at"))
                artifact_timestamp = entry.joinpath("manifest.json").lstat().st_mtime
                semantic = completed or created
                source_identity = self._result_source_identity(
                    context["validated"].request,
                    project_id=project_id,
                    project_manifest=context.get("project_manifest"),
                )
                runs.append(
                    {
                        "project_id": project_id,
                        "dataset_id": f"project:{project_id}",
                        "run_id": entry.name,
                        "state": manifest["state"],
                        "scoring_mode": result["scoring_mode"],
                        "metric_k": context["validated"].request.top_k,
                        "combination_count": int(result.get("combination_count") or manifest.get("combination_count") or 0),
                        "succeeded": int(result.get("succeeded") or manifest.get("succeeded") or 0),
                        "failed": int(result.get("failed") or manifest.get("failed") or 0),
                        "evidence_count": int(
                            sum(
                                int(value or 0)
                                for value in (result.get("evidence_counts_by_combo") or {}).values()
                            )
                        ),
                        "created_at": manifest.get("created_at") if created else None,
                        "completed_at": manifest.get("completed_at") if completed else None,
                        "timestamp_label": manifest.get("completed_at") if completed else (manifest.get("created_at") if created else "Legacy artifact time"),
                        "artifact_time": _iso_timestamp(artifact_timestamp),
                        **source_identity,
                        "_sort": semantic.timestamp() if semantic else artifact_timestamp,
                    }
                )
            except (ProjectRunResultsError, OSError, ValueError):
                continue
        runs.sort(key=lambda row: (-row["_sort"], row["run_id"]))
        for row in runs:
            row.pop("_sort", None)
        return runs

    def _parse_summary(self, context: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        project_id = context["manifest"]["project_id"]
        run_id = context["manifest"]["run_id"]
        content = self._read(context["run_layout"]["root"], "summary.csv", self.MAX_SUMMARY_BYTES)
        try:
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            headers = reader.fieldnames
            if headers is None or len(headers) != len(set(headers)):
                raise ValueError
            legacy = tuple(headers) == _LEGACY_FIELDS
            if not legacy and tuple(headers) not in {_V2_FIELDS, _V2_FIELDS_WITH_DETAIL}:
                raise ValueError
            raw_rows: list[dict[str, str]] = []
            for raw in reader:
                if len(raw_rows) >= self.MAX_SUMMARY_ROWS or None in raw:
                    raise ValueError
                raw_rows.append(raw)
        except (UnicodeError, csv.Error, ValueError):
            _fail("run_unavailable", "Run results are unavailable", 409)
        if len(raw_rows) != context["manifest"]["combination_count"]:
            _fail("run_unavailable", "Run results are unavailable", 409)

        request = context["validated"].request
        effective_rerankers = request.rerankers or ("none",)
        selected = {
            "chunker_id": set(request.chunkers),
            "embedding_id": set(request.embeddings),
            "vector_store_id": set(request.vector_stores),
            "reranker_id": set(effective_rerankers),
        }
        expected_matrix = set(
            itertools.product(
                request.chunkers,
                request.embeddings,
                request.vector_stores,
                effective_rerankers,
            )
        )
        observed_matrix: set[tuple[str, str, str, str]] = set()
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        try:
            for raw in raw_rows:
                combo_id = raw["combo_id"]
                if not isinstance(combo_id, str) or not combo_id or len(combo_id) > 512 or combo_id in seen:
                    raise ValueError
                seen.add(combo_id)
                if raw["project_id"] != project_id or raw["run_id"] != run_id:
                    raise ValueError
                if raw["status"] not in {"completed", "failed"}:
                    raise ValueError
                for field, dimension in (
                    ("chunker_id", "chunkers"),
                    ("embedding_id", "embeddings"),
                    ("vector_store_id", "vector_stores"),
                    ("reranker_id", "rerankers"),
                ):
                    canonical = self._canonical[dimension]
                    if dimension == "rerankers" and not request.rerankers:
                        canonical = ("none",)
                    if raw[field] not in canonical or raw[field] not in selected[field]:
                        raise ValueError
                matrix_identity = (
                    raw["chunker_id"],
                    raw["embedding_id"],
                    raw["vector_store_id"],
                    raw["reranker_id"],
                )
                if matrix_identity in observed_matrix:
                    raise ValueError
                observed_matrix.add(matrix_identity)
                if not raw["physical_namespace"]:
                    raise ValueError
                if not legacy and _nonnegative_int(raw["summary_schema_version"]) != 2:
                    raise ValueError
                commercial = [
                    adapter_id
                    for adapter_id in (raw["embedding_id"], raw["reranker_id"])
                    if adapter_id in self._commercial
                ]
                measured = {
                    "embedding_input_tokens": None if legacy else _nonnegative_int(raw["embedding_input_tokens"], nullable=True),
                    "embedding_usage_scope": None if legacy or not raw["embedding_usage_scope"] else raw["embedding_usage_scope"],
                    "embedding_usage_key": None if legacy or not raw["embedding_usage_key"] else raw["embedding_usage_key"],
                    "rerank_search_units": None if legacy else _nonnegative_int(raw["rerank_search_units"], nullable=True),
                    "rerank_usage_scope": None if legacy or not raw["rerank_usage_scope"] else raw["rerank_usage_scope"],
                }
                embedding_scope = measured["embedding_usage_scope"]
                embedding_key = measured["embedding_usage_key"]
                rerank_scope = measured["rerank_usage_scope"]
                if (
                    embedding_scope not in {None, "shared_embedding"}
                    or rerank_scope not in {None, "combination"}
                    or (embedding_key is not None and not isinstance(embedding_key, str))
                    or (embedding_scope is None and embedding_key is not None)
                    or (
                        embedding_scope == "shared_embedding"
                        and embedding_key != f"{raw['chunker_id']}|{raw['embedding_id']}"
                    )
                    or (
                        measured["embedding_input_tokens"] is not None
                        and embedding_scope != "shared_embedding"
                    )
                    or (
                        measured["rerank_search_units"] is not None
                        and rerank_scope != "combination"
                    )
                ):
                    raise ValueError
                query_count = _required_nonnegative_int(raw["query_count"])
                labelled_queries = _required_nonnegative_int(raw["labelled_queries"])
                unlabelled_queries = _required_nonnegative_int(raw["unlabelled_queries"])
                if query_count != labelled_queries + unlabelled_queries:
                    raise ValueError
                recall_at_k = _quality_float(raw["recall_at_k"])
                mrr_at_k = None if legacy else _quality_float(raw["mrr_at_k"])
                ndcg_at_k = None if legacy else _quality_float(raw["ndcg_at_k"])
                retrieval_latency_s = None if legacy else _nonnegative_float(raw["retrieval_latency_s"])
                rerank_latency_s = None if legacy else _nonnegative_float(raw["rerank_latency_s"])
                avg_query_latency_s = None if legacy else _nonnegative_float(raw["avg_query_latency_s"])
                evidence_count = None if legacy else _nonnegative_int(raw["evidence_count"], nullable=True)
                error_code = raw["error_code"] or None
                if (
                    (raw["status"] == "completed" and error_code is not None)
                    or (raw["status"] == "failed" and error_code not in _ERROR_CODES)
                ):
                    raise ValueError
                quality_values = (recall_at_k, mrr_at_k, ndcg_at_k)
                latency_values = (
                    retrieval_latency_s,
                    rerank_latency_s,
                    avg_query_latency_s,
                )
                if raw["status"] == "failed":
                    if (
                        any(value is not None for value in quality_values + latency_values)
                        or evidence_count not in {None, 0}
                        or (
                            measured["rerank_search_units"] is not None
                            and error_code != "combination_failed"
                        )
                        or (
                            error_code in {"chunker_failed", "embedding_failed"}
                            and measured["embedding_input_tokens"] is not None
                        )
                    ):
                        raise ValueError
                else:
                    if labelled_queries == 0 and any(value is not None for value in quality_values):
                        raise ValueError
                    if labelled_queries > 0 and (
                        recall_at_k is None
                        or (not legacy and (mrr_at_k is None or ndcg_at_k is None))
                    ):
                        raise ValueError
                    if not legacy:
                        if (
                            retrieval_latency_s is None
                            or rerank_latency_s is None
                            or avg_query_latency_s is None
                        ):
                            raise ValueError
                        if not math.isclose(
                            avg_query_latency_s,
                            retrieval_latency_s + rerank_latency_s,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            raise ValueError
                rows.append(
                    {
                        "project_id": project_id,
                        "run_id": run_id,
                        "combo_id": combo_id,
                        "status": raw["status"],
                        "summary_schema_version": 1 if legacy else 2,
                        "chunker_id": raw["chunker_id"],
                        "embedding_id": raw["embedding_id"],
                        "vector_store_id": raw["vector_store_id"],
                        "reranker_id": raw["reranker_id"],
                        "physical_namespace": raw["physical_namespace"],
                        "query_count": query_count,
                        "labelled_queries": labelled_queries,
                        "unlabelled_queries": unlabelled_queries,
                        "recall_at_k": recall_at_k,
                        "mrr_at_k": mrr_at_k,
                        "ndcg_at_k": ndcg_at_k,
                        "retrieval_latency_s": retrieval_latency_s,
                        "rerank_latency_s": rerank_latency_s,
                        "avg_query_latency_s": avg_query_latency_s,
                        "evidence_count": evidence_count,
                        "commercial_model_ids": commercial,
                        "measured_usage": measured,
                        "error_code": error_code,
                    }
                )
        except (KeyError, TypeError, ValueError):
            _fail("run_unavailable", "Run results are unavailable", 409)
        if observed_matrix != expected_matrix:
            _fail("run_unavailable", "Run results are unavailable", 409)
        return rows, legacy

    def _validate_manifest_summary(
        self,
        context: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        legacy: bool,
    ) -> None:
        manifest = context["manifest"]
        if legacy:
            succeeded = sum(row["status"] == "completed" for row in rows)
            failed = len(rows) - succeeded
            expected_state = "completed" if failed == 0 else "failed" if succeeded == 0 else "partial"
            schema_version = manifest.get("schema_version")
            summary_schema_version = manifest.get("summary_schema_version", 1)
            if (
                type(schema_version) is not int
                or schema_version != 1
                or type(summary_schema_version) is not int
                or summary_schema_version != 1
                or manifest.get("combination_count") != len(rows)
                or manifest.get("succeeded") != succeeded
                or manifest.get("failed") != failed
                or manifest.get("state") != expected_state
                or not isinstance(manifest.get("receipts"), list)
            ):
                _fail("run_unavailable", "Run results are unavailable", 409)
            return

        succeeded = sum(row["status"] == "completed" for row in rows)
        failed = len(rows) - succeeded
        expected_state = "completed" if failed == 0 else "failed" if succeeded == 0 else "partial"
        expected_scoring_mode = (
            "retrieval_labels"
            if any(row["labelled_queries"] > 0 for row in rows)
            else "evidence_only"
        )
        created_at = _safe_time(manifest.get("created_at"))
        completed_at = _safe_time(manifest.get("completed_at"))
        receipts = manifest.get("receipts")
        schema_version = manifest.get("schema_version")
        summary_schema_version = manifest.get("summary_schema_version")
        if (
            type(schema_version) is not int
            or schema_version != 2
            or type(summary_schema_version) is not int
            or summary_schema_version != 2
            or manifest.get("metric_k") != context["validated"].request.top_k
            or manifest.get("scoring_mode") != expected_scoring_mode
            or manifest.get("succeeded") != succeeded
            or manifest.get("failed") != failed
            or manifest.get("state") != expected_state
            or created_at is None
            or completed_at is None
            or completed_at < created_at
            or not isinstance(receipts, list)
            or len(receipts) != len(rows)
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)

        rows_by_combo = {row["combo_id"]: row for row in rows}
        seen_receipts: set[str] = set()
        expected_usage_keys = {
            "embedding_input_tokens",
            "embedding_usage_scope",
            "embedding_usage_key",
            "rerank_search_units",
            "rerank_usage_scope",
        }
        for receipt in receipts:
            if not isinstance(receipt, dict):
                _fail("run_unavailable", "Run results are unavailable", 409)
            combo_id = receipt.get("combo_id")
            row = rows_by_combo.get(combo_id)
            usage = receipt.get("usage")
            if (
                not isinstance(combo_id, str)
                or row is None
                or combo_id in seen_receipts
                or receipt.get("status") != row["status"]
                or not isinstance(usage, dict)
                or set(usage) != expected_usage_keys
                or usage != row["measured_usage"]
            ):
                _fail("run_unavailable", "Run results are unavailable", 409)
            seen_receipts.add(combo_id)
        if seen_receipts != set(rows_by_combo):
            _fail("run_unavailable", "Run results are unavailable", 409)

    def _v2_evidence_index(
        self,
        context: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]] | None:
        manifest = context["manifest"]
        artifacts = manifest["artifacts"]
        index_path = artifacts.get("evidence_index")
        expected_sha256 = manifest.get("evidence_index_sha256")
        if index_path is None and expected_sha256 is None:
            return None
        if (
            index_path != "evidence_index.json"
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)
        content = self._read(
            context["run_layout"]["root"],
            "evidence_index.json",
            self.MAX_EVIDENCE_INDEX_BYTES,
        )
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            _fail("run_unavailable", "Run results are unavailable", 409)
        index = _decode_json(content)
        if (
            set(index) != {"schema_version", "project_id", "run_id", "mode", "entries"}
            or index.get("schema_version") != 1
            or index.get("project_id") != manifest["project_id"]
            or index.get("run_id") != manifest["run_id"]
            or index.get("mode") != manifest.get("scoring_mode")
            or not isinstance(index.get("entries"), list)
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)
        rows_by_combo = {row["combo_id"]: row for row in rows}
        entries_by_combo: dict[str, dict[str, Any]] = {}
        for entry in index["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"combo_id", "row_count", "parts"}:
                _fail("run_unavailable", "Run results are unavailable", 409)
            combo_id = entry.get("combo_id")
            row_count = entry.get("row_count")
            parts = entry.get("parts")
            if (
                not isinstance(combo_id, str)
                or combo_id not in rows_by_combo
                or combo_id in entries_by_combo
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 0
                or not isinstance(parts, list)
            ):
                _fail("run_unavailable", "Run results are unavailable", 409)
            part_rows = 0
            for part_number, part in enumerate(parts):
                expected_path = f"reranking/{combo_id}.evidence.{part_number:05d}.jsonl"
                if not isinstance(part, dict) or set(part) != {"path", "row_count", "size_bytes", "sha256"}:
                    _fail("run_unavailable", "Run results are unavailable", 409)
                part_count = part.get("row_count")
                part_size = part.get("size_bytes")
                part_sha256 = part.get("sha256")
                if (
                    part.get("path") != expected_path
                    or isinstance(part_count, bool)
                    or not isinstance(part_count, int)
                    or part_count < 1
                    or isinstance(part_size, bool)
                    or not isinstance(part_size, int)
                    or not 1 <= part_size <= self.MAX_EVIDENCE_PART_BYTES
                    or not isinstance(part_sha256, str)
                    or len(part_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in part_sha256)
                ):
                    _fail("run_unavailable", "Run results are unavailable", 409)
                part_rows += part_count
            summary_row = rows_by_combo[combo_id]
            if (
                part_rows != row_count
                or summary_row["evidence_count"] != row_count
                or (summary_row["status"] != "completed" and row_count != 0)
            ):
                _fail("run_unavailable", "Run results are unavailable", 409)
            entries_by_combo[combo_id] = entry
        if set(entries_by_combo) != set(rows_by_combo):
            _fail("run_unavailable", "Run results are unavailable", 409)
        return entries_by_combo

    def _indexed_evidence_page(
        self,
        context: dict[str, Any],
        entry: dict[str, Any],
        *,
        combo_id: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        total = entry["row_count"]
        if offset >= total:
            selected: list[dict[str, Any]] = []
        else:
            selected = []
            cursor = 0
            requested_end = min(total, offset + limit)
            for part in entry["parts"]:
                part_end = cursor + part["row_count"]
                if part_end <= offset:
                    cursor = part_end
                    continue
                if cursor >= requested_end:
                    break
                content = self._read(
                    context["run_layout"]["root"],
                    part["path"],
                    self.MAX_EVIDENCE_PART_BYTES,
                )
                if (
                    len(content) != part["size_bytes"]
                    or hashlib.sha256(content).hexdigest() != part["sha256"]
                ):
                    _fail("run_unavailable", "Run results are unavailable", 409)
                lines = content.splitlines()
                if len(lines) != part["row_count"] or any(not line for line in lines):
                    _fail("run_unavailable", "Run results are unavailable", 409)
                parsed = [_decode_json(line) for line in lines]
                if any(row.get("combo_id") != combo_id for row in parsed):
                    _fail("run_unavailable", "Run results are unavailable", 409)
                local_start = max(0, offset - cursor)
                local_end = min(len(parsed), requested_end - cursor)
                selected.extend(parsed[local_start:local_end])
                cursor = part_end
        next_offset = offset + len(selected)
        return {
            "project_id": context["manifest"]["project_id"],
            "run_id": context["manifest"]["run_id"],
            "combo_id": combo_id,
            "rows": selected,
            "next_offset": next_offset if next_offset < total else None,
        }

    def _v2_evidence_counts(
        self,
        context: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, int]:
        index = self._v2_evidence_index(context, rows)
        if index is not None:
            return {combo_id: entry["row_count"] for combo_id, entry in index.items()}
        envelope = _decode_json(
            self._read(
                context["run_layout"]["root"],
                "evidence.json",
                self.MAX_LEGACY_EVIDENCE_BYTES,
            )
        )
        manifest = context["manifest"]
        if (
            set(envelope) != {"schema_version", "project_id", "run_id", "mode", "rows"}
            or envelope.get("schema_version") != 1
            or envelope.get("project_id") != manifest["project_id"]
            or envelope.get("run_id") != manifest["run_id"]
            or envelope.get("mode") != manifest.get("scoring_mode")
            or not isinstance(envelope.get("rows"), list)
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)
        completed = {row["combo_id"] for row in rows if row["status"] == "completed"}
        counts = {row["combo_id"]: 0 for row in rows}
        for evidence_row in envelope["rows"]:
            combo_id = evidence_row.get("combo_id") if isinstance(evidence_row, dict) else None
            if not isinstance(combo_id, str) or combo_id not in completed:
                _fail("run_unavailable", "Run results are unavailable", 409)
            _public_evidence_row(evidence_row, combo_id)
            counts[combo_id] += 1
        if any(row["evidence_count"] != counts[row["combo_id"]] for row in rows):
            _fail("run_unavailable", "Run results are unavailable", 409)
        return counts

    def _legacy_evidence_counts(
        self,
        context: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, int]:
        envelope = _decode_json(
            self._read(
                context["run_layout"]["root"],
                "evidence.json",
                self.MAX_LEGACY_EVIDENCE_BYTES,
            )
        )
        expected_mode = (
            "retrieval_labels"
            if any(row["labelled_queries"] > 0 for row in rows)
            else "evidence_only"
        )
        manifest_mode = context["manifest"].get("scoring_mode")
        if (
            set(envelope) != {"schema_version", "project_id", "run_id", "mode", "rows"}
            or envelope.get("schema_version") != 1
            or envelope.get("project_id") != context["manifest"]["project_id"]
            or envelope.get("run_id") != context["manifest"]["run_id"]
            or envelope.get("mode") != expected_mode
            or (manifest_mode is not None and manifest_mode != expected_mode)
            or not isinstance(envelope.get("rows"), list)
        ):
            _fail("run_unavailable", "Run results are unavailable", 409)
        completed = {row["combo_id"] for row in rows if row["status"] == "completed"}
        counts = {row["combo_id"]: 0 for row in rows}
        for row in envelope["rows"]:
            combo_id = row.get("combo_id") if isinstance(row, dict) else None
            if not isinstance(combo_id, str) or combo_id not in completed:
                _fail("run_unavailable", "Run results are unavailable", 409)
            _public_evidence_row(row, combo_id)
            counts[combo_id] += 1
        return counts

    def project_run_results(self, project_id: str, run_id: str) -> dict[str, Any]:
        context = self._run_context(project_id, run_id)
        rows, legacy = self._parse_summary(context)
        self._validate_manifest_summary(context, rows, legacy=legacy)
        if legacy:
            counts = self._legacy_evidence_counts(context, rows)
            for row in rows:
                row["evidence_count"] = counts[row["combo_id"]]
        else:
            counts = self._v2_evidence_counts(context, rows)

        ledger: list[dict[str, Any]] = []
        seen_usage: dict[str, tuple[str, int | None]] = {}
        for row in rows:
            usage = row["measured_usage"]
            key = usage["embedding_usage_key"]
            if usage["embedding_usage_scope"] != "shared_embedding" or not key:
                continue
            measurement = (row["embedding_id"], usage["embedding_input_tokens"])
            if key in seen_usage:
                if seen_usage[key] != measurement:
                    _fail("run_unavailable", "Run results are unavailable", 409)
                continue
            seen_usage[key] = measurement
            ledger.append(
                {
                    "embedding_id": row["embedding_id"],
                    "embedding_input_tokens": usage["embedding_input_tokens"],
                    "embedding_usage_scope": usage["embedding_usage_scope"],
                    "embedding_usage_key": key,
                }
            )

        manifest = context["manifest"]
        scoring_mode = (
            "retrieval_labels"
            if any(row["labelled_queries"] > 0 for row in rows)
            else "evidence_only"
        ) if legacy else manifest["scoring_mode"]
        succeeded = sum(row["status"] == "completed" for row in rows)
        failed = len(rows) - succeeded
        public_rows = [
            {key: value for key, value in row.items() if key != "physical_namespace"}
            for row in rows
        ]
        source_identity = self._result_source_identity(
                    context["validated"].request,
                    project_id=project_id,
                    project_manifest=context.get("project_manifest"),
                )
        return {
            "source_type": "uploaded_project",
            "project_id": project_id,
            "dataset_id": f"project:{project_id}",
            "project_label": context["project_manifest"]["label"].strip(),
            "run_id": run_id,
            "run_state": manifest["state"],
            "scoring_mode": scoring_mode,
            "metric_k": context["validated"].request.top_k,
            "combination_count": len(rows),
            "succeeded": succeeded,
            "failed": failed,
            "metric_names": ["recall_at_k", "mrr_at_k", "ndcg_at_k", "avg_query_latency_s"],
            "summary_schema_version": 1 if legacy else 2,
            "created_at": manifest.get("created_at"),
            "completed_at": manifest.get("completed_at"),
            "rows": public_rows,
            "evidence_counts_by_combo": counts,
            "measured_usage_ledger": ledger,
            **source_identity,
        }

    def project_run_evidence(
        self,
        project_id: str,
        run_id: str,
        combo_id: str,
        *,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        if (
            not isinstance(combo_id, str)
            or not combo_id
            or len(combo_id) > 512
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.MAX_EVIDENCE_PAGE
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            _fail("invalid_request", "Invalid result request", 400)
        context = self._run_context(project_id, run_id)
        rows, legacy = self._parse_summary(context)
        self._validate_manifest_summary(context, rows, legacy=legacy)
        owned = next((row for row in rows if row["combo_id"] == combo_id), None)
        if owned is None or owned["status"] != "completed":
            _fail("not_found", "Combination evidence was not found", 404)
        evidence_index = None
        if legacy:
            self._legacy_evidence_counts(context, rows)
        else:
            evidence_index = self._v2_evidence_index(context, rows)
            if evidence_index is None:
                self._v2_evidence_counts(context, rows)
        if evidence_index is not None:
            page = self._indexed_evidence_page(
                context,
                evidence_index[combo_id],
                combo_id=combo_id,
                offset=offset,
                limit=limit,
            )
        else:
            try:
                page = read_project_json_rows_page(
                    context["run_layout"]["root"],
                    "evidence.json",
                    combo_id=combo_id,
                    offset=offset,
                    limit=limit,
                    max_bytes=self.MAX_LEGACY_EVIDENCE_BYTES,
                )
            except ProjectRunArtifactError as exc:
                _artifact_error(exc)
        if page.get("project_id") != project_id or page.get("run_id") != run_id:
            _fail("run_unavailable", "Run results are unavailable", 409)
        safe_rows = [_public_evidence_row(row, combo_id) for row in page["rows"]]
        return {
            "project_id": project_id,
            "run_id": run_id,
            "combo_id": combo_id,
            "rows": safe_rows,
            "next_offset": page["next_offset"],
        }
