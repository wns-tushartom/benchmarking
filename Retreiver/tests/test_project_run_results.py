from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from source.services.project_matrix_contract import validate_project_matrix_request
from source.services.project_run_results import (
    ProjectRunResultService,
    ProjectRunResultsError,
)
from source.services.project_workspace import ProjectWorkspace


CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs" / "project_matrix_catalog.json"
V2_FIELDS = (
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
LEGACY_FIELDS = (
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


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_project(workspace: ProjectWorkspace, label: str, created_at: str) -> str:
    project_id = workspace.new_project_id(label)
    root = workspace.project_root(project_id)
    root.mkdir(parents=True)
    layout = workspace.layout(project_id)
    for name in ("raw_uploads", "extracted_text", "chunks", "questions", "indexes", "runs"):
        layout[name].mkdir()
    _json(
        root / "manifest.json",
        {
            "ok": True,
            "project_id": project_id,
            "label": label,
            "created_at": created_at,
        },
    )
    return project_id


def _request(project_id: str, *, top_k: int = 10, rerankers: list[str] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_id": project_id,
        "top_k": top_k,
        "questions_source": {"type": "typed", "query": "What is the refund policy?"},
        "selections": {
            "chunkers": ["fixed_tok1200_ov150"],
            "embeddings": ["openai_text-embedding-3-large"],
            "vector_stores": ["FAISS"],
            "rerankers": rerankers or ["Amazon Rerank v1"],
        },
        "large_matrix_confirmation": None,
    }


def _base_row(project_id: str, run_id: str, combo_id: str) -> dict[str, object]:
    return {
        "project_id": project_id,
        "run_id": run_id,
        "combo_id": combo_id,
        "status": "completed",
        "summary_schema_version": 2,
        "chunker_id": "fixed_tok1200_ov150",
        "embedding_id": "openai_text-embedding-3-large",
        "vector_store_id": "FAISS",
        "reranker_id": "Amazon Rerank v1",
        "physical_namespace": "project-run-namespace",
        "query_count": 2,
        "labelled_queries": 2,
        "unlabelled_queries": 0,
        "recall_at_k": 0.75,
        "mrr_at_k": 0.625,
        "ndcg_at_k": 0.7,
        "retrieval_latency_s": 0.1,
        "rerank_latency_s": 0.2,
        "avg_query_latency_s": 0.3,
        "evidence_count": 3,
        "embedding_input_tokens": 123,
        "embedding_usage_scope": "shared_embedding",
        "embedding_usage_key": "fixed_tok1200_ov150|openai_text-embedding-3-large",
        "rerank_search_units": 2,
        "rerank_usage_scope": "combination",
        "error_code": "",
    }


def _receipt(row: dict[str, object]) -> dict[str, object]:
    def measured(name: str) -> object | None:
        value = row.get(name)
        return None if value in {None, ""} else value

    return {
        "combo_id": row["combo_id"],
        "status": row["status"],
        "usage": {
            "embedding_input_tokens": measured("embedding_input_tokens"),
            "embedding_usage_scope": measured("embedding_usage_scope"),
            "embedding_usage_key": measured("embedding_usage_key"),
            "rerank_search_units": measured("rerank_search_units"),
            "rerank_usage_scope": measured("rerank_usage_scope"),
        },
    }


def _make_run(
    workspace: ProjectWorkspace,
    project_id: str,
    *,
    rows: list[dict[str, object]],
    request: dict[str, object] | None = None,
    schema_version: int = 2,
    state: str = "completed",
    scoring_mode: str | None = "retrieval_labels",
    created_at: str | None = "2026-07-13T10:00:00Z",
    completed_at: str | None = "2026-07-13T10:12:00Z",
    mode: str | None = None,
    evidence_rows: list[dict[str, object]] | None = None,
) -> tuple[str, Path]:
    payload = request or _request(project_id, rerankers=[str(row["reranker_id"]) for row in rows])
    validated = validate_project_matrix_request(payload, catalog_path=CATALOG_PATH)
    assert validated.combination_count == len(rows)
    run_id = workspace.new_run_id()
    root = workspace.create_run_layout(project_id, run_id)["root"]
    for row in rows:
        row["project_id"] = project_id
        row["run_id"] = run_id
    _json(
        root / "request.json",
        {
            "schema_version": 1,
            "request": payload,
            "request_fingerprint": validated.request_fingerprint,
            "combination_count": validated.combination_count,
        },
    )
    fields = V2_FIELDS if schema_version == 2 else LEGACY_FIELDS
    _csv(root / "summary.csv", fields, rows)
    evidence = evidence_rows
    if evidence is None:
        evidence = [
            {
                "combo_id": rows[0]["combo_id"],
                "query_id": f"q{index}",
                "query": "refund?",
                "source_name": "alpha.txt",
                "page_number": "1",
                "excerpt": f"evidence {index}",
                "paragraph": "must not escape",
                "base_score": 0.8,
                "rerank_score": 0.9,
                "latency_s": 0.02,
                "rank": index,
                "private_path": "/srv/secret/project",
            }
            for index in range(1, 4)
        ]
    _json(
        root / "evidence.json",
        {
            "schema_version": 1,
            "project_id": project_id,
            "run_id": run_id,
            "mode": scoring_mode or "evidence_only",
            "rows": evidence,
        },
    )
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "project_id": project_id,
        "run_id": run_id,
        "request_fingerprint": validated.request_fingerprint,
        "combination_count": validated.combination_count,
        "state": state,
        "succeeded": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "receipts": [_receipt(row) for row in rows] if schema_version == 2 else [],
        "artifacts": {
            "summary": "summary.csv",
            "evidence": "evidence.json",
            "analysis": "analysis.json" if scoring_mode == "retrieval_labels" else None,
        },
    }
    if schema_version == 2:
        manifest["summary_schema_version"] = 2
        manifest["metric_k"] = payload["top_k"]
    if scoring_mode is not None:
        manifest["scoring_mode"] = scoring_mode
    if created_at is not None:
        manifest["created_at"] = created_at
    if completed_at is not None:
        manifest["completed_at"] = completed_at
    if mode is not None:
        manifest["mode"] = mode
    _json(root / "manifest.json", manifest)
    return run_id, root


def _add_indexed_evidence(
    root: Path,
    project_id: str,
    run_id: str,
    rows: list[dict[str, object]],
    evidence: list[dict[str, object]],
    mode: str,
) -> None:
    entries = []
    for row in rows:
        combo_id = str(row["combo_id"])
        combo_rows = [item for item in evidence if item.get("combo_id") == combo_id]
        content = b"".join(
            (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            for item in combo_rows
        )
        path = root / "reranking" / f"{combo_id}.evidence.00000.jsonl"
        path.write_bytes(content)
        entries.append(
            {
                "combo_id": combo_id,
                "row_count": len(combo_rows),
                "parts": [
                    {
                        "path": f"reranking/{combo_id}.evidence.00000.jsonl",
                        "row_count": len(combo_rows),
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        )
    index = {
        "schema_version": 1,
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "entries": entries,
    }
    _json(root / "evidence_index.json", index)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["evidence_index"] = "evidence_index.json"
    manifest["evidence_index_sha256"] = hashlib.sha256(
        (root / "evidence_index.json").read_bytes()
    ).hexdigest()
    _json(root / "manifest.json", manifest)


def test_indexed_v2_results_and_pages_do_not_materialize_global_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "Indexed", "2026-07-13T09:00:00Z")
    row = _base_row(project_id, "pending", "combo_indexed")
    evidence = [
        {
            "combo_id": "combo_indexed",
            "query_id": f"q{i}",
            "query": "refund?",
            "source_name": "a.txt",
            "excerpt": f"row {i}",
            "rank": i,
        }
        for i in range(1, 4)
    ]
    run_id, root = _make_run(
        workspace,
        project_id,
        rows=[row],
        evidence_rows=evidence,
    )
    _add_indexed_evidence(
        root,
        project_id,
        run_id,
        [row],
        evidence,
        "retrieval_labels",
    )
    (root / "evidence.json").write_text(
        "this global artifact must not be read",
        encoding="utf-8",
    )

    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)
    original_read = service._read
    reads: list[str] = []

    def recording_read(read_root: Path, name: str, maximum: int) -> bytes:
        reads.append(name)
        assert name != "evidence.json"
        return original_read(read_root, name, maximum)

    monkeypatch.setattr(service, "_read", recording_read)
    result = service.project_run_results(project_id, run_id)
    page = service.project_run_evidence(
        project_id,
        run_id,
        "combo_indexed",
        limit=2,
        offset=0,
    )
    assert result["evidence_counts_by_combo"] == {"combo_indexed": 3}
    assert [item["query_id"] for item in page["rows"]] == ["q1", "q2"]
    assert page["next_offset"] == 2
    assert "evidence_index.json" in reads
    assert "reranking/combo_indexed.evidence.00000.jsonl" in reads


@pytest.fixture
def result_workspace(tmp_path: Path) -> dict[str, object]:
    workspace = ProjectWorkspace(tmp_path / "projects")
    alpha_id = _make_project(workspace, "Alpha refunds", "2026-07-13T09:00:00Z")
    beta_id = _make_project(workspace, "Beta claims", "2026-07-12T09:00:00Z")

    labelled = _base_row(alpha_id, "pending", "combo_labeled")
    labelled_run_id, labelled_root = _make_run(
        workspace,
        alpha_id,
        rows=[labelled],
    )

    legacy = _base_row(alpha_id, "pending", "combo_legacy")
    for key in set(legacy) - set(LEGACY_FIELDS):
        legacy.pop(key)
    legacy["recall_at_k"] = 0.5
    legacy_run_id, legacy_root = _make_run(
        workspace,
        alpha_id,
        rows=[legacy],
        schema_version=1,
        scoring_mode="retrieval_labels",
        created_at=None,
        completed_at=None,
        evidence_rows=[
            {
                "combo_id": "combo_legacy",
                "query_id": "q1",
                "query": "legacy?",
                "source_name": "legacy.txt",
                "page_number": "2",
                "excerpt": "legacy evidence",
                "rank": 1,
            }
        ],
    )
    legacy_time = datetime(2026, 7, 11, 10, 0, tzinfo=timezone.utc).timestamp()
    os.utime(legacy_root / "manifest.json", (legacy_time, legacy_time))

    partial_completed = _base_row(alpha_id, "pending", "combo_partial_ok")
    partial_completed["reranker_id"] = "Qwen3:4B Rerank"
    partial_completed.update(
        embedding_input_tokens="",
        embedding_usage_scope="",
        embedding_usage_key="",
        rerank_search_units="",
        rerank_usage_scope="",
        evidence_count=1,
    )
    partial_failed = dict(partial_completed)
    partial_failed.update(
        combo_id="combo_partial_failed",
        reranker_id="bge-reranker-base",
        status="failed",
        recall_at_k="",
        mrr_at_k="",
        ndcg_at_k="",
        retrieval_latency_s="",
        rerank_latency_s="",
        avg_query_latency_s="",
        evidence_count=0,
        error_code="reranker_failed",
    )
    partial_request = _request(
        alpha_id,
        rerankers=["Qwen3:4B Rerank", "bge-reranker-base"],
    )
    partial_run_id, partial_root = _make_run(
        workspace,
        alpha_id,
        rows=[partial_completed, partial_failed],
        request=partial_request,
        state="partial",
        scoring_mode="retrieval_labels",
        created_at="2026-07-13T09:00:00Z",
        completed_at="2026-07-13T09:10:00Z",
        evidence_rows=[
            {
                "combo_id": "combo_partial_ok",
                "query_id": "q1",
                "query": "partial?",
                "source_name": "alpha.txt",
                "page_number": "1",
                "excerpt": "safe",
                "rank": 1,
            }
        ],
    )

    lexical = _base_row(alpha_id, "pending", "combo_lexical")
    lexical_run_id, _ = _make_run(
        workspace,
        alpha_id,
        rows=[lexical],
        mode="lexical_preview",
    )

    beta_row = _base_row(beta_id, "pending", "combo_beta")
    beta_run_id, _ = _make_run(workspace, beta_id, rows=[beta_row])

    malformed = workspace.base / "not-a-project"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("{raw /srv/error", encoding="utf-8")
    (workspace.base / "linked-project").symlink_to(workspace.project_root(beta_id), target_is_directory=True)
    bad_run = workspace.layout(alpha_id)["runs"] / "run_not_uuid"
    bad_run.mkdir()
    (workspace.layout(alpha_id)["runs"] / "run_" + Path("x" * 32)) if False else None
    linked_run = workspace.layout(alpha_id)["runs"] / ("run_" + "f" * 32)
    linked_run.symlink_to(workspace.run_layout(beta_id, beta_run_id)["root"], target_is_directory=True)

    return {
        "workspace": workspace,
        "service": ProjectRunResultService(workspace, catalog_path=CATALOG_PATH),
        "alpha_id": alpha_id,
        "beta_id": beta_id,
        "labelled_run_id": labelled_run_id,
        "labelled_root": labelled_root,
        "legacy_run_id": legacy_run_id,
        "legacy_root": legacy_root,
        "partial_run_id": partial_run_id,
        "partial_root": partial_root,
        "lexical_run_id": lexical_run_id,
        "beta_run_id": beta_run_id,
    }


def _error(exc: pytest.ExceptionInfo[ProjectRunResultsError], code: str, status: int) -> None:
    assert exc.value.code == code
    assert exc.value.status == status
    assert exc.value.public_message == str(exc.value)
    assert "/" not in str(exc.value)
    assert "Traceback" not in str(exc.value)


def test_public_contract_lists_only_valid_projects_and_matrix_runs(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    assert isinstance(service, ProjectRunResultService)
    error = ProjectRunResultsError("not_found", "Result was not found", 404)
    assert error.code == "not_found"
    assert error.public_message == "Result was not found"
    assert error.status == 404

    sources = service.result_sources(official_configured=180, official_evaluated=175)
    assert sources["official"] == {
        "source_type": "official",
        "configured": 180,
        "evaluated": 175,
    }
    assert [row["project_id"] for row in sources["projects"]] == [
        result_workspace["alpha_id"],
        result_workspace["beta_id"],
    ]
    assert sources["projects"][0]["label"] == "Alpha refunds"
    assert sources["projects"][0]["dataset_id"] == f"project:{result_workspace['alpha_id']}"

    runs = service.project_runs(str(result_workspace["alpha_id"]))
    assert {row["run_id"] for row in runs} == {
        result_workspace["labelled_run_id"],
        result_workspace["legacy_run_id"],
        result_workspace["partial_run_id"],
    }
    assert result_workspace["lexical_run_id"] not in {row["run_id"] for row in runs}
    assert runs[0]["run_id"] == result_workspace["labelled_run_id"]
    assert runs[0]["dataset_id"] == f"project:{result_workspace['alpha_id']}"
    assert runs[0]["groundtruth_id"] == "groundtruth:none"
    assert runs[0]["groundtruth_label"] == "None (evidence-only)"
    legacy = next(row for row in runs if row["run_id"] == result_workspace["legacy_run_id"])
    assert legacy["timestamp_label"] == "Legacy artifact time"
    assert legacy["artifact_time"].endswith("Z")


@pytest.mark.parametrize(
    ("configured", "evaluated"),
    ((-1, 0), (1, -1), (True, 0), (1, 1.0)),
)
def test_result_sources_rejects_invalid_official_counts(
    result_workspace: dict[str, object], configured: object, evaluated: object
):
    with pytest.raises(ProjectRunResultsError) as caught:
        result_workspace["service"].result_sources(  # type: ignore[union-attr]
            official_configured=configured,
            official_evaluated=evaluated,
        )
    _error(caught, "invalid_request", 400)


def test_schema_v2_results_are_normalized_without_cross_source_fallback(
    result_workspace: dict[str, object],
):
    result = result_workspace["service"].project_run_results(  # type: ignore[union-attr]
        str(result_workspace["alpha_id"]),
        str(result_workspace["labelled_run_id"]),
    )

    assert result["source_type"] == "uploaded_project"
    assert result["project_id"] == result_workspace["alpha_id"]
    assert result["project_label"] == "Alpha refunds"
    assert result["run_id"] == result_workspace["labelled_run_id"]
    assert result["run_state"] == "completed"
    assert result["scoring_mode"] == "retrieval_labels"
    assert result["dataset_id"] == f"project:{result_workspace['alpha_id']}"
    assert result["groundtruth_id"] == "groundtruth:none"
    assert result["groundtruth_label"] == "None (evidence-only)"
    assert result["metric_k"] == 10
    assert result["combination_count"] == 1
    assert result["succeeded"] == 1
    assert result["failed"] == 0
    assert result["metric_names"] == [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "recall_at_k",
        "mrr_at_k",
        "precision_at_5",
        "ndcg_at_5",
        "ndcg_at_k",
        "avg_first_relevant_rank",
        "no_hit_queries",
        "avg_query_latency_s",
    ]
    assert result["evidence_counts_by_combo"] == {"combo_labeled": 3}
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert "physical_namespace" not in row
    assert row["project_id"] == result_workspace["alpha_id"]
    assert row["run_id"] == result_workspace["labelled_run_id"]
    assert row["recall_at_k"] == 0.75
    assert row["avg_query_latency_s"] == 0.3
    assert row["commercial_model_ids"] == [
        "openai_text-embedding-3-large",
        "Amazon Rerank v1",
    ]
    assert row["measured_usage"] == {
        "embedding_input_tokens": 123,
        "embedding_usage_scope": "shared_embedding",
        "embedding_usage_key": "fixed_tok1200_ov150|openai_text-embedding-3-large",
        "rerank_search_units": 2,
        "rerank_usage_scope": "combination",
    }
    assert result["measured_usage_ledger"] == [
        {
            "embedding_id": "openai_text-embedding-3-large",
            "embedding_input_tokens": 123,
            "embedding_usage_scope": "shared_embedding",
            "embedding_usage_key": "fixed_tok1200_ov150|openai_text-embedding-3-large",
        }
    ]
    assert "total_cost_usd" not in row

    with pytest.raises(ProjectRunResultsError) as caught:
        result_workspace["service"].project_run_results(  # type: ignore[union-attr]
            str(result_workspace["beta_id"]),
            str(result_workspace["labelled_run_id"]),
        )
    _error(caught, "not_found", 404)


def test_legacy_recall_is_preserved_and_missing_values_stay_none(
    result_workspace: dict[str, object],
):
    result = result_workspace["service"].project_run_results(  # type: ignore[union-attr]
        str(result_workspace["alpha_id"]),
        str(result_workspace["legacy_run_id"]),
    )
    row = result["rows"][0]
    assert result["metric_k"] == 10
    assert result["scoring_mode"] == "retrieval_labels"
    assert row["recall_at_k"] == 0.5
    assert row["mrr_at_k"] is None
    assert row["ndcg_at_k"] is None
    assert row["retrieval_latency_s"] is None
    assert row["rerank_latency_s"] is None
    assert row["avg_query_latency_s"] is None
    assert row["evidence_count"] == 1
    assert result["evidence_counts_by_combo"] == {"combo_legacy": 1}
    assert row["measured_usage"] == {
        "embedding_input_tokens": None,
        "embedding_usage_scope": None,
        "embedding_usage_key": None,
        "rerank_search_units": None,
        "rerank_usage_scope": None,
    }


def test_shared_embedding_usage_is_deduplicated_in_run_ledger(
    result_workspace: dict[str, object],
):
    root = result_workspace["labelled_root"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))  # type: ignore[operator]
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))  # type: ignore[operator]
    request["request"]["selections"]["vector_stores"] = ["FAISS", "Qdrant"]
    validated = validate_project_matrix_request(request["request"], catalog_path=CATALOG_PATH)
    request["request_fingerprint"] = validated.request_fingerprint
    request["combination_count"] = 2
    _json(root / "request.json", request)  # type: ignore[operator]

    first = _base_row(str(result_workspace["alpha_id"]), str(result_workspace["labelled_run_id"]), "combo_labeled")
    second = dict(first)
    second.update(combo_id="combo_labeled_2", vector_store_id="Qdrant", evidence_count=0)
    _csv(root / "summary.csv", V2_FIELDS, [first, second])  # type: ignore[operator]
    manifest["request_fingerprint"] = validated.request_fingerprint
    manifest["combination_count"] = 2
    manifest["succeeded"] = 2
    manifest["receipts"] = [_receipt(first), _receipt(second)]
    _json(root / "manifest.json", manifest)  # type: ignore[operator]

    result = result_workspace["service"].project_run_results(  # type: ignore[union-attr]
        str(result_workspace["alpha_id"]), str(result_workspace["labelled_run_id"])
    )
    assert len(result["rows"]) == 2
    assert len(result["measured_usage_ledger"]) == 1
    assert all("total_cost_usd" not in row for row in result["rows"])


def test_evidence_is_exact_paginated_and_whitelisted(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    page = service.project_run_evidence(  # type: ignore[union-attr]
        str(result_workspace["alpha_id"]),
        str(result_workspace["labelled_run_id"]),
        "combo_labeled",
        limit=2,
        offset=0,
    )
    assert page["project_id"] == result_workspace["alpha_id"]
    assert page["run_id"] == result_workspace["labelled_run_id"]
    assert page["combo_id"] == "combo_labeled"
    assert len(page["rows"]) == 2
    assert page["next_offset"] == 2
    assert all(row["combo_id"] == "combo_labeled" for row in page["rows"])
    assert set(page["rows"][0]) <= {
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
    }
    assert "paragraph" not in page["rows"][0]
    assert "private_path" not in page["rows"][0]

    final = service.project_run_evidence(  # type: ignore[union-attr]
        str(result_workspace["alpha_id"]),
        str(result_workspace["labelled_run_id"]),
        "combo_labeled",
        limit=2,
        offset=2,
    )
    assert len(final["rows"]) == 1
    assert final["next_offset"] is None


@pytest.mark.parametrize("combo_id", ("combo_partial_failed", "combo_foreign"))
def test_failed_or_foreign_combo_evidence_is_safe_not_found(
    result_workspace: dict[str, object], combo_id: str
):
    run_id = (
        result_workspace["partial_run_id"]
        if combo_id == "combo_partial_failed"
        else result_workspace["labelled_run_id"]
    )
    with pytest.raises(ProjectRunResultsError) as caught:
        result_workspace["service"].project_run_evidence(  # type: ignore[union-attr]
            str(result_workspace["alpha_id"]), str(run_id), combo_id, limit=2, offset=0
        )
    _error(caught, "not_found", 404)


@pytest.mark.parametrize(
    ("combo_id", "limit", "offset"),
    (("", 1, 0), ("combo_labeled", 0, 0), ("combo_labeled", 101, 0), ("combo_labeled", 1, -1), ("combo_labeled", True, 0)),
)
def test_evidence_rejects_invalid_request_values(
    result_workspace: dict[str, object], combo_id: str, limit: int, offset: int
):
    with pytest.raises(ProjectRunResultsError) as caught:
        result_workspace["service"].project_run_evidence(  # type: ignore[union-attr]
            str(result_workspace["alpha_id"]),
            str(result_workspace["labelled_run_id"]),
            combo_id,
            limit=limit,
            offset=offset,
        )
    _error(caught, "invalid_request", 400)


def test_symlinked_manifest_summary_and_evidence_fail_closed(
    result_workspace: dict[str, object], tmp_path: Path
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    outside = tmp_path / "outside"
    outside.mkdir()

    for artifact, operation in (
        ("manifest.json", lambda: service.project_run_results(project_id, run_id)),
        ("summary.csv", lambda: service.project_run_results(project_id, run_id)),
        (
            "evidence.json",
            lambda: service.project_run_evidence(
                project_id, run_id, "combo_labeled", limit=2, offset=0
            ),
        ),
    ):
        path = root / artifact  # type: ignore[operator]
        original = path.read_bytes()
        path.unlink()
        outside_path = outside / artifact
        outside_path.write_bytes(original)
        path.symlink_to(outside_path)
        try:
            with pytest.raises(ProjectRunResultsError) as caught:
                operation()
            _error(caught, "run_unavailable", 409)
            assert str(outside) not in str(caught.value)
        finally:
            path.unlink()
            path.write_bytes(original)


def test_malformed_json_csv_and_raw_errors_are_sanitized(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    summary_path = root / "summary.csv"  # type: ignore[operator]
    manifest_original = manifest_path.read_bytes()
    summary_original = summary_path.read_bytes()

    for path, content in (
        (manifest_path, b'{"raw_error":"/srv/private/tenant-alpha"'),
        (summary_path, b"project_id,run_id\nraw,/srv/private/tenant-alpha\n"),
    ):
        path.write_bytes(content)
        try:
            with pytest.raises(ProjectRunResultsError) as caught:
                service.project_run_results(project_id, run_id)
            _error(caught, "run_unavailable", 409)
            assert "private" not in str(caught.value)
            assert "tenant-alpha" not in str(caught.value)
        finally:
            manifest_path.write_bytes(manifest_original)
            summary_path.write_bytes(summary_original)

    evidence_path = root / "evidence.json"  # type: ignore[operator]
    evidence_path.write_text("{not-json /srv/private", encoding="utf-8")
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_evidence(project_id, run_id, "combo_labeled", limit=2, offset=0)
    _error(caught, "run_unavailable", 409)
    assert "private" not in str(caught.value)


def test_oversized_summary_and_row_overflow_are_rejected(
    result_workspace: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    summary_path = root / "summary.csv"  # type: ignore[operator]
    original = summary_path.read_bytes()

    monkeypatch.setattr(service, "MAX_SUMMARY_BYTES", len(original) - 1)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "artifact_too_large", 413)
    monkeypatch.setattr(service, "MAX_SUMMARY_BYTES", 16 * 1024 * 1024)

    header = ",".join(LEGACY_FIELDS) + "\n"
    row = ",".join(
        [
            project_id,
            run_id,
            "combo_overflow",
            "completed",
            "fixed_tok1200_ov150",
            "openai_text-embedding-3-large",
            "FAISS",
            "Amazon Rerank v1",
            "namespace",
            "1",
            "1",
            "0",
            "1",
            "",
        ]
    ) + "\n"
    summary_path.write_text(header + row * 3, encoding="utf-8")
    monkeypatch.setattr(service, "MAX_SUMMARY_ROWS", 2)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_header",
        "missing_header",
        "duplicate_combo",
        "identity",
        "unknown_adapter",
        "nonfinite_metric",
        "negative_metric",
        "metric_above_one",
        "query_count_mismatch",
    ),
)
def test_invalid_summary_schema_or_values_fail_closed(
    result_workspace: dict[str, object], mutation: str
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    path = root / "summary.csv"  # type: ignore[operator]
    row = _base_row(project_id, run_id, "combo_labeled")

    if mutation == "duplicate_header":
        text = path.read_text(encoding="utf-8")
        header, body = text.split("\n", 1)
        path.write_text(header + ",combo_id\n" + body.rstrip("\n") + ",duplicate\n", encoding="utf-8")
    elif mutation == "missing_header":
        fields = tuple(field for field in V2_FIELDS if field != "run_id")
        _csv(path, fields, [{key: value for key, value in row.items() if key != "run_id"}])
    elif mutation == "duplicate_combo":
        _csv(path, V2_FIELDS, [row, row])
    elif mutation == "identity":
        row["project_id"] = str(result_workspace["beta_id"])
        _csv(path, V2_FIELDS, [row])
    elif mutation == "unknown_adapter":
        row["embedding_id"] = "local-fallback-secret"
        _csv(path, V2_FIELDS, [row])
    elif mutation == "nonfinite_metric":
        row["ndcg_at_k"] = "NaN"
        _csv(path, V2_FIELDS, [row])
    elif mutation == "negative_metric":
        row["avg_query_latency_s"] = -0.1
        _csv(path, V2_FIELDS, [row])
    elif mutation == "metric_above_one":
        row["recall_at_k"] = 1.01
        _csv(path, V2_FIELDS, [row])
    else:
        row["query_count"] = 3
        _csv(path, V2_FIELDS, [row])

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)
    assert "secret" not in str(caught.value)


def test_summary_must_cover_the_exact_selected_cartesian_matrix(
    result_workspace: dict[str, object],
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["partial_run_id"])
    path = result_workspace["partial_root"] / "summary.csv"  # type: ignore[operator]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = [dict(row) for row in csv.DictReader(handle)]
    assert len(rows) == 2
    assert rows[0]["reranker_id"] != rows[1]["reranker_id"]
    rows[1]["reranker_id"] = rows[0]["reranker_id"]
    _csv(path, V2_FIELDS, rows)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


def test_request_and_manifest_identity_fingerprint_and_fixed_evidence_path_are_required(
    result_workspace: dict[str, object]
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    request_path = root / "request.json"  # type: ignore[operator]
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    request_original = request_path.read_bytes()
    manifest_original = manifest_path.read_bytes()

    request = json.loads(request_original)
    request["request_fingerprint"] = "0" * 64
    _json(request_path, request)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)
    request_path.write_bytes(request_original)

    manifest = json.loads(manifest_original)
    manifest["artifacts"]["evidence"] = "reranking/private.json"
    _json(manifest_path, manifest)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


@pytest.mark.parametrize(
    "mutation",
    (
        "schema_version",
        "summary_schema_version",
        "metric_k",
        "succeeded",
        "failed",
        "receipt_status",
        "receipt_usage",
    ),
)
def test_v2_manifest_and_receipts_must_match_summary_and_request(
    result_workspace: dict[str, object], mutation: str
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    path = result_workspace["labelled_root"] / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "schema_version":
        manifest["schema_version"] = 1
    elif mutation == "summary_schema_version":
        manifest["summary_schema_version"] = 1
    elif mutation == "metric_k":
        manifest["metric_k"] = 999
    elif mutation == "succeeded":
        manifest["succeeded"] = 0
    elif mutation == "failed":
        manifest["failed"] = 1
    elif mutation == "receipt_status":
        manifest["receipts"][0]["status"] = "failed"
    else:
        manifest["receipts"][0]["usage"]["rerank_search_units"] = 999
    _json(path, manifest)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


@pytest.mark.parametrize("mutation", ("mode", "count"))
def test_v2_evidence_mode_and_counts_must_match_summary(
    result_workspace: dict[str, object], mutation: str
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    path = result_workspace["labelled_root"] / "evidence.json"  # type: ignore[operator]
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "mode":
        evidence["mode"] = "evidence_only"
    else:
        evidence["rows"] = []
    _json(path, evidence)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


def test_run_catalog_excludes_run_with_missing_summary(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    summary = result_workspace["labelled_root"] / "summary.csv"  # type: ignore[operator]
    summary.unlink()

    listed = service.project_runs(project_id)  # type: ignore[union-attr]
    assert run_id not in {row["run_id"] for row in listed}


def test_legacy_malformed_evidence_fails_closed(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["legacy_run_id"])
    root = result_workspace["legacy_root"]
    evidence = root / "evidence.json"  # type: ignore[operator]
    evidence.write_text("{malformed", encoding="utf-8")

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


def test_legacy_manifest_counters_must_match_summary(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["legacy_run_id"])
    path = result_workspace["legacy_root"] / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["succeeded"] = 0
    _json(path, manifest)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


def test_v2_conflicting_shared_embedding_usage_fails_closed(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "Usage conflict", "2026-07-13T09:00:00Z")
    first = _base_row(project_id, "pending", "combo_usage_a")
    second = _base_row(project_id, "pending", "combo_usage_b")
    second["vector_store_id"] = "Qdrant"
    second["evidence_count"] = 0
    second["embedding_input_tokens"] = 456
    request = _request(project_id)
    request["selections"]["vector_stores"] = ["FAISS", "Qdrant"]  # type: ignore[index]
    run_id, _root = _make_run(
        workspace,
        project_id,
        rows=[first, second],
        request=request,
    )
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


def test_v2_evidence_only_row_cannot_carry_quality_metrics(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "Evidence mode", "2026-07-13T09:00:00Z")
    row = _base_row(project_id, "pending", "combo_evidence_quality")
    row["labelled_queries"] = 0
    row["unlabelled_queries"] = 2
    run_id, _root = _make_run(
        workspace,
        project_id,
        rows=[row],
        scoring_mode="evidence_only",
    )
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


def test_v2_failed_row_cannot_carry_quality_or_latency_metrics(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "Failed metrics", "2026-07-13T09:00:00Z")
    completed = _base_row(project_id, "pending", "combo_ok")
    failed = _base_row(project_id, "pending", "combo_failed_metrics")
    failed["vector_store_id"] = "Qdrant"
    failed["status"] = "failed"
    failed["error_code"] = "retrieval_failed"
    failed["evidence_count"] = 0
    request = _request(project_id)
    request["selections"]["vector_stores"] = ["FAISS", "Qdrant"]  # type: ignore[index]
    run_id, _root = _make_run(
        workspace,
        project_id,
        rows=[completed, failed],
        request=request,
        state="partial",
    )
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


def test_project_evidence_rejects_absolute_source_names(result_workspace: dict[str, object]):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    path = root / "evidence.json"  # type: ignore[operator]
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["rows"][0]["source_name"] = "/tmp/example/secret.txt"
    _json(path, evidence)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_evidence(  # type: ignore[union-attr]
            project_id,
            run_id,
            "combo_labeled",
            limit=2,
            offset=0,
        )
    _error(caught, "run_unavailable", 409)


def test_legacy_summary_version_and_evidence_mode_must_be_consistent(
    result_workspace: dict[str, object],
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["legacy_run_id"])
    root = result_workspace["legacy_root"]

    manifest_path = root / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_schema_version"] = 2
    _json(manifest_path, manifest)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)

    manifest.pop("summary_schema_version")
    _json(manifest_path, manifest)
    evidence_path = root / "evidence.json"  # type: ignore[operator]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["mode"] = "evidence_only"
    _json(evidence_path, evidence)
    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("summary_schema_version", True),
        ("summary_schema_version", 1.0),
    ],
)
def test_legacy_manifest_requires_exact_integer_schema_versions(
    result_workspace: dict[str, object],
    field: str,
    value: object,
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["legacy_run_id"])
    root = result_workspace["legacy_root"]
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    _json(manifest_path, manifest)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


@pytest.mark.parametrize("field", ["schema_version", "summary_schema_version"])
def test_v2_manifest_requires_exact_integer_schema_versions(
    result_workspace: dict[str, object],
    field: str,
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = 2.0
    _json(manifest_path, manifest)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)


def test_legacy_manifest_rejects_malformed_scoring_mode_without_crashing(
    result_workspace: dict[str, object],
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["legacy_run_id"])
    root = result_workspace["legacy_root"]
    manifest_path = root / "manifest.json"  # type: ignore[operator]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scoring_mode"] = []
    _json(manifest_path, manifest)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)  # type: ignore[union-attr]
    _error(caught, "run_unavailable", 409)
    assert run_id not in {
        item["run_id"] for item in service.project_runs(project_id)  # type: ignore[union-attr]
    }


def test_all_failed_v2_run_is_truthfully_available(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "All failed", "2026-07-13T09:00:00Z")
    row = _base_row(project_id, "pending", "combo_all_failed")
    row.update(
        status="failed",
        recall_at_k="",
        mrr_at_k="",
        ndcg_at_k="",
        retrieval_latency_s="",
        rerank_latency_s="",
        avg_query_latency_s="",
        evidence_count=0,
        rerank_search_units="",
        rerank_usage_scope="",
        error_code="reranker_failed",
    )
    run_id, _root = _make_run(
        workspace,
        project_id,
        rows=[row],
        state="failed",
        evidence_rows=[],
    )
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)

    result = service.project_run_results(project_id, run_id)
    assert result["run_state"] == "failed"
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert result["evidence_counts_by_combo"] == {"combo_all_failed": 0}
    assert {item["run_id"] for item in service.project_runs(project_id)} == {run_id}


def test_legacy_failed_combo_cannot_publish_evidence(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project_id = _make_project(workspace, "Legacy failed", "2026-07-13T09:00:00Z")
    completed = _base_row(project_id, "pending", "combo_legacy_completed")
    completed["reranker_id"] = "Qwen3:4B Rerank"
    failed = _base_row(project_id, "pending", "combo_legacy_failed")
    failed["reranker_id"] = "bge-reranker-base"
    for row in (completed, failed):
        for key in set(row) - set(LEGACY_FIELDS):
            row.pop(key)
    failed.update(status="failed", recall_at_k="", error_code="reranker_failed")
    evidence = [{
        "combo_id": "combo_legacy_failed",
        "query_id": "q1",
        "query": "failed?",
        "source_name": "legacy.txt",
        "excerpt": "must not publish",
        "rank": 1,
    }]
    run_id, _root = _make_run(
        workspace,
        project_id,
        rows=[completed, failed],
        request=_request(
            project_id,
            rerankers=["Qwen3:4B Rerank", "bge-reranker-base"],
        ),
        schema_version=1,
        state="partial",
        scoring_mode="retrieval_labels",
        evidence_rows=evidence,
    )
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_results(project_id, run_id)
    _error(caught, "run_unavailable", 409)


def test_run_listing_excludes_runs_with_missing_evidence(
    result_workspace: dict[str, object],
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    (root / "evidence.json").unlink()  # type: ignore[operator]

    assert run_id not in {item["run_id"] for item in service.project_runs(project_id)}  # type: ignore[union-attr]


def test_project_evidence_rejects_windows_drive_relative_source_names(
    result_workspace: dict[str, object],
):
    service = result_workspace["service"]
    project_id = str(result_workspace["alpha_id"])
    run_id = str(result_workspace["labelled_run_id"])
    root = result_workspace["labelled_root"]
    path = root / "evidence.json"  # type: ignore[operator]
    evidence = json.loads(path.read_text(encoding="utf-8"))
    evidence["rows"][0]["source_name"] = "C:secret.txt"
    _json(path, evidence)

    with pytest.raises(ProjectRunResultsError) as caught:
        service.project_run_evidence(  # type: ignore[union-attr]
            project_id,
            run_id,
            "combo_labeled",
            limit=2,
            offset=0,
        )
    _error(caught, "run_unavailable", 409)


def test_browser_question_set_runs_publish_project_owned_groundtruth_identity(
    tmp_path: Path,
):
    workspace = ProjectWorkspace(tmp_path / "projects")
    service = ProjectRunResultService(workspace, catalog_path=CATALOG_PATH)
    project_id = _make_project(workspace, "test1", "2026-07-24T10:00:00Z")
    gt_csv = (
        "query,source,answer\n"
        "What is the daily meal allowance for employee travel?,policy.pdf,INR 1500\n"
    ).encode("utf-8")
    questions_dir = workspace.layout(project_id)["questions"]
    (questions_dir / "northstar-demo-groundtruth.csv").write_bytes(gt_csv)
    question_set = workspace.create_question_set(
        project_id,
        "northstar-demo-groundtruth.csv",
        gt_csv,
    )
    row = _base_row(project_id, "pending", "combo_project_gt")
    request = {
        "schema_version": 1,
        "project_id": project_id,
        "top_k": 10,
        "questions_source": {
            "type": "question_set",
            "question_set_id": question_set["question_set_id"],
            "content_sha256": question_set["content_sha256"],
        },
        "selections": {
            "chunkers": ["fixed_tok1200_ov150"],
            "embeddings": ["openai_text-embedding-3-large"],
            "vector_stores": ["FAISS"],
            "rerankers": ["Amazon Rerank v1"],
        },
        "large_matrix_confirmation": None,
    }
    run_id, _root = _make_run(workspace, project_id, rows=[row], request=request)

    runs = service.project_runs(project_id)
    assert runs
    assert runs[0]["run_id"] == run_id
    assert runs[0]["groundtruth_id"] == f"groundtruth:project:{project_id}"
    assert "uploaded ground truth" in runs[0]["groundtruth_label"]

    result = service.project_run_results(project_id, run_id)
    assert result["groundtruth_id"] == f"groundtruth:project:{project_id}"
    assert result["dataset_id"] == f"project:{project_id}"
    assert result["scoring_mode"] == "retrieval_labels"
    assert result["rows"]
