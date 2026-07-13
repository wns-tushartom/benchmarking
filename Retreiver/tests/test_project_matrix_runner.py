from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

import benchmarking.adapters.remote_embeddings as remote_embeddings
import benchmarking.adapters.remote_rerankers as remote_rerankers
import benchmarking.adapters.vector_weaviate as vector_weaviate
from benchmarking.adapters.remote_embeddings import (
    OpenAIEmbeddingAdapter,
    RemoteHTTPEmbeddingAdapter,
)
from benchmarking.adapters.remote_rerankers import (
    AmazonBedrockRerankerAdapter,
    RemoteHTTPRerankerAdapter,
)
from benchmarking.adapters.vector_faiss import FaissVectorStoreAdapter
from benchmarking.adapters.vector_pgvector import PGVectorStoreAdapter
from benchmarking.adapters.vector_qdrant import QdrantVectorStoreAdapter
from benchmarking.adapters.vector_weaviate import WeaviateVectorStoreAdapter
from scripts import run_project_matrix
from scripts import serve_benchmark_dashboard as dashboard
from source.services import project_matrix_runner
from source.services.project_documents import load_project_documents
from source.services.project_matrix_contract import validate_project_matrix_request
from source.services.project_matrix_runner import (
    ProjectMatrixRunner,
    ProjectMatrixRunnerError,
    production_adapter_classes,
)
from source.services.project_workspace import ProjectWorkspace


class _VectorParams:
    def __init__(self, **values: object):
        self.values = values


class _HnswConfigDiff(_VectorParams):
    pass


class _PointStruct:
    def __init__(self, id: int, vector: list[float], payload: dict[str, object]):
        self.id = id
        self.vector = vector
        self.payload = payload


class _Distance:
    COSINE = "COSINE"


class _FakeQdrantClient:
    collections: dict[str, list[_PointStruct]] = {}

    def __init__(self, **_: object):
        pass

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def create_collection(self, *, collection_name: str, **_: object) -> None:
        self.collections[collection_name] = []

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[_PointStruct],
        wait: bool,
    ) -> None:
        assert wait is True
        self.collections[collection_name].extend(points)

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool,
    ) -> list[SimpleNamespace]:
        assert with_payload is True
        points = self.collections[collection_name]
        scored = [
            SimpleNamespace(
                payload=point.payload,
                score=sum(left * right for left, right in zip(point.vector, query_vector)),
            )
            for point in points
        ]
        return sorted(scored, key=lambda row: row.score, reverse=True)[:limit]


def _install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeQdrantClient.collections = {}
    qdrant = types.ModuleType("qdrant_client")
    qdrant.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    qdrant_http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.VectorParams = _VectorParams  # type: ignore[attr-defined]
    models.HnswConfigDiff = _HnswConfigDiff  # type: ignore[attr-defined]
    models.PointStruct = _PointStruct  # type: ignore[attr-defined]
    models.Distance = _Distance  # type: ignore[attr-defined]
    qdrant_http.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qdrant_http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)


def _install_fake_weaviate(monkeypatch: pytest.MonkeyPatch) -> None:
    objects: dict[str, list[dict[str, object]]] = {}

    def fake_request(
        method: str,
        url: str,
        payload: dict[str, object] | None = None,
        timeout: int = 60,
    ) -> dict[str, object]:
        del timeout
        if method == "DELETE":
            objects.pop(url.rsplit("/", 1)[-1], None)
            return {}
        if url.endswith("/v1/schema"):
            assert payload is not None
            objects[str(payload["class"])] = []
            return {}
        if url.endswith("/v1/batch/objects"):
            assert payload is not None
            for item in payload["objects"]:  # type: ignore[index]
                assert isinstance(item, dict)
                objects[str(item["class"])].append(item)
            return {}
        if url.endswith("/v1/graphql"):
            assert payload is not None
            query = str(payload["query"])
            class_name = next(name for name in objects if name in query)
            rows = [
                {
                    **dict(item["properties"]),  # type: ignore[arg-type]
                    "_additional": {"certainty": 1.0},
                }
                for item in objects[class_name]
            ]
            return {"data": {"Get": {class_name: rows}}}
        raise AssertionError(f"unexpected Weaviate request: {method} {url}")

    monkeypatch.setattr(vector_weaviate, "_request", fake_request)


def _embedding_vector(text: str) -> list[float]:
    folded = text.casefold()
    return [
        1.0 if "alpha_sentinel" in folded else 0.0,
        1.0 if "refund" in folded else 0.0,
        min(len(text), 1000) / 1000.0,
    ]


def _fake_embedding_post(
    _url: str,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, object]:
    del headers, timeout
    values = payload.get("input") or payload.get("inputs") or payload.get("texts")
    assert isinstance(values, list)
    return {"data": [{"embedding": _embedding_vector(str(value))} for value in values]}


def _fake_rerank_post(
    _url: str,
    payload: dict[str, object],
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, object]:
    del headers, timeout
    model = str(payload.get("model") or "")
    if model == "bge-reranker-base":
        raise RuntimeError("synthetic provider failure")
    documents = payload.get("documents") or payload.get("texts") or payload.get("passages")
    assert isinstance(documents, list)
    return {
        "scores": [
            10.0 if "ALPHA_SENTINEL" in str(document) else 0.0
            for document in documents
        ],
        "request_id": "rerank-request-1",
    }


@pytest.fixture
def project_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    projects = tmp_path / "user_projects"
    monkeypatch.setattr(dashboard, "USER_PROJECTS_DIR", projects)
    monkeypatch.setenv("JINA_EMBEDDING_URL", "http://embedding.test")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test")
    monkeypatch.setenv("WEAVIATE_URL", "http://weaviate.test")
    monkeypatch.setenv("QWEN_RERANK_URL", "http://qwen.test")
    monkeypatch.setenv("BGE_RERANK_URL", "http://bge.test")
    monkeypatch.setattr(remote_embeddings, "_post_json", _fake_embedding_post)
    monkeypatch.setattr(remote_rerankers, "_post_json", _fake_rerank_post)
    _install_fake_qdrant(monkeypatch)
    _install_fake_weaviate(monkeypatch)
    return projects, ProjectWorkspace(projects)


def _upload_project(name: str, sentinel: str) -> dict[str, object]:
    text = (
        f"REFUND POLICY\n\n{sentinel} controls the refund decision. "
        f"{sentinel} requires identity verification and receipt review. "
        f"{sentinel} records the approved amount and settlement date."
    )
    return dashboard.create_user_project_upload(name, text.encode("utf-8"), name)


def _request_payload(
    project_id: str,
    *,
    question_source: dict[str, str] | None = None,
    rerankers: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_id": project_id,
        "top_k": 3,
        "questions_source": question_source
        or {"type": "typed", "query": "What does ALPHA_SENTINEL say about refund?"},
        "selections": {
            "chunkers": ["entity_heuristic_w6", "fixed_tok1200_ov150"],
            "embeddings": ["jina_v3"],
            "vector_stores": ["Qdrant"],
            "rerankers": rerankers
            if rerankers is not None
            else ["Qwen3:4B Rerank", "bge-reranker-base"],
        },
        "large_matrix_confirmation": None,
    }


def _create_run_request(
    workspace: ProjectWorkspace,
    payload: dict[str, object],
) -> tuple[str, Path]:
    validated = validate_project_matrix_request(payload)
    run_id = workspace.new_run_id()
    layout = workspace.create_run_layout(validated.request.project_id, run_id)
    envelope = {
        "schema_version": 1,
        "request": validated.request.to_dict(),
        "request_fingerprint": validated.request_fingerprint,
        "combination_count": validated.combination_count,
    }
    request_path = layout["root"] / "request.json"
    with request_path.open("x", encoding="utf-8") as output:
        json.dump(envelope, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return run_id, layout["root"]


def _json_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_cli_accepts_only_project_and_run_ids():
    parsed = run_project_matrix.build_parser().parse_args(
        ["--project-id", "alpha_123e4567e89b42d3a456426614174000", "--run-id", "run_123e4567e89b42d3a456426614174000"]
    )
    assert parsed.project_id.startswith("alpha_")
    assert parsed.run_id.startswith("run_")
    with pytest.raises(SystemExit):
        run_project_matrix.build_parser().parse_args(
            [
                "--project-id",
                parsed.project_id,
                "--run-id",
                parsed.run_id,
                "--output",
                "/tmp/escape",
            ]
        )


def test_catalog_ids_resolve_to_intended_production_adapter_classes():
    classes = production_adapter_classes()
    assert classes["embeddings"] == {
        "jina_v3": RemoteHTTPEmbeddingAdapter,
        "gte_multilingual_base": RemoteHTTPEmbeddingAdapter,
        "openai_text-embedding-3-large": OpenAIEmbeddingAdapter,
    }
    assert classes["vector_stores"] == {
        "Qdrant": QdrantVectorStoreAdapter,
        "PGVector": PGVectorStoreAdapter,
        "Weaviate": WeaviateVectorStoreAdapter,
        "FAISS": FaissVectorStoreAdapter,
    }
    assert classes["rerankers"] == {
        "Amazon Rerank v1": AmazonBedrockRerankerAdapter,
        "Qwen3:4B Rerank": RemoteHTTPRerankerAdapter,
        "bge-reranker-base": RemoteHTTPRerankerAdapter,
    }


def test_real_matrix_is_project_scoped_honest_and_failure_isolated(project_environment):
    projects, workspace = project_environment
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    beta = _upload_project("beta.txt", "BETA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    beta_id = str(beta["project_id"])

    first_run_id, first_root = _create_run_request(workspace, _request_payload(alpha_id))
    result = ProjectMatrixRunner(workspace).run(alpha_id, first_run_id)

    assert result == {
        "project_id": alpha_id,
        "run_id": first_run_id,
        "state": "partial",
        "combination_count": 4,
        "succeeded": 2,
        "failed": 2,
    }
    summary = _summary_rows(first_root / "summary.csv")
    assert len(summary) == 4
    assert {row["status"] for row in summary} == {"completed", "failed"}
    assert len({row["combo_id"] for row in summary}) == 4
    assert all(row["project_id"] == alpha_id for row in summary)
    assert all(row["run_id"] == first_run_id for row in summary)
    completed = [row for row in summary if row["status"] == "completed"]
    failed = [row for row in summary if row["status"] == "failed"]
    assert {row["reranker_id"] for row in completed} == {"Qwen3:4B Rerank"}
    assert {row["reranker_id"] for row in failed} == {"bge-reranker-base"}
    assert all(row["error_code"] == "reranker_failed" for row in failed)
    assert all(row["labelled_queries"] == "0" for row in summary)
    assert all(row["unlabelled_queries"] == "1" for row in summary)
    assert all(not row["recall_at_k"] for row in summary)

    evidence = json.loads((first_root / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["mode"] == "evidence_only"
    assert "leaderboard" not in evidence
    assert "metrics" not in evidence
    assert evidence["rows"]
    assert all("ALPHA_SENTINEL" in row["excerpt"] for row in evidence["rows"])
    assert all("BETA_SENTINEL" not in row["excerpt"] for row in evidence["rows"])
    assert {row["combo_id"] for row in evidence["rows"]}.isdisjoint(
        {row["combo_id"] for row in failed}
    )

    details = _json_rows(first_root / "details.jsonl")
    assert details
    assert all(row["project_id"] == alpha_id for row in details)
    assert all(row["query_id"] == "q_000001" for row in details)
    assert all("base_score" in row and "rerank_score" in row for row in details)
    assert all("source_name" in row and "page_number" in row for row in details)

    manifest = json.loads((first_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["request_fingerprint"]
    assert len(manifest["receipts"]) == 4
    serialized_receipts = json.dumps(manifest["receipts"], sort_keys=True).casefold()
    assert "fallback" not in serialized_receipts
    for receipt in manifest["receipts"]:
        assert receipt["combo_id"]
        assert receipt["config_sha256"]
        assert receipt["adapters"]["embedding"]["canonical_id"] == "jina_v3"
        assert receipt["adapters"]["embedding"]["adapter_class"] == "RemoteHTTPEmbeddingAdapter"
        assert receipt["adapters"]["embedding"]["dimensions"] == 3
        assert receipt["adapters"]["embedding"]["vector_count"] >= 1
        assert receipt["adapters"]["vector_store"]["canonical_id"] == "Qdrant"
        assert receipt["adapters"]["vector_store"]["adapter_class"] == "QdrantVectorStoreAdapter"
        assert receipt["adapters"]["vector_store"]["physical_namespace"]
        assert receipt["adapters"]["reranker"]["canonical_id"] in {
            "Qwen3:4B Rerank",
            "bge-reranker-base",
        }
        assert (first_root / receipt["receipt_path"]).is_file()
        if receipt["status"] == "completed":
            assert receipt["adapters"]["reranker"]["provider_metadata"] == {
                "request_id": "rerank-request-1"
            }
    physical = {
        receipt["adapters"]["vector_store"]["physical_namespace"]
        for receipt in manifest["receipts"]
    }
    assert len(physical) == 2  # retrieval is reused across the two rerankers

    chunk_manifest = json.loads((first_root / "chunks" / "manifest.json").read_text(encoding="utf-8"))
    assert set(chunk_manifest["artifacts"]) == {
        "entity_heuristic_w6",
        "fixed_tok1200_ov150",
    }
    assert all((first_root / artifact["path"]).is_file() for artifact in chunk_manifest["artifacts"].values())
    assert not (first_root / "analysis.json").exists()
    assert not (projects.parent / "modular_runs").exists()
    assert workspace.layout(beta_id)["documents"].is_file()

    second_run_id, second_root = _create_run_request(
        workspace,
        _request_payload(alpha_id, rerankers=["Qwen3:4B Rerank"]),
    )
    second_result = ProjectMatrixRunner(workspace).run(alpha_id, second_run_id)
    assert second_result["state"] == "completed"
    second_manifest = json.loads((second_root / "manifest.json").read_text(encoding="utf-8"))
    second_physical = {
        receipt["adapters"]["vector_store"]["physical_namespace"]
        for receipt in second_manifest["receipts"]
    }
    assert physical.isdisjoint(second_physical)


def test_embeddings_are_cached_by_chunker_and_embedding_across_stores(
    project_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    _, workspace = project_environment
    calls: list[dict[str, object]] = []

    def counted_post(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> dict[str, object]:
        calls.append(payload)
        return _fake_embedding_post(url, payload, headers, timeout)

    monkeypatch.setattr(remote_embeddings, "_post_json", counted_post)
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    payload = _request_payload(
        alpha_id,
        rerankers=["Qwen3:4B Rerank"],
    )
    payload["selections"]["chunkers"] = ["entity_heuristic_w6"]  # type: ignore[index]
    payload["selections"]["vector_stores"] = ["Qdrant", "Weaviate"]  # type: ignore[index]
    run_id, root = _create_run_request(workspace, payload)

    result = ProjectMatrixRunner(workspace).run(alpha_id, run_id)

    assert result["state"] == "completed"
    assert result["combination_count"] == 2
    assert len(calls) == 2  # one document batch and one query batch, reused by both stores
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert {
        receipt["adapters"]["vector_store"]["canonical_id"]
        for receipt in manifest["receipts"]
    } == {"Qdrant", "Weaviate"}


def test_chunker_failure_is_isolated_without_cloned_evidence(
    project_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    _, workspace = project_environment
    real_chunker = project_matrix_runner.chunk_project_documents

    def fail_one_chunker(documents, chunker_id):
        if chunker_id == "entity_heuristic_w6":
            raise RuntimeError("provider detail must not escape")
        return real_chunker(documents, chunker_id)

    monkeypatch.setattr(
        project_matrix_runner,
        "chunk_project_documents",
        fail_one_chunker,
    )
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    payload = _request_payload(alpha_id, rerankers=["Qwen3:4B Rerank"])
    payload["selections"]["chunkers"] = [  # type: ignore[index]
        "entity_heuristic_w6",
        "fixed_tok1200_ov150",
    ]
    run_id, root = _create_run_request(workspace, payload)

    result = ProjectMatrixRunner(workspace).run(alpha_id, run_id)

    assert result == {
        "project_id": alpha_id,
        "run_id": run_id,
        "state": "partial",
        "combination_count": 2,
        "succeeded": 1,
        "failed": 1,
    }
    summary = _summary_rows(root / "summary.csv")
    assert {row["status"] for row in summary} == {"completed", "failed"}
    failed = next(row for row in summary if row["status"] == "failed")
    assert failed["chunker_id"] == "entity_heuristic_w6"
    assert failed["error_code"] == "chunker_failed"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failed_receipt = next(
        receipt for receipt in manifest["receipts"] if receipt["status"] == "failed"
    )
    assert failed_receipt["adapters"]["chunker"]["artifact_sha256"] is None
    assert failed_receipt["artifacts"] == {}
    assert "provider detail" not in json.dumps(manifest)
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    completed_combo = next(
        receipt["combo_id"]
        for receipt in manifest["receipts"]
        if receipt["status"] == "completed"
    )
    assert evidence["rows"]
    assert {row["combo_id"] for row in evidence["rows"]} == {completed_combo}


def test_mid_rerank_failure_discards_partial_combo_evidence(
    project_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    _, workspace = project_environment
    calls = 0

    def fail_second_rerank(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("second query provider failure")
        return _fake_rerank_post(url, payload, headers, timeout)

    monkeypatch.setattr(remote_rerankers, "_post_json", fail_second_rerank)
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    question_set = workspace.create_question_set(
        alpha_id,
        "questions.txt",
        b"What controls refunds?\nHow is identity verified?\n",
    )
    payload = _request_payload(
        alpha_id,
        question_source={
            "type": "question_set",
            "question_set_id": str(question_set["question_set_id"]),
            "content_sha256": str(question_set["content_sha256"]),
        },
        rerankers=["Qwen3:4B Rerank"],
    )
    payload["selections"]["chunkers"] = ["fixed_tok1200_ov150"]  # type: ignore[index]
    run_id, root = _create_run_request(workspace, payload)

    result = ProjectMatrixRunner(workspace).run(alpha_id, run_id)

    assert result["state"] == "failed"
    assert result["failed"] == 1
    assert json.loads((root / "evidence.json").read_text(encoding="utf-8"))["rows"] == []
    assert (root / "details.jsonl").read_text(encoding="utf-8") == ""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["receipts"][0]["error_code"] == "reranker_failed"
    assert "second query provider failure" not in json.dumps(manifest)


def test_labelled_run_scores_only_applicable_retrieval_labels(project_environment):
    _, workspace = project_environment
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    manifest = json.loads(
        (workspace.project_root(alpha_id) / "manifest.json").read_text(encoding="utf-8")
    )
    documents = load_project_documents(
        workspace.layout(alpha_id)["documents"],
        expected_sha256=manifest["corpus_sha256"],
    )
    source_id = documents[0].source_id
    question_set = workspace.create_question_set(
        alpha_id,
        "questions.csv",
        (
            "question,source_id,answer\n"
            f"What controls the refund decision?,{source_id},The sentinel policy does.\n"
        ).encode("utf-8"),
    )
    payload = _request_payload(
        alpha_id,
        question_source={
            "type": "question_set",
            "question_set_id": question_set["question_set_id"],
            "content_sha256": question_set["content_sha256"],
        },
        rerankers=["Qwen3:4B Rerank"],
    )
    payload["selections"]["chunkers"] = ["entity_heuristic_w6"]  # type: ignore[index]
    run_id, root = _create_run_request(workspace, payload)

    result = ProjectMatrixRunner(workspace).run(alpha_id, run_id)

    assert result["state"] == "completed"
    summary = _summary_rows(root / "summary.csv")
    assert len(summary) == 1
    assert summary[0]["labelled_queries"] == "1"
    assert summary[0]["unlabelled_queries"] == "0"
    assert summary[0]["recall_at_k"] == "1.0"
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["scoring_mode"] == "retrieval_labels"
    assert analysis["rows"][0]["labelled_queries"] == 1
    assert analysis["rows"][0]["unlabelled_queries"] == 0
    assert analysis["rows"][0]["recall_at_k"] == 1.0


def test_runner_rejects_corpus_manifest_copied_from_another_project(
    project_environment,
):
    _, workspace = project_environment
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    beta = _upload_project("beta.txt", "BETA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    beta_id = str(beta["project_id"])
    run_id, _ = _create_run_request(workspace, _request_payload(alpha_id))
    alpha_layout = workspace.layout(alpha_id)
    beta_layout = workspace.layout(beta_id)
    alpha_layout["documents"].write_bytes(beta_layout["documents"].read_bytes())
    (alpha_layout["root"] / "manifest.json").write_bytes(
        (beta_layout["root"] / "manifest.json").read_bytes()
    )

    with pytest.raises(ProjectMatrixRunnerError, match="extraction is incomplete"):
        ProjectMatrixRunner(workspace).run(alpha_id, run_id)


def test_runner_rejects_request_fingerprint_or_project_substitution(project_environment):
    _, workspace = project_environment
    alpha = _upload_project("alpha.txt", "ALPHA_SENTINEL")
    beta = _upload_project("beta.txt", "BETA_SENTINEL")
    alpha_id = str(alpha["project_id"])
    beta_id = str(beta["project_id"])
    run_id, root = _create_run_request(workspace, _request_payload(alpha_id))
    request_path = root / "request.json"
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    envelope["request_fingerprint"] = "0" * 64
    request_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        ProjectMatrixRunner(workspace).run(alpha_id, run_id)
    with pytest.raises((FileNotFoundError, ValueError)):
        ProjectMatrixRunner(workspace).run(beta_id, run_id)
