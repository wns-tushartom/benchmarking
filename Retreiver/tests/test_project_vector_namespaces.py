from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from benchmarking.adapters.vector_faiss import FaissVectorStoreAdapter
from benchmarking.adapters.vector_pgvector import PGVectorStoreAdapter
from benchmarking.adapters.vector_qdrant import QdrantVectorStoreAdapter
from benchmarking.adapters.vector_weaviate import WeaviateVectorStoreAdapter
from source.services.project_workspace import ProjectWorkspace


UUID_A = "123e4567e89b42d3a456426614174000"
UUID_B = "223e4567e89b42d3a456426614174000"
PROJECT_A = f"alpha_{UUID_A}"
PROJECT_B = f"beta_{UUID_B}"
RUN_A = f"run_{UUID_A}"
RUN_B = f"run_{UUID_B}"


class _QdrantClient:
    def __init__(self, **_: object):
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)


class _Psycopg:
    @staticmethod
    def connect(*_: object, **__: object):
        raise AssertionError("database connection is not needed for namespace construction")


def _install_fake_provider_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    faiss = types.ModuleType("faiss")
    monkeypatch.setitem(sys.modules, "faiss", faiss)

    qdrant = types.ModuleType("qdrant_client")
    qdrant.QdrantClient = _QdrantClient  # type: ignore[attr-defined]
    qdrant_http = types.ModuleType("qdrant_client.http")
    qdrant_models = types.ModuleType("qdrant_client.http.models")
    qdrant_http.models = qdrant_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", qdrant_http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", qdrant_models)

    psycopg = types.ModuleType("psycopg")
    psycopg.connect = _Psycopg.connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)


def _identity(project_id: str, run_id: str, combo: str = "combo-1") -> str:
    return f"{project_id}|{run_id}|{combo}"


def test_explicit_namespaces_are_backend_safe_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://test")
    identities = (
        _identity(PROJECT_A, RUN_A),
        _identity(PROJECT_A, RUN_B),
        _identity(PROJECT_B, RUN_A),
        _identity(PROJECT_B, RUN_B),
    )
    run_indexes = tmp_path / "runs" / RUN_A / "indexes"
    run_indexes.mkdir(parents=True)

    qdrant = [QdrantVectorStoreAdapter("Qdrant", namespace=value) for value in identities]
    pgvector = [PGVectorStoreAdapter("PGVector", namespace=value) for value in identities]
    weaviate = [WeaviateVectorStoreAdapter("Weaviate", namespace=value) for value in identities]
    faiss = [
        FaissVectorStoreAdapter(
            "FAISS",
            namespace=value,
            index_root=str(run_indexes),
        )
        for value in identities
    ]

    assert len({adapter.physical_namespace for adapter in qdrant}) == 4
    assert len({adapter.physical_namespace for adapter in pgvector}) == 4
    assert len({adapter.physical_namespace for adapter in weaviate}) == 4
    assert len({adapter.physical_namespace for adapter in faiss}) == 4
    assert all(re.fullmatch(r"[a-z0-9_]{1,63}", item.physical_namespace) for item in qdrant)
    assert all(re.fullmatch(r"[a-z0-9_]{1,63}", item.physical_namespace) for item in pgvector)
    assert all(re.fullmatch(r"[A-Z][A-Za-z0-9]{0,62}", item.physical_namespace) for item in weaviate)
    assert all(
        Path(item.physical_namespace).is_relative_to(run_indexes)
        for item in faiss
    )


@pytest.mark.parametrize("unsafe", ("../other", "a/b", "a\\b", "x;DROP TABLE y", "{ Get }"))
def test_raw_namespace_syntax_never_reaches_physical_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str
):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://test")
    index_root = tmp_path / "indexes"
    index_root.mkdir()

    adapters = (
        QdrantVectorStoreAdapter("Qdrant", namespace=unsafe),
        PGVectorStoreAdapter("PGVector", namespace=unsafe),
        WeaviateVectorStoreAdapter("Weaviate", namespace=unsafe),
        FaissVectorStoreAdapter("FAISS", namespace=unsafe, index_root=str(index_root)),
    )
    for adapter in adapters[:3]:
        assert not any(token in adapter.physical_namespace for token in ("/", "\\", ";", "{", "}", " "))
    assert Path(adapters[3].physical_namespace).is_relative_to(index_root)


def test_omitting_namespace_preserves_official_default_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://test")
    explicit_faiss_dir = tmp_path / "official-faiss"

    qdrant = QdrantVectorStoreAdapter("Qdrant", collection_prefix="official")
    pgvector = PGVectorStoreAdapter("PGVector", table_prefix="official")
    weaviate = WeaviateVectorStoreAdapter("Weaviate", class_prefix="Official")
    faiss = FaissVectorStoreAdapter("FAISS", index_dir=str(explicit_faiss_dir))

    assert qdrant.collection.startswith("official_")
    assert pgvector.table.startswith("official_")
    assert weaviate.class_name.startswith("Official")
    assert faiss.index_dir == explicit_faiss_dir
    assert qdrant.physical_namespace == qdrant.collection
    assert pgvector.physical_namespace == pgvector.table
    assert weaviate.physical_namespace == weaviate.class_name
    assert faiss.physical_namespace == str(explicit_faiss_dir)


def test_faiss_explicit_namespace_rejects_untrusted_or_symlinked_index_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "indexes"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="index_root"):
        FaissVectorStoreAdapter(
            "FAISS",
            namespace=_identity(PROJECT_A, RUN_A),
            index_root=str(link),
        )
    with pytest.raises(ValueError, match="index_root"):
        FaissVectorStoreAdapter(
            "FAISS",
            namespace=_identity(PROJECT_A, RUN_A),
            index_root=str(tmp_path / "missing"),
        )


def test_faiss_explicit_namespace_is_reserved_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    identity = _identity(PROJECT_A, RUN_A)

    first = FaissVectorStoreAdapter(
        "FAISS", namespace=identity, index_root=str(index_root)
    )
    assert Path(first.physical_namespace).is_dir()
    with pytest.raises(FileExistsError):
        FaissVectorStoreAdapter(
            "FAISS", namespace=identity, index_root=str(index_root)
        )


def test_faiss_load_existing_reopens_reserved_explicit_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    identity = _identity(PROJECT_A, RUN_A)

    created = FaissVectorStoreAdapter(
        "FAISS", namespace=identity, index_root=str(index_root)
    )
    loaded_paths: list[Path | None] = []
    monkeypatch.setattr(FaissVectorStoreAdapter, "load", lambda self: loaded_paths.append(self.index_dir))

    reopened = FaissVectorStoreAdapter(
        "FAISS",
        namespace=identity,
        index_root=str(index_root),
        load_existing=True,
    )

    assert reopened.index_dir == created.index_dir
    assert reopened.physical_namespace == created.physical_namespace
    assert loaded_paths == [created.index_dir]


def test_drop_namespace_is_idempotent_for_all_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_provider_modules(monkeypatch)
    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://test")
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    identity = _identity(PROJECT_A, RUN_A)

    qdrant = QdrantVectorStoreAdapter("Qdrant", namespace=identity)
    pgvector = PGVectorStoreAdapter("PGVector", namespace=identity)
    weaviate = WeaviateVectorStoreAdapter("Weaviate", namespace=identity)
    faiss = FaissVectorStoreAdapter("FAISS", namespace=identity, index_root=str(index_root))

    monkeypatch.setattr(pgvector, "reset_collection", lambda *_: None)
    monkeypatch.setattr(weaviate, "reset_collection", lambda *_: None)
    for adapter in (qdrant, pgvector, weaviate, faiss):
        adapter.drop_namespace()
        adapter.drop_namespace()


def test_workspace_creates_run_directory_exclusively_and_rejects_cross_run_symlink(
    tmp_path: Path,
):
    workspace = ProjectWorkspace(tmp_path / "projects")
    project = workspace.create_upload("alpha.txt", b"alpha", "Alpha")
    first = workspace.create_run_layout(project["project_id"], RUN_A)

    assert first["root"].is_dir()
    assert first["indexes"].is_dir()
    with pytest.raises(FileExistsError):
        workspace.create_run_layout(project["project_id"], RUN_A)

    other = workspace.create_run_layout(project["project_id"], RUN_B)
    first["indexes"].rmdir()
    first["indexes"].symlink_to(other["indexes"], target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.run_layout(project["project_id"], RUN_A)


def test_workspace_run_layout_rejects_project_and_canonical_child_symlinks(tmp_path: Path):
    workspace = ProjectWorkspace(tmp_path / "projects")
    alpha = workspace.create_upload("alpha.txt", b"alpha", "Alpha")
    beta = workspace.create_upload("beta.txt", b"beta", "Beta")
    beta_run = workspace.create_run_layout(beta["project_id"], RUN_B)

    alpha_runs = workspace.layout(alpha["project_id"])["runs"]
    alpha_run = alpha_runs / RUN_A
    alpha_run.symlink_to(beta_run["root"], target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.run_layout(alpha["project_id"], RUN_A)

    alpha_run.unlink()
    alpha_questions = workspace.layout(alpha["project_id"])["questions"]
    beta_questions = workspace.layout(beta["project_id"])["questions"]
    alpha_questions.rmdir()
    alpha_questions.symlink_to(beta_questions, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace.layout(alpha["project_id"])
