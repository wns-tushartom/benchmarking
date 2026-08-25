from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from benchmarking.core.schemas import Chunk


class FakeIdMapIndex:
    """Small behavioral double for the evaluated TurboVec 1.0.0 API."""

    instances: list["FakeIdMapIndex"] = []

    def __init__(self, *, dim: int, bit_width: int) -> None:
        self.dim = dim
        self.bit_width = bit_width
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.uint64)
        self.add_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.search_calls: list[tuple[np.ndarray, int]] = []
        type(self).instances.append(self)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert vectors.dtype == np.float32
        assert vectors.flags.c_contiguous
        assert ids.dtype == np.uint64
        assert ids.flags.c_contiguous
        self.add_calls.append((vectors.copy(), ids.copy()))
        self.vectors = vectors.copy()
        self.ids = ids.copy()

    def search(self, queries: np.ndarray, *, k: int):
        assert queries.dtype == np.float32
        assert queries.flags.c_contiguous
        self.search_calls.append((queries.copy(), k))
        similarities = queries @ self.vectors.T
        order = np.argsort(-similarities, axis=1)[:, :k]
        return np.take_along_axis(similarities, order, axis=1), self.ids[order]

    def sync(self, path: str) -> None:
        payload = {
            "dim": self.dim,
            "bit_width": self.bit_width,
            "vectors": self.vectors.tolist(),
            "ids": [int(value) for value in self.ids],
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "FakeIdMapIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(dim=int(payload["dim"]), bit_width=int(payload["bit_width"]))
        instance.vectors = np.ascontiguousarray(payload["vectors"], dtype=np.float32)
        instance.ids = np.ascontiguousarray(payload["ids"], dtype=np.uint64)
        return instance

    def __len__(self) -> int:
        return len(self.ids)


class ForbiddenTurboQuantIndex:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("adapter must use IdMapIndex for stable external IDs")


@pytest.fixture
def fake_turbovec(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    FakeIdMapIndex.instances.clear()
    module = types.ModuleType("turbovec")
    setattr(module, "__version__", "1.0.0")
    setattr(module, "IdMapIndex", FakeIdMapIndex)
    setattr(module, "TurboQuantIndex", ForbiddenTurboQuantIndex)
    monkeypatch.setitem(sys.modules, "turbovec", module)
    return module


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            id=9001,
            pdf_name="alpha.pdf",
            paragraph="alpha",
            parent_id="parent-a",
            metadata={"section": "A", "store": "old"},
        ),
        Chunk(
            id=42,
            pdf_name="beta.pdf",
            paragraph="beta",
            parent_id="parent-b",
            metadata={"section": "B"},
        ),
    ]


def new_adapter(**kwargs: object):
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    return TurboVecVectorStoreAdapter("TurboVec", **kwargs)


def test_module_import_is_lazy_and_missing_dependency_errors_only_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "turbovec", raising=False)
    module = importlib.import_module("benchmarking.adapters.vector_turbovec")
    importlib.reload(module)

    assert "turbovec" not in sys.modules
    real_import_module = importlib.import_module

    def import_without_turbovec(name: str, package: str | None = None):
        if name == "turbovec":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(module.importlib, "import_module", import_without_turbovec)
    with pytest.raises(RuntimeError, match="turbovec==1.0.0"):
        module.TurboVecVectorStoreAdapter("TurboVec")


def test_constructor_rejects_unverified_turbovec_version(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("turbovec")
    setattr(module, "__version__", "9.9.9")
    setattr(module, "IdMapIndex", FakeIdMapIndex)
    monkeypatch.setitem(sys.modules, "turbovec", module)
    with pytest.raises(RuntimeError, match="turbovec==1.0.0"):
        new_adapter()


def test_benchmark_requirements_pin_evaluated_turbovec_version() -> None:
    requirements = (Path(__file__).resolve().parents[1] / "requirements-benchmark.txt").read_text(
        encoding="utf-8"
    )
    assert "turbovec==1.0.0" in requirements.splitlines()


def test_accepts_only_the_four_bit_candidate_configuration(fake_turbovec: types.ModuleType) -> None:
    adapter = new_adapter()
    assert adapter.index_type == "TurboQuant4bit"
    assert adapter.bits == 4

    with pytest.raises(ValueError, match="TurboQuant4bit"):
        new_adapter(index_type="TurboQuant2bit")
    with pytest.raises(ValueError, match="bits=4"):
        new_adapter(bits=2)


def test_upsert_normalizes_vectors_and_uses_external_chunk_ids(fake_turbovec: types.ModuleType) -> None:
    adapter = new_adapter()
    chunks = make_chunks()

    result = adapter.upsert(chunks, [[3.0, 0.0], [0.0, 7.0]])

    assert result["vector_count"] == 2
    assert result["upsert_latency_s"] >= 0.0
    index = FakeIdMapIndex.instances[-1]
    assert (index.dim, index.bit_width) == (2, 4)
    vectors, ids = index.add_calls[0]
    np.testing.assert_allclose(vectors, np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(ids, np.array([9001, 42], dtype=np.uint64))


def test_float32_max_vectors_do_not_overflow_to_zero_during_normalization(
    fake_turbovec: types.ModuleType,
) -> None:
    maximum = float(np.finfo(np.float32).max)
    adapter = new_adapter()
    adapter.upsert([make_chunks()[0]], [[maximum, maximum]])
    vectors, _ = FakeIdMapIndex.instances[-1].add_calls[0]
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert np.count_nonzero(vectors[0]) == 2


def test_search_normalizes_query_maps_external_ids_and_preserves_metadata(fake_turbovec: types.ModuleType) -> None:
    adapter = new_adapter()
    adapter.upsert(make_chunks(), [[2.0, 0.0], [0.0, 5.0]])

    hits = adapter.search([20.0, 0.0], top_k=5)

    assert [hit.chunk.id for hit in hits] == [9001, 42]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].chunk.metadata == {"section": "A", "store": "TurboVec"}
    assert hits[1].chunk.metadata == {"section": "B", "store": "TurboVec"}
    query, requested_k = FakeIdMapIndex.instances[-1].search_calls[-1]
    np.testing.assert_allclose(query, np.array([[1.0, 0.0]], dtype=np.float32))
    assert requested_k == 2


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        (make_chunks(), [[1.0, 2.0]], "chunk/vector mismatch"),
        ([make_chunks()[0]], [1.0, 2.0], "2D"),
        ([make_chunks()[0]], [[]], "dimension"),
        ([make_chunks()[0]], [[float("nan"), 1.0]], "non-finite"),
        ([make_chunks()[0]], [[0.0, 0.0]], "zero-norm"),
        ([], [], "2D"),
    ],
)
def test_upsert_rejects_invalid_vector_inputs(
    fake_turbovec: types.ModuleType,
    chunks: list[Chunk],
    vectors: list[list[float]] | list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        new_adapter().upsert(chunks, vectors)  # type: ignore[arg-type]


def test_upsert_rejects_duplicate_or_non_uint64_chunk_ids(fake_turbovec: types.ModuleType) -> None:
    duplicate = make_chunks()
    duplicate[1].id = duplicate[0].id
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        new_adapter().upsert(duplicate, [[1.0, 0.0], [0.0, 1.0]])

    invalid = [Chunk(id=-1, pdf_name="x", paragraph="x")]
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        new_adapter().upsert(invalid, [[1.0, 0.0]])


def test_search_validates_state_top_k_and_query(fake_turbovec: types.ModuleType) -> None:
    adapter = new_adapter()
    with pytest.raises(RuntimeError, match="upsert.*load"):
        adapter.search([1.0, 0.0], 1)

    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    for top_k in (0, -1):
        with pytest.raises(ValueError, match="top_k"):
            adapter.search([1.0, 0.0], top_k)
    with pytest.raises(ValueError, match="dimension mismatch"):
        adapter.search([1.0, 0.0, 0.0], 1)
    with pytest.raises(ValueError, match="non-finite"):
        adapter.search([float("inf"), 0.0], 1)
    with pytest.raises(ValueError, match="zero-norm"):
        adapter.search([0.0, 0.0], 1)


@pytest.mark.parametrize(
    ("scores", "external_ids", "message"),
    [
        ([[1.0]], [[9001]], "result count"),
        ([[1.0, 0.5], [0.4, 0.3]], [[9001, 42], [42, 9001]], "result shapes"),
        ([[float("nan"), 0.5]], [[9001, 42]], "non-finite"),
        ([["1.0", "0.5"]], [[9001, 42]], "non-numeric"),
        ([[1.0, 0.5]], [[9001, 9001]], "duplicate"),
    ],
)
def test_search_rejects_incomplete_nonfinite_or_duplicate_results(
    fake_turbovec: types.ModuleType,
    scores: list[list[object]],
    external_ids: list[list[int]],
    message: str,
) -> None:
    adapter = new_adapter()
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    assert adapter.index is not None
    adapter.index.search = lambda query, *, k: (scores, external_ids)

    with pytest.raises(ValueError, match=message):
        adapter.search([1.0, 0.0], 2)


def test_save_writes_tvim_chunks_and_hashed_manifest_then_loads(fake_turbovec: types.ModuleType, tmp_path: Path) -> None:
    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    index_path = index_dir / "index.tvim"
    chunks_path = index_dir / "chunks.json"
    manifest_path = index_dir / "manifest.json"
    assert index_path.is_file() and chunks_path.is_file() and manifest_path.is_file()
    assert index_dir.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (index_path, chunks_path, manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "format_version": 1,
        "index_type": "TurboQuant4bit",
        "bits": 4,
        "dimensions": 2,
        "vector_count": 2,
        "files": {
            "index.tvim": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "chunks.json": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        },
    }

    restored = new_adapter(index_dir=str(index_dir), load_existing=True)
    hits = restored.search([1.0, 0.0], 1)
    assert hits[0].chunk == Chunk(
        id=9001,
        pdf_name="alpha.pdf",
        paragraph="alpha",
        parent_id="parent-a",
        metadata={"section": "A", "store": "TurboVec"},
    )


def test_load_rejects_hash_tampering_and_metadata_index_mismatch(fake_turbovec: types.ModuleType, tmp_path: Path) -> None:
    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    chunks_path = index_dir / "chunks.json"
    chunks_path.write_bytes(chunks_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch.*chunks.json"):
        new_adapter(index_dir=str(index_dir), load_existing=True)

    adapter.save()
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["vector_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="vector count mismatch"):
        new_adapter(index_dir=str(index_dir), load_existing=True)


def test_load_requires_exact_persisted_chunk_schema(
    fake_turbovec: types.ModuleType, tmp_path: Path
) -> None:
    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    chunks_path = index_dir / "chunks.json"
    manifest_path = index_dir / "manifest.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks[0].pop("paragraph")
    chunks_path.write_text(json.dumps(chunks, sort_keys=True), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["chunks.json"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="chunk metadata entry"):
        new_adapter(index_dir=str(index_dir), load_existing=True)


def test_failed_save_preserves_previous_complete_generation(
    fake_turbovec: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    original_atomic_bytes = TurboVecVectorStoreAdapter._atomic_bytes_at

    def fail_chunks(directory_fd: int, name: str, payload: bytes) -> str:
        if name == "chunks.json":
            raise OSError("injected chunks publication failure")
        return original_atomic_bytes(directory_fd, name, payload)

    monkeypatch.setattr(
        TurboVecVectorStoreAdapter,
        "_atomic_bytes_at",
        staticmethod(fail_chunks),
    )
    replacement = [Chunk(id=77, pdf_name="new.pdf", paragraph="new")]
    with pytest.raises(OSError, match="injected chunks"):
        adapter.upsert(replacement, [[1.0, 0.0]])

    restored = new_adapter(index_dir=str(index_dir), load_existing=True)
    assert restored.search([1.0, 0.0], 1)[0].chunk.id == 9001
    assert [path.name for path in tmp_path.iterdir()] == ["candidate"]


def test_load_rejects_real_index_bit_width_that_disagrees_with_manifest(
    fake_turbovec: types.ModuleType, tmp_path: Path
) -> None:
    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    index_path = index_dir / "index.tvim"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["bit_width"] = 2
    index_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["index.tvim"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="bit width"):
        new_adapter(index_dir=str(index_dir), load_existing=True)


def test_load_does_not_follow_file_swapped_to_symlink_after_validation(
    fake_turbovec: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    chunks_path = index_dir / "chunks.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(chunks_path.read_bytes())
    original = TurboVecVectorStoreAdapter._require_regular_file

    def validate_then_swap(path: Path) -> None:
        original(path)
        if path == chunks_path and not path.is_symlink():
            path.unlink()
            path.symlink_to(outside)

    monkeypatch.setattr(
        TurboVecVectorStoreAdapter,
        "_require_regular_file",
        staticmethod(validate_then_swap),
    )
    with pytest.raises(ValueError, match="symlink|regular"):
        new_adapter(index_dir=str(index_dir), load_existing=True)


def test_save_does_not_follow_file_swapped_to_symlink_after_validation(
    fake_turbovec: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    index_dir = tmp_path / "candidate"
    index_dir.mkdir()
    chunks_path = index_dir / "chunks.json"
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    original = TurboVecVectorStoreAdapter._reject_existing_symlink

    def validate_then_swap(path: Path) -> None:
        original(path)
        if path == chunks_path and not path.exists():
            path.symlink_to(outside)

    monkeypatch.setattr(
        TurboVecVectorStoreAdapter,
        "_reject_existing_symlink",
        staticmethod(validate_then_swap),
    )
    adapter = new_adapter(index_dir=str(index_dir))
    with pytest.raises(ValueError, match="symlink|regular"):
        adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_save_does_not_follow_parent_swapped_to_symlink_after_validation(
    fake_turbovec: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    parent = tmp_path / "owned"
    parent.mkdir()
    original_parent = tmp_path / "owned-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    adapter = new_adapter(index_dir=str(parent / "candidate"))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    adapter.drop_namespace()
    original_reject = TurboVecVectorStoreAdapter._reject_symlink_parents
    swapped = False

    def validate_then_swap(path: Path) -> None:
        nonlocal swapped
        original_reject(path)
        if path == parent and not swapped:
            parent.rename(original_parent)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(
        TurboVecVectorStoreAdapter,
        "_reject_symlink_parents",
        staticmethod(validate_then_swap),
    )
    with pytest.raises(ValueError, match="namespace.*symlink|real directory"):
        adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    assert list(outside.iterdir()) == []


def test_persistence_rejects_symlink_namespace_and_files(fake_turbovec: types.ModuleType, tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="namespace.*symlink"):
        new_adapter(index_dir=str(linked_dir))

    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    chunks_path = index_dir / "chunks.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(chunks_path.read_bytes())
    chunks_path.unlink()
    chunks_path.symlink_to(outside)
    with pytest.raises(ValueError, match="persisted file.*symlink"):
        new_adapter(index_dir=str(index_dir), load_existing=True)


def test_reset_and_drop_clear_memory_and_owned_persistence(fake_turbovec: types.ModuleType, tmp_path: Path) -> None:
    index_dir = tmp_path / "candidate"
    adapter = new_adapter(index_dir=str(index_dir))
    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])

    adapter.reset_collection()
    assert adapter.index is None
    assert adapter.chunks == []
    assert adapter.dimensions == 0
    assert index_dir.is_dir()
    assert list(index_dir.iterdir()) == []

    adapter.upsert(make_chunks(), [[1.0, 0.0], [0.0, 1.0]])
    adapter.drop_namespace()
    assert adapter.index is None
    assert not index_dir.exists()
    adapter.drop_namespace()


def test_real_turbovec_smoke_when_installed(tmp_path: Path) -> None:
    pytest.importorskip("turbovec")
    from benchmarking.adapters.vector_turbovec import TurboVecVectorStoreAdapter

    chunks = [Chunk(id=101 + i, pdf_name="smoke", paragraph=str(i)) for i in range(3)]
    vectors = np.eye(3, 8, dtype=np.float32).tolist()
    adapter = TurboVecVectorStoreAdapter("TurboVec", index_dir=str(tmp_path / "real"))
    adapter.upsert(chunks, vectors)
    assert adapter.search(vectors[0], 1)[0].chunk.id == 101
    restored = TurboVecVectorStoreAdapter("TurboVec", index_dir=str(tmp_path / "real"), load_existing=True)
    assert restored.search(vectors[1], 1)[0].chunk.id == 102
    replacement = [Chunk(id=201 + i, pdf_name="replacement", paragraph=str(i)) for i in range(3)]
    restored.upsert(replacement, vectors)
    reloaded = TurboVecVectorStoreAdapter("TurboVec", index_dir=str(tmp_path / "real"), load_existing=True)
    assert reloaded.search(vectors[2], 1)[0].chunk.id == 203
