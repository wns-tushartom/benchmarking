from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from benchmarking.core.registry import default_registry
from scripts.verify_official_artifacts_unchanged import (
    ArtifactVerificationError,
    check_runtime_baseline,
    check_tracked_baseline,
    write_runtime_baseline,
    write_tracked_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "configs" / "project_matrix_catalog.json"

EXPECTED_MATRIX = {
    "chunkers": [
        "entity_heuristic_w6",
        "entity_heuristic_w5",
        "entity_heuristic_w4",
        "Heading_sections_l2",
        "fixed_tok1200_ov150",
    ],
    "embeddings": [
        "jina_v3",
        "gte_multilingual_base",
        "openai_text-embedding-3-large",
    ],
    "vector_stores": ["Qdrant", "PGVector", "Weaviate", "FAISS"],
    "rerankers": [
        "Amazon Rerank v1",
        "Qwen3:4B Rerank",
        "bge-reranker-base",
    ],
}

EXPECTED_CHUNKERS = {
    "entity_heuristic_w6": (
        "source.services.project_chunking",
        "entity_heuristic_chunking",
        {"window_size": 6},
    ),
    "entity_heuristic_w5": (
        "source.services.project_chunking",
        "entity_heuristic_chunking",
        {"window_size": 5},
    ),
    "entity_heuristic_w4": (
        "source.services.project_chunking",
        "entity_heuristic_chunking",
        {"window_size": 4},
    ),
    "Heading_sections_l2": (
        "source.services.project_chunking",
        "heading_sections_chunking",
        {"level": 2},
    ),
    "fixed_tok1200_ov150": (
        "source.services.project_chunking",
        "fixed_token_chunks",
        {"chunk_tokens": 1200, "overlap_tokens": 150},
    ),
}

EXPECTED_ADAPTERS = {
    "embeddings": {
        "jina_v3": ("remote_http", "RemoteHTTPEmbeddingAdapter"),
        "gte_multilingual_base": ("remote_http", "RemoteHTTPEmbeddingAdapter"),
        "openai_text-embedding-3-large": ("openai", "OpenAIEmbeddingAdapter"),
    },
    "vector_stores": {
        "Qdrant": ("qdrant", "QdrantVectorStoreAdapter"),
        "PGVector": ("pgvector", "PGVectorStoreAdapter"),
        "Weaviate": ("weaviate", "WeaviateVectorStoreAdapter"),
        "FAISS": ("faiss", "FaissVectorStoreAdapter"),
    },
    "rerankers": {
        "Amazon Rerank v1": ("amazon_bedrock", "AmazonBedrockRerankerAdapter"),
        "Qwen3:4B Rerank": ("remote_http", "RemoteHTTPRerankerAdapter"),
        "bge-reranker-base": ("remote_http", "RemoteHTTPRerankerAdapter"),
    },
}

REGISTRY_KINDS = {
    "embeddings": "embedding",
    "vector_stores": "vector_store",
    "rerankers": "reranker",
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _runtime_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data" / "runtime"
    roots_file = tmp_path / "configs" / "roots.txt"
    baseline = tmp_path / "artifacts" / "runtime.sha256"
    _write(root / "nested" / "result.json", '{"ok": true}\n')
    _write(roots_file, "data/runtime\ndata/not-created\n")
    baseline.parent.mkdir(parents=True)
    return root, roots_file, baseline


def test_catalog_is_exact_canonical_180_product() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    assert catalog["schema_version"] == 1
    assert catalog["matrix"] == EXPECTED_MATRIX
    assert (
        len(catalog["matrix"]["chunkers"])
        * len(catalog["matrix"]["embeddings"])
        * len(catalog["matrix"]["vector_stores"])
        * len(catalog["matrix"]["rerankers"])
        == 180
    )
    for dimension, labels in EXPECTED_MATRIX.items():
        assert set(catalog["techniques"][dimension]) == set(labels)


def test_catalog_chunkers_resolve_to_exact_existing_production_functions() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    for chunker_id, (module_name, function_name, params) in EXPECTED_CHUNKERS.items():
        spec = catalog["techniques"]["chunkers"][chunker_id]
        assert spec == {
            "module": module_name,
            "callable": function_name,
            "params": params,
        }
        implementation = getattr(importlib.import_module(module_name), function_name)
        assert callable(implementation)


def test_catalog_adapters_resolve_to_registered_production_classes_and_config() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    official = json.loads(
        (REPO_ROOT / "configs" / "benchmark.local.json").read_text(encoding="utf-8")
    )
    registry = default_registry()

    for dimension, expected in EXPECTED_ADAPTERS.items():
        for label, (adapter_id, class_name) in expected.items():
            spec = catalog["techniques"][dimension][label]
            assert spec == official["techniques"][dimension][label]
            assert spec["adapter"] == adapter_id
            adapter_class = registry.get(REGISTRY_KINDS[dimension], adapter_id)
            assert adapter_class.__name__ == class_name

    serialized = json.dumps(catalog).casefold()
    for forbidden in (
        "fallback",
        "local_hash",
        "local-hash",
        "local_vector",
        "local-vector",
        "weighted_overlap",
        "weighted-overlap",
    ):
        assert forbidden not in serialized


def test_checked_in_protection_scope_is_exact_and_tracked_baseline_is_valid() -> None:
    roots = (
        REPO_ROOT / "configs" / "protected_runtime_artifact_roots.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert roots == [
        "data/modular_runs",
        "data/full_benchmark",
        "data/evaluation",
        "data/retrieval_smoke",
        "data/reranker_smoke",
    ]

    manifest = REPO_ROOT / "configs" / "protected_official_artifacts.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("  configs/benchmark.local.json")
    check_tracked_baseline(manifest, repo_root=REPO_ROOT)


def test_tracked_baseline_round_trip_detects_changed_and_missing_files(tmp_path: Path) -> None:
    tracked = tmp_path / "configs" / "benchmark.local.json"
    manifest = tmp_path / "configs" / "protected.sha256"
    _write(tracked, '{"matrix": 180}\n')

    write_tracked_baseline(
        manifest,
        [Path("configs/benchmark.local.json")],
        repo_root=tmp_path,
    )
    expected_hash = hashlib.sha256(tracked.read_bytes()).hexdigest()
    assert manifest.read_text(encoding="utf-8") == (
        f"{expected_hash}  configs/benchmark.local.json\n"
    )
    check_tracked_baseline(manifest, repo_root=tmp_path)

    tracked.write_text('{"matrix": 1}\n', encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="changed"):
        check_tracked_baseline(manifest, repo_root=tmp_path)

    tracked.unlink()
    with pytest.raises(ArtifactVerificationError, match="missing"):
        check_tracked_baseline(manifest, repo_root=tmp_path)


def test_baseline_creation_is_exclusive_without_explicit_replace(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    manifest = tmp_path / "tracked.sha256"
    _write(tracked, "v1\n")
    write_tracked_baseline(manifest, [Path("tracked.txt")], repo_root=tmp_path)
    original = manifest.read_bytes()

    tracked.write_text("v2\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="already exists"):
        write_tracked_baseline(manifest, [Path("tracked.txt")], repo_root=tmp_path)
    assert manifest.read_bytes() == original

    write_tracked_baseline(
        manifest,
        [Path("tracked.txt")],
        repo_root=tmp_path,
        replace=True,
    )
    assert manifest.read_bytes() != original
    check_tracked_baseline(manifest, repo_root=tmp_path)


def test_manifest_sorting_is_by_normalized_repository_path(tmp_path: Path) -> None:
    _write(tmp_path / "a.txt", "a")
    _write(tmp_path / "b.txt", "b")
    manifest = tmp_path / "tracked.sha256"

    write_tracked_baseline(
        manifest,
        [Path("b.txt"), Path("a.txt")],
        repo_root=tmp_path,
    )

    assert [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()] == [
        "a.txt",
        "b.txt",
    ]
    check_tracked_baseline(manifest, repo_root=tmp_path)


def test_runtime_baseline_records_missing_roots_and_round_trips(tmp_path: Path) -> None:
    _, roots_file, baseline = _runtime_fixture(tmp_path)

    write_runtime_baseline(baseline, roots_file, repo_root=tmp_path)

    lines = baseline.read_text(encoding="utf-8").splitlines()
    assert "!MISSING data/not-created" in lines
    assert any(line.endswith("  data/runtime/nested/result.json") for line in lines)
    assert lines == sorted(lines)
    check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)


def test_runtime_comparison_rejects_changed_missing_and_added_files(tmp_path: Path) -> None:
    root, roots_file, baseline = _runtime_fixture(tmp_path)
    result = root / "nested" / "result.json"
    original = result.read_bytes()
    write_runtime_baseline(baseline, roots_file, repo_root=tmp_path)

    result.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="runtime artifact baseline differs"):
        check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)

    result.write_bytes(original)
    result.unlink()
    with pytest.raises(ArtifactVerificationError, match="runtime artifact baseline differs"):
        check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)

    result.write_bytes(original)
    added = root / "added.txt"
    added.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="runtime artifact baseline differs"):
        check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)


def test_runtime_comparison_rejects_missing_root_state_change(tmp_path: Path) -> None:
    _, roots_file, baseline = _runtime_fixture(tmp_path)
    write_runtime_baseline(baseline, roots_file, repo_root=tmp_path)

    newly_created = tmp_path / "data" / "not-created"
    newly_created.mkdir()
    with pytest.raises(ArtifactVerificationError, match="runtime artifact baseline differs"):
        check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)


def test_manifests_fail_closed_on_malformed_duplicate_and_outside_entries(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.sha256"
    digest = "0" * 64

    manifest.write_text("not-a-manifest-line\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="malformed"):
        check_tracked_baseline(manifest, repo_root=tmp_path)

    manifest.write_text(
        f"{digest}  file.txt\n{digest}  file.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactVerificationError, match="duplicate"):
        check_tracked_baseline(manifest, repo_root=tmp_path)

    manifest.write_text(f"{digest}  ../outside.txt\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="normalized repository-relative"):
        check_tracked_baseline(manifest, repo_root=tmp_path)


def test_tracked_paths_and_runtime_roots_must_stay_inside_repository(tmp_path: Path) -> None:
    tracked_manifest = tmp_path / "tracked.sha256"
    with pytest.raises(ArtifactVerificationError, match="normalized repository-relative"):
        write_tracked_baseline(
            tracked_manifest,
            [tmp_path.parent / "outside.txt"],
            repo_root=tmp_path,
        )

    roots_file = tmp_path / "roots.txt"
    roots_file.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(ArtifactVerificationError, match="normalized repository-relative"):
        write_runtime_baseline(
            tmp_path / "runtime.sha256",
            roots_file,
            repo_root=tmp_path,
        )


def test_control_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    tracked = tmp_path / "configs" / "benchmark.local.json"
    real_manifest = tmp_path / "configs" / "real.sha256"
    alias_manifest = tmp_path / "configs" / "alias.sha256"
    _write(tracked, '{"matrix": 180}\n')
    write_tracked_baseline(
        real_manifest,
        [Path("configs/benchmark.local.json")],
        repo_root=tmp_path,
    )
    alias_manifest.symlink_to(real_manifest)

    with pytest.raises(ArtifactVerificationError, match="symlink"):
        check_tracked_baseline(alias_manifest, repo_root=tmp_path)


def test_dangling_symlink_is_not_accepted_as_missing_runtime_root(tmp_path: Path) -> None:
    root, roots_file, baseline = _runtime_fixture(tmp_path)
    write_runtime_baseline(baseline, roots_file, repo_root=tmp_path)
    dangling = tmp_path / "data" / "not-created"
    dangling.symlink_to(tmp_path / "data" / "still-missing", target_is_directory=True)

    with pytest.raises(ArtifactVerificationError, match="symlink"):
        check_runtime_baseline(baseline, roots_file, repo_root=tmp_path)


def test_baseline_publication_rejects_symlinked_parent(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    physical_parent = tmp_path / "physical"
    alias_parent = tmp_path / "alias"
    _write(tracked, "content\n")
    physical_parent.mkdir()
    alias_parent.symlink_to(physical_parent, target_is_directory=True)

    with pytest.raises(ArtifactVerificationError, match="symlink"):
        write_tracked_baseline(
            alias_parent / "baseline.sha256",
            [Path("tracked.txt")],
            repo_root=tmp_path,
        )
    assert not (physical_parent / "baseline.sha256").exists()


def test_protected_path_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    _write(physical / "tracked.txt", "content\n")
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(ArtifactVerificationError, match="symlink"):
        write_tracked_baseline(
            tmp_path / "baseline.sha256",
            [Path("alias/tracked.txt")],
            repo_root=tmp_path,
        )
