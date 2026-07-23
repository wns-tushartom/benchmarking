from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import dashboard_source_catalog as catalog


def _write_gt(path: Path, *, query: str = "What is the policy?", answer: str = "Policy text") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "answer", "source"])
        writer.writeheader()
        writer.writerow({"query": query, "answer": answer, "source": "policy.pdf"})


def _write_project(root: Path, project_id: str, *, with_groundtruth: bool) -> None:
    project = root / "data" / "user_projects" / project_id
    raw = project / "raw_uploads" / "policy.txt"
    corpus = project / "extracted_text" / "documents.jsonl"
    raw.parent.mkdir(parents=True)
    corpus.parent.mkdir(parents=True)
    raw.write_text(f"{project_id} policy", encoding="utf-8")
    corpus.write_text("{\"schema_version\":1}\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "project_id": project_id,
        "label": project_id.title(),
        "saved_path": f"data/user_projects/{project_id}/raw_uploads/policy.txt",
        "canonical_corpus": f"data/user_projects/{project_id}/extracted_text/documents.jsonl",
        "extraction_status": "complete",
        "document_count": 1,
    }
    if with_groundtruth:
        gt = project / "questions" / "groundtruth.csv"
        _write_gt(gt, query=f"Question for {project_id}?", answer=f"Answer for {project_id}")
        manifest["question_file"] = f"data/user_projects/{project_id}/questions/groundtruth.csv"
    (project / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_every_dataset_catalog_row_owns_exactly_one_linked_groundtruth_id(tmp_path: Path) -> None:
    _write_project(tmp_path, "alpha", with_groundtruth=True)
    _write_project(tmp_path, "beta", with_groundtruth=False)

    rows = catalog.build_source_catalog(tmp_path)["datasets"]

    assert rows
    assert all(set(row).issuperset({"id", "linked_groundtruth_id"}) for row in rows)
    assert all(isinstance(row["linked_groundtruth_id"], str) for row in rows)
    assert all(row["linked_groundtruth_id"].startswith("groundtruth:") for row in rows)


def test_official_dataset_links_only_to_valid_official_repository_groundtruth(tmp_path: Path) -> None:
    _write_gt(tmp_path / "data" / "groundtruth" / "other.csv")
    official_path = tmp_path / "data" / "groundtruth" / "qa_text_test.csv"
    _write_gt(official_path)

    default = next(
        row for row in catalog.build_source_catalog(tmp_path)["datasets"]
        if row["id"] == catalog.DEFAULT_DATASET_ID
    )

    assert default["linked_groundtruth_id"] == "groundtruth:repository:qa_text_test.csv"
    assert default.get("mode") != "evidence_only"
    dataset, groundtruth = catalog.resolve_source_pair(
        tmp_path, default["id"], default["linked_groundtruth_id"]
    )
    assert dataset.id == catalog.DEFAULT_DATASET_ID
    assert groundtruth is not None
    assert groundtruth.path == official_path.resolve()


def test_official_dataset_prefers_cleaned_groundtruth_500_over_legacy_name(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "data" / "groundtruth" / "qa_text_test.csv"
    canonical = tmp_path / "data" / "groundtruth" / "groundtruth_500.csv"
    _write_gt(legacy, query="legacy question", answer="legacy answer")
    _write_gt(canonical, query="canonical question", answer="canonical answer")

    default = next(
        row for row in catalog.build_source_catalog(tmp_path)["datasets"]
        if row["id"] == catalog.DEFAULT_DATASET_ID
    )

    assert catalog.OFFICIAL_GROUNDTRUTH_ID == "groundtruth:repository:groundtruth_500.csv"
    assert default["linked_groundtruth_id"] == catalog.OFFICIAL_GROUNDTRUTH_ID
    _dataset, groundtruth = catalog.resolve_source_pair(
        tmp_path, default["id"], default["linked_groundtruth_id"]
    )
    assert groundtruth is not None
    assert groundtruth.path == canonical.resolve()


def test_invalid_official_groundtruth_does_not_become_a_link(tmp_path: Path) -> None:
    invalid = tmp_path / "data" / "groundtruth" / "qa_text_test.csv"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("query,answer\n,missing query\n", encoding="utf-8")

    default = catalog.build_source_catalog(tmp_path)["datasets"][0]

    assert default["linked_groundtruth_id"] == catalog.NONE_GROUNDTRUTH_ID
    assert default["mode"] == "evidence_only"


def test_project_links_only_to_its_own_valid_groundtruth_and_no_gt_is_evidence_only(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path, "alpha", with_groundtruth=True)
    _write_project(tmp_path, "beta", with_groundtruth=False)

    datasets = {
        row["id"]: row for row in catalog.build_source_catalog(tmp_path)["datasets"]
    }

    assert datasets["project:alpha"]["linked_groundtruth_id"] == "groundtruth:project:alpha"
    assert datasets["project:alpha"].get("mode") != "evidence_only"
    assert datasets["project:beta"]["linked_groundtruth_id"] == catalog.NONE_GROUNDTRUTH_ID
    assert datasets["project:beta"]["mode"] == "evidence_only"


def test_two_projects_cannot_cross_select_each_others_groundtruth(tmp_path: Path) -> None:
    _write_project(tmp_path, "alpha", with_groundtruth=True)
    _write_project(tmp_path, "beta", with_groundtruth=True)

    with pytest.raises(ValueError, match="linked ground truth"):
        catalog.resolve_source_pair(
            tmp_path, "project:alpha", "groundtruth:project:beta"
        )
    with pytest.raises(ValueError, match="linked ground truth"):
        catalog.resolve_source_pair(
            tmp_path, "project:beta", "groundtruth:project:alpha"
        )


def test_project_without_groundtruth_rejects_unrelated_repository_groundtruth(tmp_path: Path) -> None:
    _write_project(tmp_path, "alpha", with_groundtruth=False)
    _write_gt(tmp_path / "data" / "groundtruth" / "qa_text_test.csv")

    with pytest.raises(ValueError, match="linked ground truth"):
        catalog.resolve_source_pair(
            tmp_path,
            "project:alpha",
            "groundtruth:repository:qa_text_test.csv",
        )

    dataset, groundtruth = catalog.resolve_source_pair(
        tmp_path, "project:alpha", catalog.NONE_GROUNDTRUTH_ID
    )
    assert dataset.id == "project:alpha"
    assert groundtruth is None
