from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from pypdf import PdfReader

from scripts.dashboard_source_catalog import build_source_catalog
from source.services.project_questions import parse_question_bytes


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "demo" / "northstar_services"
EXPECTED_DOCUMENTS = {
    "refund_cancellation_policy.pdf",
    "security_incident_response.pdf",
    "travel_expense_policy.pdf",
    "customer_support_sla.pdf",
    "data_retention_deletion.pdf",
}


def test_northstar_demo_pack_is_complete_readable_and_two_path(tmp_path: Path) -> None:
    documents = sorted((PACK / "documents").glob("*.pdf"))
    assert {document.name for document in documents} == EXPECTED_DOCUMENTS
    extracted = {
        document.name: "\n".join(page.extract_text() or "" for page in PdfReader(document).pages)
        for document in documents
    }
    assert all(len(text.strip()) >= 120 for text in extracted.values())

    archive_path = PACK / "northstar-demo-documents.zip"
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        assert set(members) == EXPECTED_DOCUMENTS
        assert all(not Path(member).is_absolute() and ".." not in Path(member).parts for member in members)
        assert archive.testzip() is None
        assert all(archive.read(member).startswith(b"%PDF-") for member in members)

    ground_truth_path = PACK / "northstar-demo-groundtruth.csv"
    with ground_truth_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 11
    assert set(rows[0]) == {"query", "reference_context", "answer"}
    parsed = parse_question_bytes(ground_truth_path.name, ground_truth_path.read_bytes())
    assert len(parsed) == len(rows)
    labelled = [question for question in parsed if not question.labels.is_empty()]
    assert len(labelled) >= 10
    assert len(labelled) == len(parsed)
    assert any(len(question.labels.reference_contexts) >= 2 for question in labelled)

    catalog_groundtruth = tmp_path / "data" / "groundtruth" / ground_truth_path.name
    catalog_groundtruth.parent.mkdir(parents=True)
    shutil.copy2(ground_truth_path, catalog_groundtruth)
    catalog = build_source_catalog(tmp_path)
    discovered = next(
        row
        for row in catalog["groundtruth"]
        if row["id"] == "groundtruth:repository:northstar-demo-groundtruth.csv"
    )
    assert discovered["valid"] is True
    assert discovered["row_count"] == len(rows)

    normalized_documents = [" ".join(text.split()) for text in extracted.values()]
    for question in labelled:
        for context in question.labels.reference_contexts:
            assert any(" ".join(context.split()) in document for document in normalized_documents)

    query_path = PACK / "northstar-demo-queries.txt"
    evidence_queries = [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(evidence_queries) >= 5
    assert any("holiday bonus" in query.casefold() for query in evidence_queries)
    assert all(question.labels.is_empty() for question in parse_question_bytes(query_path.name, query_path.read_bytes()))

    diagnostics = json.loads((PACK / "expected_results.json").read_text(encoding="utf-8"))
    assert diagnostics["document_count"] == 5
    assert diagnostics["labelled_query_count"] == len(labelled)
    assert diagnostics["unanswerable_query_count"] == 1
    assert diagnostics["evidence_only_query_count"] == len(evidence_queries)
    assert set(diagnostics["expected_top_documents"].values()).issubset(EXPECTED_DOCUMENTS)

    readme = (PACK / "README.md").read_text(encoding="utf-8")
    assert "Scored path" in readme
    assert "Evidence-only path" in readme
    assert "does not produce quality metrics" in readme
    assert "Northstar Services" in readme
    for required_name in (
        "northstar-demo-documents.zip",
        "northstar-demo-groundtruth.csv",
        "northstar-demo-queries.txt",
        "EXPECTED_RESULTS.md",
    ):
        assert required_name in readme

    expected_results = (PACK / "EXPECTED_RESULTS.md").read_text(encoding="utf-8")
    assert "Multi-document comparison" in expected_results
    assert "Deliberately unanswerable" in expected_results
    assert "not applicable" in expected_results.casefold()
    assert "failure indicators" in expected_results.casefold()
