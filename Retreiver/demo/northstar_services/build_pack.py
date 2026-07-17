from __future__ import annotations

import csv
import json
from pathlib import Path
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


PACK = Path(__file__).resolve().parent
DOCUMENTS = PACK / "documents"
ARCHIVE = PACK / "northstar-demo-documents.zip"
GROUND_TRUTH = PACK / "northstar-demo-groundtruth.csv"
QUERY_FILE = PACK / "northstar-demo-queries.txt"

DOCS: dict[str, tuple[str, list[str]]] = {
    "refund_cancellation_policy.pdf": (
        "Northstar Services Refund and Cancellation Policy",
        [
            "Customers may request a full refund within 30 calendar days of the original purchase date.",
            "Cancellation requests received before service activation incur no fee; requests after activation are reviewed for consumed usage.",
            "A refund request must include the order number and the email address used for the purchase.",
            "Approved refunds return to the original payment method within five to seven business days.",
            "Enterprise customers submit cancellation requests through their named account manager.",
        ],
    ),
    "security_incident_response.pdf": (
        "Northstar Services Security Incident Response Procedure",
        [
            "Employees must report suspected security incidents to the security hotline immediately and must not investigate alone.",
            "The incident commander classifies severity, assigns owners, preserves evidence, and starts the response timeline.",
            "Containment actions may include credential rotation, network isolation, access revocation, and temporary service restrictions.",
            "For confirmed customer-data exposure, affected customers are notified without undue delay after facts are validated.",
            "The response team completes a lessons-learned review within ten business days after incident closure.",
        ],
    ),
    "travel_expense_policy.pdf": (
        "Northstar Services Employee Travel and Expense Policy",
        [
            "International airfare requires manager approval before booking and must use the contracted travel portal.",
            "The daily meal allowance is 75 US dollars and excludes alcoholic beverages.",
            "An itemized receipt is required for every individual expense above 25 US dollars.",
            "Expense reports must be submitted within ten business days after the employee returns from travel.",
            "Personal vehicle mileage is reimbursed at 67 cents per approved business mile.",
        ],
    ),
    "customer_support_sla.pdf": (
        "Northstar Services Customer Support SLA Tiers",
        [
            "Priority One incidents are acknowledged within 15 minutes and receive continuous support until service is restored.",
            "Priority Two incidents are acknowledged within one hour during the customer's contracted support window.",
            "Priority Three incidents are acknowledged within four business hours.",
            "Enterprise cases may be opened through the support portal, emergency phone line, or technical account manager.",
            "A written root-cause analysis is delivered within five business days after a Priority One incident is resolved.",
        ],
    ),
    "data_retention_deletion.pdf": (
        "Northstar Services Data Retention and Deletion Standard",
        [
            "Production application logs are retained for 90 days and then permanently deleted from active systems.",
            "Encrypted daily backups are retained for 35 days using a rolling expiration schedule.",
            "Customer account records are retained while the account is active and for 12 months after account closure.",
            "A verified deletion request is completed within 30 days unless a legal hold requires continued retention.",
            "Legal-hold records are isolated from routine deletion and released only with written approval from legal counsel.",
        ],
    ),
}

# query, supporting documents, exact supporting contexts, reference answer
QUESTIONS: list[tuple[str, list[str], list[str], str]] = [
    ("How long after purchase can a customer request a full refund?", ["refund_cancellation_policy.pdf"], [DOCS["refund_cancellation_policy.pdf"][1][0]], "Within 30 calendar days."),
    ("What must a refund request include?", ["refund_cancellation_policy.pdf"], [DOCS["refund_cancellation_policy.pdf"][1][2]], "The order number and purchase email address."),
    ("What must an employee do after noticing a suspected security incident?", ["security_incident_response.pdf"], [DOCS["security_incident_response.pdf"][1][0]], "Report it immediately and do not investigate alone."),
    ("When is the incident lessons-learned review due?", ["security_incident_response.pdf"], [DOCS["security_incident_response.pdf"][1][4]], "Within ten business days after closure."),
    ("What is the daily meal allowance for employee travel?", ["travel_expense_policy.pdf"], [DOCS["travel_expense_policy.pdf"][1][1]], "75 US dollars, excluding alcohol."),
    ("When is an itemized travel receipt required?", ["travel_expense_policy.pdf"], [DOCS["travel_expense_policy.pdf"][1][2]], "For each individual expense above 25 US dollars."),
    ("What is the Priority One acknowledgement target?", ["customer_support_sla.pdf"], [DOCS["customer_support_sla.pdf"][1][0]], "Within 15 minutes."),
    ("When is the Priority One root-cause analysis delivered?", ["customer_support_sla.pdf"], [DOCS["customer_support_sla.pdf"][1][4]], "Within five business days after resolution."),
    ("How long are production application logs retained?", ["data_retention_deletion.pdf"], [DOCS["data_retention_deletion.pdf"][1][0]], "90 days."),
    ("What can delay a verified deletion request beyond 30 days?", ["data_retention_deletion.pdf"], [DOCS["data_retention_deletion.pdf"][1][3]], "A legal hold."),
    (
        "Compare the refund-request window with the verified deletion-request deadline.",
        ["refund_cancellation_policy.pdf", "data_retention_deletion.pdf"],
        [DOCS["refund_cancellation_policy.pdf"][1][0], DOCS["data_retention_deletion.pdf"][1][3]],
        "Both are 30 days, measured from purchase and verified request respectively.",
    ),
]

EVIDENCE_QUERIES = [
    "A traveller lost a receipt for a 40 dollar taxi. What policy applies?",
    "Which actions can contain a confirmed security incident?",
    "Compare Priority One and Priority Two acknowledgement times.",
    "Which records survive routine deletion during a legal hold?",
    "A customer cancelled after activation. What review should they expect?",
    "What holiday bonus does Northstar Services guarantee to every employee?",
]


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    lines = [title, "", "Northstar Services - fictional stakeholder demonstration corpus", ""]
    for paragraph in paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=82))
        lines.append("")
    operators = ["BT", "/F1 11 Tf", "14 TL", "72 730 Td"]
    for line in lines:
        operators.extend((f"({_pdf_escape(line)}) Tj", "T*"))
    operators.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(operators).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def _context_cell(contexts: list[str]) -> str:
    if not contexts:
        return ""
    if len(contexts) == 1:
        return contexts[0]
    return json.dumps(contexts, ensure_ascii=False)


def main() -> None:
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    for old in DOCUMENTS.glob("*.pdf"):
        old.unlink()
    for obsolete in (PACK / "ground_truth.csv", PACK / "evidence_only_queries.txt"):
        obsolete.unlink(missing_ok=True)

    for filename, (title, paragraphs) in DOCS.items():
        write_pdf(DOCUMENTS / filename, title, paragraphs)

    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
        for filename in sorted(DOCS):
            archive.write(DOCUMENTS / filename, arcname=filename)

    with GROUND_TRUTH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "reference_context", "answer"])
        writer.writerows(
            (query, _context_cell(contexts), answer)
            for query, _documents, contexts, answer in QUESTIONS
        )

    QUERY_FILE.write_text("\n".join(EVIDENCE_QUERIES) + "\n", encoding="utf-8")

    labelled = [row for row in QUESTIONS if row[2]]
    diagnostics = {
        "document_count": len(DOCS),
        "labelled_query_count": len(labelled),
        "unanswerable_query_count": 1,
        "evidence_only_query_count": len(EVIDENCE_QUERIES),
        "expected_top_documents": {
            query: documents[0] for query, documents, contexts, _answer in labelled if contexts
        },
        "expected_support_documents": {
            query: documents for query, documents, _contexts, _answer in QUESTIONS
        },
        "integrity_notes": [
            "Scored runs compute quality metrics only from rows with relevance labels.",
            "The holiday-bonus query is deliberately unanswerable and has no relevance labels.",
            "Evidence-only runs expose passages and latency without quality metrics or winners.",
        ],
    }
    diagnostics["expected_support_documents"][EVIDENCE_QUERIES[-1]] = []
    (PACK / "expected_results.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    expected_lines = [
        "# Northstar Services expected results",
        "",
        "## Labelled retrieval checks",
        "",
    ]
    for query, documents, _contexts, answer in labelled:
        expected_lines.append(f"- **{query}** — `{', '.join(documents)}` — {answer}")
    expected_lines.extend(
        [
            "",
            "## Multi-document comparison",
            "",
            "The comparison query must retrieve both the refund/cancellation and data-retention documents.",
            "",
            "## Deliberately unanswerable",
            "",
            "The holiday-bonus query has no supporting document. Returning text is not a retrieval-quality success.",
            "",
            "## Evidence-only expectations",
            "",
            "Recall, MRR, nDCG, accuracy, winner score, and quality recommendations are **not applicable** without labels.",
            "",
            "## Failure indicators",
            "",
            "- Official WNS or NVIDIA artifacts change during this isolated demo.",
            "- The unanswerable query is counted as a successful relevant hit.",
            "- Evidence-only mode displays quality metrics or a best-quality winner.",
            "- A result cites a document outside this five-document corpus.",
            "",
        ]
    )
    (PACK / "EXPECTED_RESULTS.md").write_text("\n".join(expected_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
