# Northstar Services expected results

## Labelled retrieval checks

- **How long after purchase can a customer request a full refund?** — `refund_cancellation_policy.pdf` — Within 30 calendar days.
- **What must a refund request include?** — `refund_cancellation_policy.pdf` — The order number and purchase email address.
- **What must an employee do after noticing a suspected security incident?** — `security_incident_response.pdf` — Report it immediately and do not investigate alone.
- **When is the incident lessons-learned review due?** — `security_incident_response.pdf` — Within ten business days after closure.
- **What is the daily meal allowance for employee travel?** — `travel_expense_policy.pdf` — 75 US dollars, excluding alcohol.
- **When is an itemized travel receipt required?** — `travel_expense_policy.pdf` — For each individual expense above 25 US dollars.
- **What is the Priority One acknowledgement target?** — `customer_support_sla.pdf` — Within 15 minutes.
- **When is the Priority One root-cause analysis delivered?** — `customer_support_sla.pdf` — Within five business days after resolution.
- **How long are production application logs retained?** — `data_retention_deletion.pdf` — 90 days.
- **What can delay a verified deletion request beyond 30 days?** — `data_retention_deletion.pdf` — A legal hold.
- **Compare the refund-request window with the verified deletion-request deadline.** — `refund_cancellation_policy.pdf, data_retention_deletion.pdf` — Both are 30 days, measured from purchase and verified request respectively.

## Multi-document comparison

The comparison query must retrieve both the refund/cancellation and data-retention documents.

## Deliberately unanswerable

The holiday-bonus query has no supporting document. Returning text is not a retrieval-quality success.

## Evidence-only expectations

Recall, MRR, nDCG, accuracy, winner score, and quality recommendations are **not applicable** without labels.

## Failure indicators

- Official WNS or NVIDIA artifacts change during this isolated demo.
- The unanswerable query is counted as a successful relevant hit.
- Evidence-only mode displays quality metrics or a best-quality winner.
- A result cites a document outside this five-document corpus.
