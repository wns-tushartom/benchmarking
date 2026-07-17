# Northstar Services RAG Demo

A compact **fictional** corpus for demonstrating the uploaded-project retrieval pipeline without touching official WNS or NVIDIA artifacts.

## Deliverables

- `northstar-demo-documents.zip` — exactly five readable PDFs:
  - refund and cancellation policy
  - security incident response procedure
  - employee travel and expense policy
  - customer support SLA tiers
  - data retention and deletion policy
- `northstar-demo-groundtruth.csv` — 11 labelled retrieval checks, including one multi-document comparison.
- `northstar-demo-queries.txt` — six unlabelled queries for manual evidence review, including one deliberately unanswerable query.
- `EXPECTED_RESULTS.md` — expected supporting documents, not-applicable behavior, and failure indicators.
- `expected_results.json` — machine-readable diagnostics used by regression tests.
- `build_pack.py` — deterministic local builder using the repository's `pypdf` dependency.

## Scored path

1. Upload `northstar-demo-documents.zip` as a **Dataset**.
2. Select the resulting uploaded project in the global **Dataset** selector.
3. Upload/select `northstar-demo-groundtruth.csv` as **Ground truth**.
4. Select a small component matrix and run **Check readiness**.
5. Confirm the server-validated Cartesian combination count, then run the project matrix.
6. Inspect Metrics, Recommendations, and Top Hits Evidence for the exact returned `project_id` + `run_id`.
7. Compare supporting documents with `EXPECTED_RESULTS.md`.

Only labelled queries are eligible for Recall@K, MRR@K, nDCG@K, and scored recommendations. The holiday-bonus query is deliberately unanswerable, appears only in the evidence-only set, and has no relevance labels.

## Evidence-only path

1. Keep the same uploaded project selected.
2. Select **None — evidence-only** for Ground truth.
3. Paste or upload `northstar-demo-queries.txt`.
4. Choose retrieval only or explicitly select a reranker.
5. Run **Check readiness**, confirm the server count, and launch.
6. Inspect source documents, excerpts, retrieval scores, optional reranker scores, latency, and evidence coverage.

The evidence-only path **does not produce quality metrics**, winners, Recall, MRR, nDCG, accuracy, or best-quality claims. Reranking may change operational evidence ordering but does not create semantic quality labels.

## Integrity expectations

- Uploaded-project artifacts remain under the project-owned workspace.
- Official WNS matrix artifacts remain unchanged.
- NVIDIA text and multimodal lanes remain separate and unchanged.
- Invalid, incompatible, or path-like source identifiers fail closed.
- Partial and failed rows remain diagnostic and cannot influence final recommendations.
- The selected uploaded result replaces stale official/project rows in source-sensitive views.

## Rebuild and verify

```bash
python demo/northstar_services/build_pack.py
python -m pytest -q tests/test_northstar_demo_pack.py
```
