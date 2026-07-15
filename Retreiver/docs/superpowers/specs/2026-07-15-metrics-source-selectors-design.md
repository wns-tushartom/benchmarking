# Metrics source selectors design

## Decision
Metrics and Recommendations expose the same Dataset, Ground truth, and Result set controls backed by one active source selection. They must not maintain independent selection state.

## Behavior
- Official WNS + official ground truth can switch between the reranked matrix and no-reranker baseline.
- Uploaded datasets expose only ground truths and completed runs actually associated with that dataset.
- A selection change on either page synchronizes the other page and renders both from the same result payload.
- Labelled runs populate scored Metrics; evidence-only runs explicitly suppress Recall/MRR/nDCG/score claims.
- Existing component filters remain scoped to the selected result payload.

## Integrity
The browser continues to send opaque dataset/ground-truth/run IDs. Server-side allowlisted resolution and project/run identity checks remain unchanged.

## Verification
Static UI contract tests, Node behavior tests for bidirectional synchronization and row scoping, existing recommendation/source tests, full runnable pytest suite, JavaScript syntax check, API smoke, desktop/mobile screenshots, and browser-console inspection.
