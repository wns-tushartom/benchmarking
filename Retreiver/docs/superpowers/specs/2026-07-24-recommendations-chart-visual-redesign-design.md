# Recommendations Chart Visual Redesign

**Date:** 2026-07-24  
**Status:** Approved direction  
**Target:** Canonical WNS benchmark dashboard, Recommendations page

## Goal

Make the Recommendations analytics immediately useful for decisions. A reviewer must be able to identify related pipeline variants, distinguish the strongest and weakest combinations, and inspect exact metrics without decoding a dense monochrome dot cloud.

## Current weaknesses

- Category colours repeat after four values and can map unrelated components to the same colour.
- Every point uses the same circular marker, so reranker variants are indistinguishable.
- Winners receive limited text labels, but weak combinations have no clear visual treatment.
- Native SVG titles are too slow and sparse for practical inspection.
- Dense or overlapping points do not expose their related family.
- The heatmap uses coarse buckets without a strong low-to-high semantic scale.

## Design direction

Use a relationship-first scatter plot. Colour communicates the active component grouping, marker shape communicates reranker, and ranking treatments expose the best and worst combinations. Colour remains meaningful rather than decorative.

### Visual encoding

| Visual channel | Meaning |
|---|---|
| X position | Selected X metric, default average latency per query |
| Y position | Selected Y metric, default Recall@5 |
| Point colour | Selected grouping component, default vector database |
| Point shape | Canonical reranker |
| Point size | Fixed by default; evaluated-query sizing remains optional |
| Mint numbered halo | Top five rows under the canonical recommendation ranking |
| Rose dashed ring | Bottom five rows under the same ranking |
| Family connector | Variants sharing chunker, embedding, and vector database |

The canonical recommendation ranking is the same ordering used by the active Recommendations table. The chart must not invent a second winner score. Changing X or Y metrics changes plotted position but not the published recommendation rank.

### Colour system

- Replace index modulo four with a stable, deterministic category-to-colour map.
- Provide at least six accessible hues for uploaded/custom categories: cyan, mint, violet, amber, rose, and blue.
- Use the same category colour throughout points, legend markers, selected-state details, and family connectors.
- Category mappings must not change when filters hide values or when rows arrive in a different order.
- Colours must remain distinguishable against the existing dark technical dashboard and meet non-text contrast expectations where practical.

### Marker shapes

- `none`: circle.
- BGE reranker: diamond.
- Amazon reranker: triangle.
- Any additional reranker: square, then a documented fallback shape.
- Every point retains an accessible text title and keyboard focus target, so shape and colour are never the only identifiers.

### Rank and state treatment

- Top five: mint outer halo, rank number, elevated opacity, subtle glow.
- Bottom five: rose dashed outer ring and slightly reduced opacity.
- Middle rows: full category colour with quieter stroke.
- Recommendation-role labels remain pinned only to Quality, Speed, and Value winners.
- Failed, running, incomplete, and not-run rows remain excluded from the scatter and represented honestly in the heatmap.

### Related-family interaction

A family is the exact tuple `(chunker, embedding, vector database)`, with reranker as its variant.

- Default view shows no persistent connector web at 180-row scale.
- Hover, keyboard focus, or click highlights every variant in the selected family.
- Highlighted variants are connected using a thin line in the family colour.
- Non-family points fade without disappearing.
- Click locks the family selection; Escape or clicking empty chart space clears it.
- Exact or near-identical coordinates receive a small deterministic collision spread while the tooltip continues to show unmodified metric values.

### Exact-data inspection

Hover or focus opens a compact tooltip containing:

- recommendation rank,
- complete pipeline label,
- chunker, embedding, vector database, and reranker,
- selected X and Y metric values,
- Recall@5, MRR, nDCG@5, average latency, and evaluated-query count when recorded,
- Quality, Speed, or Value role badge when applicable.

Clicking a point locks the detail state for presentation use.

### Quick views

Add a compact `Show` control:

- All combinations,
- Top 10,
- Bottom 10,
- Pareto frontier.

The Pareto frontier contains rows not dominated on the current X/Y orientation. For default latency versus Recall@5, lower X and higher Y are better. When axis metrics change, orientation must come from the metric contract rather than an assumption.

### Plot guidance

- Retain explicit `Faster ←` and `Better ↑` guidance for the default axes.
- Add a subtle ideal-zone background only when the active metric directions support it.
- Add restrained grid lines and readable min/max ticks.
- Keep the chart title and active-source context outside dense plotting regions.

## Heatmap redesign

Use an ordered performance scale instead of unrelated category colours:

`rose → amber → cyan → mint`

Requirements:

- exact metric value remains visible in every measured cell,
- strongest measured cell receives a mint outline,
- weakest measured cell receives a rose outline,
- hover highlights the active row and column,
- `not run`, `running`, `incomplete`, and `failed` use distinct neutral/state treatments,
- missing values never render as zero,
- legend shows exact low and high values plus named state swatches.

## Responsive behavior

- Desktop: full plot, adjacent or anchored tooltip, multi-column legend.
- Tablet: full-width plot with wrapped legend and controls.
- Mobile: stacked controls, horizontally scrollable plot only when necessary, tooltip/details below the plot, and touch-friendly legend filters.
- All point interaction must work with keyboard focus and touch selection.

## Technical boundaries

- Preserve the no-build, dependency-free frontend architecture.
- Use inline SVG and existing DOM rendering patterns.
- Do not alter backend result contracts, official metric admission, recommendation scoring, or source isolation.
- Preserve official, uploaded labelled, baseline, and evidence-only behavior.
- Bump the served asset cache key after the final frontend change.

## Verification contract

### Automated

- Stable colour mapping is independent of row/filter order and supports more than four categories.
- Canonical rerankers render distinct marker shapes.
- Top five and bottom five classes follow the existing recommendation order.
- Family keys and selected-family highlighting use chunker, embedding, and vector database.
- Quick views return the expected row subsets.
- Pareto logic respects metric direction.
- Heatmap cells preserve exact values and distinct lifecycle states.
- Existing Recommendations, metrics, source-isolation, release-contract, and 180-row tests remain green.
- `node --check web/app.js`, Python release verifier, and `git diff --check` pass.

### Browser QA

Capture and inspect:

- desktop Recommendations page with all 180 rows,
- desktop family hover/locked selection,
- Top 10 and Bottom 10 quick views,
- redesigned heatmap with exact values,
- mobile Recommendations page at approximately 390 px,
- keyboard focus and tooltip state,
- no console errors or failed API requests.

The visual target is at least 8.5/10 for a client-facing demonstration, with no clipped labels, unreadable points, or misleading missing values.

## Non-goals

- No new backend charting service.
- No third-party chart library.
- No change to recommendation formulas or benchmark results.
- No decorative animation beyond short opacity/transform transitions.
- No permanent connector lines across all 180 points.
