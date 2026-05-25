# WNS Benchmark Dashboard Design

## Purpose
Help WNS benchmarking work feel operational: compare benchmark configurations, quickly identify winners, and inspect tradeoffs without opening raw CSVs.

## Visual direction
- Dark technical dashboard, high contrast, compact but premium.
- Emphasize ranking, filters, recall-vs-latency tradeoffs, and chunking comparison.
- Avoid decorative noise; use clear metrics and usable controls.

## Palette
- Background: #050913 / #07111f
- Panel: #0e1a2b / #13233a
- Border: #243850
- Primary cyan: #7dd3fc
- Success mint: #a7f3d0
- Warning amber: #fbbf24
- Risk rose: #fda4af
- Text: #edf5ff
- Muted: #9fb1c8

## Typography
- System sans-serif.
- Page title: 40-44px desktop, 30-34px mobile.
- Section title: 18-22px.
- Dense data text: 12-13px.

## Components
- KPI cards for runs, queries, best recall, best latency.
- Customization controls: rank metric, DB, embedding, chunking, reranker, top N.
- Visualizations:
  - Recall vs latency scatter.
  - Top recall bar chart.
  - Chunking comparison bars.
  - Matrix heatmap by chunking × vector DB.
- Tables remain available for exact values.

## Interaction rules
- Filters update client-side instantly.
- Benchmark run buttons should remain visible near controls.
- Dashboard should work with static `/api/results` JSON and no frontend build step.
- No external CDN dependencies; local/offline safe.

## Responsive rules
- Desktop: two-column analytical grid.
- Mobile: single-column stacked cards and horizontally scrollable tables.
