# Global Dataset and Ground-Truth Selectors

## Goal

Make the standard and NVIDIA Run Pipeline pages operate on explicit, independently selectable data sources and ground-truth files. Preserve benchmark integrity: a run without ground truth is evidence-only and must never display Recall, MRR, nDCG, accuracy, or a winner score.

## Product decisions

- Data sources and ground-truth sources are global and independently selectable.
- The browser sends stable source IDs, never arbitrary filesystem paths.
- The server resolves IDs only from approved roots.
- Default WNS data remains available alongside uploaded projects.
- Existing files under `data/groundtruth/` and uploaded ground-truth CSV/XLSX files are selectable.
- Existing uploaded projects remain isolated on disk but appear in one global dataset selector.
- The NVIDIA lane uses the same source registry while remaining separate from the official modular benchmark matrix.
- Baseline and reranked NVIDIA results remain separately visible; a reranked run does not erase baseline evidence.

## Source registry

The dashboard API will expose a `source_catalog` containing two lists.

### Data sources

Each entry includes:

- `id`: opaque stable ID, such as `dataset:wns-default` or `project:<project_id>`
- `label`
- `kind`: `default` or `uploaded_project`
- `document_count`
- `chunk_count`
- readiness/status information

Filesystem paths are retained only in the server-side resolver and are not returned in the public catalog. The default WNS source resolves internally to `data/pdfs` and the canonical chunk workbook. Uploaded projects resolve internally from their manifest and search index under `data/user_projects/<project_id>/`.

### Ground-truth sources

Each entry includes:

- `id`: `groundtruth:none`, `groundtruth:repository:<name>`, or `groundtruth:upload:<id>`
- `label`
- `kind`: `none`, `repository`, or `uploaded`
- `row_count`
- validation state and detected query/answer/source columns

As with datasets, ground-truth filesystem paths remain private resolver state. The server discovers repository files under `data/groundtruth/` and validated uploaded files under a dedicated global ground-truth root. Existing project question files that satisfy the ground-truth schema may also be exposed as global entries.

## Upload flow

The upload form gains an explicit upload type:

1. **Document dataset** creates an isolated uploaded project and rebuilds its searchable chunks/workbook.
2. **Ground-truth file** accepts CSV/XLSX, validates that a query column exists, stores it in the global ground-truth repository, records row count/schema, and does not index it as document evidence.

The UI refreshes the catalog after upload and selects the newly created source.

## Standard Run Pipeline UI

Replace the free-text ground-truth path with:

- **Data source** selector showing label, document count, and chunk count.
- **Ground truth** selector starting with `None — evidence-only`, then repository and uploaded files with row counts.

Existing component selectors remain: chunker, embedding, vector DB, reranker, Top K, and limits.

### Standard execution modes

| Dataset | Ground truth | Mode |
|---|---|---|
| selected | selected | ingest → retrieve → rerank → evaluate |
| selected | none | ingest/retrieve typed queries → evidence-only results |
| missing | any | block with clear validation error |

The backend resolves source IDs before constructing commands. Default WNS data uses the existing complete pipeline. Uploaded projects pass their resolved workbook/chunks into the pipeline adapter. Evidence-only mode does not launch evaluation or grounding-score stages.

## NVIDIA RAG UI

Replace `Current data path` and free-text `Ground truth file` with the same global selectors:

- **NVIDIA data source**
- **NVIDIA ground truth**

The selected data source resolves to its real document directory for ingestion. The selected ground truth resolves to its CSV/XLSX path for benchmarking.

### NVIDIA actions

- **Ingest selected data** requires a dataset and records source ID, counts, collection, and real task status in `ingestion_latest.json`.
- **Test one query** works with any ingested collection and remains evidence-only.
- **Run NVIDIA benchmark** requires a non-`none` ground-truth selection.
- If ground truth is `none`, the benchmark button is disabled and the interface states `Evidence-only — quality metrics unavailable`.

## NVIDIA result visibility

The frontend must render available modes independently:

- baseline report/summary/details from `benchmark_baseline_*`
- reranked report/summary/details from `benchmark_reranked_*`
- compatibility/latest files as secondary aliases only

The completed baseline artifact with 500 evaluated and 500 successful queries must render even when no reranked artifact exists. Cards and tables identify the active mode and collection. A mode switch or adjacent sections may be used, but one mode must not hide or overwrite the other.

## API and safety

- `/api/results` includes `source_catalog` and both NVIDIA result modes.
- Run endpoints accept `dataset_id` and `groundtruth_id`.
- Resolver functions reject unknown IDs and paths outside approved roots.
- Existing raw path query parameters are removed from normal UI flows; compatibility may remain server-side only where required.
- Upload filenames are sanitized and stored under generated IDs.
- CSV/XLSX ground-truth validation occurs before catalog registration.

## Error and empty states

- No datasets: explain how to upload one.
- Invalid ground truth: show missing query-column/schema reason.
- Dataset without extractable chunks: block standard execution with its readiness reason.
- NVIDIA dataset not ingested into selected collection: smoke results show the real retrieval error; no quality score is inferred.
- Partial benchmark calls remain labeled partial and expose failed query counts.

## Testing

### Backend

- Catalog lists default WNS data, uploaded projects, repository ground truth, uploaded ground truth, and `none`.
- Counts come from real manifests/indexes/files.
- Unknown or traversal-style IDs are rejected.
- Ground-truth uploads are validated and are not indexed as documents.
- Command builders resolve selected sources correctly.
- No-ground-truth requests dispatch evidence-only behavior and never evaluation.
- NVIDIA baseline artifacts remain available without reranked artifacts.

### Frontend

- Both pages render dataset and ground-truth selectors.
- Labels include document/chunk/row counts.
- Selecting `none` changes mode copy, disables scored benchmark actions, and suppresses metric cards.
- Selected source IDs are sent to standard and NVIDIA endpoints.
- Completed NVIDIA baseline results render with 500/500 query evidence.

### Verification

- Run focused unit tests, then the full dashboard test suite.
- Start the dashboard and inspect `/api/results` schema.
- Capture desktop screenshots of Run Pipeline and NVIDIA RAG pages.
- Verify browser console has no errors.
- Verify baseline metrics appear and evidence-only mode shows no quality metrics.

## Out of scope

- Automatically claiming that arbitrary ground truth is semantically compatible with any dataset.
- Merging NVIDIA results into the official 180-combination matrix.
- Deleting or migrating existing uploaded projects.
- Fabricating metrics for evidence-only runs.
