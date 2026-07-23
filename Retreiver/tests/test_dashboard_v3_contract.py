import csv
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app.js"
SOURCE_STATE = ROOT / "web" / "source-state.js"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _complete_metrics(query_count: str = "5") -> dict[str, str]:
    return {
        "status": "completed",
        "query_count": query_count,
        "recall_at_1": "0.50",
        "recall_at_3": "0.60",
        "recall_at_5": "0.70",
        "recall_at_10": "0.80",
        "mrr": "0.65",
        "precision_at_5": "0.40",
        "ndcg_at_5": "0.68",
        "avg_first_relevant_rank": "1.5",
        "no_hit_queries": "1",
        "avg_latency_seconds": "0.03",
    }


def _call_handler(dashboard, monkeypatch: pytest.MonkeyPatch, path: str):
    captured: list[tuple[object, int]] = []
    handler = object.__new__(dashboard.Handler)
    handler.path = path
    handler.send_json = lambda payload, status=200: captured.append((payload, status))
    dashboard.Handler.do_GET(handler)
    assert len(captured) == 1
    return captured[0]


def _stub_results_payload_dependencies(dashboard, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard, "read_csv", lambda _path: [])
    monkeypatch.setattr(dashboard, "sort_summary", lambda rows: rows)
    monkeypatch.setattr(dashboard, "read_ingestion_summaries", lambda: [])
    monkeypatch.setattr(dashboard, "read_retrieval_smokes", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "read_reranker_smokes", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "benchmark_detail_evidence", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "retrieval_smoke_count", lambda: 0)
    monkeypatch.setattr(dashboard, "reranker_smoke_count", lambda: 0)
    monkeypatch.setattr(dashboard, "read_hallucination", lambda: {})
    monkeypatch.setattr(dashboard, "read_pdf_audit", lambda: {})
    monkeypatch.setattr(dashboard, "read_document_repository", lambda: {})
    monkeypatch.setattr(dashboard, "publish_document_readiness", lambda _root, payload: payload)
    monkeypatch.setattr(dashboard, "list_user_projects", lambda: [])
    monkeypatch.setattr(dashboard, "read_nvidia_rag", lambda: {})
    monkeypatch.setattr(dashboard, "read_reranker_analysis", lambda: {})
    monkeypatch.setattr(dashboard, "read_snapshot", lambda: {})
    monkeypatch.setattr(dashboard, "live_service_health", lambda _snapshot: [])
    monkeypatch.setattr(dashboard, "benchmark_options", lambda: {"matrix_count": 1})
    monkeypatch.setattr(dashboard, "build_source_catalog", lambda _root: {"datasets": [], "groundtruth": []})


def test_canonical_backend_exposes_result_sources_project_results_source_catalog_and_pipeline_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    project_id = "alpha_0123456789abcdef0123456789abcdef"
    run_id = "run_0123456789abcdef0123456789abcdef"

    class FakeResults:
        def result_sources(self, *, official_configured: int, official_evaluated: int):
            return {
                "official": {
                    "source_type": "official",
                    "configured": official_configured,
                    "evaluated": official_evaluated,
                },
                "projects": [{"project_id": project_id}],
            }

        def project_runs(self, supplied_project_id: str):
            assert supplied_project_id == project_id
            return [{"project_id": project_id, "run_id": run_id}]

        def project_run_results(self, supplied_project_id: str, supplied_run_id: str):
            assert (supplied_project_id, supplied_run_id) == (project_id, run_id)
            return {"source_type": "uploaded_project", "project_id": project_id, "run_id": run_id, "rows": []}

        def project_run_evidence(
            self,
            supplied_project_id: str,
            supplied_run_id: str,
            combo_id: str,
            *,
            limit: int,
            offset: int,
        ):
            assert (supplied_project_id, supplied_run_id, combo_id) == (project_id, run_id, "combo")
            return {"project_id": project_id, "run_id": run_id, "combo_id": combo_id, "limit": limit, "offset": offset, "rows": []}

    evaluation = {
        "summary": [],
        "reranked": {"summary": []},
        "benchmark_reference": {"summary": [], "report": {"official_matrix_rows": 180}},
    }
    catalog = {"datasets": [{"id": "dataset:wns-default"}], "groundtruth": []}
    pipeline_state = [{"key": "c|e|s|r", "state": "not_run"}]

    monkeypatch.setattr(dashboard, "project_result_service", lambda: FakeResults())
    monkeypatch.setattr(dashboard, "_resolve_pipeline_state", lambda: (pipeline_state, 0, []))
    monkeypatch.setattr(dashboard, "read_evaluation", lambda _snapshot=None: evaluation)
    monkeypatch.setattr(dashboard, "official_evaluated_count", lambda _evaluation: 0)
    monkeypatch.setattr(dashboard, "read_csv", lambda _path: [])
    monkeypatch.setattr(dashboard, "sort_summary", lambda rows: rows)
    monkeypatch.setattr(dashboard, "read_ingestion_summaries", lambda: [])
    monkeypatch.setattr(dashboard, "read_retrieval_smokes", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "read_reranker_smokes", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "benchmark_detail_evidence", lambda **_kwargs: [])
    monkeypatch.setattr(dashboard, "retrieval_smoke_count", lambda: 0)
    monkeypatch.setattr(dashboard, "reranker_smoke_count", lambda: 0)
    monkeypatch.setattr(dashboard, "read_hallucination", lambda: {})
    monkeypatch.setattr(dashboard, "read_pdf_audit", lambda: {})
    monkeypatch.setattr(dashboard, "read_document_repository", lambda: {})
    monkeypatch.setattr(dashboard, "publish_document_readiness", lambda _root, payload: payload)
    monkeypatch.setattr(dashboard, "list_user_projects", lambda: [])
    monkeypatch.setattr(dashboard, "read_nvidia_rag", lambda: {})
    monkeypatch.setattr(dashboard, "read_reranker_analysis", lambda: {})
    monkeypatch.setattr(dashboard, "read_snapshot", lambda: {})
    monkeypatch.setattr(dashboard, "live_service_health", lambda _snapshot: [])
    monkeypatch.setattr(dashboard, "build_source_catalog", lambda _root: catalog)
    if hasattr(dashboard, "pipeline_state_rows"):
        monkeypatch.setattr(dashboard, "pipeline_state_rows", lambda _snapshot=None: pipeline_state)

    result_payload, status = _call_handler(dashboard, monkeypatch, "/api/results")
    assert status == 200
    assert result_payload["source_catalog"] == catalog
    assert result_payload["operational"]["pipeline_state"] == pipeline_state

    source_payload, status = _call_handler(dashboard, monkeypatch, "/api/result-sources")
    assert status == 200
    assert source_payload["official"] == {"source_type": "official", "configured": 180, "evaluated": 0}

    runs_payload, status = _call_handler(dashboard, monkeypatch, f"/api/project-runs?project_id={project_id}")
    assert status == 200
    assert runs_payload == [{"project_id": project_id, "run_id": run_id}]

    run_payload, status = _call_handler(
        dashboard,
        monkeypatch,
        f"/api/project-run-results?project_id={project_id}&run_id={run_id}",
    )
    assert status == 200
    assert run_payload["source_type"] == "uploaded_project"

    evidence_payload, status = _call_handler(
        dashboard,
        monkeypatch,
        f"/api/project-run-evidence?project_id={project_id}&run_id={run_id}&combo_id=combo&limit=2&offset=0",
    )
    assert status == 200
    assert evidence_payload["limit"] == 2


def test_official_options_use_existing_config_authority_and_define_exactly_180_pipelines() -> None:
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import load_benchmark_config

    config = load_benchmark_config(dashboard.CONFIG_PATH)
    matrix = config["matrix"]
    options = dashboard.benchmark_options()

    assert options["chunkers"] == matrix["chunkers"]
    assert options["embeddings"] == matrix["embeddings"]
    assert options["vector_stores"] == matrix["vector_stores"]
    assert options["rerankers"] == matrix["rerankers"]
    assert tuple(map(len, (options["chunkers"], options["embeddings"], options["vector_stores"], options["rerankers"]))) == (5, 3, 4, 3)
    assert options["matrix_count"] == 5 * 3 * 4 * 3 == 180
    assert len(dashboard.official_matrix_keys()) == 180


def test_results_request_uses_one_pipeline_snapshot_for_reference_and_operational_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    first_key = ("first", "embed", "FAISS", "rerank")
    second_key = ("second", "embed", "FAISS", "rerank")

    def state_row(key: tuple[str, str, str, str]) -> dict:
        return {
            "key": "|".join(key),
            "state": "complete",
            "status": "completed",
            "official_provenance": "trusted",
            "sheet": key[0],
            "embedding": key[1],
            "store": key[2],
            "reranker": key[3],
            "evaluated_queries": 5,
            **{name: float(value) for name, value in _complete_metrics().items() if name not in {"status", "query_count"}},
        }

    snapshots = [([state_row(first_key)], 0, []), ([state_row(second_key)], 0, [])]
    resolve_calls = 0

    def resolve():
        nonlocal resolve_calls
        snapshot = snapshots[min(resolve_calls, len(snapshots) - 1)]
        resolve_calls += 1
        return snapshot

    monkeypatch.setattr(dashboard, "_resolve_pipeline_state", resolve)
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {first_key})
    monkeypatch.setattr(dashboard, "read_evaluation_dir", lambda _path: {"summary": [], "details": [], "report": {}})
    _stub_results_payload_dependencies(dashboard, monkeypatch)

    payload, status = _call_handler(dashboard, monkeypatch, "/api/results")

    assert status == 200
    assert resolve_calls == 1
    reference_keys = [row["key"] for row in payload["operational"]["evaluation"]["benchmark_reference"]["summary"]]
    operational_keys = [row["key"] for row in payload["operational"]["pipeline_state"]]
    assert reference_keys == operational_keys == ["|".join(first_key)]


def test_pipeline_resolution_computes_manifest_selection_once_for_multiple_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    keys = {
        ("chunk-a", "embed", "FAISS", "rerank"),
        ("chunk-b", "embed", "FAISS", "rerank"),
    }
    path = tmp_path / "run-one" / "modular_summary.csv"
    _write_csv(path, [
        {"chunker": key[0], "embedding": key[1], "vector_store": key[2], "reranker": key[3], **_complete_metrics()}
        for key in sorted(keys)
    ])
    calls = 0

    def selected_keys(_manifest):
        nonlocal calls
        calls += 1
        return keys

    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: keys)
    monkeypatch.setattr(dashboard, "benchmark_reference_sources", lambda: [("run-one", path)])
    monkeypatch.setattr(dashboard, "official_artifact_provenance", lambda _path: (True, "trusted", {"run_id": "run-one"}))
    monkeypatch.setattr(dashboard, "manifest_selected_matrix_keys", selected_keys)

    rows, _skipped, _diagnostics = dashboard._resolve_pipeline_state()

    assert len(rows) == 2
    assert calls == 1


def test_benchmark_reference_reconciles_running_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    states = ["complete", "incomplete", "failed", "running", "not_run"]
    keys = {(state, "embed", "FAISS", "rerank") for state in states}
    rows = []
    for key in sorted(keys):
        row = {
            "key": "|".join(key),
            "sheet": key[0],
            "embedding": key[1],
            "store": key[2],
            "reranker": key[3],
            "state": key[0],
            "evaluated_queries": 5 if key[0] == "complete" else "",
        }
        if key[0] == "complete":
            row.update({
                name: float(value)
                for name, value in _complete_metrics().items()
                if name not in {"status", "query_count"}
            })
        rows.append(row)
    monkeypatch.setattr(dashboard, "_resolve_pipeline_state", lambda: (rows, 0, []))
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: keys)

    report = dashboard.read_benchmark_reference()["report"]

    assert report["running_rows"] == 1
    assert sum(report[name] for name in (
        "complete_metric_rows",
        "incomplete_metric_rows",
        "failed_rows",
        "running_rows",
        "not_run_rows",
    )) == report["official_matrix_rows"]


def test_pipeline_state_keeps_latest_complete_by_nanosecond_and_preserves_nonranking_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    complete_key = ("chunk-a", "embed-a", "FAISS", "rerank-a")
    failed_key = ("chunk-b", "embed-b", "Qdrant", "rerank-b")
    missing_key = ("chunk-c", "embed-c", "Weaviate", "rerank-c")
    base = {
        "chunker": complete_key[0],
        "embedding": complete_key[1],
        "vector_store": complete_key[2],
        "reranker": complete_key[3],
    }
    older_complete = tmp_path / "runs" / "z-older-complete" / "modular_summary.csv"
    newer_complete = tmp_path / "runs" / "a-newer-complete" / "modular_summary.csv"
    incomplete_retry = tmp_path / "runs" / "incomplete-retry" / "modular_summary.csv"
    failed = tmp_path / "runs" / "failed" / "modular_summary.csv"
    _write_csv(older_complete, [{**base, **_complete_metrics(), "recall_at_5": "0.61"}])
    _write_csv(newer_complete, [{**base, **_complete_metrics(), "recall_at_5": "0.72"}])
    _write_csv(incomplete_retry, [{**base, **_complete_metrics(), "mrr": ""}])
    _write_csv(failed, [{
        "chunker": failed_key[0],
        "embedding": failed_key[1],
        "vector_store": failed_key[2],
        "reranker": failed_key[3],
        "status": "failed",
        "error_code": "runner_failed",
    }])
    second = 1_800_000_000_000_000_000
    os.utime(older_complete, ns=(second + 100, second + 100))
    os.utime(newer_complete, ns=(second + 200, second + 200))
    os.utime(incomplete_retry, ns=(second + 300, second + 300))
    os.utime(failed, ns=(second + 400, second + 400))

    sources = [
        ("modular_run:z-older-complete", older_complete),
        ("modular_run:a-newer-complete", newer_complete),
        ("modular_run:incomplete-retry", incomplete_retry),
        ("modular_run:failed", failed),
    ]
    monkeypatch.setattr(dashboard, "official_matrix_keys", lambda: {complete_key, failed_key, missing_key})
    monkeypatch.setattr(dashboard, "benchmark_reference_sources", lambda: sources)
    monkeypatch.setattr(
        dashboard,
        "official_artifact_provenance",
        lambda path: (True, "trusted", {"run_id": path.parent.name}),
    )
    monkeypatch.setattr(dashboard, "manifest_selected_matrix_keys", lambda _manifest: None)

    rows = dashboard.pipeline_state_rows()
    by_key = {row["key"]: row for row in rows}

    assert len(rows) == 3
    complete = by_key["|".join(complete_key)]
    assert complete["state"] == "complete"
    assert complete["source"] == "modular_run:a-newer-complete"
    assert complete["artifact"] == dashboard.display_path(newer_complete)
    assert complete["recall_at_5"] == pytest.approx(0.72)
    assert complete["candidate_count"] == 3
    assert by_key["|".join(failed_key)]["state"] == "failed"
    assert by_key["|".join(missing_key)]["state"] == "not_run"


def test_frontend_quality_uses_complete_pipeline_state_and_keeps_other_states_visible() -> None:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const elements = {{}};
function element(id) {{
  return elements[id] ||= {{
    id, value: 'all', textContent: '', innerHTML: '', options: [],
    classList: {{toggle() {{}}, add() {{}}, remove() {{}}}},
    addEventListener() {{}}, querySelectorAll() {{return [];}}, setAttribute() {{}}, replaceChildren() {{}},
  }};
}}
const context = {{
  console,
  document: {{getElementById: element, querySelectorAll() {{return [];}}, addEventListener() {{}}, body: {{insertAdjacentHTML() {{}}}}}},
  window: {{}}, location: {{hash: '#metrics'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok: true, json: async () => ({{}}), text: async () => ''}}),
  setTimeout() {{}}, AbortController, URLSearchParams,
  PipelineRecommendations: {{
    recommendationsForSource(source) {{return {{mode: 'labelled', roles: {{}}, completed: source.rows || [], failed: [], rows: source.rows || []}};}},
    pricingLedger() {{return [];}}, metricLabels() {{return {{}};}}, rowBadges() {{return [];}}, pricingForRow() {{return {{}};}},
  }},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const app = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(app + `
const complete = {{
  key:'complete|embed|FAISS|rerank', state:'complete', status:'completed', official_provenance:'trusted',
  sheet:'complete', embedding:'embed', store:'FAISS', reranker:'rerank',
  evaluated_queries:500, recall_at_1:.5, recall_at_3:.6, recall_at_5:.70, recall_at_10:.8,
  mrr:.65, precision_at_5:.4, ndcg_at_5:.68, avg_first_relevant_rank:1.5,
  no_hit_queries:1, avg_latency_seconds:.03,
}};
const operationalRows = [
  complete,
  {{key:'incomplete|embed|Qdrant|rerank',state:'incomplete',sheet:'incomplete',embedding:'embed',store:'Qdrant',reranker:'rerank',evaluated_queries:5,recall_at_5:.99,mrr:.99,ndcg_at_5:.99,avg_latency_seconds:.01}},
  {{key:'failed|embed|PGVector|rerank',state:'failed',sheet:'failed',embedding:'embed',store:'PGVector',reranker:'rerank',evaluated_queries:5,recall_at_5:1,mrr:1,ndcg_at_5:1,avg_latency_seconds:.01}},
  {{key:'not-run|embed|Weaviate|rerank',state:'not_run',sheet:'not-run',embedding:'embed',store:'Weaviate',reranker:'rerank'}},
  {{key:'zero|embed|FAISS|rerank',state:'complete',sheet:'zero',embedding:'embed',store:'FAISS',reranker:'rerank',evaluated_queries:0,recall_at_5:1,mrr:1,ndcg_at_5:1,avg_latency_seconds:.01}},
];
state = {{operational: {{pipeline_state: operationalRows}}, files: [], options: {{}}}};
const evaluation = {{summary:[{{sheet:'raw-incomplete',embedding:'raw',store:'Qdrant',reranker:'none',evaluated_queries:500,recall_at_5:1,mrr:1,ndcg_at_5:1,avg_latency_seconds:.001}}]}};
const quality = evaluatedRows(evaluation);
renderEvaluation(evaluation, false);
const payload = officialRecommendationPayload('official');
globalThis.__contract = {{
  qualityKeys: quality.map(row => row.key),
  winner: document.getElementById('qualityInsight').innerHTML,
  incompleteStates: payload.incomplete_rows.map(row => row.state).sort(),
  payloadKeys: payload.rows.map(row => row.key),
}};
`, context);
const result = context.__contract;
if (JSON.stringify(result.qualityKeys) !== JSON.stringify(['complete|embed|FAISS|rerank'])) throw new Error(JSON.stringify(result));
if (!result.winner.includes('complete') || result.winner.includes('raw-incomplete')) throw new Error(JSON.stringify(result));
if (JSON.stringify(result.payloadKeys) !== JSON.stringify(['complete|embed|FAISS|rerank'])) throw new Error(JSON.stringify(result));
if (JSON.stringify(result.incompleteStates) !== JSON.stringify(['complete','failed','incomplete','not_run'])) throw new Error(JSON.stringify(result));
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_official_recommendations_use_the_same_composite_winner_and_evidence_as_metrics() -> None:
    recommendations = ROOT / "web" / "recommendations.js"
    script = f"""
const fs = require('fs');
const vm = require('vm');
function element(id) {{
  return {{
    id, value: 'all', textContent: '', innerHTML: '', options: [], selectedOptions: [],
    classList: {{toggle() {{}}, add() {{}}, remove() {{}}}}, addEventListener() {{}},
    querySelectorAll() {{return [];}}, setAttribute() {{}}, replaceChildren() {{}},
  }};
}}
const context = {{
  console,
  PipelineRecommendations: require({str(recommendations)!r}),
  document: {{getElementById: element, querySelectorAll() {{return [];}}, addEventListener() {{}}, body: {{insertAdjacentHTML() {{}}}}}},
  window: {{}}, location: {{hash: '#metrics'}}, history: {{replaceState() {{}}}},
  fetch: async () => ({{ok: true, json: async () => ({{}}), text: async () => ''}}),
  setTimeout() {{}}, AbortController, URLSearchParams,
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({str(SOURCE_STATE)!r}, 'utf8'), context);
const app = fs.readFileSync({str(APP)!r}, 'utf8').split('loadOptions().then(refresh)')[0];
vm.runInContext(app + `
function completeRow(sheet, evaluatedQueries, recall5, mrr, ndcg, latency) {{
  return {{
    key: sheet + '|embed|FAISS|rerank', state: 'complete', status: 'completed',
    official_provenance: 'trusted', sheet, embedding: 'embed', store: 'FAISS', reranker: 'rerank',
    evaluated_queries: evaluatedQueries, recall_at_1: .5, recall_at_3: .6, recall_at_5: recall5,
    recall_at_10: .95, mrr, precision_at_5: .4, ndcg_at_5: ndcg,
    avg_first_relevant_rank: 1.5, no_hit_queries: 1, avg_latency_seconds: latency,
    _quality_complete: true,
  }};
}}
const shortHighScore = completeRow('five-query-high-score', 5, .99, .99, .99, .001);
const canonicalWinner = completeRow('canonical-500-query', 500, .70, .80, .80, .10);
const evidence = [
  {{sheet:'canonical-500-query',embedding:'embed',store:'FAISS',reranker:'rerank',query_id:'q1'}},
  {{sheet:'canonical-500-query',embedding:'embed',store:'FAISS',reranker:'rerank',query_id:'q2'}},
];
state = {{operational: {{
  pipeline_state: [shortHighScore, canonicalWinner],
  evaluation: {{benchmark_reference: {{report: {{expected_keys: [shortHighScore.key, canonicalWinner.key]}}}}}},
  benchmark_detail_evidence: evidence,
}}, files: [], options: {{}}}};
recommendationState.official = {{configured: 2, evaluated: 2}};
const metricRows = evaluatedRows(state.operational.evaluation);
const payload = officialRecommendationPayload('official');
const recommendation = PipelineRecommendations.recommendationsForSource(payload);
globalThis.__contract = {{
  metricKeys: metricRows.map(row => row.key),
  payloadKeys: payload.rows.map(row => row.key),
  metricWinner: metricRows[0]?.key,
  recommendationWinner: recommendation.roles.quality?.combo_id,
  winnerScore: recommendation.roles.quality?.winner_score,
  expectedScore: DashboardSourceState.metricScore(canonicalWinner),
  evidenceCount: recommendation.roles.quality?.evidence_count,
  leakedPrivateKeys: payload.rows.flatMap(row => Object.keys(row).filter(key => key.startsWith('_'))),
}};
`, context);
const result = context.__contract;
const expectedKeys = ['canonical-500-query|embed|FAISS|rerank'];
if (JSON.stringify(result.metricKeys) !== JSON.stringify(expectedKeys)) throw new Error(JSON.stringify(result));
if (JSON.stringify(result.payloadKeys) !== JSON.stringify(expectedKeys)) throw new Error(JSON.stringify(result));
if (result.metricWinner !== expectedKeys[0]) throw new Error(JSON.stringify(result));
if (result.recommendationWinner !== result.metricWinner) throw new Error(JSON.stringify(result));
if (result.winnerScore !== result.expectedScore) throw new Error(JSON.stringify(result));
if (result.evidenceCount !== 2) throw new Error(JSON.stringify(result));
if (result.leakedPrivateKeys.length) throw new Error(JSON.stringify(result));
"""
    proc = subprocess.run(["node", "-e", script], text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_global_dataset_selection_owns_one_linked_groundtruth() -> None:
    app = APP.read_text(encoding="utf-8")
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    assert 'id="globalDataset"' in index
    assert 'id="globalGroundtruth"' in index
    assert "selectedDatasetRecord" in app
    assert "linkedGroundtruthId" in app
    assert "syncLinkedGroundtruth" in app
    assert "linked_groundtruth_id" in app
    assert "Evidence-only" in app or "evidence-only" in app


def test_mineru_project_contract_is_visible_and_fail_closed() -> None:
    parser = (ROOT / "source" / "services" / "document_parser.py").read_text(encoding="utf-8")
    documents = (ROOT / "source" / "services" / "project_documents.py").read_text(encoding="utf-8")
    server = (ROOT / "scripts" / "serve_benchmark_dashboard.py").read_text(encoding="utf-8")

    assert "find_mineru_cli" in parser
    assert "DocumentParserService" in documents
    assert "MinerU" in documents
    assert "resolve_source_pair" in server
    assert "allow_fallback=False" in documents
