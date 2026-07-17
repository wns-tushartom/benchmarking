import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_STATE = ROOT / "web" / "source-state.js"
APP = ROOT / "web" / "app.js"
INDEX = ROOT / "web" / "index.html"


def _run_node(body: str) -> Any:
    script = f"""
const SourceState = require({json.dumps(str(SOURCE_STATE))});
{body}
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout.strip()
    return json.loads(output) if output else None


def test_initial_official_and_toggle_return_have_identical_canonical_rows_and_winner() -> None:
    result = _run_node(
        """
const completeMetrics = {
  evaluated_queries:500,recall_at_1:.7,recall_at_3:.8,recall_at_5:.9,recall_at_10:.95,
  mrr:.82,precision_at_5:.4,ndcg_at_5:.86,avg_first_relevant_rank:1.7,
  no_hit_queries:10,avg_latency_seconds:.03,official_provenance:'trusted',
};
const evaluation = {
  summary: [
    {sheet:'stale',embedding:'stale',store:'Qdrant',reranker:'none',winner_score:.99,status:'completed'},
    {sheet:'mixed',embedding:'mixed',store:'Qdrant',reranker:'qwen3_4b_rerank',winner_score:.98,status:'completed'},
  ],
  reranked: {summary: [
    {sheet:'base',embedding:'base',store:'FAISS',reranker:'baseline',winner_score:.4,status:'completed'},
  ]},
  benchmark_reference: {summary: [
    {...completeMetrics,sheet:'official',embedding:'bge',store:'FAISS',reranker:'bge_reranker_base',winner_score:.9,status:'completed'},
    {...completeMetrics,sheet:'official',embedding:'bge',store:'FAISS',reranker:'bge-reranker-base',winner_score:.1,status:'completed'},
    {...completeMetrics,sheet:'official',embedding:'amazon',store:'Qdrant',reranker:'amazon_rerank_v1',winner_score:.8,status:'completed'},
    {sheet:'failed',embedding:'x',store:'Qdrant',reranker:'qwen',winner_score:1,status:'failed'},
    {sheet:'nvidia',embedding:'x',store:'NVIDIA',reranker:'qwen',winner_score:1,status:'completed',source_type:'nvidia'},
    {sheet:'not-reranked',embedding:'x',store:'Qdrant',reranker:'none',winner_score:1,status:'completed'},
  ]},
};
evaluation.summary[0] = {...completeMetrics,...evaluation.summary[0]};
evaluation.reranked.summary[0] = {...completeMetrics,...evaluation.reranked.summary[0]};
evaluation.benchmark_reference.report = {
  expected_keys:['official|bge|FAISS|bge-reranker-base','official|amazon|Qdrant|Amazon Rerank v1'],
  baseline_expected_keys:['stale|stale|Qdrant|none','base|base|FAISS|none'],
};
const initial = SourceState.officialResultPayload(evaluation, {configured:2,evaluated:2});
const baseline = SourceState.baselineResultPayload(evaluation);
const returned = SourceState.officialResultPayload(evaluation, {configured:2,evaluated:2});
console.log(JSON.stringify({
  initialIds: initial.rows.map(row => row.combo_id),
  returnedIds: returned.rows.map(row => row.combo_id),
  initialWinner: initial.rows[0].combo_id,
  returnedWinner: returned.rows[0].combo_id,
  officialConfigured: initial.configured,
  baselineIds: baseline.rows.map(row => row.combo_id),
  officialRerankers: initial.rows.map(row => row.reranker),
}));
"""
    )

    assert result == {
        "initialIds": [
            "official|bge|FAISS|bge-reranker-base",
            "official|amazon|Qdrant|Amazon Rerank v1",
        ],
        "returnedIds": [
            "official|bge|FAISS|bge-reranker-base",
            "official|amazon|Qdrant|Amazon Rerank v1",
        ],
        "initialWinner": "official|bge|FAISS|bge-reranker-base",
        "returnedWinner": "official|bge|FAISS|bge-reranker-base",
        "officialConfigured": 2,
        "baselineIds": [
            "stale|stale|Qdrant|none",
            "base|base|FAISS|none",
        ],
        "officialRerankers": ["bge-reranker-base", "Amazon Rerank v1"],
    }


def test_active_source_generation_identity_and_evidence_only_payload_are_preserved() -> None:
    result = _run_node(
        """
const first = SourceState.createSelection({dataset_id:'project:one',groundtruth_id:'groundtruth:none',result_set_id:'run:1',source_type:'uploaded_project',scoring_mode:'evidence_only'});
const same = SourceState.advanceSelection(first, {dataset_id:'project:one'});
const second = SourceState.advanceSelection(same, {result_set_id:'run:2'});
const payload = SourceState.normalizeResultPayload({
  source_type:'uploaded_project',scoring_mode:'evidence_only',project_id:'one',run_id:'run:2',
  rows:[{combo_id:'uploaded-row',status:'completed',chunker_id:'c',embedding_id:'e',vector_store_id:'s',reranker_id:'r'}],
});
console.log(JSON.stringify({
  first, same, second,
  firstMatches: SourceState.matchesSelection(first, same),
  staleMatches: SourceState.matchesSelection(second, first),
  payload: {source_type:payload.source_type,scoring_mode:payload.scoring_mode,rowIds:payload.rows.map(row => row.combo_id)},
}));
"""
    )

    assert result["first"]["generation"] == 0
    assert result["same"]["generation"] == 0
    assert result["second"]["generation"] == 1
    assert result["firstMatches"] is True
    assert result["staleMatches"] is False
    assert result["payload"] == {
        "source_type": "uploaded_project",
        "scoring_mode": "evidence_only",
        "rowIds": ["uploaded-row"],
    }


def test_server_diagnostic_rows_are_reported_but_never_ranked() -> None:
    result = _run_node(
        """
const complete = {
  sheet:'complete',embedding:'gte',store:'FAISS',reranker:'bge-reranker-base',status:'completed',official_provenance:'trusted',
  evaluated_queries:500,recall_at_1:.7,recall_at_3:.8,recall_at_5:.9,recall_at_10:.95,
  mrr:.82,precision_at_5:.4,ndcg_at_5:.86,avg_first_relevant_rank:1.7,
  no_hit_queries:10,avg_latency_seconds:.03,winner_score:.88,
};
const incomplete = {
  sheet:'incomplete',embedding:'bge',store:'Qdrant',reranker:'qwen3_4b_rerank',status:'completed',official_provenance:'trusted',
  evaluated_queries:500,recall_at_1:.7,recall_at_3:.8,recall_at_5:.91,recall_at_10:.95,
  mrr:'',precision_at_5:.4,ndcg_at_5:null,avg_first_relevant_rank:1.8,
  no_hit_queries:12,avg_latency_seconds:'',admission_reason:'metrics_incomplete',
};
const expected = ['complete|gte|FAISS|bge-reranker-base'];
const payload = SourceState.officialResultPayload(
  {benchmark_reference:{summary:[complete],diagnostics:[incomplete],report:{expected_keys:expected}}},
  {configured:1,evaluated:1,expected_keys:expected},
);
let mutationBlocked = false;
try { payload.incomplete_rows[0].status = 'failed'; } catch (_) { mutationBlocked = true; }
const immutableStatus = payload.incomplete_rows[0].status;
console.log(JSON.stringify({
  ranked:payload.rows.map(row => row.combo_id),
  incomplete:payload.incomplete_rows.map(row => ({id:row.combo_id,missing:row.missing_metrics})),
  discovered:payload.discovered,
  evaluated:payload.evaluated,
  incompleteScore:SourceState.metricScore(incomplete),
  diagnosticFrozen:Object.isFrozen(payload.incomplete_rows) && Object.isFrozen(payload.incomplete_rows[0]),
  immutableStatus,
  mutationBlocked,
}));
"""
    )

    assert result == {
        "ranked": ["complete|gte|FAISS|bge-reranker-base"],
        "incomplete": [
            {
                "id": "incomplete|bge|Qdrant|Qwen3:4B Rerank",
                "missing": ["mrr", "ndcg_at_5", "avg_latency_seconds"],
            }
        ],
        "discovered": 1,
        "evaluated": 1,
        "incompleteScore": None,
        "diagnosticFrozen": True,
        "immutableStatus": "completed",
        "mutationBlocked": False,
    }


def test_baseline_admission_requires_trusted_complete_exact_official_rows() -> None:
    result = _run_node(
        """
const metrics = {
  evaluated_queries:5,recall_at_1:.5,recall_at_3:.6,recall_at_5:.7,recall_at_10:.8,
  mrr:.65,precision_at_5:.4,ndcg_at_5:.68,avg_first_relevant_rank:1.5,
  no_hit_queries:1,avg_latency_seconds:.03,
};
const valid = {...metrics,sheet:'c',embedding:'e',store:'FAISS',reranker:'none',status:'completed',official_provenance:'trusted',winner_score:.7};
const expected = ['c|e|FAISS|none'];
const evaluation = {summary:[
  {...valid,status:''},
  {...valid,official_provenance:'untrusted',winner_score:.99},
  {...valid,evaluated_queries:0},
  {...valid,mrr:''},
  {...valid,sheet:'wrong'},
  {...valid,source_type:'nvidia'},
  {...valid,source_type:'uploaded_project'},
  valid,
],benchmark_reference:{report:{baseline_expected_keys:expected}}};
const payload = SourceState.baselineResultPayload(evaluation);
console.log(JSON.stringify({
  rows:payload.rows.map(row => row.combo_id),
  diagnostics:payload.incomplete_rows.map(row => row.combo_id),
  winner:payload.rows[0]?.winner_score ?? null,
}));
"""
    )

    assert result == {
        "rows": ["c|e|FAISS|none"],
        "diagnostics": [
            "c|e|FAISS|none",
            "c|e|FAISS|none",
            "c|e|FAISS|none",
            "c|e|FAISS|none",
            "wrong|e|FAISS|none",
            "c|e|FAISS|none",
            "c|e|FAISS|none",
        ],
        "winner": 0.7,
    }


def test_browser_load_order_and_initial_render_use_one_canonical_transaction() -> None:
    index = INDEX.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")

    assert index.index("/source-state.js?") < index.index("/recommendations.js?") < index.index("/app.js?")
    assert "function renderCanonicalResultPayload(payload)" in app
    render_operational = app.split("function renderOperational()", 1)[1].split("function renderFiles", 1)[0]
    assert "officialRecommendationPayload('official')" in render_operational
    assert "renderCanonicalResultPayload(officialPayload)" in render_operational
    assert "renderEvaluation(op.evaluation)" not in render_operational


def test_official_status_and_exact_key_admission_fail_closed() -> None:
    result = _run_node(
        """
const metrics = {
  evaluated_queries:5,recall_at_1:.5,recall_at_3:.6,recall_at_5:.7,recall_at_10:.8,
  mrr:.65,precision_at_5:.4,ndcg_at_5:.68,avg_first_relevant_rank:1.5,
  no_hit_queries:1,avg_latency_seconds:.03,official_provenance:'trusted',
};
const expected = ['c|e|FAISS|bge-reranker-base'];
const payload = SourceState.officialResultPayload({benchmark_reference:{summary:[
  {...metrics,sheet:'c',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',status:''},
  {...metrics,sheet:'c',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',status:'unknown'},
  {...metrics,sheet:'wrong',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',status:'completed'},
  {...metrics,sheet:'c',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',status:'completed'},
  {...metrics,sheet:'extra',embedding:'e',store:'FAISS',reranker:'bge-reranker-base',status:'completed'},
]}}, {configured:1,evaluated:99,expected_keys:expected});
const normalized = SourceState.normalizeResultPayload({rows:[{combo_id:'blank'},{combo_id:'unknown',status:'unknown'}]});
console.log(JSON.stringify({
  rows: payload.rows.map(row => row.combo_id),
  diagnostics: payload.incomplete_rows.map(row => row.combo_id),
  matrix_complete: payload.matrix_complete,
  evaluated: payload.evaluated,
  statuses: normalized.rows.map(row => row.status ?? null),
}));
"""
    )

    assert result == {
        "rows": ["c|e|FAISS|bge-reranker-base"],
        "diagnostics": [
            "c|e|FAISS|bge-reranker-base",
            "c|e|FAISS|bge-reranker-base",
            "wrong|e|FAISS|bge-reranker-base",
            "extra|e|FAISS|bge-reranker-base",
        ],
        "matrix_complete": True,
        "evaluated": 1,
        "statuses": [None, "unknown"],
    }
