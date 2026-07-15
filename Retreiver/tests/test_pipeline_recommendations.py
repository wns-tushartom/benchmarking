import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATIONS = ROOT / "web" / "recommendations.js"


def _run_node(body: str) -> Any:
    script = f"""
const R = require({json.dumps(str(RECOMMENDATIONS))});
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


def test_exports_primitives_completion_rules_and_pricing_constants_are_frozen():
    result = _run_node(
        """
const official = R.completed([
  {combo_id:'missing'}, {combo_id:'done',status:'completed'},
  {combo_id:'failed',status:'failed'}, null,
], 'official');
const uploaded = R.completed([
  {combo_id:'missing'}, {combo_id:'done',status:'completed'},
  {combo_id:'failed',status:'failed'}, null,
], 'uploaded_project');
console.log(JSON.stringify({
  finite:[R.finite('2.5'), R.finite(''), R.finite('not-a-number'), R.finite(Infinity)],
  positive:[R.positive('2.5'), R.positive(0), R.positive(-1), R.positive('not-a-number')],
  stable:R.stableKey({sheet:'chunk',embedding:'embed',store:'store'}),
  official:official.map(row => row.combo_id),
  uploaded:uploaded.map(row => row.combo_id),
  frozen:[Object.isFrozen(R), Object.isFrozen(R.PRICING), ...Object.values(R.PRICING).map(Object.isFrozen)],
  pricing:R.PRICING,
}));
"""
    )

    assert result["finite"] == [2.5, 0, None, None]
    assert result["positive"] == [2.5, None, None, None]
    assert result["stable"] == "chunk|embed|store|none"
    assert result["official"] == ["missing", "done"]
    assert result["uploaded"] == ["done"]
    assert result["frozen"] == [True, True, True, True]
    assert result["pricing"] == {
        "openai_text-embedding-3-large": {
            "unit": "million_input_tokens",
            "usd_rate": 0.13,
        },
        "Amazon Rerank v1": {"unit": "search_unit", "usd_rate": 0.001},
    }


def test_pricing_states_use_exact_canonical_ids_and_never_zero_missing_usage():
    result = _run_node(
        """
const noFee = R.pricingForRow({commercial_model_ids:[], measured_usage:{}});
const missing = R.pricingForRow({
  commercial_model_ids:['Amazon Rerank v1'], measured_usage:{},
});
const unknown = R.pricingForRow({
  commercial_model_ids:['custom Amazon Rerank v1 compatible'],
  measured_usage:{rerank_search_units:7,rerank_usage_scope:'combination'},
});
const amazon = R.pricingForRow({
  combo_id:'amazon', commercial_model_ids:['Amazon Rerank v1'],
  measured_usage:{rerank_search_units:7,rerank_usage_scope:'combination'},
});
console.log(JSON.stringify({noFee, missing, unknown, amazon}));
"""
    )

    assert result["noFee"]["state"] == "no_api_fee"
    assert result["noFee"]["total_cost_usd"] is None
    assert result["missing"]["state"] == "usage_missing"
    assert result["missing"]["total_cost_usd"] is None
    assert result["unknown"]["state"] == "pricing_unavailable"
    assert result["unknown"]["total_cost_usd"] is None
    assert result["amazon"]["state"] == "measured_usage"
    assert result["amazon"]["total_cost_usd"] == 0.007
    assert result["amazon"]["charges"] == [
        {
            "model_id": "Amazon Rerank v1",
            "unit": "search_unit",
            "usd_rate": 0.001,
            "usage": 7,
            "usage_scope": "combination",
            "cost_usd": 0.007,
        }
    ]


def test_shared_openai_usage_is_ledgered_once_and_never_becomes_a_row_total():
    result = _run_node(
        """
const sharedUsage = {
  embedding_input_tokens:1000000,
  embedding_usage_scope:'shared_embedding',
  embedding_usage_key:'chunk|openai',
};
const rows = [
  {combo_id:'a',chunker_id:'chunk',embedding_id:'openai_text-embedding-3-large',vector_store_id:'A',reranker_id:'local',commercial_model_ids:['openai_text-embedding-3-large'],measured_usage:sharedUsage},
  {combo_id:'b',chunker_id:'chunk',embedding_id:'openai_text-embedding-3-large',vector_store_id:'B',reranker_id:'local',commercial_model_ids:['openai_text-embedding-3-large'],measured_usage:sharedUsage},
];
console.log(JSON.stringify({
  rowPricing:rows.map(R.pricingForRow),
  ledger:R.pricingLedger({source_type:'uploaded_project',rows}),
}));
"""
    )

    assert [entry["state"] for entry in result["rowPricing"]] == [
        "measured_usage",
        "measured_usage",
    ]
    assert [entry["total_cost_usd"] for entry in result["rowPricing"]] == [None, None]
    assert all(
        charge["cost_usd"] is None
        for entry in result["rowPricing"]
        for charge in entry["charges"]
    )
    assert result["ledger"] == [
        {
            "key": "shared_embedding:chunk|openai",
            "model_id": "openai_text-embedding-3-large",
            "state": "measured_usage",
            "unit": "million_input_tokens",
            "usd_rate": 0.13,
            "usage": 1_000_000,
            "usage_scope": "shared_embedding",
            "cost_usd": 0.13,
            "combo_id": None,
        }
    ]


def test_official_winner_uses_existing_score_and_official_metric_labels():
    result = _run_node(
        """
const source = {source_type:'official',rows:[
  {combo_id:'high-recall',winner_score:0.8,recall_at_5:0.99,avg_latency_seconds:0.1,sheet:'z',embedding:'e',store:'s',reranker:'r'},
  {combo_id:'winner',winner_score:0.9,recall_at_5:0.1,avg_latency_seconds:0.4,sheet:'a',embedding:'e',store:'s',reranker:'r'},
  {combo_id:'failed',status:'failed',winner_score:1,recall_at_5:1,avg_latency_seconds:0.01,sheet:'f',embedding:'e',store:'s',reranker:'r'},
]};
const recommendations = R.recommendationsForSource(source);
console.log(JSON.stringify({
  labels:R.metricLabels(source),
  quality:recommendations.roles.quality.combo_id,
  speed:recommendations.roles.speed.combo_id,
  rows:recommendations.rows.map(row => row.combo_id),
}));
"""
    )

    assert result["labels"] == {
        "recall": "Recall@5",
        "mrr": "MRR",
        "ndcg": "nDCG@5",
        "latency": "Avg sec/query",
    }
    assert result["quality"] == "winner"
    assert result["speed"] == "high-recall"
    assert result["rows"] == ["winner", "high-recall", "failed"]


def test_project_quality_order_is_ndcg_mrr_recall_latency_then_stable_key():
    order = _run_node(
        """
const base = {status:'completed',commercial_model_ids:[],measured_usage:{}};
const rows = [
  {...base,combo_id:'stable-z',chunker_id:'z',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.8,mrr_at_k:.7,recall_at_k:.6,avg_query_latency_s:.2},
  {...base,combo_id:'recall',chunker_id:'d',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.8,mrr_at_k:.7,recall_at_k:.7,avg_query_latency_s:.3},
  {...base,combo_id:'mrr',chunker_id:'c',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.8,mrr_at_k:.8,recall_at_k:.1,avg_query_latency_s:.5},
  {...base,combo_id:'ndcg',chunker_id:'b',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.9,mrr_at_k:.1,recall_at_k:.1,avg_query_latency_s:1},
  {...base,combo_id:'latency',chunker_id:'y',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.8,mrr_at_k:.7,recall_at_k:.6,avg_query_latency_s:.1},
  {...base,combo_id:'stable-a',chunker_id:'a',embedding_id:'e',vector_store_id:'s',reranker_id:'r',ndcg_at_k:.8,mrr_at_k:.7,recall_at_k:.6,avg_query_latency_s:.2},
];
console.log(JSON.stringify(rows.sort(R.projectQualityCompare).map(row => row.combo_id)));
"""
    )

    assert order == ["ndcg", "mrr", "recall", "latency", "stable-a", "stable-z"]


def test_labelled_roles_exclude_failed_rows_and_fall_back_to_best_no_api_fee():
    result = _run_node(
        """
const labelled = {
  source_type:'uploaded_project',metric_k:10,scoring_mode:'retrieval_labels',
  rows:[
    {combo_id:'quality',status:'completed',chunker_id:'q',ndcg_at_k:.9,mrr_at_k:.8,recall_at_k:.95,avg_query_latency_s:.4,commercial_model_ids:[],measured_usage:{}},
    {combo_id:'speed',status:'completed',chunker_id:'s',ndcg_at_k:.7,mrr_at_k:.7,recall_at_k:.8,avg_query_latency_s:.1,commercial_model_ids:[],measured_usage:{}},
    {combo_id:'failed',status:'failed',chunker_id:'f',ndcg_at_k:1,mrr_at_k:1,recall_at_k:1,avg_query_latency_s:.01,commercial_model_ids:[],measured_usage:{}},
  ],
};
const recommendations=R.recommendationsForSource(labelled);
console.log(JSON.stringify({
  mode:recommendations.mode,
  roleKeys:Object.keys(recommendations.roles).sort(),
  quality:recommendations.roles.quality.combo_id,
  speed:recommendations.roles.speed.combo_id,
  value:recommendations.roles.value.combo_id,
  valueKind:recommendations.roles.value.recommendation_kind,
  completed:recommendations.completed.map(row => row.combo_id),
  failed:recommendations.failed.map(row => row.combo_id),
  badges:R.rowBadges(recommendations.roles.quality,recommendations.roles),
  labels:R.metricLabels(labelled),
}));
"""
    )

    assert result == {
        "mode": "labelled",
        "roleKeys": ["quality", "speed", "value"],
        "quality": "quality",
        "speed": "speed",
        "value": "quality",
        "valueKind": "no_api_fee",
        "completed": ["quality", "speed"],
        "failed": ["failed"],
        "badges": ["quality", "no_api_fee"],
        "labels": {
            "recall": "Recall@10",
            "mrr": "MRR@10",
            "ndcg": "nDCG@10",
            "latency": "Avg sec/query",
        },
    }


def test_null_project_metrics_do_not_create_quality_or_value_winners():
    result = _run_node(
        """
const source={source_type:'uploaded_project',metric_k:10,scoring_mode:'retrieval_labels',rows:[
  {combo_id:'unlabelled',status:'completed',chunker_id:'a',ndcg_at_k:null,mrr_at_k:null,recall_at_k:null,avg_query_latency_s:.2,commercial_model_ids:[],measured_usage:{}},
]};
const result=R.recommendationsForSource(source);
console.log(JSON.stringify({quality:result.roles.quality,value:result.roles.value,speed:result.roles.speed.combo_id}));
"""
    )

    assert result == {"quality": None, "value": None, "speed": "unlabelled"}


def test_value_role_uses_quality_per_comparable_combination_cost():
    result = _run_node(
        """
const source={source_type:'uploaded_project',metric_k:5,scoring_mode:'retrieval_labels',rows:[
  {combo_id:'better-ratio',status:'completed',chunker_id:'a',ndcg_at_k:.8,mrr_at_k:.8,recall_at_k:.8,avg_query_latency_s:.2,commercial_model_ids:['Amazon Rerank v1'],measured_usage:{rerank_search_units:100,rerank_usage_scope:'combination'}},
  {combo_id:'higher-quality',status:'completed',chunker_id:'b',ndcg_at_k:.9,mrr_at_k:.9,recall_at_k:.9,avg_query_latency_s:.2,commercial_model_ids:['Amazon Rerank v1'],measured_usage:{rerank_search_units:200,rerank_usage_scope:'combination'}},
]};
const result=R.recommendationsForSource(source);
console.log(JSON.stringify({id:result.roles.value.combo_id,kind:result.roles.value.recommendation_kind,badges:R.rowBadges(result.roles.value,result.roles)}));
"""
    )

    assert result == {"id": "better-ratio", "kind": "value", "badges": ["value"]}


def test_evidence_only_roles_never_expose_quality_or_value_and_handle_missing_latency():
    result = _run_node(
        """
const missingLatency={
  source_type:'uploaded_project',scoring_mode:'evidence_only',succeeded:1,failed:1,
  evidence_counts_by_combo:{done:4},
  rows:[
    {combo_id:'done',status:'completed',chunker_id:'b',query_count:3,avg_query_latency_s:null,evidence_count:4,commercial_model_ids:[],measured_usage:{}},
    {combo_id:'failed',status:'failed',chunker_id:'a',query_count:3,avg_query_latency_s:.01,evidence_count:0,commercial_model_ids:[],measured_usage:{}},
  ],
};
const withLatency={...missingLatency,failed:0,rows:[
  {...missingLatency.rows[0],combo_id:'slow',avg_query_latency_s:.4},
  {...missingLatency.rows[0],combo_id:'fast',avg_query_latency_s:.1},
]};
const missing=R.recommendationsForSource(missingLatency);
const measured=R.recommendationsForSource(withLatency);
console.log(JSON.stringify({
  mode:missing.mode,
  roleKeys:Object.keys(missing.roles).sort(),
  fastest:missing.roles.fastest,
  health:missing.roles.run_health,
  coverage:missing.roles.evidence_coverage,
  rows:missing.rows.map(row=>row.combo_id),
  measuredFastest:measured.roles.fastest.combo_id,
  forbidden:['quality','value'].some(key => key in missing.roles),
}));
"""
    )

    assert result == {
        "mode": "evidence_only",
        "roleKeys": ["evidence_coverage", "fastest", "run_health"],
        "fastest": {"state": "latency_not_recorded"},
        "health": {"state": "run_health", "succeeded": 1, "failed": 1, "total": 2},
        "coverage": {"state": "evidence_coverage", "query_count": 3, "evidence_count": 4},
        "rows": ["done", "failed"],
        "measuredFastest": "fast",
        "forbidden": False,
    }
