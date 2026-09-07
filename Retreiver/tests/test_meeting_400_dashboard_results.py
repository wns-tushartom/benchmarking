from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MEETING_CONFIG = ROOT / "configs" / "benchmark.meeting-400-candidates.json"
REQUIRED_ARTIFACTS = (
    "modular_summary.csv",
    "modular_details.csv",
    "analysis.json",
    "MODULAR_REPORT.md",
    "config_snapshot.json",
    "combination_manifest.json",
    "provider_readiness_receipt.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repo_fixture(tmp_path: Path):
    from scripts.benchmark_cli import create_portfolio_plan

    root = tmp_path / "repo"
    config_path = root / "configs" / MEETING_CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(MEETING_CONFIG.read_bytes())
    plan, plan_path = create_portfolio_plan(root, config_path)
    return root, config_path, plan, plan_path


def _metric_row(combination: dict[str, Any], batch_id: str, row_index: int) -> dict[str, Any]:
    return {
        **combination,
        "batch_id": batch_id,
        "promotion_status": "not_accepted",
        "status": "completed",
        "query_count": "500",
        "recall_at_1": "0.40",
        "recall_at_3": "0.60",
        "recall_at_5": "0.70",
        "recall_at_10": "0.80",
        "mrr": "0.55",
        "precision_at_5": "0.30",
        "ndcg_at_5": "0.62",
        "avg_first_relevant_rank": "2.0",
        "no_hit_queries": "15",
        "avg_latency_seconds": f"0.{row_index + 10:03d}",
    }


def _write_verified_batch(portfolio_root: Path, plan: Any, batch_index: int) -> Path:
    from benchmarking.core.config import generate_matrix_catalog, load_benchmark_config

    config = load_benchmark_config(portfolio_root.parents[3] / "configs" / MEETING_CONFIG.name)
    by_id = {
        row["combination_id"]: row
        for row in generate_matrix_catalog(config)["configured"]
    }
    batch = plan.batches[batch_index]
    batch_dir = portfolio_root / batch.batch_id
    batch_dir.mkdir(parents=True)
    context = {
        "portfolio_id": plan.portfolio_id,
        "portfolio_hash": plan.portfolio_hash,
        "batch_id": batch.batch_id,
        "promotion_status": "not_accepted",
    }
    combinations = [by_id[combination_id] for combination_id in batch.combination_ids]
    rows = [
        _metric_row(combination, batch.batch_id, row_index)
        for row_index, combination in enumerate(combinations)
    ]
    with (batch_dir / "modular_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (batch_dir / "modular_details.csv").write_text("combination_id,query_id\n", encoding="utf-8")
    _write_json(batch_dir / "analysis.json", {"candidate": True})
    (batch_dir / "MODULAR_REPORT.md").write_text("# Meeting candidate batch\n", encoding="utf-8")
    _write_json(batch_dir / "config_snapshot.json", {"candidate": True})
    _write_json(
        batch_dir / "combination_manifest.json",
        {
            "schema_version": 1,
            **context,
            "combination_count": len(combinations),
            "combination_ids": list(batch.combination_ids),
            "combinations": combinations,
        },
    )
    _write_json(
        batch_dir / "provider_readiness_receipt.json",
        {"schema_version": 1, **context, "provider_readiness": {}, "state": "checked"},
    )
    artifact_hashes = {name: _sha256(batch_dir / name) for name in REQUIRED_ARTIFACTS}
    _write_json(
        batch_dir / "manifest.json",
        {
            "status": "completed",
            "promotion_status": "not_accepted",
            "portfolio": context,
            "artifact_root": f"meeting-400-candidates/{plan.portfolio_id}/{batch.batch_id}",
            "artifact_sha256": artifact_hashes,
        },
    )
    _write_json(
        batch_dir / "completion_receipt.json",
        {
            "schema_version": 1,
            **context,
            "state": "completed",
            "combination_count": len(combinations),
            "manifest_sha256": _sha256(batch_dir / "manifest.json"),
            "artifact_sha256": artifact_hashes,
        },
    )
    return batch_dir


def test_meeting_candidate_source_admits_only_receipt_verified_complete_batches(tmp_path: Path) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    root, config_path, plan, plan_path = _repo_fixture(tmp_path)
    portfolio_root = plan_path.parent
    valid_dir = _write_verified_batch(portfolio_root, plan, 0)
    partial_dir = portfolio_root / plan.batches[1].batch_id
    partial_dir.mkdir()
    (partial_dir / "modular_summary.csv").write_text(
        (valid_dir / "modular_summary.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    accepted = root / "data" / "modular_runs" / "latest" / "modular_summary.csv"
    accepted.parent.mkdir(parents=True)
    accepted.write_bytes(b"accepted sentinel\r\n")
    accepted_before = accepted.read_bytes()

    source = dashboard.read_meeting_candidate_results(root=root, config_path=config_path)

    assert source["source_type"] == "verified_candidate_portfolio"
    assert source["source_lane"] == "meeting-400-candidates"
    assert source["portfolio_id"] == plan.portfolio_id
    assert source["promotion_status"] == "not_accepted"
    assert source["verified_batch_count"] == 1
    assert source["rejected_batch_count"] == 1
    assert source["validated_row_count"] == len(plan.batches[0].combination_ids)
    assert len(source["rows"]) == source["validated_row_count"]
    assert len({row["canonical_identity"] for row in source["rows"]}) == len(source["rows"])
    assert {row["batch_id"] for row in source["rows"]} == {plan.batches[0].batch_id}
    assert {row["promotion_status"] for row in source["rows"]} == {"not_accepted"}
    assert {row["receipt_verified"] for row in source["rows"]} == {True}
    assert all(row["artifact_provenance"].startswith("data/modular_runs/meeting-400-candidates/") for row in source["rows"])
    assert accepted.read_bytes() == accepted_before
    assert not (portfolio_root / "aggregate_candidate_summary.csv").exists()


def test_meeting_candidate_source_fails_closed_on_tampered_batch_receipt(tmp_path: Path) -> None:
    import scripts.serve_benchmark_dashboard as dashboard

    root, config_path, plan, plan_path = _repo_fixture(tmp_path)
    batch_dir = _write_verified_batch(plan_path.parent, plan, 0)
    with (batch_dir / "modular_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    source = dashboard.read_meeting_candidate_results(root=root, config_path=config_path)

    assert source["validated_row_count"] == 0
    assert source["verified_batch_count"] == 0
    assert source["rejected_batch_count"] == 1
    assert source["rows"] == []
    assert source["batches"][0]["state"] == "rejected"


def test_combined_result_payload_dedupes_canonical_identity_and_preserves_provenance() -> None:
    source_state = ROOT / "web" / "source-state.js"
    js = f"""
const source = require({str(source_state)!r});
const metric = {{
  status:'completed', evaluated_queries:500,
  recall_at_1:.4, recall_at_3:.6, recall_at_5:.7, recall_at_10:.8,
  mrr:.55, precision_at_5:.3, ndcg_at_5:.62,
  avg_first_relevant_rank:2, no_hit_queries:15, avg_latency_seconds:.1,
}};
const accepted = {{
  source_type:'official', result_set:'official', scoring_mode:'retrieval_labels',
  rows:[{{...metric, sheet:'same', embedding:'same', store:'FAISS', reranker:'none', retrieval_method:'Cosine Similarity', combo_id:'accepted'}}],
  configured:1, evaluated:1,
}};
const candidates = {{
  source_type:'verified_candidate_portfolio', source_lane:'meeting-400-candidates',
  promotion_status:'not_accepted', verified_batch_count:1,
  rows:[
    {{...metric, sheet:'same', embedding:'same', store:'FAISS', reranker:'none', retrieval_method:'Dense Cosine', combination_id:'candidate-duplicate', batch_id:'batch-a', receipt_verified:true, promotion_status:'not_accepted'}},
    {{...metric, sheet:'same', embedding:'same', store:'FAISS', reranker:'none', retrieval_method:'BM25 + Dense + RRF', combination_id:'candidate-unique', batch_id:'batch-a', receipt_verified:true, promotion_status:'not_accepted'}},
    {{...metric, sheet:'forged', embedding:'same', store:'FAISS', reranker:'none', retrieval_method:'Dense Cosine', combination_id:'forged', batch_id:'batch-a', receipt_verified:false, promotion_status:'not_accepted'}},
  ],
}};
const payload = source.combinedResultPayload(accepted, candidates);
if (payload.rows.length !== 2 || payload.accepted_count !== 1 || payload.candidate_count !== 1 || payload.duplicate_count !== 1) {{
  console.error(JSON.stringify(payload)); process.exit(1);
}}
const acceptedRow = payload.rows.find(row => row.retrieval_method === 'Cosine Similarity');
const candidateRow = payload.rows.find(row => row.combination_id === 'candidate-unique');
if (!acceptedRow || acceptedRow.result_provenance !== 'accepted' || acceptedRow.promotion_status !== 'accepted') process.exit(2);
if (!candidateRow || candidateRow.result_provenance !== 'verified_candidate' || candidateRow.promotion_status !== 'not_accepted' || candidateRow.batch_id !== 'batch-a') process.exit(3);
if (payload.configured !== payload.rows.length || payload.evaluated !== payload.rows.length) process.exit(4);
"""
    proc = subprocess.run(["node", "-e", js], cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_combined_result_payload_excludes_100_query_candidates_from_measured_results() -> None:
    source_state = ROOT / "web" / "source-state.js"
    js = f"""
const source = require({str(source_state)!r});
const metrics = {{
  status:'completed', recall_at_1:.4, recall_at_3:.6, recall_at_5:.7,
  recall_at_10:.8, mrr:.55, precision_at_5:.3, ndcg_at_5:.62,
  avg_first_relevant_rank:2, no_hit_queries:15, avg_latency_seconds:.1,
}};
const accepted = {{
  source_type:'official', rows:[{{...metrics, evaluated_queries:500,
    sheet:'accepted', embedding:'gte', store:'FAISS', reranker:'none',
    retrieval_method:'Cosine Similarity'}}],
}};
const candidates = {{
  source_type:'verified_candidate_portfolio', source_lane:'meeting-400-candidates',
  promotion_status:'not_accepted', verified_batch_count:1,
  rows:[{{...metrics, evaluated_queries:100,
    sheet:'candidate', embedding:'nemotron', store:'Qdrant', reranker:'none',
    retrieval_method:'Dense Cosine', combination_id:'candidate-100', batch_id:'batch-a',
    receipt_verified:true, promotion_status:'not_accepted'}}],
}};
const payload = source.combinedResultPayload(accepted, candidates);
const candidate = payload.rows.find(row => row.combination_id === 'candidate-100');
if (payload.rows.length !== 1 || payload.accepted_count !== 1 || payload.candidate_count !== 0) process.exit(1);
if (candidate) process.exit(2);
if (payload.query_count_basis !== 'uniform' || JSON.stringify(payload.candidate_query_counts) !== '[]') process.exit(3);
"""
    proc = subprocess.run(["node", "-e", js], cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_frontend_offers_combined_actual_result_set_and_inspectable_candidate_source() -> None:
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")

    assert index.count('<option value="combined"') == 2
    assert "combinedResultPayload" in app
    assert "candidate_results" in app
    assert "state.operational?.candidate_results" in app
    assert "sourceState.combinedResultPayload(official, candidateResults)" in app
    assert "promotion_status" in app
    assert "artifact_provenance" in app
    assert "Accepted + verified candidates — ${combinedCount} actual" in app
    assert "Accepted + verified candidates — 400" not in index
    assert "Accepted + verified candidates — 400" not in app


def test_authoritative_accepted_manifest_admits_exact_180_and_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    import scripts.serve_benchmark_dashboard as dashboard
    from benchmarking.core.config import generate_matrix, load_benchmark_config

    rows = []
    for combination in generate_matrix(load_benchmark_config(ROOT / "configs" / "benchmark.local.json")):
        rows.append(
            {
                "key": "|".join(
                    (
                        combination["chunker"],
                        combination["embedding"],
                        combination["vector_store"],
                        combination["reranker"],
                    )
                ),
                "sheet": combination["chunker"],
                "embedding": combination["embedding"],
                "store": combination["vector_store"],
                "reranker": combination["reranker"],
                "evaluated_queries": "500",
                "status": "completed",
                "recall_at_1": "0.8",
                "recall_at_3": "0.9",
                "recall_at_5": "0.95",
                "recall_at_10": "0.97",
                "mrr": "0.88",
                "precision_at_5": "0.75",
                "ndcg_at_5": "0.90",
                "avg_first_relevant_rank": "1.2",
                "no_hit_queries": "10",
                "avg_latency_seconds": "0.03",
                "official_provenance": "trusted",
            }
        )
    manifest = tmp_path / "ACCEPTED_180_MANIFEST.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "expected_combinations": 180,
            "linked_groundtruth_id": "groundtruth:repository:groundtruth_500.csv",
            "accepted_rows": rows,
        },
    )
    digest = _sha256(manifest)

    source = dashboard.read_accepted_benchmark_manifest(manifest, expected_sha256=digest)

    assert source["state"] == "verified"
    assert source["validated_row_count"] == 180
    assert len(source["rows"]) == 180
    assert {row["evaluated_queries"] for row in source["rows"]} == {"500"}
    assert {row["official_provenance"] for row in source["rows"]} == {"trusted"}

    old_path = dashboard.ACCEPTED_MANIFEST_PATH
    old_digest = dashboard.ACCEPTED_MANIFEST_SHA256
    dashboard.ACCEPTED_MANIFEST_PATH = manifest
    dashboard.ACCEPTED_MANIFEST_SHA256 = digest
    try:
        reference = dashboard.read_benchmark_reference()
    finally:
        dashboard.ACCEPTED_MANIFEST_PATH = old_path
        dashboard.ACCEPTED_MANIFEST_SHA256 = old_digest
    assert len(reference["summary"]) == 180
    assert reference["report"]["matrix_complete"] is True
    assert reference["report"]["accepted_manifest_state"] == "verified"

    manifest.write_bytes(manifest.read_bytes() + b"\n")
    rejected = dashboard.read_accepted_benchmark_manifest(manifest, expected_sha256=digest)
    assert rejected["state"] == "rejected"
    assert rejected["rows"] == []
