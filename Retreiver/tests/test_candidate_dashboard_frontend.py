from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


def test_candidate_retrieval_selection_runs_isolated_modular_benchmark() -> None:
    assert "function runCandidateBenchmark" in APP
    assert "/api/run/selected?${p.toString()}" in APP
    assert "const candidate = isCandidateRetrievalMethod();" in APP
    assert "if (candidate) return runCandidateBenchmark();" in APP
    assert "Candidate lane: running selected retrieval/reranker combination" in APP


def test_candidate_retrieval_selection_does_not_use_complete_pipeline() -> None:
    start = APP.index("async function runCompletePipeline")
    end = APP.index("function runAction", start)
    body = APP[start:end]
    assert "const preflight = await runPreflight();" in body
    assert "if (candidate) return runCandidateBenchmark();" in body
    assert "/api/run/complete-pipeline" in body


def test_candidate_preflight_uses_candidate_only_endpoint() -> None:
    start = APP.index("async function runPreflight")
    end = APP.index("function parseLiveOutputRows", start)
    body = APP[start:end]
    assert "/api/run/preflight-candidate" in body


def test_candidate_all_selection_is_sent_as_all_not_expanded_duplicate_parameters() -> None:
    assert "const selectedChunkers = candidate ? selectedValues('runSheet') : expandedSelection('runSheet', activeDatasetSheets());" in APP


def test_candidate_completion_does_not_claim_official_quality_or_evidence_views() -> None:
    assert "pollRunJob(payload.job_id, {candidate: true})" in APP
    assert "Candidate artifacts" in APP
