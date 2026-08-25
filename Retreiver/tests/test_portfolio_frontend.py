from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
STYLES = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")


def test_overview_exposes_candidate_portfolio_and_adapter_workbench() -> None:
    for marker in (
        'id="candidateOperations"',
        'id="portfolioBatchList"',
        'id="adapterSlotList"',
        'id="portfolioOperatorToken"',
        'id="portfolioRunNext"',
        'id="portfolioStartRequired"',
        'id="portfolioConfiguredCount"',
        'id="portfolioExcludedCount"',
        'id="portfolioPromotionStatus"',
        'id="portfolioCombinationDetails"',
        'id="portfolioCombinationRows"',
    ):
        assert marker in HTML
    assert "Candidate only" in HTML
    assert "not accepted" in HTML.lower()


def test_frontend_loads_server_owned_adapter_and_portfolio_state() -> None:
    assert "async function loadCandidateOperations" in APP
    assert "api('/api/adapters')" in APP
    assert "api('/api/portfolio')" in APP
    assert "api('/api/portfolio/batches')" in APP
    assert "renderAdapterSlots" in APP
    assert "renderPortfolioBatches" in APP
    assert "slot_count" in APP
    assert "batch_count" in APP


def test_mutations_use_bearer_auth_without_persisting_operator_token() -> None:
    start = APP.index("function candidateOperatorHeaders")
    end = APP.index("async function loadCandidateOperations", start)
    body = APP[start:end]
    assert "Authorization: `Bearer ${token}`" in body
    assert "portfolioOperatorToken" in body
    assert "localStorage" not in body
    assert "sessionStorage" not in body
    assert "document.cookie" not in body


def test_start_required_and_batch_execution_send_only_server_known_identifiers() -> None:
    assert "candidateMutation('/api/adapters/start-required', {batch_id: batchId})" in APP
    assert "candidateMutation('/api/portfolio/run-batch', {batch_id: batchId})" in APP
    assert "candidateMutation('/api/portfolio/run-next', {})" in APP
    assert "service_ids" not in APP[APP.index("async function startRequiredAdapters"):APP.index("async function runPortfolioBatch")]


def test_candidate_operations_mobile_layout_does_not_require_page_overflow() -> None:
    assert ".candidate-operations" in STYLES
    assert ".adapter-slot-row" in STYLES
    assert ".portfolio-batch-row" in STYLES
    assert "@media(max-width:760px)" in STYLES
    assert "grid-template-columns:1fr" in STYLES
